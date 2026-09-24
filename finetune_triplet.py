"""Fine-tuning léger de FashionCLIP par triplet loss (batch-hard), sur des embeddings gelés.

Étape 2 : lit le fichier produit par extraire_embeddings.py (embeddings FashionCLIP déjà
calculés et recadrés) et entraîne une petite tête de projection par-dessus. Le backbone
FashionCLIP reste gelé : on n'entraîne que la tête, sur des vecteurs déjà calculés, donc
c'est rapide même sur CPU (pas de nouveau passage dans les images).

Installation :
    pip install torch numpy pandas

Usage :
    python finetune_triplet.py --embeddings data/embeddings_fashionclip.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recall_clip import KS, evaluer_recall


class TeteProjection(nn.Module):
    """Petit MLP entraîné par-dessus l'embedding gelé du backbone. Sortie normalisée (norme 1),
    pour que la distance euclidienne utilisée dans le triplet loss reste cohérente avec la
    similarité cosinus utilisée par evaluer_recall.

    Le LayerNorm d'entrée n'est pas décoratif : les embeddings arrivent normalisés (norme 1),
    donc chaque coordonnée vaut ~1/sqrt(dim_entree) ≈ 0.036, soit l'ordre de grandeur des biais
    tirés par l'initialisation par défaut de nn.Linear. Ces biais, identiques pour toutes les
    entrées, écrasent alors le signal : sans LayerNorm, une tête même PAS entraînée ramène
    l'écart-type des similarités de 0.26 à 0.01, et l'arrêt anticipé sauvegarde cet état
    effondré avant que l'entraînement n'ait eu le temps d'en sortir."""

    def __init__(self, dim_entree: int, dim_cachee: int = 256, dim_sortie: int = 128):
        super().__init__()
        self.norme_entree = nn.LayerNorm(dim_entree)
        self.net = nn.Sequential(
            nn.Linear(dim_entree, dim_cachee),
            nn.ReLU(),
            nn.Linear(dim_cachee, dim_sortie),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(self.norme_entree(x)), dim=-1)


def encoder_labels(labels: np.ndarray) -> np.ndarray:
    """Convertit des labels texte en entiers consécutifs (même texte -> même entier)."""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse


def perte_supcon(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Perte contrastive supervisée (Khosla et al., 2020) : rapproche chaque ancre de TOUS ses
    positifs du batch en la repoussant de tous les autres, au lieu de ne retenir qu'un triplet.

    C'est ce qui la rend insensible à l'effondrement, contrairement à perte_triplet_batch_hard :
    si tous les vecteurs se confondent, le rapport de la softmax vaut 1/(N-1) et la perte reste
    haute avec un gradient qui sépare, là où la triplet loss atteint un plateau plat à la valeur
    de la marge et n'a plus de raison d'en sortir."""
    n = len(labels)
    diag = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = emb @ emb.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # stabilité numérique

    exp = torch.exp(sim).masked_fill(diag, 0.0)
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)

    memes = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag
    n_pos = memes.sum(dim=1)
    garde = n_pos > 0  # une ancre sans positif dans le batch n'apporte rien
    if not garde.any():
        return torch.zeros((), device=emb.device, requires_grad=True)
    return -((log_prob * memes).sum(dim=1)[garde] / n_pos[garde]).mean()


def perte_triplet_batch_hard(emb: torch.Tensor, labels: torch.Tensor, marge: float = 0.2) -> torch.Tensor:
    """Triplet loss "batch-hard" (Hermans et al., 2017) : pour chaque ancre du batch, on prend
    le positif le plus difficile (même classe, le plus loin) et le négatif le plus difficile
    (classe différente, le plus proche), plutôt que d'énumérer tous les triplets possibles.

    ATTENTION : s'effondre sur nos volumes (mesuré sur 78 montures comme sur un jeu synthétique,
    avec deux backbones différents). Tout rapprocher amène la perte au plateau `marge`, dont le
    gradient ne fait plus rien sortir. Conservée pour comparaison ; `entrainer` utilise
    perte_supcon."""
    dist = torch.cdist(emb, emb, p=2)
    memes = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    memes_hors_diag = memes & ~diag

    dist_pos = dist.masked_fill(~memes_hors_diag, -1.0)
    plus_dur_positif, _ = dist_pos.max(dim=1)

    dist_neg = dist.masked_fill(memes, float("inf"))
    plus_dur_negatif, _ = dist_neg.min(dim=1)

    return F.relu(marge + plus_dur_positif - plus_dur_negatif).mean()


def echantillonner_batch_pk(labels: np.ndarray, p: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Tire un batch "P classes x K images" : p classes au hasard parmi celles qui ont au moins
    2 images (pour qu'un positif existe), k images par classe (avec remise si besoin)."""
    assert k >= 2, "k doit être >= 2 pour qu'une paire positive existe dans chaque classe tirée"
    valeurs, comptes = np.unique(labels, return_counts=True)
    classes_dispo = valeurs[comptes >= 2]
    classes = rng.choice(classes_dispo, size=min(p, len(classes_dispo)), replace=False)
    idx = []
    for c in classes:
        pool = np.nonzero(labels == c)[0]
        idx.extend(rng.choice(pool, size=k, replace=len(pool) < k))
    return np.array(idx)


def _projeter(tete: TeteProjection, emb: np.ndarray, device: str) -> np.ndarray:
    tete.eval()
    with torch.no_grad():
        e = tete(torch.tensor(emb, dtype=torch.float32, device=device)).cpu().numpy()
    tete.train()
    return e


def evaluer(tete: TeteProjection, emb: np.ndarray, labels: np.ndarray, device: str) -> dict:
    return evaluer_recall(_projeter(tete, emb, device), labels)["recall"]


def etalement(emb: np.ndarray) -> float:
    """1 - similarité cosinus moyenne entre paires. Détecte l'effondrement : une tête dégénérée
    renvoie des vecteurs quasi identiques (similarité moyenne ~0.99, donc étalement ~0.006),
    alors que son recall peut rester trompeusement correct -- le classement garde un reste
    d'ordre même quand les écarts de similarité sont devenus insignifiants.

    On mesure la similarité MOYENNE et non son écart-type : des vecteurs parfaitement séparés
    (classes orthogonales) ont eux aussi un écart-type nul, et seraient signalés à tort."""
    if len(emb) < 2:
        return 1.0
    s = emb @ emb.T
    return float(1.0 - s[~np.eye(len(emb), dtype=bool)].mean())


# En dessous, les vecteurs sont trop proches les uns des autres pour porter une information
# exploitable (mesuré : tête effondrée ~0.006, backbone sain 0.42 à 0.72).
SEUIL_EFFONDREMENT = 0.05


def entrainer(emb_train, labels_train, emb_val, labels_val, dim_sortie=128, p=16, k=4,
              pas=800, lr=1e-3, temperature=0.07, patience=8, eval_tous_les=50,
              device="cpu", seed=0, verbose=True,
              tete_initiale: TeteProjection | None = None) -> tuple[TeteProjection, dict]:
    """`tete_initiale` permet de reprendre une tête déjà entraînée (pré-entraînement sur un jeu
    plus large avant spécialisation sur le stock réel) plutôt que de repartir de zéro.

    `emb_train` accepte aussi un tableau (V, N, D) : V encodages de la même photo sous des
    variations de prise de vue. Chaque élément du batch est alors tiré au hasard parmi ses
    variantes, ce qui apprend à la tête à les ignorer."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    emb_train = np.asarray(emb_train)
    if emb_train.ndim == 2:
        emb_train = emb_train[None]   # une seule vue : même chemin de code

    tete = (tete_initiale if tete_initiale is not None
            else TeteProjection(emb_train.shape[2], dim_sortie=dim_sortie)).to(device)
    opt = torch.optim.Adam(tete.parameters(), lr=lr)
    emb_train_t = torch.tensor(emb_train, dtype=torch.float32, device=device)
    labels_train_int = encoder_labels(labels_train)
    n_vues = emb_train_t.shape[0]

    meilleur_recall5, meilleur_etat, attente = -1.0, {k_: v.clone() for k_, v in tete.state_dict().items()}, 0
    for pas_i in range(1, pas + 1):
        idx = echantillonner_batch_pk(labels_train, p, k, rng)
        vues = rng.integers(n_vues, size=len(idx))
        batch_emb = tete(emb_train_t[vues, idx])
        batch_labels = torch.tensor(labels_train_int[idx], device=device)
        perte = perte_supcon(batch_emb, batch_labels, temperature)

        opt.zero_grad()
        perte.backward()
        opt.step()

        if pas_i % eval_tous_les == 0:
            projete = _projeter(tete, emb_val, device)
            recalls = evaluer_recall(projete, labels_val)["recall"]
            etal = etalement(projete)
            if verbose:
                print(f"  pas {pas_i:4d}  perte={perte.item():.4f}  "
                      f"recall@5(val)={recalls[5]:.3f}  étalement={etal:.4f}")
            # Un état effondré n'est jamais retenu, même si son recall est le meilleur vu :
            # ses similarités sont toutes quasi égales, donc inexploitables en production
            # (pas de seuil de confiance possible, classement instable).
            if etal >= SEUIL_EFFONDREMENT and recalls[5] > meilleur_recall5:
                meilleur_recall5 = recalls[5]
                meilleur_etat = {k_: v.clone() for k_, v in tete.state_dict().items()}
                attente = 0
            else:
                attente += 1
                if attente >= patience:
                    if verbose:
                        print(f"  arrêt anticipé (pas d'amélioration depuis {patience} évaluations)")
                    break

    if meilleur_recall5 < 0:
        raise RuntimeError(
            "Aucun état exploitable : la tête est restée effondrée (étalement < "
            f"{SEUIL_EFFONDREMENT}) pendant tout l'entraînement. Vérifier le LayerNorm d'entrée, "
            "le taux d'apprentissage, ou augmenter `pas`."
        )
    tete.load_state_dict(meilleur_etat)
    return tete, {"recall@5_val": meilleur_recall5}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--embeddings", type=Path, default=Path("data/embeddings_fashionclip.npz"))
    p.add_argument("--dim-sortie", type=int, default=128)
    p.add_argument("--p", type=int, default=16, help="classes par batch")
    p.add_argument("--k", type=int, default=4, help="images par classe et par batch")
    p.add_argument("--pas", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sortie", type=Path, default=Path("data/tete_finetuned.pt"))
    args = p.parse_args()

    d = np.load(args.embeddings, allow_pickle=True)
    emb, labels, splits = d["emb"], d["labels"], d["splits"]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device : {device}")

    def sous_ensemble(nom):
        m = splits == nom
        return emb[m], labels[m]

    emb_train, labels_train = sous_ensemble("train")
    emb_val, labels_val = sous_ensemble("val")
    emb_test, labels_test = sous_ensemble("test")
    print(f"train : {len(labels_train)} images, {len(set(labels_train))} produits")
    print(f"val   : {len(labels_val)} images, {len(set(labels_val))} produits")
    print(f"test  : {len(labels_test)} images, {len(set(labels_test))} produits")

    print("\nBaseline (FashionCLIP gelé, sans fine-tuning) sur le test :")
    base = evaluer_recall(emb_test, labels_test)["recall"]
    for k_ in KS:
        print(f"  recall@{k_} = {base[k_]:.3f}")

    print("\nEntraînement de la tête de projection...")
    tete, _ = entrainer(
        emb_train, labels_train, emb_val, labels_val,
        dim_sortie=args.dim_sortie, p=args.p, k=args.k, pas=args.pas,
        lr=args.lr, temperature=args.temperature, patience=args.patience, device=device, seed=args.seed,
    )

    print("\nAprès fine-tuning, sur le test :")
    apres = evaluer(tete, emb_test, labels_test, device)
    for k_ in KS:
        delta = apres[k_] - base[k_]
        signe = "+" if delta >= 0 else ""
        print(f"  recall@{k_} = {apres[k_]:.3f}  ({signe}{delta:.3f} vs baseline)")

    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tete.state_dict(), args.sortie)
    print(f"\nTête sauvegardée : {args.sortie}")


if __name__ == "__main__":
    main()
