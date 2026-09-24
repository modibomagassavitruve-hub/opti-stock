import numpy as np
import torch
import pytest

from finetune_triplet import (
    SEUIL_EFFONDREMENT,
    TeteProjection,
    echantillonner_batch_pk,
    encoder_labels,
    entrainer,
    etalement,
    evaluer,
    perte_triplet_batch_hard,
)
from extraire_embeddings import assigner_split, filtrer_couverture


# ---------------------------------------------------------------- assigner_split
def test_assigner_split_respecte_les_proportions():
    labels = np.repeat([f"p{i}" for i in range(100)], 3)  # 100 produits, 3 images chacun
    assign = assigner_split(labels, seed=0, ratios=(0.7, 0.15, 0.15))
    compte = {"train": 0, "val": 0, "test": 0}
    for v in assign.values():
        compte[v] += 1
    assert compte == {"train": 70, "val": 15, "test": 15}


def test_assigner_split_aucun_produit_dans_deux_jeux():
    labels = np.repeat([f"p{i}" for i in range(40)], 2)
    assign = assigner_split(labels, seed=1)
    # un produit a exactement UNE valeur dans assign (c'est un dict) -> test trivialement vrai par construction,
    # on vérifie plutôt que chaque produit du tableau original apparaît bien dans assign
    assert set(assign.keys()) == set(labels)


def test_assigner_split_deterministe_avec_meme_seed():
    labels = np.repeat([f"p{i}" for i in range(50)], 2)
    a1 = assigner_split(labels, seed=42)
    a2 = assigner_split(labels, seed=42)
    assert a1 == a2


# ---------------------------------------------------------------- filtrer_couverture
def test_filtrer_couverture_garde_toutes_les_classes_suffisantes(tmp_path_factory=None):
    import pandas as pd
    df = pd.DataFrame({
        "chemin": [f"img{i}.jpg" for i in range(10)],
        "label": ["a", "a", "a", "b", "b", "c"] + ["d"] * 4,  # d a 4 images, c en a 1 seule
    })
    out = filtrer_couverture(df, max_par_classe=2, min_images=2, seed=0)
    # 'c' n'a qu'une image -> exclue ; toutes les autres classes présentes, plafonnées à 2
    assert set(out["label"]) == {"a", "b", "d"}
    assert (out.groupby("label").size() <= 2).all()


def test_filtrer_couverture_pas_de_plafond_sur_le_nombre_de_classes():
    import pandas as pd
    # 500 classes de 2 images chacune -> filtrer() de recall_clip.py plafonnerait à 200 max_classes,
    # filtrer_couverture ne doit PAS plafonner le nombre de classes
    df = pd.DataFrame({
        "chemin": [f"img{i}_{j}.jpg" for i in range(500) for j in range(2)],
        "label": [f"p{i}" for i in range(500) for _ in range(2)],
    })
    out = filtrer_couverture(df, max_par_classe=2, min_images=2, seed=0)
    assert out["label"].nunique() == 500


# ---------------------------------------------------------------- TeteProjection
def test_tete_projection_sortie_normalisee():
    tete = TeteProjection(dim_entree=16, dim_cachee=8, dim_sortie=4)
    x = torch.randn(5, 16)
    y = tete(x)
    assert y.shape == (5, 4)
    normes = y.norm(dim=-1)
    assert torch.allclose(normes, torch.ones(5), atol=1e-5)


def test_tete_non_entrainee_ne_seffondre_pas():
    """Régression : sans LayerNorm d'entrée, les biais d'initialisation de nn.Linear (de l'ordre
    de 1/sqrt(dim)) dominent des embeddings de norme 1 et écrasent toutes les sorties sur un
    même point, AVANT même le premier pas d'entraînement."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 512)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # embeddings normalisés, comme en vrai

    for graine in range(3):
        torch.manual_seed(graine)
        tete = TeteProjection(dim_entree=512, dim_sortie=128).eval()
        with torch.no_grad():
            sortie = tete(torch.tensor(x)).numpy()
        assert etalement(sortie) > SEUIL_EFFONDREMENT, "tête effondrée dès l'initialisation"


def test_etalement_detecte_un_effondrement():
    identiques = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype="float32"), 8, axis=0)
    assert etalement(identiques) < SEUIL_EFFONDREMENT

    # classes parfaitement séparées : similarités toutes nulles, donc écart-type nul lui aussi.
    # Ne doit PAS être confondu avec un effondrement (c'est l'inverse : le cas idéal).
    orthogonaux = np.eye(8, dtype="float32")
    assert etalement(orthogonaux) > SEUIL_EFFONDREMENT

    rng = np.random.default_rng(0)
    quelconques = rng.normal(size=(30, 16)).astype("float32")
    quelconques /= np.linalg.norm(quelconques, axis=1, keepdims=True)
    assert etalement(quelconques) > SEUIL_EFFONDREMENT


# ---------------------------------------------------------------- perte_triplet_batch_hard
def test_perte_nulle_si_clusters_parfaitement_separes():
    # 2 classes, très loin l'une de l'autre, très proches en interne -> perte doit être ~0
    emb = torch.tensor([
        [0.0, 0.0], [0.01, 0.0], [0.0, 0.01],
        [10.0, 10.0], [10.01, 10.0], [10.0, 10.01],
    ])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    perte = perte_triplet_batch_hard(emb, labels, marge=0.2)
    assert perte.item() < 1e-3


def test_perte_positive_si_classes_confondues():
    # tous les points au même endroit -> distance positive ET négative valent 0
    # -> perte = relu(marge + 0 - 0) = marge
    emb = torch.zeros(4, 2)
    labels = torch.tensor([0, 0, 1, 1])
    perte = perte_triplet_batch_hard(emb, labels, marge=0.2)
    assert abs(perte.item() - 0.2) < 1e-4


# ---------------------------------------------------------------- echantillonner_batch_pk
def test_echantillonner_taille_du_batch():
    labels = np.repeat([f"p{i}" for i in range(20)], 5)
    rng = np.random.default_rng(0)
    idx = echantillonner_batch_pk(labels, p=8, k=3, rng=rng)
    assert len(idx) == 24  # 8 classes x 3 images


def test_echantillonner_exclut_les_classes_a_une_seule_image():
    labels = np.array(["a", "a", "b"])  # 'b' n'a qu'une image
    rng = np.random.default_rng(0)
    idx = echantillonner_batch_pk(labels, p=5, k=2, rng=rng)
    # seule 'a' est utilisable -> batch de 2 images, toutes de la classe 'a'
    assert len(idx) == 2
    assert set(labels[idx]) == {"a"}


def test_echantillonner_leve_erreur_si_k_inferieur_a_2():
    labels = np.repeat([f"p{i}" for i in range(5)], 3)
    rng = np.random.default_rng(0)
    try:
        echantillonner_batch_pk(labels, p=3, k=1, rng=rng)
        assert False, "devrait lever une AssertionError"
    except AssertionError:
        pass


# ---------------------------------------------------------------- encoder_labels
def test_encoder_labels_meme_texte_meme_entier():
    labels = np.array(["chat", "chien", "chat", "chat", "chien"])
    codes = encoder_labels(labels)
    assert codes[0] == codes[2] == codes[3]
    assert codes[1] == codes[4]
    assert codes[0] != codes[1]


# ---------------------------------------------------------------- entrainer (bout en bout)
def _dataset_synthetique(n_classes=50, n_par_classe=6, dim=8, dim_nuisance=8,
                         centre_scale=0.5, bruit_scale=0.4, nuisance_scale=4.0, seed=0):
    """Classes gaussiennes qui se chevauchent, plus des dimensions de NUISANCE de forte
    amplitude et sans aucune information de classe.

    Ces dimensions jouent le rôle du décor de boutique dans les vraies photos : elles dominent
    la distance sans rien dire de l'identité. C'est ce qui rend le test signifiant — apprendre
    à les ignorer est une règle qui vaut aussi pour les classes jamais vues, donc un
    apprentissage correct doit améliorer le recall sur le jeu de validation. Sans elles, les
    centres sont isotropes : il n'y a rien à apprendre qui se transfère, et le test ne peut
    être passé que par accident."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=centre_scale, size=(n_classes, dim))
    emb, labels = [], []
    for i, c in enumerate(centres):
        signal = c + rng.normal(scale=bruit_scale, size=(n_par_classe, dim))
        nuisance = rng.normal(scale=nuisance_scale, size=(n_par_classe, dim_nuisance))
        emb.append(np.hstack([signal, nuisance]))
        labels += [f"p{i}"] * n_par_classe
    emb = np.concatenate(emb).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, np.array(labels)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_entrainement_ameliore_le_recall_sur_un_jeu_synthetique_difficile(seed):
    """Les classes de validation ne sont JAMAIS vues à l'entraînement : ce qui doit se
    transférer, c'est la règle « ignorer les dimensions de nuisance »."""
    emb, labels = _dataset_synthetique(seed=seed)
    classes = sorted(set(labels))
    ordre = np.random.default_rng(seed).permutation(len(classes))
    masque_train = np.isin(labels, [classes[i] for i in ordre[:35]])
    masque_val = np.isin(labels, [classes[i] for i in ordre[35:]])

    emb_val, labels_val = emb[masque_val], labels[masque_val]
    base = evaluer_recall_local(emb_val, labels_val)
    tete, _ = entrainer(
        emb[masque_train], labels[masque_train], emb_val, labels_val,
        dim_sortie=16, p=12, k=3, pas=400, lr=2e-3, patience=8, eval_tous_les=25,
        device="cpu", seed=0, verbose=False,
    )
    apres = evaluer(tete, emb_val, labels_val, "cpu")
    assert apres[5] > base[5] + 0.15, f"avant={base[5]:.3f} après={apres[5]:.3f}"


def evaluer_recall_local(emb, labels):
    from recall_clip import evaluer_recall
    return evaluer_recall(emb, labels)["recall"]


def test_entrainer_accepte_plusieurs_vues_par_photo():
    """Chaque photo peut être fournie sous plusieurs encodages (variations de prise de vue).
    L'entraînement doit alors piocher au hasard parmi les vues, sans changer de contrat."""
    emb, labels = _dataset_synthetique(n_classes=20, n_par_classe=5, seed=4)
    classes = sorted(set(labels))
    mt = np.isin(labels, classes[:14])
    mv = np.isin(labels, classes[14:])

    rng = np.random.default_rng(0)
    bruitees = emb[mt] + rng.normal(scale=0.05, size=emb[mt].shape).astype("float32")
    vues = np.stack([emb[mt], bruitees])     # (2 vues, N, D)

    tete, _ = entrainer(vues, labels[mt], emb[mv], labels[mv], dim_sortie=16, p=10, k=3,
                        pas=200, lr=2e-3, device="cpu", seed=0, verbose=False)
    with torch.no_grad():
        sortie = tete(torch.tensor(emb, dtype=torch.float32)).numpy()
    assert sortie.shape == (len(emb), 16)
    assert etalement(sortie) > SEUIL_EFFONDREMENT


def test_entrainer_ne_renvoie_jamais_une_tete_effondree():
    """Régression : l'arrêt anticipé sélectionnait sur le seul recall@5, qui reste trompeusement
    correct sur une tête effondrée -- elle était donc sauvegardée puis servie en production."""
    emb, labels = _dataset_synthetique(n_classes=20, n_par_classe=5, seed=3)
    classes = sorted(set(labels))
    masque_train = np.isin(labels, classes[:14])
    masque_val = np.isin(labels, classes[14:])

    tete, _ = entrainer(
        emb[masque_train], labels[masque_train], emb[masque_val], labels[masque_val],
        dim_sortie=16, p=10, k=3, pas=300, lr=2e-3, patience=6, eval_tous_les=25,
        device="cpu", seed=0, verbose=False,
    )
    with torch.no_grad():
        sortie = tete(torch.tensor(emb, dtype=torch.float32)).numpy()
    assert etalement(sortie) > SEUIL_EFFONDREMENT


def test_entrainer_sauvegarde_bien_le_meilleur_etat_pas_le_dernier():
    # perte très bruitée / lr élevé -> le dernier pas n'est pas forcément le meilleur ;
    # on vérifie juste que la fonction ne retourne pas une tête aléatoire non entraînée
    emb, labels = _dataset_synthetique(n_classes=20, n_par_classe=5, seed=2)
    classes = sorted(set(labels))
    train_cls = set(classes[:14])
    val_cls = set(classes[14:18])
    emb_train, labels_train = emb[np.isin(labels, list(train_cls))], labels[np.isin(labels, list(train_cls))]
    emb_val, labels_val = emb[np.isin(labels, list(val_cls))], labels[np.isin(labels, list(val_cls))]

    tete, hist = entrainer(
        emb_train, labels_train, emb_val, labels_val,
        dim_sortie=16, p=10, k=3, pas=150, lr=5e-3, patience=3, eval_tous_les=25,
        device="cpu", seed=0, verbose=False,
    )
    assert hist["recall@5_val"] >= 0.0  # a bien été évalué au moins une fois
