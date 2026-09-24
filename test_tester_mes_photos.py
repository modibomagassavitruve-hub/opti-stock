import numpy as np
import pandas as pd
import torch

from tester_mes_photos import construire_detail, filtrer_min_images
from finetune_triplet import TeteProjection
from recall_clip import evaluer_recall


# ---------------------------------------------------------------- filtrer_min_images
def test_filtrer_min_images_exclut_les_montures_isolees():
    df = pd.DataFrame({
        "chemin": [f"img{i}.jpg" for i in range(6)],
        "label": ["a", "a", "a", "b", "b", "c"],  # c n'a qu'une photo
    })
    out = filtrer_min_images(df, min_images=2)
    assert set(out["label"]) == {"a", "b"}
    assert len(out) == 5


def test_filtrer_min_images_dataframe_vide_si_rien_de_suffisant():
    df = pd.DataFrame({"chemin": ["img0.jpg", "img1.jpg"], "label": ["a", "b"]})
    out = filtrer_min_images(df, min_images=2)
    assert out.empty


# ---------------------------------------------------------------- construire_detail
def test_construire_detail_colonnes_et_alignement():
    chemins = np.array(["p1_a.jpg", "p1_b.jpg", "p2_a.jpg", "p2_b.jpg"])
    labels = np.array(["p1", "p1", "p2", "p2"])
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype="float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    res = evaluer_recall(emb, labels, ks=(1, 5))

    detail = construire_detail(chemins, labels, res)
    assert list(detail.columns) == ["photo", "monture", "hit@5", "top1_photo", "top1_monture", "top1_similarite"]
    assert len(detail) == 4
    # chaque monture n'a qu'un seul autre exemplaire -> le top1 doit être son propre "jumeau"
    assert detail.loc[0, "top1_monture"] == "p1"
    assert detail.loc[2, "top1_monture"] == "p2"
    assert detail["hit@5"].all()  # avec k=(1,5) et 2 classes bien séparées, tout doit matcher


# ---------------------------------------------------------------- intégration légère avec TeteProjection
def test_tete_chargee_produit_des_embeddings_valides_pour_evaluer_recall():
    # vérifie que le format sauvegardé/chargé (state_dict) reste compatible bout en bout
    tete = TeteProjection(dim_entree=8, dim_cachee=6, dim_sortie=4)
    etat = tete.state_dict()

    tete2 = TeteProjection(dim_entree=8, dim_cachee=6, dim_sortie=4)
    tete2.load_state_dict(etat)

    x = torch.randn(6, 8)
    with torch.no_grad():
        y1, y2 = tete(x), tete2(x)
    assert torch.allclose(y1, y2)  # même poids -> même sortie
    assert torch.allclose(y2.norm(dim=-1), torch.ones(6), atol=1e-5)
