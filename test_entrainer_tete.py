import numpy as np
import pytest

from entrainer_tete import calibrer_seuil, verifier_preentrainement
from recall_grid import BACKBONE_PROD


def test_seuil_calibre_atteint_la_precision_visee():
    """Les scores élevés sont justes, les bas sont faux : le seuil doit se poser à la frontière."""
    sims = np.concatenate([np.linspace(0.90, 0.99, 50), np.linspace(0.40, 0.60, 50)])
    justes = np.concatenate([np.ones(50, bool), np.zeros(50, bool)])
    seuil = calibrer_seuil(sims, justes, precision_visee=0.90)
    garde = sims >= seuil
    assert justes[garde].mean() >= 0.90   # la précision visée est tenue
    assert garde.sum() >= 50              # sans sacrifier la couverture


def test_seuil_calibre_prend_le_plus_bas_qui_convient():
    """À précision égale on veut la meilleure couverture, donc le seuil le plus bas."""
    sims = np.linspace(0.5, 1.0, 100)
    justes = np.ones(100, bool)       # tout est juste : le seuil doit descendre au minimum
    assert calibrer_seuil(sims, justes, precision_visee=0.90) == pytest.approx(0.5, abs=0.02)


def test_seuil_inatteignable_ne_declare_jamais_fiable():
    """Une précision parfaite sur 2 cas ne prouve rien. Si la cible est hors de portée, il vaut
    mieux ne jamais se déclarer fiable que de l'affirmer sur un échantillon minuscule."""
    sims = np.concatenate([[0.99, 0.98], np.linspace(0.3, 0.7, 98)])
    justes = np.concatenate([[True, True], np.random.default_rng(0).random(98) < 0.5])
    seuil = calibrer_seuil(sims, justes, precision_visee=0.95)
    assert seuil > 1.0, "aucune similarité cosinus ne peut dépasser 1"
    assert (sims >= seuil).sum() == 0


def test_preentrainement_du_bon_backbone_accepte():
    verifier_preentrainement(BACKBONE_PROD, 1024, 1024, "kaggle.npz")


def test_preentrainement_dun_autre_backbone_refuse():
    """Même dimension mais backbone différent : le test dimensionnel seul laisserait passer des
    espaces incompatibles, et le pré-entraînement dégraderait la tête sans rien signaler."""
    with pytest.raises(RuntimeError, match="autre_backbone"):
        verifier_preentrainement("autre_backbone", 1024, 1024, "kaggle.npz")


def test_preentrainement_de_dimension_incompatible_refuse():
    with pytest.raises(RuntimeError, match="512"):
        verifier_preentrainement(BACKBONE_PROD, 512, 1024, "kaggle.npz")


def test_preentrainement_sans_backbone_declare_refuse():
    """Un fichier produit avant que le backbone ne soit tracé : impossible de savoir comment il
    a été encodé, donc on refuse plutôt que de parier."""
    with pytest.raises(RuntimeError):
        verifier_preentrainement(None, 1024, 1024, "ancien.npz")
