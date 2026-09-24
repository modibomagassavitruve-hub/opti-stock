import pytest

from entrainer_tete import verifier_preentrainement
from recall_grid import BACKBONE_PROD


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
