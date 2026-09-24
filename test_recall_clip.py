import numpy as np

from recall_clip import evaluer_recall, filtrer, lister_images


def test_recall_parfait_sur_clusters_separes():
    rng = np.random.default_rng(0)
    centres = np.eye(4) * 10
    emb = np.vstack([centres[c] + rng.normal(0, 0.01, (5, 4)) for c in range(4)]).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.repeat(["a", "b", "c", "d"], 5)
    res = evaluer_recall(emb, labels)
    assert res["recall"][1] == 1.0 and res["recall"][5] == 1.0


def test_recall_proche_du_hasard_sur_bruit():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(600, 32)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.repeat(np.arange(120), 5)
    res = evaluer_recall(emb, labels)
    assert abs(res["recall"][5] - res["hasard"][5]) < 0.05


def test_formule_du_hasard():
    emb = np.eye(4, dtype="float32")
    res = evaluer_recall(emb, ["a", "a", "b", "b"], ks=(1,))
    assert abs(res["hasard"][1] - 1 / 3) < 1e-9


def test_la_requete_est_exclue_des_resultats():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype="float32")
    res = evaluer_recall(emb, ["a", "a", "b"], ks=(1,))
    assert res["top_idx"][0, 0] == 1 and res["top_idx"][1, 0] == 0


def test_lister_et_filtrer(tmp_path):
    for produit, nb in {"p1": 3, "p2": 2, "p3": 1}.items():
        (tmp_path / produit).mkdir()
        for i in range(nb):
            (tmp_path / produit / f"{i}.jpg").write_bytes(b"")
    (tmp_path / "p1" / "avis.txt").write_text("bien")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x.jpg").write_bytes(b"")
    (tmp_path / "orpheline.jpg").write_bytes(b"")

    df = lister_images(tmp_path)
    assert len(df) == 6 and set(df["label"]) == {"p1", "p2", "p3"}
    garde = filtrer(df, max_classes=10, max_par_classe=2)
    assert set(garde["label"]) == {"p1", "p2"} and len(garde) == 4
