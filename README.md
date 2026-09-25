# Opti-Stock

Identifier une monture d'optique à partir d'une photo, et pointer le stock en inventaire.

```bash
uvicorn app:app --port 8000     # puis ouvrir http://localhost:8000
```

---

## Utiliser

**Identifier une monture.** Photographier, choisir la marque si elle est connue, valider la
bonne proposition d'un clic. Préciser la marque fait passer le recall@5 de 0.79 à 0.86.

**Inventaire.** Démarrer une session, parcourir les rayons, identifier chaque monture : la
validation la pointe automatiquement. À la fin, la liste des montures **non trouvées** —
disparues, vendues sans saisie, ou rangées ailleurs.

**Fiabilité constatée.** Chaque validation alimente un journal. C'est la seule mesure prise
dans vos conditions réelles ; tous les autres chiffres de ce document viennent de photos
prises en une seule séance, et sont donc optimistes.

## Ajouter des montures

```bash
# 1. déposer les photos dans data/mes_montures/<identifiant>/
# 2. renseigner la marque dans data/montures.csv
python mettre_a_jour.py
# 3. redémarrer l'API
```

Le catalogue et la tête de projection sont **indissociables**. Les reconstruire séparément ne
lève aucune erreur mais dégrade silencieusement les résultats : `mettre_a_jour.py` enchaîne les
deux, et l'API avertit au démarrage si la tête est plus ancienne que le catalogue.

## Déployer

```bash
docker build -t opti-stock .
docker run -p 8000:8000 -v opti_journal:/journal opti-stock
```

Le volume est indispensable : sans lui le journal disparaît à chaque redéploiement, et c'est
lui qui porte la mesure de fiabilité réelle.

---

## Décisions mesurées

Elles sont contre-intuitives, et chacune a coûté une mesure. Les défaire sans remesurer ferait
régresser le projet.

| Décision | Mesure |
|---|---|
| **Ne pas recadrer** sur la monture détectée | recall@5 0.33 avec, **0.80** sans |
| Backbone **FashionCLIP** | 0.841 contre 0.807 pour DINOv2-large |
| **Augmentation** à l'entraînement | 0.786 → **0.847** sur requête dégradée |
| Correction **EXIF** des photos | sans elle, tout est traité couché |
| Pas de pré-entraînement Kaggle | dégrade le résultat |

Deux pièges à connaître :

**Le classement des backbones dépend du prétraitement.** DINOv2 gagnait sur images recadrées,
FashionCLIP gagne sans recadrage. Comparer des backbones sans fixer le prétraitement n'a pas
de sens.

**Le recadrage échouait sur 29 % des photos** — bandes très allongées, fragments minuscules.
Le détecteur segmente la monture sans les verres : sur des lunettes de soleil à monture fine,
il ne reste qu'un liseré. L'image entière, elle, contient toujours la monture.

## Mesurer

```bash
python evaluer.py --par-taille --seuils
```

Le protocole évite trois pièges qui donnent tous des chiffres flatteurs, et qui m'ont eu
successivement : l'auto-appariement (interroger avec une photo du catalogue), la fuite par
l'entraînement (recall@1 de 0.77 avec, 0.26 sans), et le raccourci du décor (le modèle
reconnaît l'endroit du présentoir, pas la monture). Détails dans `evaluer.py`.

État actuel, sur 115 montures : **recall@5 = 0.861**, 0.855 avec filtre par marque. L'API
répond dans 74 % des cas et a raison 91 % du temps. Le seuil de confiance est recalibré
automatiquement à chaque entraînement — il dépend de la distribution des scores, et une
constante écrite en dur s'est périmée quatre fois.

## Fichiers

| | |
|---|---|
| `app.py` | API et interface |
| `construire_catalogue.py` | photos → `data/catalogue.npz` + vignettes |
| `entrainer_tete.py` | catalogue → `data/tete.pt` + seuil calibré |
| `mettre_a_jour.py` | enchaîne les deux, dans l'ordre |
| `evaluer.py` | mesure sans fuite |
| `journal.py` | collecte des identifications validées |
| `inventaire.py` | sessions d'inventaire |
| `recall_grid.py` | backbone de production, comparaison de backbones |
| `finetune_triplet.py` | tête de projection et sa perte |
| `parse_etiquette.py` | lecture d'étiquette (OCR → champs) |

`recall_clip.py`, `extraire_embeddings.py` et `tester_mes_photos.py` datent de
l'exploration initiale sur le jeu Kaggle ; ils ne servent plus à la production.

## Limites

**Les mesures sont optimistes.** Toutes les photos d'une monture ont été prises en quelques
secondes : même lumière, même fond, même angle. Une photo prise un autre jour est un cas que
le jeu de test ne contient pas. Le journal existe pour combler ça à l'usage.

**39 montures sur 115 n'ont pas de marque.** Les renseigner dans `data/montures.csv` est le
gain le plus accessible : le recall@5 passe de 0.79 à 0.86 quand la marque est connue.

**Photographier plus de montures n'améliorera pas le modèle** — la courbe d'apprentissage est
plate entre 10 et 80 montures d'entraînement. C'est utile pour couvrir le stock, pas pour la
qualité.

**Les modules 2 (inventaire) et 4 (recherche visuelle) du schéma sont faits.** Le module 3
(réseau entre opticiens) ne l'est pas : il suppose plusieurs boutiques, des comptes, une
authentification et de la modération — et il est intestable avec une seule boutique.
