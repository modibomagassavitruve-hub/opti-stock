# Opti-Stock

Identifier une monture d'optique à partir d'une photo, et pointer le stock en inventaire.

```bash
uvicorn app:app --port 8000     # puis ouvrir http://localhost:8000
```

---

## Utiliser

**Identifier une monture.** Photographier, choisir la marque si elle est connue, valider la
bonne proposition d'un clic. Préciser la marque fait passer le recall@5 de 0.807 à 0.867.

**Entrer en stock.** Après validation, la monture s'entre en rayon avec sa quantité et son
emplacement. C'est ce qui fait de l'identification autre chose qu'une curiosité : sans stock,
elle ne débouchait sur rien.

**Inventaire.** Démarrer une session, parcourir les rayons, compter. À la fin, l'**écart** :
combien il devrait y en avoir, combien il y en a, référence par référence. Clôturer applique
les comptages au stock — le rayon fait foi, et chaque correction garde sa trace et son écart.

**Fiabilité constatée.** Chaque validation alimente un journal. C'est la seule mesure prise
dans vos conditions réelles ; tous les autres chiffres de ce document viennent de photos
prises en une seule séance, et sont donc optimistes.

**Réseau.** Un client demande une monture que vous n'avez plus : cherchez-la chez un confrère
inscrit et écrivez-lui directement. Vous choisissez monture par monture ce que vous exposez ;
le reste de votre stock n'est jamais visible. Ce qui circule est la marque et la référence,
jamais vos étiquettes internes — le « 50 » d'une boutique n'est pas celui d'une autre.

La disponibilité est vérifiée **à la recherche**, pas au partage : le partage est un choix
durable, le stock bouge. Partager le matin et vendre à midi ne doit pas faire se déplacer un
confrère l'après-midi.

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

État actuel, sur 115 montures : **recall@5 = 0.861**, recall@1 = 0.739. Avec filtre par marque,
sur les 83 requêtes dont la marque est connue : 0.867 contre 0.807 sans. L'API répond dans
74 % des cas et a raison 91 % du temps. Le seuil de confiance est recalibré automatiquement à
chaque entraînement — il dépend de la distribution des scores, et une constante écrite en dur
s'est périmée quatre fois.

**Ne pas confondre avec le chiffre affiché en fin d'entraînement** (0.940). Celui-là vient de
la validation interne, dont le découpage est plus favorable. Seul `evaluer.py` applique le
protocole sans fuite, et c'est lui qui fait foi : l'écart entre les deux est de 8 points.

**Renseigner 7 marques de plus n'a pas bougé le recall global** (0.861 avant comme après). Ce
que ça change est ailleurs : 83 requêtes au lieu de 76 peuvent utiliser le filtre par marque,
et 7 montures de plus sont proposables au réseau.

## Fichiers

| | |
|---|---|
| `app.py` | API et interface |
| `construire_catalogue.py` | photos → `data/catalogue.npz` + vignettes |
| `entrainer_tete.py` | catalogue → `data/tete.pt` + seuil calibré |
| `mettre_a_jour.py` | enchaîne les deux, dans l'ordre |
| `evaluer.py` | mesure sans fuite |
| `journal.py` | collecte des identifications validées |
| `stock.py` | quantités et emplacements, par mouvements |
| `inventaire.py` | sessions d'inventaire et écarts |
| `reseau.py` | réseau entre opticiens : partage, recherche, messagerie |
| `metadonnees.py` | lecture/écriture de `data/montures.csv` (page `/saisie`) |
| `recall_grid.py` | backbone de production, comparaison de backbones |
| `finetune_triplet.py` | tête de projection et sa perte |
| `parse_etiquette.py` | lecture d'étiquette (OCR → champs) |

`recall_clip.py`, `extraire_embeddings.py` et `tester_mes_photos.py` datent de
l'exploration initiale sur le jeu Kaggle ; ils ne servent plus à la production.

## Limites

**Les mesures sont optimistes.** Toutes les photos d'une monture ont été prises en quelques
secondes : même lumière, même fond, même angle. Une photo prise un autre jour est un cas que
le jeu de test ne contient pas. Le journal existe pour combler ça à l'usage.

**32 montures sur 115 n'ont pas de marque, et l'information n'est pas dans les photos.**
Passées à l'OCR, elles ne rendent que les autocollants de verres (« UV 100% cat.3 ») : la
gravure est à l'intérieur de la branche, qui n'a pas été photographiée. Seules 5 marques ont
pu être lues — et une, RAY-BAN, à sa signature sur le verre.

C'est donc une saisie humaine, et `http://localhost:8000/saisie` la sert : la vignette de
chaque monture, un champ marque avec les marques déjà connues en autocomplétion. L'opticien
reconnaît son propre stock à l'œil, sans avoir besoin de la gravure. Le gain vaut le quart
d'heure : le recall@5 passe de 0.807 à 0.867, et une monture sans marque ne peut pas être
proposée au réseau.

Une marque devinée y reste marquée « à vérifier » tant qu'elle n'a pas été cochée :
enregistrer la page ne vaut pas relecture. Après saisie, `python mettre_a_jour.py` — le
catalogue embarque les marques au moment où il est construit.

**Photographier plus de montures n'améliorera pas le modèle** — la courbe d'apprentissage est
plate entre 10 et 80 montures d'entraînement. C'est utile pour couvrir le stock, pas pour la
qualité.

**Le réseau n'a pas de véritable authentification.** Le jeton remis à l'inscription suffit à
agir au nom d'une boutique : ni mot de passe, ni vérification d'e-mail ou de SIRET, et il est
gardé dans le navigateur. Cela suffit à montrer le produit ; cela ne suffit pas à porter de
vrais échanges entre entreprises — qui intercepte un jeton lit les messages d'une boutique et
écrit en son nom. **Avant d'ouvrir le réseau à de vraies boutiques**, il faut au minimum :

| | |
|---|---|
| Authentification externe | le schéma la délègue explicitement ; tout passe par `_boutique()` dans `app.py` |
| HTTPS | le jeton circule en clair aujourd'hui |
| CORS restreint | `allow_origins=["*"]` est là pour le développement |
| Modération | les signalements sont conservés, pas traités |
| RGPD | information, conservation, effacement — rien n'est prévu |

Les quatre modules du schéma sont en place ; le module 3 reste celui qui demande ce travail
avant mise en service, les trois autres ne concernent que la boutique elle-même.
