# Mettre l'application en ligne — Hugging Face Spaces (gratuit)

Une adresse stable qui tient la journée sans votre ordinateur. Gratuit, 16 Go de RAM, pas de
mise en veille tant qu'on s'en sert.

Comptez dix minutes, dont sept d'attente pendant la construction.

---

## 1. Un compte (2 min)

<https://huggingface.co/join> — e-mail, mot de passe, c'est tout.

## 2. Un jeton d'écriture (1 min)

<https://huggingface.co/settings/tokens> → **New token** → type **Write** → copiez-le.

## 3. Créer le Space (1 min)

<https://huggingface.co/new-space>

| Champ | Valeur |
|---|---|
| Space name | `opti-stock` |
| License | `mit` |
| SDK | **Docker** → *Blank* |
| Hardware | **CPU basic** (gratuit) |
| Visibilité | **Public** |

## 4. Envoyer le code (1 min + 7 de construction)

Dans un terminal, en remplaçant `VOTRE-PSEUDO` :

```bash
cd ~/code/opti-stock
pip install -q huggingface_hub
huggingface-cli login          # collez le jeton de l'étape 2
git remote add space https://huggingface.co/spaces/VOTRE-PSEUDO/opti-stock
git push space master:main
```

Suivez la construction dans l'onglet **Logs** du Space. Au bout de quelques minutes :

```
[léger] Reconnaissance par similarité désactivée (data/catalogue.npz absent).
Saisie, étiquette, stock, inventaire et réseau fonctionnent.
```

C'est normal, et c'est voulu — voir plus bas.

## 5. Votre adresse

```
https://VOTRE-PSEUDO-opti-stock.hf.space
```

Elle ne change plus. Faites-en un QR :

```bash
python -c "import segno; segno.make('https://VOTRE-PSEUDO-opti-stock.hf.space', error='h').save('qr_silmo.png', scale=16, border=4)"
```

---

## Ce qui marche, et ce qui ne marche pas

**Marche** : créer sa boutique, photographier une monture, lire l'étiquette par OCR, tenir son
stock, inventorier avec écarts chiffrés, le réseau entre opticiens et sa messagerie.

**Ne marche pas** : la reconnaissance d'une monture par photo. Elle exige `data/catalogue.npz`
et le modèle entraîné, qui ne sont pas dans le dépôt public — ce sont les photos et
l'inventaire d'une boutique réelle. L'onglet Saisir le signale proprement plutôt que d'échouer.

Ce n'est pas une perte pour une démonstration : le modèle ne connaît que VOS 115 montures, et
n'aurait rien reconnu du stock d'un autre opticien. Le parcours qui convainc — photographier,
laisser l'étiquette remplir les champs, voir la monture entrer en stock — n'en a pas besoin.

Pour l'activer quand même : déployer depuis un dépôt **privé** contenant `data/catalogue.npz`,
`data/tete.pt`, `data/tete.json` et `data/vignettes/`. Le Dockerfile les détecte et embarque
le backbone automatiquement.

## Le point à connaître : les données sont éphémères

Sur un Space gratuit, le disque est remis à neuf si Hugging Face redémarre le service. Les
boutiques inscrites et leur stock disparaîtraient alors.

En une journée de salon, c'est peu probable mais pas impossible. **Les inscriptions sont vos
contacts** : notez-les au fur et à mesure autrement, ou consultez la liste de temps en temps
sur `https://VOTRE-PSEUDO-opti-stock.hf.space/reseau/boutiques`.

La persistance existe en option payante chez Hugging Face (Persistent Storage, ~5 $/mois), ou
en déployant ailleurs avec un disque monté — voir `render.yaml`.

## Mettre à jour après coup

```bash
git push space master:main
```

Le Space se reconstruit tout seul. Attention : une reconstruction efface les données.
