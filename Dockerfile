FROM python:3.12-slim

# Port et chemin de données paramétrables : Hugging Face Spaces impose 7860 et un utilisateur
# non root, un serveur classique attend 8000 et un volume monté. Un seul Dockerfile pour les
# deux, plutôt que deux qui divergeront.
# 7860 par défaut : c'est ce que route Hugging Face Spaces (voir app_port dans le README).
# Ailleurs, passer -e PORT=8000 ou publier le port : docker run -p 8000:7860 …
ENV PORT=7860

# libgl1 / libglib2 : requis par opencv, dont dépend easyocr
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non root : exigé par Hugging Face Spaces, et de toute façon préférable ailleurs.
# HOME pointe vers son dossier pour que les caches (torch, easyocr) soient accessibles en
# écriture -- sans ça le téléchargement des poids échoue au démarrage.
RUN useradd -m -u 1000 opti
ENV HOME=/home/opti
WORKDIR /app

# torch en version CPU : l'image passe de ~6 Go à ~1,5 Go, et un serveur d'appli n'a pas de GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# Les poids de l'OCR sont téléchargés au premier démarrage, ce qui rend le premier appel très
# lent et exige un réseau en production. On les embarque dans l'image.
#
# Le backbone de reconnaissance n'est embarqué que s'il y a un catalogue à interroger : sans
# data/catalogue.npz -- le cas d'un déploiement depuis le dépôt public -- l'application
# démarre en mode léger et n'en a pas l'usage. Ça évite 600 Mo d'image pour rien.
RUN python -c "import easyocr; easyocr.Reader(['fr','en'], gpu=False)" && \
    if [ -f data/catalogue.npz ]; then \
      python -c "from recall_grid import BACKBONE_PROD, charger_backbone; \
                 charger_backbone(BACKBONE_PROD, 'cpu')"; \
    else echo "pas de catalogue : backbone non embarqué (mode léger)"; fi

# Le stock des boutiques, leurs fiches et le journal de fiabilité vivent ici. Sans volume monté,
# tout disparaît à chaque redéploiement :
#     docker run -p 8000:8000 -v opti_journal:/journal opti-stock
ENV OPTI_STOCK_JOURNAL=/journal
RUN mkdir -p /journal && chown -R opti:opti /journal /app
VOLUME /journal

USER opti
EXPOSE ${PORT}
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
