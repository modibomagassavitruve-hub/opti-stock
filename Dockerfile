FROM python:3.12-slim

WORKDIR /app

# libgl1 / libglib2 : requis par opencv, dont dépend easyocr
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch en version CPU : l'image passe de ~6 Go à ~1,5 Go, et un serveur d'appli n'a pas de GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# Les poids du backbone et de l'OCR sont téléchargés au premier démarrage, ce qui rend le
# premier appel très lent et exige un réseau en production. On les embarque dans l'image.
RUN python -c "\
from recall_grid import BACKBONE_PROD, charger_backbone; charger_backbone(BACKBONE_PROD, 'cpu'); \
import easyocr; easyocr.Reader(['fr','en'], gpu=False)"

# Le journal des identifications validées est la seule mesure de fiabilité en conditions
# réelles, et il s'accumule sur des semaines. Sans volume monté ici, il disparaîtrait à chaque
# redéploiement :
#     docker run -p 8000:8000 -v opti_journal:/journal opti-stock
ENV OPTI_STOCK_JOURNAL=/journal
VOLUME /journal

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
