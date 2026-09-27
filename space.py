"""Point d'entrée Hugging Face Spaces (SDK gradio).

Pourquoi ce fichier existe. Hugging Face ne propose gratuitement que les SDK « static » et
« gradio » : Docker est passé payant. Or notre application est une API FastAPI qui sert sa
propre interface -- rien à voir avec Gradio.

La solution tient au fait que Gradio est lui-même bâti sur FastAPI. `mount_gradio_app` greffe
une application Gradio DANS une application FastAPI existante : la nôtre garde la racine et
sert l'interface des opticiens, Gradio occupe /gradio et satisfait l'hébergeur.

Le Space exécute ce fichier (voir `app_file` dans l'en-tête du README).
"""
from __future__ import annotations

import os

import gradio as gr

from app import app as api

# Page d'accueil Gradio : elle n'est là que pour l'hébergeur et pour qui arriverait sur
# /gradio par hasard. Tout le produit est à la racine.
with gr.Blocks(title="Opti-Stock") as accueil:
    gr.Markdown(
        "# Opti-Stock\n"
        "Gestion de stock pour opticiens.\n\n"
        "**L'application est à la racine de ce Space** : [ouvrir](/)\n"
    )

application = gr.mount_gradio_app(api, accueil, path="/gradio")


if __name__ == "__main__":
    import uvicorn

    # 7860 : le port que Hugging Face route. GRADIO_SERVER_PORT est posé par l'hébergeur ;
    # on le respecte plutôt que de le supposer.
    uvicorn.run(application, host="0.0.0.0",
                port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860))))
