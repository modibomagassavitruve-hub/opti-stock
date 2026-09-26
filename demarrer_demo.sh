#!/bin/bash
# Lance la démonstration : API + tunnel public + QR code à scanner.
#
# Pensé pour un salon : un opticien scanne le QR affiché sur l'écran, et l'application s'ouvre
# sur SON téléphone. Aucune installation de son côté.
#
#   ./demarrer_demo.sh
#
# L'adresse publique change à chaque lancement (tunnel éphémère Cloudflare). Le script
# régénère donc le QR code à chaque fois. Laisser cette fenêtre ouverte : fermer le terminal
# ou laisser le Mac s'endormir coupe le tunnel.

set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$HOME/.pyenv/versions/opti-stock-312/bin/python}"
PORT="${PORT:-8000}"
JOURNAL_CF="$(mktemp -t cloudflared)"

nettoyer() {
  echo
  echo "Arrêt…"
  [[ -n "${PID_API:-}" ]] && kill "$PID_API" 2>/dev/null
  [[ -n "${PID_CF:-}" ]] && kill "$PID_CF" 2>/dev/null
  exit 0
}
trap nettoyer INT TERM

command -v cloudflared >/dev/null || { echo "cloudflared manquant : brew install cloudflared"; exit 1; }

echo "Démarrage de l'API (chargement du modèle, ~10 s)…"
"$PYTHON" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" --log-level warning &
PID_API=$!

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/sante" >/dev/null && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/sante" >/dev/null || { echo "L'API n'a pas démarré."; nettoyer; }
echo "API prête."

echo "Ouverture du tunnel public…"
cloudflared tunnel --url "http://localhost:$PORT" >"$JOURNAL_CF" 2>&1 &
PID_CF=$!

URL=""
for _ in $(seq 1 45); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$JOURNAL_CF" | head -1)
  [[ -n "$URL" ]] && break
  sleep 1
done
[[ -n "$URL" ]] || { echo "Tunnel non établi. Voir $JOURNAL_CF"; nettoyer; }

# Le QR dans le terminal suffit à faire scanner quelqu'un en face de soi ; le PNG sert à
# l'afficher en grand sur l'écran, ce qui marche mieux sur un stand.
"$PYTHON" - "$URL" <<'PY'
import sys
import segno
url = sys.argv[1]
qr = segno.make(url, error="m")
print()
qr.terminal(compact=True, border=2)
qr.save("qr_demo.png", scale=12, border=3)
PY

cat <<FIN

  ═══════════════════════════════════════════════════════════
   $URL
  ═══════════════════════════════════════════════════════════

  Faites scanner le QR ci-dessus, ou ouvrez qr_demo.png en
  plein écran pour le montrer de loin.

  Laissez cette fenêtre ouverte et le Mac éveillé :
  fermer le terminal ou laisser l'écran s'endormir coupe tout.

  Ctrl+C pour arrêter.

FIN

wait
