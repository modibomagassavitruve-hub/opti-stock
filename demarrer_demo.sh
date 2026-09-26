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
  for p in "${PID_API:-}" "${PID_CF:-}" "${PID_VEILLE:-}"; do
    [[ -n "$p" ]] && kill "$p" 2>/dev/null
  done
  exit 0
}
trap nettoyer INT TERM

# Le Mac qui s'endort coupe le tunnel, et la démonstration meurt sans prévenir au milieu d'un
# salon. caffeinate l'en empêche tant que ce script tourne, et rend la main à l'arrêt.
if command -v caffeinate >/dev/null; then
  caffeinate -dimsu &
  PID_VEILLE=$!
fi

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

# Ouvre un tunnel et renvoie son adresse, ou vide si elle n'apparaît pas.
ouvrir_tunnel() {
  : >"$JOURNAL_CF"
  cloudflared tunnel --url "http://localhost:$PORT" >"$JOURNAL_CF" 2>&1 &
  PID_CF=$!
  for _ in $(seq 1 45); do
    URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$JOURNAL_CF" | head -1)
    [[ -n "$URL" ]] && return 0
    sleep 1
  done
  return 1
}

# Le QR dans le terminal suffit à faire scanner quelqu'un en face de soi ; le PNG sert à
# l'afficher en grand sur l'écran, ce qui marche mieux sur un stand.
annoncer() {
  "$PYTHON" - "$1" <<'PY'
import sys
import segno
qr = segno.make(sys.argv[1], error="m")
print()
qr.terminal(compact=True, border=2)
qr.save("qr_demo.png", scale=12, border=3)
PY
  cat <<FIN

  ═══════════════════════════════════════════════════════════
   $1
  ═══════════════════════════════════════════════════════════

  Faites scanner le QR ci-dessus, ou ouvrez qr_demo.png en
  plein écran pour le montrer de loin.

  Laissez cette fenêtre ouverte : la fermer coupe tout.
  La mise en veille est déjà bloquée pendant ce temps.

  Ctrl+C pour arrêter.

FIN
}

echo "Ouverture du tunnel public…"
ouvrir_tunnel || { echo "Tunnel non établi. Voir $JOURNAL_CF"; nettoyer; }
annoncer "$URL"

# Surveillance. Un tunnel éphémère Cloudflare peut perdre son adresse sans que le processus
# s'arrête : constaté après trois heures. Le QR affiché devient alors muet, et sur un stand
# on ne s'en aperçoit qu'en voyant un opticien s'éloigner. On vérifie donc l'adresse depuis
# l'extérieur, et on en rouvre une en annonçant clairement le changement.
while sleep 30; do
  kill -0 "$PID_API" 2>/dev/null || { echo "L'API s'est arrêtée."; nettoyer; }
  curl -sf --max-time 10 -o /dev/null "$URL/sante" && continue

  echo
  echo "  ⚠  Le tunnel ne répond plus — réouverture…"
  kill "$PID_CF" 2>/dev/null
  if ouvrir_tunnel; then
    echo "  ⚠  NOUVELLE ADRESSE : l'ancien QR ne marche plus, refaites scanner."
    annoncer "$URL"
  else
    echo "  ⚠  Réouverture impossible. Vérifiez la connexion, puis relancez ce script."
    sleep 30
  fi
done
