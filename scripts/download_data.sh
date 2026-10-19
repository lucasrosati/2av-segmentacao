#!/usr/bin/env bash
# Baixa o kvasir-seg.zip oficial para data/raw/ e confere SHA-256 e contagem (1000 imagens + 1000 máscaras).
set -euo pipefail
URL=https://datasets.simula.no/downloads/kvasir-seg.zip
SHA256=03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7
OUT=${1:-data/raw}
ZIP="$OUT/kvasir-seg.zip"

mkdir -p "$OUT"
[ -f "$ZIP" ] || curl -fL --retry 3 -o "$ZIP" "$URL"
got=$( (command -v sha256sum >/dev/null && sha256sum "$ZIP" || shasum -a 256 "$ZIP") | cut -d' ' -f1)
[ "$got" = "$SHA256" ] || { echo "SHA-256 diverge: $got" >&2; exit 1; }
unzip -oq "$ZIP" -d "$OUT/kvasir-seg"
for d in images masks; do
  n=$(find "$OUT/kvasir-seg/Kvasir-SEG/$d" -name '*.jpg' | wc -l | tr -d ' ')
  [ "$n" -eq 1000 ] || { echo "$d: $n arquivos, esperado 1000" >&2; exit 1; }
done
echo "ok: $OUT/kvasir-seg/Kvasir-SEG (1000 imagens + 1000 máscaras, hash confere)"
