#!/usr/bin/env bash
# Build ../submission/ in the shape the Kuku form asks for. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../submission
rm -rf "$OUT" && mkdir -p "$OUT"

# 1. Solution doc (PDF, <= 2 pages)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/docs/solution.pdf" "$PWD/docs/solution.html" 2>/dev/null
cp docs/solution.pdf "$OUT/1_Solution_Doc.pdf"

# 2. Prototype: code only. No key, no cache, no confidential scripts/outputs.
mkdir -p "$OUT/2_Prototype"
rsync -a --exclude '.env' --exclude '.cache' --exclude 'outputs' --exclude '__pycache__' \
  --exclude 'eval/gold_*.json' --exclude '.DS_Store' --exclude '.git' ./ "$OUT/2_Prototype/promo-finder/"
(cd "$OUT/2_Prototype" && zip -qr promo-finder.zip promo-finder)

# 3. Raw outputs: unedited run folders + evaluation + the gold labels they were scored against
mkdir -p "$OUT/3_Raw_Outputs"
cp -R outputs/*/* "$OUT/3_Raw_Outputs/"
mkdir -p "$OUT/3_Raw_Outputs/gold_labels" && cp eval/gold_*.json "$OUT/3_Raw_Outputs/gold_labels/"
(cd "$OUT" && zip -qr 3_Raw_Outputs.zip 3_Raw_Outputs)

# 4. Walkthrough video script
cp docs/WALKTHROUGH.md "$OUT/4_Walkthrough_Script.md"

# Safety: the key must never ship
KEY=$(grep -s GEMINI_API_KEY .env | cut -d= -f2 || true)
if [ -n "$KEY" ] && grep -rqF "$KEY" "$OUT"; then echo "ERROR: API key found in submission" >&2; exit 1; fi
echo "built $OUT"; find "$OUT" -maxdepth 2 | sort
