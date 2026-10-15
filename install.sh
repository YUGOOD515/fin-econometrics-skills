#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)/skills/fin-econometrics"
for DEST in "$HOME/.claude/skills" "$HOME/.agents/skills"; do
  mkdir -p "$DEST" && rm -rf "$DEST/fin-econometrics" && cp -R "$SRC" "$DEST/"
  echo "installed -> $DEST/fin-econometrics"
done
python -c "import statsmodels, linearmodels" 2>/dev/null || pip install -r "$SRC/requirements.txt"
