#!/usr/bin/env bash
# ============================================================================
#  run.sh  -  rung 3 (context window) on its own: train, generalize, hit walls.
# ----------------------------------------------------------------------------
#  Self-contained. Keeps a fixed window of the last 3 words and predicts from
#  their glue. It generalizes -- but reads by position, not by relevance.
# ============================================================================
set -uo pipefail

CORPUS="corpus.txt"
EPOCHS="${EPOCHS:-4000}"

line () { printf '%s\n' "------------------------------------------------------------------------"; }
note () { printf '\n>> %s\n' "$1"; }
show () { printf '\n$ %s\n' "$*"; "$@"; }

if [ ! -x "./v3" ]; then echo "missing ./v3 -- run ./compile.sh first."; exit 1; fi

clear 2>/dev/null || true
echo "RUNG 3 of 4 : WINDOW  (keep the last 3 words, glue them)"
echo
echo "training text:"
echo
cat "$CORPUS"
echo

note "It keeps a fixed window of recent words and predicts from their glue."
show ./v3 train "$CORPUS" v3.model --epochs "$EPOCHS"
note "It generalizes -- it answers a phrase even with a leading word dropped:"
show ./v3 gen v3.model "the capital of france is"
show ./v3 gen v3.model "of france is"
note "THE WALL #1 -- BOUNDED WINDOW: push the country past the window and it is"
note "gone. Different countries collapse to one answer:"
show ./v3 gen v3.model "france is the capital of"
show ./v3 gen v3.model "japan is the capital of"
note "THE WALL #2 -- NO RELEVANCE: it reads by position, not content. Reorder the"
note "same words and it breaks, because the country left the slot it learned:"
show ./v3 gen v3.model "france is capital of"
echo
line
echo "  Rung 4 fixes the relevance problem: it learns to look at the word"
echo "  that matters."
line
echo
