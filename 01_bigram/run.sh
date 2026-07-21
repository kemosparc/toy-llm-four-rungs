#!/usr/bin/env bash
# ============================================================================
#  run.sh  -  rung 1 (bigram) on its own: train, inspect, hit the wall.
# ----------------------------------------------------------------------------
#  Self-contained. Predicts the next word from ONLY the single word before it.
#  Maximum verbosity on purpose: the model file is plain text, so we print it.
# ============================================================================
set -uo pipefail

CORPUS="corpus.txt"
EPOCHS="${EPOCHS:-4000}"

line () { printf '%s\n' "------------------------------------------------------------------------"; }
note () { printf '\n>> %s\n' "$1"; }
show () { printf '\n$ %s\n' "$*"; "$@"; }

if [ ! -x "./v1" ]; then echo "missing ./v1 -- run ./compile.sh first."; exit 1; fi

clear 2>/dev/null || true
echo "RUNG 1 of 4 : BIGRAM  (one token of memory)"
echo
echo "training text:"
echo
cat "$CORPUS"
echo
echo "Vocabulary is 25 words. The model uses D = 8 numbers per word."

note "It predicts the next word from ONLY the single word just before it."
show ./v1 train "$CORPUS" v1.model --epochs "$EPOCHS"
note "The trained model is plain text. Here it is (E = word table, W = next-word weights):"
show cat v1.model
note "It can follow short, local patterns:"
show ./v1 gen v1.model "the capital of"
note "THE WALL: with only one word of memory, it stands on 'is' and cannot see"
note "which country came before. So it cannot finish this correctly:"
show ./v1 gen v1.model "the capital of france is"
echo
line
echo "  One token of memory is not enough. Rung 2 tries to fix that by"
echo "  memorizing whole phrases."
line
echo
