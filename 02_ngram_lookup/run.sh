#!/usr/bin/env bash
# ============================================================================
#  run.sh  -  rung 2 (n-gram lookup foil) on its own: train, inspect, crack.
# ----------------------------------------------------------------------------
#  Self-contained. Stores each exact phrase it saw and looks the answer up.
#  It gets every capital right -- and is blind to any phrase it never saw.
# ============================================================================
set -uo pipefail

CORPUS="corpus.txt"
EPOCHS="${EPOCHS:-4000}"

line () { printf '%s\n' "------------------------------------------------------------------------"; }
note () { printf '\n>> %s\n' "$1"; }
show () { printf '\n$ %s\n' "$*"; "$@"; }

if [ ! -x "./v2" ]; then echo "missing ./v2 -- run ./compile.sh first."; exit 1; fi

clear 2>/dev/null || true
echo "RUNG 2 of 4 : LOOKUP  (memorize whole phrases)"
echo
echo "training text:"
echo
cat "$CORPUS"
echo

note "It stores each exact phrase it saw and looks the answer up."
show ./v2 train "$CORPUS" v2.model --epochs "$EPOCHS"
note "Unlike the bigram, it now gets EVERY capital right, because it remembers the"
note "whole phrase that came before -- the country is part of the stored key:"
show ./v2 gen v2.model "the capital of france is"
show ./v2 gen v2.model "the capital of japan is"
show ./v2 gen v2.model "the capital of egypt is"
show ./v2 gen v2.model "the capital of italy is"
show ./v2 gen v2.model "the capital of spain is"
show ./v2 gen v2.model "the capital of germany is"
note "It also reproduces the unrelated gym sentence it memorized:"
show ./v2 gen v2.model "i go everyday to the gym for"
note "THE FIRST CRACK -- BLINDNESS: drop one word and there is no stored row,"
note "so it goes silent and stops. The gym sentence with one word missing from the front:"
show ./v2 gen v2.model "go everyday to the gym for"
note "And the same blindness on the capitals when a leading word is dropped:"
show ./v2 gen v2.model "france is the capital"
echo
line
echo "  THE SECOND CRACK -- EXPLOSION: it keeps one row per phrase, so its memory"
echo "  grows with the length of the text, not with the size of the language."
echo "  Rung 3 fixes both at once by keeping only the last few words."
line
echo
