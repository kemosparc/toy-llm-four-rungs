#!/usr/bin/env bash
# ============================================================================
#  run.sh  (top level)  -  walk every rung that is present, in order.
# ----------------------------------------------------------------------------
#  Runs each present rung's own run.sh, one after another, pausing between
#  rungs. Like the top-level compile.sh, this discovers the rung directories
#  that exist rather than hardcoding four, so it tours exactly what has been
#  uploaded so far -- one rung, or the full V1 -> V4 ladder.
#
#  Requires the binaries to be built first: run ./compile.sh.
# ============================================================================
set -uo pipefail

mapfile -t RUNGS < <(find . -maxdepth 1 -type d -name '[0-9][0-9]_*' -printf '%f\n' | sort)

if [ "${#RUNGS[@]}" -eq 0 ]; then
    echo "no rung directories found (expected e.g. 01_bigram/). nothing to run."
    exit 1
fi

pause () { echo; read -rsn1 -p "    [ press any key for the next rung ] "; echo; echo; }

clear 2>/dev/null || true
echo "TOY LANGUAGE MODEL -- ${#RUNGS[@]} RUNG(S) PRESENT: ${RUNGS[*]}"
echo
echo "Each rung trains on the same corpus, prints its model, answers a few"
echo "prompts, and runs straight into the wall the NEXT rung is built to climb."
pause

count="${#RUNGS[@]}"
i=0
for d in "${RUNGS[@]}"; do
    i=$((i+1))
    if [ ! -f "$d/run.sh" ]; then
        echo "!! $d has no run.sh -- skipping"
        continue
    fi
    ( cd "$d" && bash run.sh )
    # pause between rungs, but not after the last one
    if [ "$i" -lt "$count" ]; then
        pause
    fi
done
