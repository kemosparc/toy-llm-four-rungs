#!/usr/bin/env bash
# ============================================================================
#  compile.sh  (top level)  -  build every rung that is present in the repo.
# ----------------------------------------------------------------------------
#  Each rung lives in its own self-contained directory with its own compile.sh.
#  This script does NOT hardcode the four rungs: it discovers the rung
#  directories that actually exist and builds each one. So the same script
#  works whether the repo currently holds one rung or all four -- useful while
#  the rungs are being published and uploaded incrementally.
# ============================================================================
set -uo pipefail

# rung directories are named NN_something; match that pattern, in order
mapfile -t RUNGS < <(find . -maxdepth 1 -type d -name '[0-9][0-9]_*' -printf '%f\n' | sort)

if [ "${#RUNGS[@]}" -eq 0 ]; then
    echo "no rung directories found (expected e.g. 01_bigram/). nothing to build."
    exit 1
fi

echo "found ${#RUNGS[@]} rung(s): ${RUNGS[*]}"
echo

failed=0
for d in "${RUNGS[@]}"; do
    if [ ! -x "$d/compile.sh" ] && [ ! -f "$d/compile.sh" ]; then
        echo "!! $d has no compile.sh -- skipping"
        failed=1
        continue
    fi
    echo "========================================================================"
    echo "  building $d"
    echo "========================================================================"
    # run each rung's own compile.sh from inside its directory
    ( cd "$d" && bash compile.sh ) || { echo "!! build failed in $d"; failed=1; }
    echo
done

if [ "$failed" -ne 0 ]; then
    echo "one or more rungs failed to build."
    exit 1
fi

echo "all present rungs built. next: ./run.sh"
