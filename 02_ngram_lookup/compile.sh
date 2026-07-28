#!/usr/bin/env bash
# ============================================================================
#  compile.sh  -  build rung 2 (n-gram lookup foil) on its own
# ----------------------------------------------------------------------------
#  Self-contained: this directory carries its own toy_llm.hpp and corpus.txt,
#  so it builds with nothing from the other rungs present. Output binary: v2
# ============================================================================
set -euo pipefail

CXX="${CXX:-g++}"
FLAGS="-std=c++17 -O2 -Wall -Wextra"

echo "compiler: $($CXX --version | head -1)"
echo "flags:    $FLAGS"
echo
printf 'building %-28s -> %s ... ' "llm_v2_ngram_lookup.cpp" "v2"
$CXX $FLAGS llm_v2_ngram_lookup.cpp -o v2
echo "ok   (rung 2: lookup foil, memorizes phrases)"
echo
echo "done. next: ./run.sh"
