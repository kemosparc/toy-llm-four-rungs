#!/usr/bin/env bash
# ============================================================================
#  compile.sh  -  build rung 1 (bigram) on its own
# ----------------------------------------------------------------------------
#  Self-contained: this directory carries its own toy_llm.hpp and corpus.txt,
#  so it builds with nothing from the other rungs present. Output binary: v1
# ============================================================================
set -euo pipefail

CXX="${CXX:-g++}"
FLAGS="-std=c++17 -O2 -Wall -Wextra"

echo "compiler: $($CXX --version | head -1)"
echo "flags:    $FLAGS"
echo
printf 'building %-28s -> %s ... ' "llm_v1_bigram.cpp" "v1"
$CXX $FLAGS llm_v1_bigram.cpp -o v1
echo "ok   (rung 1: bigram, one token of memory)"
echo
echo "done. next: ./run.sh"
