#!/usr/bin/env bash
# ============================================================================
#  compile.sh  -  build rung 3 (context window) on its own
# ----------------------------------------------------------------------------
#  Self-contained: this directory carries its own toy_llm.hpp and corpus.txt,
#  so it builds with nothing from the other rungs present. Output binary: v3
# ============================================================================
set -euo pipefail

CXX="${CXX:-g++}"
FLAGS="-std=c++17 -O2 -Wall -Wextra"

echo "compiler: $($CXX --version | head -1)"
echo "flags:    $FLAGS"
echo
printf 'building %-28s -> %s ... ' "llm_v3_context_window.cpp" "v3"
$CXX $FLAGS llm_v3_context_window.cpp -o v3
echo "ok   (rung 3: fixed window, first to generalize)"
echo
echo "done. next: ./run.sh"
