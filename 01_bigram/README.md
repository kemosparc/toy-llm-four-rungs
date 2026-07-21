# Rung 1 — Bigram (one token of memory)

The smallest neural language model that runs: predict the next word from the
**one** word before it. Embed → score → softmax → loss → gradient, and nothing
else.

This directory is self-contained. It carries its own copy of the shared header
`toy_llm.hpp` and the `corpus.txt`, so it builds and runs with none of the other
rungs present.

## Build and run

```sh
./compile.sh    # builds the binary: v1
./run.sh        # trains on corpus.txt, prints the model, hits the wall
```

## What it shows

It follows short, local patterns — but with only one word of memory it stands on
`is` when it must answer, and cannot see which country came before. So it cannot
finish *"the capital of france is ___"* correctly. That wall is the whole point:
it is what rung 2 is built to climb.

## Files

- `llm_v1_bigram.cpp` — the model (about forty lines) plus the CLI driver
- `toy_llm.hpp` — shared plumbing: tokenizer, softmax, file I/O, generation loop
- `corpus.txt` — the seven-line training corpus
- `compile.sh` / `run.sh` — build and demonstrate this rung alone
