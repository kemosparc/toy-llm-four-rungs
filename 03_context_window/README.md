# Rung 3 — Context Window (keep the last few words, glue them)

Same learning loop as the bigram, but the context is the last N tokens with their
embeddings concatenated into one long vector. Now the country word is still inside
the window when the model must answer, so it can finally finish *"the capital of
france is ___"* correctly.

This directory is self-contained. It carries its own copy of the shared header
`toy_llm.hpp` and the `corpus.txt`, so it builds and runs with none of the other
rungs present.

## Build and run

```sh
./compile.sh    # builds the binary: v3
./run.sh        # trains, generalizes, then hits two walls
```

## What it shows

It **generalizes** — it answers even with a leading word dropped. But two walls
remain: the window is **bounded** (push the country past it and different
countries collapse to one answer), and it reads by **position, not relevance**
(reorder the same words and it breaks). Rung 4 fixes the relevance problem.

## Files

- `llm_v3_context_window.cpp` — the model plus the CLI driver
- `toy_llm.hpp` — shared plumbing: tokenizer, softmax, file I/O, generation loop
- `corpus.txt` — the seven-line training corpus
- `compile.sh` / `run.sh` — build and demonstrate this rung alone
