# Rung 2 — N-gram Lookup (the foil: memorize whole phrases)

The tempting wrong turn. Instead of one row per **token**, give one row per whole
**prefix**. It then memorizes each context's continuation perfectly — and is
utterly blind to any prefix it never saw, because that prefix simply has no row.
It also explodes: the number of rows grows with the text, not with the language.

This directory is self-contained. It carries its own copy of the shared header
`toy_llm.hpp` and the `corpus.txt`, so it builds and runs with none of the other
rungs present.

## Build and run

```sh
./compile.sh    # builds the binary: v2
./run.sh        # trains, gets every capital right, then cracks
```

## What it shows

It gets **every** capital right, because the whole phrase is the stored key. Then
two cracks: **blindness** (drop a leading word and there is no row, so it goes
silent) and **explosion** (one row per phrase, memory growing with the text).
Rung 3 fixes both at once by keeping only the last few words.

## Files

- `llm_v2_ngram_lookup.cpp` — the model plus the CLI driver
- `toy_llm.hpp` — shared plumbing: tokenizer, softmax, file I/O, generation loop
- `corpus.txt` — the seven-line training corpus
- `compile.sh` / `run.sh` — build and demonstrate this rung alone
