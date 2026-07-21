# A Toy Language Model in Four Rungs

A tiny language model built from scratch in C++, in four steps. Each step is a
single, readable program in its own self-contained directory. Together they climb
from one word of memory to attention, the mechanism every modern language model is
built on. Every model trains on the same seven-line corpus and uses eight numbers
per word (D = 8), so you can read the model files by hand and check the math.

This is the code companion to the "AI-Native Systems Engineering" series. The four
rungs match four articles, and the repository grows one rung at a time as each
article is published.

| Rung | Directory | Idea | Wall it hits |
|------|-----------|------|--------------|
| 1 | `01_bigram/` | predict from the one previous word | can't see past one token |
| 2 | `02_ngram_lookup/` | memorize whole phrases | goes blind on unseen phrases; memory explodes |
| 3 | `03_context_window/` | keep the last few words, glue them | bounded window; reads by position, not relevance |
| 4 | `04_attention/` | weigh the window by relevance (attention) | window still bounded; one fixed global query |

## Requirements

A C++17 compiler (`g++` or `clang++`). Nothing else. No libraries, no build system.

## Two ways to use this repo

### Build and run everything at once

From the repository root:

```sh
./compile.sh    # builds every rung present, into per-directory binaries v1..v4
./run.sh        # walks every rung present, in order, pausing between them
```

Both top-level scripts **discover the rung directories that actually exist** rather
than assuming all four are here. So they build and tour exactly what has been
published so far — one rung, or the full ladder — which is what makes the
incremental uploads work. To use a different compiler: `CXX=clang++ ./compile.sh`.

### Build and run a single rung

Each rung directory is fully self-contained — its own `toy_llm.hpp`, its own
`corpus.txt`, its own `compile.sh` and `run.sh`. Step into any one and it works on
its own:

```sh
cd 01_bigram
./compile.sh
./run.sh
```

## How each rung is organized

Every rung's `.cpp` contains only the model itself. All the plumbing that never
changes between rungs — the tokenizer, the softmax, reading a file, saving and
loading a model, the generation loop — lives in the shared header `toy_llm.hpp`,
which each directory carries its own copy of. That is deliberate: the difference
between rung one and rung four is the idea that changed, not boilerplate.

The model files are plain text on purpose. Open a trained `v4.model` in any editor
and you will see the three tables the model learned: `E` (a row of numbers per
word), `Q` (the single query vector), and `W` (the next-word weights). Nothing is
hidden.

## The corpus

`corpus.txt` is seven lines: six capital-city facts and one unrelated sentence
about a gym, which keeps the vocabulary honest at twenty-five words.

```
the capital of france is paris.
the capital of japan is tokyo.
the capital of egypt is cairo.
the capital of italy is rome.
the capital of spain is madrid.
the capital of germany is berlin.
i go everyday to the gym for squats.
```

## License

Provided for learning. Use it freely.

---

Authored by Karim Sobh. Editorial polishing and proofreading were performed with
AI assistance.
