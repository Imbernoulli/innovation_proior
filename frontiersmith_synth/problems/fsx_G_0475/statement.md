# Length-Generalization Probe: Formal-Language Transducers

## Background

A recurring result in the *tiny transformer probe* literature is that a model can
perfectly fit a synthetic formal-language task on the string **lengths it was
trained on**, and still collapse on **longer held-out lengths** — because it latched
onto a length-specific shortcut instead of the true, length-independent rule. This
problem distills that phenomenon into a deterministic, CPU-only probe.

Each instance is one hidden **transduction rule** `f` mapping an input string to an
output string (e.g. reversal, sorting, bitwise complement, a caesar shift, modular
symbol-counting, de-duplication, ...). You are shown a handful of `(input, output)`
**training pairs at short lengths**, plus a set of **longer, held-out test inputs**.
Your job is to predict the test outputs. Memorizing the training strings is useless:
the test lengths are strictly longer and never appear in training. You must recover
the underlying rule and **extrapolate it to longer lengths**.

Several tasks are deliberate length-generalization **traps**: e.g. a count-modulo
rule looks exactly like plain counting until the count exceeds the modulus — which
only happens at the longer test lengths — and a composite rule (reverse-then-
complement) is not something a single-rule library recognizes.

## Candidate program contract (isolated stdin → stdout)

Your program reads ONE JSON **public instance** from stdin and writes ONE JSON
answer to stdout. It runs in a fresh OS sandbox and only ever sees the public view.

### Input (stdin) — the public instance
```json
{
  "name": "probe1103_reverse",
  "alphabet": ["a", "b", "c"],
  "train": [["abc", "cba"], ["aab", "baa"], ...],
  "tests": ["abcab...(len 13-22)", "..."]
}
```
- `alphabet`  — the ordered input alphabet.
- `train`     — `[input, output]` pairs at **short** lengths (≈3–7).
- `tests`     — held-out input strings at **longer** lengths (≈13–22); predict these.

### Output (stdout)
```json
{"pred": ["...", "...", ...]}
```
- `pred` must be a list of **exactly `len(tests)` strings**, one predicted output per
  test input, in the same order.

Any malformed answer (wrong length, non-string element, non-dict, non-JSON, a crash,
a timeout, or a non-finite value) makes that instance score **0**.

## Objective — MAXIMIZE

Per instance the score is the fraction of test inputs whose predicted output
**exactly equals** the true output `f(input)`:
```
q = (# exact matches) / (# tests)          in [0, 1]
```
The reported **Ratio** is the mean of `q` over all 16 instances; the **Vector** holds
the per-instance `q`.

## What good looks like

- A pure echo/memorizer only ever gets the identity instances right → ≈ 0.1.
- A rule-library extrapolator that infers the length-independent rule from the short
  training pairs and applies it to the longer tests scores high — but **not 1.0**: the
  count-modulo traps and the composite rule leave genuine headroom.
- A true length-generalizer that infers the hidden modulus and *composes* primitives
  can close that gap.

## Scoring is deterministic

All instances are generated from fixed seeds; there is no wall-time, GPU, or
randomness in scoring. The evaluator holds the true test outputs and the hidden rule
in the parent process only — the sandboxed candidate never sees them.
