# Deep-Sea Cable Splice: Length-OOD Generalization

## Story

A deep-sea cable network is assembled from **connector tokens**. There are `K`
connector **types**; each type has an **opening token** (a lowercase letter) and a
matching **closing token** (the same letter, uppercased). For example with `K = 3`
the opening tokens are `a b c` and the closing tokens are `A B C`. A **cable** is a
string of such tokens, e.g. `abBA` or `aAbcCB`.

A cable is **well-spliced** if and only if it obeys a fixed, deterministic rule that
you are **not told directly** and must infer from labelled examples. The rule has two
parts:

1. **Proper matched nesting.** Reading the cable left to right with a stack: an
   opening token pushes its type; a closing token must close the most-recently-opened,
   still-open connector **of the same type** (otherwise the cable is not well-spliced),
   and the stack must be empty at the end. (Equivalently: the cable is a valid Dyck
   word over `K` bracket types.)
2. **A splice-depth budget.** The nesting must never run deeper than a hidden budget
   `D` — i.e. the maximum stack depth reached must be `<= D`. The same `D` is used
   for every instance, but its value is not disclosed.

## Your task

For each instance you receive a **labelled training set** of **short** cables
(in-distribution) and must predict the well-spliced label for a **query set** that
mixes short cables with much **longer** cables (out-of-distribution in length).
A learner that memorises the training strings, or that keys on length-correlated
signals, will fail on the longer cables — only a learner that recovers the actual
matching rule generalises. The training cables never nest deep enough to reveal the
depth budget `D`, so `D` is genuinely under-determined: extrapolating it is an
inductive-bias choice you must make.

## Input (stdin): one JSON object — the PUBLIC instance

```json
{
  "name": "cable7002",
  "n_types": 3,
  "open_symbols":  ["a", "b", "c"],
  "close_symbols": ["A", "B", "C"],
  "train":   [["abBA", 1], ["abA", 0], ...],
  "queries": ["aAbBcC", "abcCBA", ...]
}
```

- `n_types` — the number of connector types `K`.
- `open_symbols` / `close_symbols` — the `K` opening tokens and their matching
  closing tokens (index `i` of one matches index `i` of the other).
- `train` — a list of `[cable_string, label]` pairs; `label` is `1` (well-spliced) or
  `0` (not). This is your learning signal.
- `queries` — cable strings **without** labels. Predict each one.

## Output (stdout): one JSON object

```json
{"labels": [0, 1, 1, 0, ...]}
```

- `labels` must be a list of **exactly `len(queries)`** integers, each `0` or `1`,
  in the **same order** as `queries`.
- Wrong length, non-integer or out-of-`{0,1}` entries, non-JSON, a crash, or a
  timeout ⇒ that instance scores **0**.

## Objective & scoring (deterministic)

Let `acc_cand` be the fraction of queries whose predicted label matches the hidden
truth, and let `acc_base` be the accuracy of the best **constant** classifier on the
hidden query labels. Every instance is constructed so that "not well-spliced" is the
strict majority, hence `acc_base > 0.5`. Each instance is normalised with an affine
anchor:

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (1 - acc_base), 0, 1 )
```

- Predicting the majority class ⇒ `acc_cand = acc_base` ⇒ `r ≈ 0.1`.
- Doing worse than the majority constant ⇒ `r < 0.1` (floored at 0).
- Correctly generalising the matching rule to the OOD cables ⇒ higher `r`.
- A perfect score requires **guessing the hidden depth budget `D`**, which the
  training set does not reveal — so there is real headroom.

Your score is the mean of `r` over all `10` instances. **Maximize it.**

## Isolation

Your program runs in an isolated subprocess. It only ever sees the JSON above (train
labels and query **strings**). The hidden query labels, the true rule, and the depth
budget `D` never leave the evaluator process — introspection buys you nothing.
