# Tiny Sprite Classifier: Design the Augmentation Policy

## Setting

A hobbyist is training a **tiny image classifier on a CPU**. The data are `12x12`
grayscale **sprites** in `K = 4` shape classes:

| label | shape          |
|-------|----------------|
| 0     | vertical bar   |
| 1     | horizontal bar |
| 2     | diagonal       |
| 3     | anti-diagonal  |

Every sprite is a class prototype jittered by an **unknown** amount of integer
**translation**, additive **Gaussian noise**, and a global **brightness** offset. Only a
handful of labelled sprites per class are available for training; a much larger,
independently-jittered **test set is held out** and never shown to you.

The classifier is **fixed**: a **1-nearest-neighbour** matcher. At test time each sprite is
labelled by its Euclidean-nearest neighbour in the training **gallery** (ties broken by lowest
gallery index). The only thing you control is the **data-augmentation policy** — a short list
of label-preserving transforms the trainer applies to each training sprite to enlarge the
gallery before matching.

Your goal: **design an augmentation policy that maximizes held-out accuracy.**

Because a shifted test sprite can only match a shifted gallery sprite, manufacturing gallery
variants that resemble the **test-time jitter** (small shifts / noise / brightness) raises
accuracy. Injecting transforms the test distribution never contains (mirror **flips**, heavy
**contrast**, **cutout**) plants confusers and *drops* accuracy below no-augmentation.

## You write a program (stdin -> stdout)

Read ONE JSON object (the public instance) from stdin and write ONE JSON object (your policy)
to stdout. You never see the held-out test set or the true jitter parameters — infer what you
need from the training sprites.

### Input (public instance)

```json
{
  "name": "sprite101",
  "H": 12, "W": 12, "K": 4,
  "n_train": 16,
  "train_images": [[<144 floats, row-major 12x12>], ...],
  "train_labels": [<int 0..3>, ...],
  "vocab": {
    "shift":    {"mag":    "int in [0,4]"},
    "noise":    {"std":    "float in [0.0,0.5]"},
    "bright":   {"delta":  "float in [0.0,0.5]"},
    "contrast": {"factor": "float in [0.0,1.0]"},
    "flip":     {"axis":   "'h' or 'v'"},
    "cutout":   {"size":   "int in [1,6]"}
  },
  "limits": {"max_ops": 6, "max_copies_per_op": 8, "total_copies": 10}
}
```

### Output (your policy)

```json
{"ops": [
  {"type": "shift",  "mag": 3,     "copies": 4},
  {"type": "noise",  "std": 0.15,  "copies": 1},
  {"type": "bright", "delta": 0.12, "copies": 1}
]}
```

Each op adds `copies` freshly-transformed variants of **every** training sprite to the
gallery (labels preserved). The transforms:

- `shift`   — translate by a random integer `dx, dy` in `[-mag, mag]` (zero fill).
- `noise`   — add `Gaussian(0, std)` per pixel.
- `bright`  — add one random constant in `[-delta, delta]` to the whole sprite.
- `contrast`— scale pixels around their mean by a random factor in `[1-factor, 1+factor]`.
- `flip`    — mirror horizontally (`"h"`) or vertically (`"v"`).
- `cutout`  — zero a random `size x size` patch.

An **empty** policy (`{"ops": []}`) is legal and means *no augmentation*.

## Validity

A policy is valid iff it is an object with an `"ops"` list of at most `max_ops` entries; each
entry has a known `"type"`, the required **finite, in-range** parameter, and an integer
`"copies"` in `[1, max_copies_per_op]`; and the **sum of all `copies` is `<= total_copies`**.
Any violation, a crash, a timeout, or non-JSON output scores **0.0** on that instance.

## Objective & scoring (deterministic)

Maximize held-out 1-NN accuracy. Per instance the evaluator computes three accuracies on the
same hidden test set:

- `a_base`   — no augmentation (weak reference),
- `a_oracle` — augmenting with the **true** jitter parameters at a copy budget far above your
  cap (a generally-unreachable strong reference),
- `a_cand`   — augmenting with **your** policy,

and normalizes

```
r = clamp( 0.1 + 0.9 * (a_cand - a_base) / max(1e-9, a_oracle - a_base), 0, 1 )
```

An empty policy scores `0.1`; a policy worse than no-augmentation scores below `0.1`; the
oracle's large copy budget (never granted to you) keeps even excellent policies below `1.0`.
Your score is the **mean of `r` over all instances** (including harder, larger-jitter held-out
ones). Everything is deterministic and seeded — no wall-clock, GPU, or randomness in scoring.

## Isolation

Your program runs in an isolated sandbox and only ever sees the public instance. The held-out
test set, the true jitter parameters, and all references live in the evaluator process; there
is no way to read them — you must design the policy from the training sprites alone.
