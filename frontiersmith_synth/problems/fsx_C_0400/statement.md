# Wildlife Corridor Flow-Balance: Length Generalization

## Story

Field ecologists have instrumented a **wildlife corridor** with a chain of monitoring
**gates**. Every animal crossing is logged with a **gate tag** — an integer symbol in
`0 .. A-1` recording which kind of gate was used. Each tag carries a *hidden* **net-flow
weight** `w[tag] ∈ {-3,-2,-1,0,1,2,3}`: a positive weight means that gate tends to route
animals **into** the protected core, a negative one **out** of it.

Reading a whole corridor log (a sequence of tags), the **net flow** of that segment is
the sum of the weights of its tags. Given a disclosed **tolerance** `T`, the segment is
classified into three states:

| label | name     | condition               |
|-------|----------|-------------------------|
| 0     | sink     | `net_flow < -T`         |
| 1     | balanced | `-T <= net_flow <= T`   |
| 2     | source   | `net_flow > T`          |

The per-tag weights `w[.]` are **hidden**. A survey hands you a batch of **short,
already-classified** logs (the *training* split, at *in-distribution* lengths) and a
batch of **unlabelled** logs to classify (the *test* split). Crucially, the test split
mixes in-distribution lengths with **much longer, out-of-distribution logs**. The
tolerance `T` is given; the weights are not.

This is a pure **length-generalization** task. A predictor that merely *memorizes* the
short training logs matches their surface statistics but cannot extrapolate to long
logs; a predictor that *recovers the underlying per-tag flow rule* classifies any length
exactly. The in-distribution-vs-length-OOD gap is exactly what the score rewards.

## Your program (isolated stdin -> stdout)

You write a standalone program. It reads ONE JSON object (the **public instance**) from
stdin and writes ONE JSON object to stdout. It runs in an isolated sandbox and only ever
sees the public instance below — the hidden weights and the true test labels never leave
the grader.

### Input (stdin) — the public instance
```json
{
  "name": "corridor10101",
  "A": 4,
  "T": 2,
  "train": [ {"seq": [0,3,1,2,0], "label": 2}, ... ],
  "test":  [ [1,0,2,3,0,1], ... ]
}
```
- `A` — number of distinct gate tags; every symbol is an integer in `[0, A-1]`.
- `T` — the disclosed tolerance (a non-negative integer).
- `train` — labelled **short** logs; each is `{"seq": [ints], "label": 0|1|2}`.
- `test` — a list of **unlabelled** logs (lists of ints) to classify. Some are at the
  same short lengths as training; others are **much longer**.

### Output (stdout)
```json
{"labels": [l_0, l_1, ..., l_{M-1}]}
```
Exactly `M = len(test)` integers, each in `{0, 1, 2}`, where `l_i` is your predicted
class for `test[i]`, in the same order.

Any wrong length, an out-of-range / boolean / non-integer label, a crash, a timeout, or
non-JSON output makes that instance score **0.0**.

## Objective & scoring (deterministic, exact-match)

For each instance the grader knows the true test labels and computes:

- `acc_cand` — the fraction of test logs your program classifies correctly;
- `acc_base` — the accuracy of the **majority-of-train** constant predictor (predict the
  most frequent training label — ties broken toward the smaller label — for every test
  log).

It then normalizes with an affine anchor (majority baseline → `0.1`, perfect → `1.0`):

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1e-9, 1 - acc_base), 0, 1 )
```

The reported `Ratio` is the mean of `r` over all 12 instances; `Vector` lists the
per-instance `r`. A program that echoes the majority label scores about `0.1`; one that
classifies every log exactly scores `1.0`; one worse than the majority baseline scores
below `0.1`.

Because the short training logs only pin the weights down to a **band** (the tolerance
`T` leaves integer slack, and short logs rarely exercise every tag heavily), even a
principled weight-recovery solver disagrees with the truth on some long OOD logs — so
there is genuine headroom below `1.0`, and several distinct strategies (memorize vs.
recover-the-rule vs. average over consistent rules) trade off differently.

## Notes
- Deterministic: the instance distribution is fixed and seeded; re-running the grader
  reproduces the identical `Ratio` and `Vector`.
- Isolation: your program runs in a fresh sandbox and sees only the public instance.
  The hidden weights and true labels stay in the grader process.
