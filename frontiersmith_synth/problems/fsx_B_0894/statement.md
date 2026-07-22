# Binder Hub: Crosstalk-Coupled Line Vectoring

A vectoring hub serves `N` subscriber lines bundled together in the same cable
binder. The hub has one shared transmit-power budget `P` to split across the
lines, `p_0, ..., p_{N-1}` with `sum p_i <= P` and every `p_i >= 0`.

Lines are **not** independent: line `j`'s transmitted power leaks into line
`i`'s receiver as crosstalk, at a fixed gain `A[i][j] >= 0` (a hidden `N x N`
coupling matrix, `A[i][i] = 0`, generally **asymmetric** — `A[i][j]` need not
equal `A[j][i]`). The hub also has a joint decoder that can perform
**successive interference cancellation (SIC)**: it commits to one global
**decoding order** over the `N` lines. Once a line is decoded, its exact
transmitted signal is known, so every *later* line's receiver can have that
line's crosstalk contribution subtracted for free. A line decoded **first**
still sees crosstalk from all `N-1` other lines (nothing cancelled yet); a
line decoded **last** sees none (everyone else already cancelled). Formally,
if line `c` sits at position `k` in the order, the interference it suffers is

```
I_c = sum over lines d decoded strictly AFTER c of  A[c][d] * p_d
```

and its achievable rate is `R_c = log2(1 + gain[c]*p_c / (noise[c] + I_c))`.
Your job: choose **both** the power split and the decode order to maximize
the total sum-rate `sum_c R_c` across the binder.

## Candidate program contract

Standalone program: read ONE JSON object (public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs in an isolated
subprocess, seeing only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... choose power[] and order[] ...
print(json.dumps({"power": power, "order": order}))
```

### Public instance (stdin)

```json
{
  "name": "cwf7004",
  "n": 6,
  "budget": 9.6,
  "gain":  [g_0, ..., g_5],
  "noise": [n_0, ..., n_5],
  "coupling": [[0, A01, ...], [A10, 0, ...], ...]
}
```
`gain[i] > 0`, `noise[i] > 0`, `coupling` is `n x n`, non-negative, zero
diagonal, generally asymmetric.

### Answer (stdout)

```json
{ "power": [p_0, ..., p_5], "order": [3, 5, 0, 1, 4, 2] }
```
- `power`: exactly `n` finite non-negative numbers with `sum(power) <= budget`
  (a tiny numerical tolerance is allowed).
- `order`: exactly `n` values forming a **permutation** of `0..n-1`.
  `order[0]` decodes first (sees everyone's interference); `order[-1]`
  decodes last (sees none).

Any wrong length, non-permutation `order`, an over-budget or negative
`power`, a crash, a timeout, or non-JSON output scores that instance `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes, itself, two references using the
**same** rate model described above:

- `UB` — the coupling-**free** convex upper bound: the global optimum of
  `sum_i log2(1 + gain_i*p_i/noise_i)` subject to `sum p_i <= P` (closed-form
  water-filling). Since every `A[i][j] >= 0`, no real (power, order) can ever
  beat `UB` — it is a valid, generally-unreachable ideal.
- `BASE` — the "obvious" recipe scored with the **real** (coupled) rate
  model: the coupling-free water-filling power split, decoded in plain index
  order `0,1,...,n-1` — i.e. treating the `n` lines as if they didn't
  interfere at all, then just handing that answer to the real system.

```
r = clamp( 0.1 + 0.9 * (R_yours - BASE) / max(1e-9, UB - BASE), 0, 1 )
```

Matching `BASE` scores ≈ 0.1; approaching `UB` (rarely fully reachable when
crosstalk is nonzero) scores near 1.0; doing worse than `BASE` scores below
0.1. **Ratio** is the mean of `r` over 10 fixed instances; **Vector** lists
the per-instance scores. Several instances have deliberately strong,
**asymmetric** crosstalk concentrated on a small cluster of boosted-gain
lines — exactly the case where ignoring the coupling matrix and decoding in
plain index order is costly.

## Suggested strategies

1. Split the budget uniformly, decode in index order (do-nothing baseline).
2. Textbook single-user water-filling over `gain`/`noise`, ignoring
   `coupling` entirely, decode in index order — the obvious first move.
3. Probe the coupling matrix to see which lines impose the most crosstalk
   on the rest, pick a decode order informed by that, then **iterate**:
   recompute each line's effective noise from the crosstalk still
   uncancelled under the chosen order, re-water-fill against that effective
   noise, and repeat to a fixed point.
