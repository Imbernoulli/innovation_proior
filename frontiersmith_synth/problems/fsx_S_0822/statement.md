# The Gene You Can Lose

A body plan grows along a line of `L = 2^T` positions, `p = 0..L-1`. Each
position reads its own location from a **positional morphogen**: the binary
digits of `p` itself, `x_0..x_{T-1}` with `x_t = (p >> t) & 1` (bit `t`,
least-significant first). Your job is to design a small **gene regulatory
network (GRN)** that reads this morphogen and develops every position into a
tissue **type** in `[0, K)`, so the resulting body matches a fixed **target**
plan — not only when the genome is intact, but also after losing **any single
gene**.

## Candidate program contract

Standalone program: read ONE JSON object (public instance) from **stdin**,
write ONE JSON object (your GRN) to **stdout**. Runs isolated; sees only the
public instance.

### Public instance (stdin)
```json
{"name": "grn101", "L": 8, "T": 3, "K": 4, "G_max": 8, "iters": 3,
 "target": [0, 0, 2, 2, 1, 1, 1, 3]}
```
`target[p]` is the desired type at position `p`.

### Your answer (stdout)
```json
{"G": 4, "Win": [[...T ints...], ...G rows...],
 "W": [[...G ints...], ...G rows...],
 "bias": [...G ints...], "decode": [...2**G ints in [0,K)...]}
```
- `1 <= G <= G_max` — the number of genes you use.
- `Win` is `G x T`, `W` is `G x G`, each entry an integer in `[-3, 3]`.
- `bias` has `G` entries, each an integer in `[-8, 8]`.
- `decode` has exactly `2**G` entries, each an integer in `[0, K)`.

Any wrong shape, out-of-range value, non-integer, crash, timeout, or
non-JSON output scores that instance `0.0`.

## Development (run by the evaluator, deterministic)

For position `p`, form `x_0..x_{T-1}` as above. State `s` (length `G`) starts
all-zero. For `iters` synchronous rounds:
```
raw_i  = bias[i] + sum_t Win[i][t]*x[t] + sum_j W[i][j]*s[j]
s_i'   = 0                          if gene i is knocked out
       = 1 if raw_i > 0 else 0      otherwise
```
After `iters` rounds, form the code `c = sum_i s_i * 2^i` (gene 0 = least
significant bit) and the position's type is `decode[c]`. A **knockout of gene
g** means gene `g` is forced to `0` in every round, at every position, for the
whole development. The **wild type (WT)** is development with no gene
knocked out.

## Objective

For a fixed, seeded family of 10 instances (varying `T`, `K`, and target
shape — some larger/held-out), develop your GRN as the WT **and** under every
single-gene knockout `g = 0..G-1`. Let `match(phenotype)` be the fraction of
positions where the phenotype's type equals `target[p]`. Your **raw score**
for an instance is the mean of `match` over `{WT} ∪ {knockout(g) : all g}`.

**Maximize** the mean raw score over all 10 instances.

## Scoring (deterministic)

The evaluator computes, itself, `base` = the frequency of the single most
common type in the target (what a network that ignores position and just
always guesses the mode would score), and normalizes:
```
r = clamp(0.1 + 0.9 * (raw - base) / (1.2 - base), 0, 1)
```
Matching the mode-guess baseline scores ≈0.1; a network that reconstructs the
target perfectly in the WT **and** survives every knockout without losing any
information scores well below 1.0 — there is real headroom. The reported
**Ratio** is the mean of `r` over all 10 instances; **Vector** lists the
per-instance scores.

## Why this is harder than "just fit the target"

A network using exactly `T` genes, one per positional bit, always fits the WT
perfectly (its expression vector literally IS `p`'s binary address). But it
has a single point of failure per bit: knocking out the gene for bit `t`
forces that bit to `0` everywhere, re-mapping every position whose true bit
was `1` onto a *different* position's identity. Surviving knockouts needs
**regulatory redundancy** — extra genes so no single gene's loss destroys
positional information (developmental canalization) — which a minimal,
exactly-fitting network cannot provide.

## Suggested strategies

1. **Mode guess**: ignore everything, always output the most common type.
2. **Direct address**: one gene per bit, decode fit to the WT target exactly.
3. **Redundant / duplicated encoding**: spend extra genes so each positional
   bit is carried by more than one gene, and fit `decode` so every code
   reachable under any single knockout still resolves to the right type.
4. Search over gene budgets, weights, and decode tables to trade off WT fit
   against knockout robustness under a bounded `G`.
