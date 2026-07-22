# Consignment Sort: Hashing the Cartel's Manifest

A parcel-forwarding depot has quietly been laundering shipments for a cartel.
Every parcel is stamped with an integer **tracking code**. To keep the manifest
looking innocuous, the cartel issues codes in tidy internal batches: a single
arithmetic run, a two-axis batch grid (carton-in-pallet stride and
pallet-in-truck stride composed together), or a repeated "lifted" set of
allowed check-digit residues. Whatever the scheme, **every pair of codes in an
instance differs by a multiple of one fixed BATCH STRIDE** — the instance's
hidden lattice determinant. You never see this stride directly; you only see
the codes.

The depot must route each parcel to one of `B` outbound trucks using a
**composed affine hash**: pick integers `a`, `c`, `M` and route code `x` to
truck

```
bucket(x) = ((a*x + c) mod M) mod B
```

Overloading a truck (far more parcels than its fair share) blows the day's
schedule and draws exactly the scrutiny the cartel wants to avoid. Your job:
choose `(a, c, M)` so that the **largest truck load is as small as possible**.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. It runs in an isolated
subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a, c, M ...
print(json.dumps({"a": a, "c": c, "M": M}))
```

### Public instance (stdin)

```json
{
  "name": "manifest90101",
  "n": 3000,
  "B": 100,
  "codes": [1044213, 2092789, 3141365, ...]
}
```

### Answer (stdout)

```json
{ "a": 733695931, "c": 999331, "M": 4000003 }
```

- `M` must be an integer with `2 <= M <= 5000000000`.
- `a` must be an integer with `1 <= a < M`.
- `c` must be an integer with `0 <= c < M`.

Any invalid output (missing/non-integer field, `M`/`a`/`c` out of range), a
crash, a timeout, or non-JSON output makes that instance score `0.0`.

## Objective

**Minimize**, across a fixed seeded family of 10 instances (varying code
count, truck count `B`, and batching scheme — single runs, two-axis grids,
lifted-subgroup cosets, and a few instances with no planted structure at all),
the largest truck load your composed hash produces.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb` = `ceil(n / B)` — the pigeonhole lower bound (a loose, generally
  unreachable ideal),
- `q_base` = the max truck load of routing every parcel by **raw `code mod B`**
  (a weak reference that ignores all hash parameters),
- `q_cand` = the max truck load your `(a, c, M)` actually produces,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.75 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

Matching the raw baseline scores ≈ `0.1`; reaching the pigeonhole bound scores
≈ `0.85` (real headroom is left above it, since it is rarely achievable
exactly). Doing *worse* than the raw baseline scores below `0.1`.

The reported **Ratio** is the mean of `r` over all 10 instances; **Vector**
lists the per-instance scores.

## Why a fixed hash can fail badly

Every instance's codes lie on an affine lattice with pairwise-GCD stride `D`.
An `M` sharing a prime factor with `D` cannot unwind that periodicity before
the final `mod B` — many codes collide onto a few residues and a truck gets
crushed. This is especially brutal for the "power-of-two table size" idiom,
since several instances plant `D` as a large power of two on purpose. A
modulus with no shared prime factor against `D` restores full spread — but
`D` differs (and is unknown) per instance, so `M` cannot be fixed in advance.

## Suggested strategies

1. **Raw mod-B routing** (baseline): ignore all structure.
2. **Static multiplicative hashing**: one large odd multiplier, one
   power-of-two modulus, reused for every instance.
3. **Lattice detection**: compute the GCD of pairwise code differences to
   recover the batch stride, factor it, and pick a modulus with no shared
   prime factor.
4. **Detection + local refinement**: after finding a coprime modulus, try a
   few nearby multipliers/shifts and keep whichever minimizes the observed
   max load.
