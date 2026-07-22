# Cascade Pricing: Igniting Network Adoption Across Regions

A product launches across **M regional markets** sitting on a fixed **influence
network**. Region `i`'s population has a base willingness-to-pay `base_i`, but that
willingness **rises with how much other regions have already adopted** — a
positive **network externality**. Formally, the perceived value in region `i` given
the adoption fractions `x = (x_0, ..., x_{M-1})` (each `x_j` in `[0,1]`) is

```
v_i(x) = base_i + gamma_i * sum_j W[i][j] * x_j
```

where `W[i][j] >= 0` is how much region `j`'s adoption feeds region `i`'s value
(the network need not be symmetric — a "hub" region can shape everyone downstream
while depending on no one itself). Given region `i`'s price `p_i`, its population's
idiosyncratic taste (spread `spread_i`) settles the adoption fraction at the
logistic best response

```
x_i = 1 / (1 + exp((p_i - v_i(x)) / spread_i)).
```

Because each `x_i` depends on every other `x_j`, the market-wide adoption vector is
a **best-response equilibrium**: starting from zero adoption, every region
repeatedly best-responds to everyone else's current adoption until nothing moves.
This process is monotone non-decreasing, so it converges to a unique fixed point —
but *which* fixed point depends entirely on how prices sit relative to each
region's network-boosted value.

## Your job

Choose **one price per region**. You are graded on the total revenue at the
induced equilibrium:

```
Revenue(p) = sum_i p_i * x_i * pop_i
```

where `pop_i` is region `i`'s market size. Higher prices earn more per adopter but
suppress adoption locally *and, through the network, downstream*; lower prices
spread adoption but sacrifice margin. A region with modest local value but large
downstream influence can be worth far more to the network than to itself.

## Input (public instance, one JSON object on stdin)

```
{"name": str, "m": M,
 "base":   [base_0, ..., base_{M-1}],     # base_i > 0
 "gamma":  [gamma_0, ..., gamma_{M-1}],   # gamma_i >= 0, externality sensitivity
 "spread": [spread_0, ..., spread_{M-1}], # spread_i > 0
 "pop":    [pop_0, ..., pop_{M-1}],       # pop_i > 0
 "W":      [[W_00, ..., W_0(M-1)], ...]}  # M x M, W[i][j] >= 0
```

## Output (one JSON object on stdout)

```
{"prices": [p_0, ..., p_{M-1}]}     # p_i finite, 0 <= p_i <= 1e6
```

Wrong length, a negative / NaN / infinite entry, a crash, a timeout, or non-JSON
output makes that instance score **0.0**.

## Scoring (deterministic)

For each instance we precomputed, once while authoring this problem (never at
grading time, from the same public data you see), two reference revenues:
`R_flat` — the best revenue reachable by a **single flat price** applied to every
region (network-blind, region-blind), and `R_ceil` — the best revenue an extensive
coordinate-ascent search found by explicitly trying sacrifice-and-cascade
strategies. At grading time we compute `R_cand`, your revenue at the equilibrium
your prices induce, and normalise

```
r = clamp( 0.1 + 0.9 * (R_cand - R_flat) / (1.3 * (R_ceil - R_flat)), 0, 1 ).
```

The flat baseline scores ~0.1; the search reference scores ~0.77 (headroom is
deliberately left above it); pricing worse than flat scores below 0.1. Your final
score is the mean of `r` over the 10 instances.

## What matters

A region's **own** revenue-maximizing price (computed pretending no one else ever
adopts) ignores what its adoption is *worth to the regions that depend on it*.
The quantity that matters for a region's strategic value is its **downstream
influence**, roughly `sum_j gamma_j * W[j][i] * pop_j` — how much value region
`i`'s adoption injects into everyone else once it fires. Pricing a
high-downstream-influence region at its own local optimum can leave the whole
network below the threshold needed for a cascade; sacrificing its margin — even
toward zero — to push its adoption toward 1, then repricing the rest of the
network against the value that cascade delivers, is where the real revenue lives.
The exact `base`, `gamma`, `spread`, `pop`, and `W` values live in the input —
read them and exploit them.
