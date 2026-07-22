# Bolting Lamps to a Rotating Cliff

A lighthouse keeper serves a request trace over `N` lamp keys. The cliff rotates,
so which lamps are *hot* drifts over time. To answer a request fast the keeper can
**bolt** (pin) a lamp into one of `p` fast L1 slots. There is also a fixed **L2**
cache of size `q` that the keeper does **not** control — it is a plain LRU that
holds only the most recently *served-not-bolted* lamps.

The trace is split into `E` equal **epochs** of `L` requests each (request `i` lies
in epoch `⌊i/L⌋`). You choose, for every epoch, the set of bolted lamps.

## Answering a request in epoch `t`
For a request to lamp `x`:
- if `x` is bolted in epoch `t` → cost `C_PIN`;
- else if `x` is in L2 → cost `C_L2`, and `x` becomes most-recently-used;
- else → cost `C_MISS`, and `x` is inserted into L2 (evicting its LRU entry if full).

Bolted lamps never touch L2.

## Reconfiguration
Bolting a lamp that was **not** bolted in the previous epoch costs `C_SWAP`, and at
most **`Bmax` new lamps may be bolted per epoch boundary** (this is a hard limit —
un-bolting is free and unlimited). Epoch 0 is a free install: you may bolt up to `p`
lamps, each still charged `C_SWAP`.

Because `Bmax < p`, when several lamps turn hot in the *same* epoch you cannot bolt
them all at that boundary: to have a lamp bolted when it peaks you may have to bolt
it **one epoch early**, splitting the reconfiguration across two epochs. Frequency
ranking never reveals this.

## Input (stdin)
```
testId
N E L p q Bmax
C_PIN C_L2 C_MISS C_SWAP
T
r_0 r_1 ... r_{T-1}      (T = E*L request keys, whitespace separated)
```

## Output (stdout)
Exactly `E` lines. Line `t` describes the bolt-set for epoch `t`:
```
m  k_1 k_2 ... k_m
```
with `0 ≤ m ≤ p`, the `k_j` distinct and in `[0, N)`.

## Feasibility
A submission is rejected (score `0`) unless every line is well-formed as above and,
for every `t ≥ 1`, the number of newly bolted lamps
`|bolt[t] \ bolt[t-1]| ≤ Bmax`.

## Objective (minimize)
Total cost = sum of per-request access costs + `C_SWAP ×` (total lamps ever bolted,
counting each epoch's new bolts). Let `F` be your total cost and `B` the cost of the
**bolt-nothing** schedule (all epochs empty), replayed identically. Your score is
```
Ratio = min(1000, 100 · B / F) / 1000
```
so bolting nothing scores `0.1` and driving cost down raises the ratio.

## What makes this hard
- The hot set **drifts**: a lamp set bolted once goes stale.
- The lamp with the **highest access frequency** is served almost for free by L2
  (it arrives in tight bursts). Bolting it by frequency wastes a slot; what matters
  is *avoided cost* = frequency × cost-if-unbolted, not frequency.
- At **regime** epochs more than `Bmax` lamps turn hot at once, so a purely reactive
  schedule is always behind. The exact bonus/weight structure is only visible in the
  trace — read it and exploit it.

## Constraints
`E ≤ 18`, `L ≤ 1300`, `p = 5`, `q = 4`, `Bmax = 2`, `N = 6000`.
Costs: `C_PIN=1, C_L2=4, C_MISS=40, C_SWAP=150`. Time limit 5 s, memory 512 MB.

## Example (scoring only)
With `E=3` and a bolt-nothing baseline cost `B = 9000`, a schedule costing
`F = 3000` scores `min(1000, 100·9000/3000)/1000 = 0.300`. (Illustrative numbers,
not a real instance.)
