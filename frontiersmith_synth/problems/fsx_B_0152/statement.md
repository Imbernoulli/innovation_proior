# Deep-Sea Cable Spare-Parts Positioning on a Supply Tree

A submarine-cable operator keeps spare parts (optical repeaters, branching units,
cable drums) at a network of maintenance depots. The depots form a **supply tree**
rooted at a central manufacturing hub: a part is shipped hub → regional depot → …
→ forward depot, and every hop adds a fixed transit/processing time. The
**net replenishment lead time** `L_i` of depot `i` is the cumulative time along its
path from the hub, so depots deeper in the tree face longer, riskier lead times.

Over its lead time, depot `i` faces uncertain spares demand, modelled as
`Normal(mu_i, sd_i^2)` with `sd_i = sigma_i * sqrt(L_i)`. You must decide the
**safety-stock quantity** `stock_i ≥ 0` held at every depot. Writing the safety
factor `k_i = stock_i / sd_i`, the expected unmet demand (backorders) at the depot is

```
B_i = sd_i * Loss(k_i),      Loss(k) = phi(k) - k * (1 - Phi(k))
```

where `phi`/`Phi` are the standard-normal pdf/cdf and `Loss` is the standard-normal
loss function (decreasing in `k`).

## Objective (minimize)

A composite holding-plus-shortage cost over all depots:

```
cost = sum_i [ h_i * stock_i  +  p_i * B_i ]
```

subject to a **network service-level (fill-rate) constraint**:

```
fill = 1 - (sum_i B_i) / (sum_i mu_i)  >=  beta
```

Any answer whose aggregate fill rate is below `beta`, or that is malformed / has a
negative or non-finite `stock_i`, is **infeasible and scores 0**.

There is no easy optimum. Each depot alone has a newsvendor optimum
`k*_i = Phi^{-1}(1 - h_i/p_i)`, but on tight instances that node-wise choice
violates the network fill-rate floor, so service must be re-allocated across the
tree — pooling extra stock onto cheap, low-variance depots — to satisfy `beta` at
least total cost.

## Public instance (stdin JSON)

```json
{
  "N": 8,                       // number of depots (node 0 = hub / root)
  "parent": [-1, 0, 0, 1, ...], // parent[i] = supplier of depot i (-1 for the root)
  "L":     [2, 5, 3, ...],      // net replenishment lead time per depot
  "mu":    [40.0, ...],         // mean lead-time demand per depot
  "sigma": [30.0, ...],         // per-period demand std per depot
  "h":     [1.5, ...],          // holding cost per unit of safety stock
  "p":     [9.0, ...],          // shortage penalty per expected backorder unit
  "sd":    [42.4, ...],         // = sigma_i * sqrt(L_i) (lead-time demand std)
  "beta":  0.92                 // network fill-rate floor
}
```

## Answer (stdout JSON)

```json
{"stock": [s_0, s_1, ..., s_{N-1}]}    // safety stock per depot, each >= 0
```

## Scoring

The evaluator computes a **baseline** `b` = the cost of the "gold-plated" design that
stocks every depot to safety factor `4` (always feasible, heaviest holding cost). For
a feasible answer with objective `obj`:

```
r = min(1, 0.1 * b / obj)
```

so the gold-plate design scores exactly `0.1`, and a design `k×` cheaper than baseline
scores `min(1, 0.1k)`. The reported `Ratio` is the mean of `r` over 10 deterministic,
seeded instances (including larger held-out ones). Infeasible or malformed answers
score `0` on that instance.

Your program reads one public instance JSON from stdin and writes one answer JSON to
stdout. It runs in an isolated subprocess and only ever sees the public instance.
