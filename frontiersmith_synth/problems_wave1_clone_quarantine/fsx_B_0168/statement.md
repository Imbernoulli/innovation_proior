# Winter Provisioning of a Polar Research Base Supply Tree

A national polar programme resupplies a network of interior stations from a single
coastal logistics **hub** (node 0). Resupply flows hub → main base → traverse depot →
… → deep-field camp, and every leg adds a fixed transit/processing time. The
**net replenishment lead time** `L_i` of station `i` is the cumulative time along its
path from the hub, so deep-field camps face the longest, riskiest lead times.

Over its lead time each station `i` faces uncertain consumable demand (fuel, food,
medical, spares), modelled as `Normal(mu_i, sd_i^2)` with `sd_i = sigma_i * sqrt(L_i)`.
Before the winter freeze you must decide the **safety-stock quantity** `stock_i ≥ 0`
held at every station. Writing the safety factor `k_i = stock_i / sd_i`, the expected
over-winter shortfall (backorders) at the station is

```
B_i = sd_i * Loss(k_i),      Loss(k) = phi(k) - k * (1 - Phi(k))
```

where `phi`/`Phi` are the standard-normal pdf/cdf and `Loss` is the standard-normal
loss function (decreasing in `k`).

## Objective (minimize)

A composite holding-plus-shortage cost over all stations:

```
cost = sum_i [ h_i * stock_i  +  p_i * B_i ]
```

subject to **two coupled service constraints**:

1. a **programme-wide fill-rate floor**

   ```
   fill = 1 - (sum_i B_i) / (sum_i mu_i)  >=  beta
   ```

2. for every designated **life-support station** `c` (`crit[c] == 1`) a **local
   fill-rate floor** (a med / power / fuel post may not run out even if the network as
   a whole is fine)

   ```
   fill_c = 1 - B_c / mu_c  >=  alpha
   ```

Any answer that violates constraint (1) or (2), or that is malformed / has a negative
or non-finite `stock_i`, is **infeasible and scores 0**.

There is no easy optimum. Each station alone has a newsvendor optimum
`k*_i = Phi^{-1}(1 - h_i/p_i)`, but that node-wise choice usually violates both floors.
The classic single-multiplier "water-filling" fix for the programme floor (1) alone
tends to *starve* expensive, high-variance life-support stations, breaking their local
floors (2). A good design must reconcile both: pin each critical station to its local
requirement `k_c^min = Loss^{-1}((1 - alpha) * mu_c / sd_c)`, then re-allocate the
remaining service across the tree to satisfy `beta` at least total cost.

## Public instance (stdin JSON)

```json
{
  "N": 28,                      // number of stations (node 0 = hub / port)
  "parent": [-1, 0, 0, 1, ...], // parent[i] = supplier of station i (-1 for the hub)
  "L":     [2, 5, 3, ...],      // net replenishment lead time per station
  "mu":    [40.0, ...],         // mean lead-time demand per station
  "sigma": [30.0, ...],         // per-period demand std per station
  "h":     [1.5, ...],          // holding cost per unit of safety stock
  "p":     [9.0, ...],          // shortage penalty per expected backorder unit
  "sd":    [42.4, ...],         // = sigma_i * sqrt(L_i) (lead-time demand std)
  "beta":  0.92,                // programme fill-rate floor
  "alpha": 0.98,                // local fill-rate floor for life-support stations
  "crit":  [0, 1, 0, ...]       // crit[i] == 1 iff station i is life-support
}
```

## Answer (stdout JSON)

```json
{"stock": [s_0, s_1, ..., s_{N-1}]}    // safety stock per station, each >= 0
```

## Scoring

The evaluator computes a **baseline** `b` = the cost of the "top-off-every-locker"
design that stocks every station to safety factor `4` (always feasible for both floors,
heaviest holding cost). For a feasible answer with objective `obj`:

```
r = min(1, 0.1 * b / obj)
```

so the top-off design scores exactly `0.1`, and a design `k×` cheaper than baseline
scores `min(1, 0.1k)`. The reported `Ratio` is the mean of `r` over 12 deterministic,
seeded instances (including larger held-out ones with more life-support stations).
Infeasible or malformed answers score `0` on that instance.

Your program reads one public instance JSON from stdin and writes one answer JSON to
stdout. It runs in an isolated subprocess and only ever sees the public instance.
