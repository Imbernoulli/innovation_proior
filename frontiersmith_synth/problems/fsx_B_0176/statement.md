# Glacier Sensor Net: Spare-Module Staging on a Supply Tree with Storage-Locker Caps

A glacier-monitoring program runs a network of unmanned field stations (GPS pingers,
seismographs, weather masts) scattered across an ice sheet. Each station needs spare
hardware modules (battery packs, comms boards, heater cartridges) to survive between
service flights. Spares are flown from a **base camp** and staged forward through
relay caches, so the stations form a **supply tree** rooted at base camp (node 0):
a spare travels base camp → relay cache → … → forward station, and every hop across
the ice adds a fixed transit/staging delay. The **net replenishment lead time** `L_i`
of station `i` is the cumulative delay along its path from base camp, so stations
deep on the glacier face longer, riskier lead times.

Over its lead time station `i` faces uncertain module demand, modelled as
`Normal(mu_i, sd_i^2)` with `sd_i = sigma_i * sqrt(L_i)`. You choose the
**safety-stock quantity** `stock_i` held at each station. Writing the safety factor
`k_i = stock_i / sd_i`, the expected unmet demand (backorders) is

```
B_i = sd_i * Loss(k_i),      Loss(k) = phi(k) - k * (1 - Phi(k))
```

where `phi`/`Phi` are the standard-normal pdf/cdf and `Loss` is the standard-normal
loss function (decreasing in `k`).

## The twist: hard storage-locker capacities

Each station has a **heated equipment locker of fixed size**, so its safety stock is
capped:

```
0 <= stock_i <= cap_i
```

Crucially, the **cheapest, lowest-variance caches** — exactly the places an
unconstrained design would want to pile extra service onto — have the **tightest
lockers**. So the usual trick of pooling service onto cheap nodes overflows their
lockers and forces service to be re-routed onto more expensive stations.

## Objective (minimize)

A composite holding-plus-shortage cost over all stations:

```
cost = sum_i [ h_i * stock_i  +  p_i * B_i ]
```

subject to a **network availability (fill-rate) constraint**:

```
fill = 1 - (sum_i B_i) / (sum_i mu_i)  >=  beta
```

Any answer that violates a locker capacity (`stock_i > cap_i`), whose aggregate fill
rate is below `beta`, or that is malformed / has a negative or non-finite `stock_i`,
is **infeasible and scores 0**.

There is no easy optimum. Each station alone has a newsvendor optimum
`k*_i = Phi^{-1}(1 - h_i/p_i)`, but on tight instances that node-wise choice
violates the availability floor, and once you push extra service onto the cheap
caches their lockers saturate — so service must be re-allocated across the tree,
onto roomier but pricier stations, to satisfy `beta` at least total cost.

## Public instance (stdin JSON)

```json
{
  "N": 8,                        // number of stations (node 0 = base camp / root)
  "parent": [-1, 0, 0, 1, ...],  // parent[i] = supplier of station i (-1 for the root)
  "L":     [2, 5, 3, ...],       // net replenishment lead time per station
  "mu":    [40.0, ...],          // mean lead-time demand per station
  "sigma": [30.0, ...],          // per-period demand std per station
  "h":     [1.5, ...],           // holding cost per unit of safety stock
  "p":     [9.0, ...],           // shortage penalty per expected backorder unit
  "sd":    [42.4, ...],          // = sigma_i * sqrt(L_i) (lead-time demand std)
  "cap":   [95.0, ...],          // hard storage-locker capacity per station
  "beta":  0.93                  // network availability (fill-rate) floor
}
```

## Answer (stdout JSON)

```json
{"stock": [s_0, s_1, ..., s_{N-1}]}    // safety stock per station, 0 <= s_i <= cap_i
```

## Scoring

The evaluator computes a **baseline** `b` = the cost of the "full-locker gold-plate"
design that stocks every station to safety factor `4` clamped to its locker
(`stock_i = min(4*sd_i, cap_i)`); this is always feasible and carries the heaviest
holding cost. For a feasible answer with objective `obj`:

```
r = min(1, 0.1 * b / obj)
```

so the full-locker design scores exactly `0.1`, and a design `k×` cheaper than
baseline scores `min(1, 0.1k)`. The reported `Ratio` is the mean of `r` over 10
deterministic, seeded instances (including larger, tighter held-out ones).
Infeasible or malformed answers score `0` on that instance.

Your program reads one public instance JSON from stdin and writes one answer JSON to
stdout. It runs in an **isolated subprocess** and only ever sees the public instance.
