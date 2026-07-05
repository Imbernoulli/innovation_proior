# Pandemic Supply-Tree Stockpile — Safety-Stock Placement (Format B, isolated)

A pandemic-response agency must pre-position a fixed stockpile of a perishable
critical supply (rapid tests / antiviral courses) across a **supply tree** of health
districts before a demand surge. The districts form a rooted tree: **node 0 is the
central depot** (a pure warehouse with no local patients); every other node `i` is a
district with a parent `parent[i]`. You decide **how many units of stock to place at
each node** — an integer safety-stock level — subject to a total budget.

Demand is random. You are told each district's demand **mean and standard deviation**,
but NOT the actual demand realizations — those are hidden, seeded scenarios the grader
uses to score you. You must choose a **robust** allocation from the distribution alone.

## How the supply tree serves demand (per scenario)

Stock flows **only downstream** (from a node to itself and its descendants), never to
siblings. For each demand scenario the grader runs this deterministic cascade:

1. Every district first uses its own stock to cover its own demand.
2. Any **unmet** demand escalates **up** the tree. An ancestor with leftover stock can
   cover the shortfall of any district in its subtree (risk pooling) — but **every unit
   served this way costs an escalation fee `e`** (cross-district transfer / delay).
3. Demand still unmet at the depot is lost — an **unserved-case shortage penalty `p`** per
   unit.
4. Any stock never used anywhere incurs a **holding/spoilage cost `h`** per unit.

Because a node's stock only reaches its own subtree, **where** in the tree you hold
stock matters: local stock avoids escalation fees but cannot help other districts and is
wasted if that district's surge is small; central (depot / internal-node) stock pools
risk across a whole subtree but pays the escalation fee whenever it is tapped.

## Objective

Minimize the **expected total cost** `E[ h·(unused) + e·(escalated) + p·(unserved) ]`,
averaged over the hidden scenarios. **Lower is better.**

## Feasibility (any violation → score 0)

- **Budget:** `sum(stock) ≤ B`.
- **Integrality & range:** each `stock[i]` an integer in `[0, B]`.
- **Service level:** the network **fill rate** `1 − (total unserved)/(total demand)`
  over all scenarios must be `≥ target`.

An ill-formed answer, a non-finite / non-integer value, an over-budget plan, or a plan
that misses the service target all score **0**.

## Public instance (stdin JSON)
```json
{
  "parent": [-1, 0, 0, 1, ...],   // parent[i]; parent[0] = -1 (depot is the root)
  "means":  [0.0, m1, m2, ...],   // per-node mean demand (means[0] = 0, depot has none)
  "stds":   [0.0, s1, s2, ...],   // per-node demand std dev
  "N":      <int>,                 // number of districts; nodes are 0..N
  "B":      <int>,                 // total stockpile budget (units)
  "h":      1.0,                   // holding/spoilage cost per unused unit
  "e":      4.0,                   // escalation fee per cross-district unit
  "p":      25.0,                  // shortage penalty per unserved unit
  "target": 0.80,                  // required network fill rate
  "K":      200                    // number of hidden scenarios (values not shown)
}
```

## Answer (stdout JSON)
```json
{"stock": [x0, x1, ..., xN]}     // exactly N+1 integers, xi in [0,B], sum <= B
```

## Scoring

Per instance: `score = min(1, 0.1 · C_baseline / C_yours)`, where `C_baseline` is the
expected cost of the **uniform split** (budget divided evenly across depot + all
districts). The final score is the mean over 10 fixed, seeded instances that vary in
tree shape, district count, budget tightness, and demand spread. Beating the uniform
baseline requires matching supply to each district's need AND deciding how much risk to
pool centrally — there is no closed-form optimum.

## Suggested strategies (increasing sophistication)

- **Uniform split** — one share per node; ignores demand heterogeneity (the baseline).
- **Demand-proportional local sizing** — allocate in proportion to mean demand; no buffer
  and no central reserve.
- **Distribution-driven marginal allocation** — sample your own demand from the given
  mean/std and add stock one unit at a time where expected cost drops most, discovering
  per-district buffers and pooling reserves.
- **Newsvendor + pooling reserve** — set each district's buffer from the critical ratio
  and hold a variance-scaled reserve at the depot / internal nodes, tuned to the
  escalation-vs-shortage trade-off.
