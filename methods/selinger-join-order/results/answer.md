# Selinger-style cost-based join-order optimization

**Problem.** Given a non-procedural relational query — a FROM-list of n tables plus a WHERE-clause of
predicates — choose an access path for each table, a method for each join, and the *order* in which to join
the tables, so as to minimize total execution cost. The result of joining n tables is order-independent, but
the cost is not (intermediate cardinalities swing by orders of magnitude), and the number of candidate plans
is n!·(n−1)! (orderings × binary-combination sequences) — ≈ 2.9×10³⁵ for n=20 — so exhaustive enumeration is
impossible.

## Key idea

Three moves collapse the search:

1. **Restrict to left-deep trees.** Always join one *base* table (the inner) onto the running composite (the
   outer): (((A⋈B)⋈C)⋈D). This eliminates the (n−1)! nesting factor, lets the inner use its own indexes,
   and pipelines the composite without materializing intermediates (unless a sort is wanted).

2. **Dynamic programming over subsets (principle of optimality).** The cheapest plan that produces a *set* S
   of tables — for a given output order — depends only on S, not on the internal order in which S was built
   (the applicable predicates, the composite cardinality, the available join methods, and the interesting
   orders are all functions of S alone). So memoize the best plan per subset and build bottom-up by subset
   size:

   ```
   optjoin(S) = min over a in S of
                  cost(optjoin(S − {a})) + join-cost(optjoin(S − {a}), a) + access-cost(a)
   ```

   Subsets number Σ C(n,k) = 2ⁿ; each tries ≤ n peels × m methods → **O(n·m·2ⁿ)** plan evaluations
   (≈ 4.1×10⁷ for n=20, m=2). The compact implementation form uses the additive recurrence
   `cost(join) = output-cardinality + cost(left) + cost(right)`, with the DP table updated whenever
   `new_cost < old_cost`.

3. **Keep interesting orders.** An output order named by ORDER BY/GROUP BY, or usable as a merge-join column,
   is *interesting*: a plan that arrives already sorted can save a downstream sort even if it is costlier on
   its own. So the DP keeps, per subset, the cheapest *unordered* plan **and** the cheapest plan per
   interesting order; a plan is only ever compared within its own order-slot. At the end, for an ORDER BY,
   compare (cheapest unordered + sort) vs (cheapest already in order) and take the min. Orders equated by
   join predicates collapse into one equivalence class, bounding the bookkeeping.

Two prunings sharpen it: **don't consider cross products** — only extend the composite along a join edge,
deferring Cartesian products as late as possible, so the DP visits only *connected* subsets — and the two
**join methods** are nested-loop and sort-merge (a two-way-join study showed one of these is essentially
always optimal, so a third buys nothing).

## Cost and selectivity model

Cost is `COST = PAGE FETCHES + W·(tuple-interface calls)`, W tuning I/O vs CPU. A per-boolean-factor
**selectivity** F estimates the surviving fraction: equality on an indexed column F = 1/ICARD (else 1/10);
column = column join F = 1/MAX(ICARD₁,ICARD₂), or 1/ICARD when only one side has an index; range column >
value F = (high−value)/(high−low) (else 1/3); BETWEEN uses (value₂−value₁)/(high−low) (else 1/4);
conjunction F₁·F₂, disjunction F₁+F₂−F₁·F₂. QCARD is the product of table cardinalities and all applicable
selectivities; RSICARD uses the sargable factors that filter inside the storage scan. Single-table access-path
costs (segment scan TCARD/P + W·RSICARD; clustered index F·(NINDX+TCARD) + W·RSICARD; non-clustered
F·(NINDX+NCARD) + W·RSICARD) come straight from the catalog statistics. The model is crude (uniform
distributions, independence, fixed defaults), but it is used only to **rank** plans, and the ranking it
induces matches the ranking of true costs in the large majority of cases — which is all an optimizer needs.

For joins, nested-loop costs `C-outer(path1) + N·C-inner(path2)`, where N is the estimated outer composite
cardinality. Merge scan has the same merge-work shape plus any required sorting; if the inner side has been
sorted into a temporary list, `C-inner(sorted list) = TEMPPAGES/N + W·RSICARD`, so the inner pages are fetched
once across the merge rather than once per outer tuple.

## Code

```python
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
import math

W = 0.1  # COST = PAGE FETCHES + W * (RSI CALLS)

@dataclass
class Index:
    name: str; column: str
    icard: int; nindx: int; clustered: bool

@dataclass
class Relation:
    name: str; ncard: int; tcard: int; npages: int
    indexes: dict = field(default_factory=dict)
    @property
    def p(self): return self.tcard / max(1, self.npages)

@dataclass
class Predicate:
    kind: str                 # 'eq_val' | 'range' | 'between' | 'join'
    rels: frozenset
    column: Optional[str] = None
    index_icard: Optional[int] = None
    other_icard: Optional[int] = None
    frac: Optional[float] = None
    def selectivity(self):
        if self.kind == "eq_val":
            return 1.0/self.index_icard if self.index_icard else 0.1
        if self.kind == "range":
            return self.frac if self.frac is not None else 1.0/3.0
        if self.kind == "between":
            return self.frac if self.frac is not None else 1.0/4.0
        if self.kind == "join":
            if self.index_icard and self.other_icard:
                return 1.0/max(self.index_icard, self.other_icard)
            if self.index_icard:
                return 1.0/self.index_icard
            return 0.1
        raise ValueError(self.kind)

def applicable(preds, S):
    return [p for p in preds if p.rels <= S]

def cardinality(rels, preds, S):
    c = 1.0
    for n in S: c *= rels[n].ncard
    for p in applicable(preds, S): c *= p.selectivity()
    return c

@dataclass
class Plan:
    rel_set: frozenset; cost: float; card: float
    order: Optional[str]; desc: str

def access_paths(rel, preds):
    local = applicable(preds, frozenset([rel.name]))
    out = cardinality({rel.name: rel}, preds, frozenset([rel.name]))
    plans = [Plan(frozenset([rel.name]), rel.tcard/rel.p + W*out, out, None,
                  f"segscan({rel.name})")]
    for col, idx in rel.indexes.items():
        matching = [p for p in local if p.column == col]
        f = 1.0
        for p in matching:
            f *= p.selectivity()
        pages = rel.tcard if idx.clustered else rel.ncard
        cost = (f*(idx.nindx+pages) if matching else (idx.nindx+pages)) + W*out
        plans.append(Plan(frozenset([rel.name]), cost, out, col,
                          f"index({rel.name}.{col})"))
    return plans

def sort_cost(card, pages):
    return pages*max(1.0, math.log2(max(2.0, pages))) + W*card

def nested_loop(left, inner, preds, combo, rels):
    N = left.card
    inner_scan = min(access_paths(inner, preds), key=lambda p: p.cost)
    return Plan(combo, left.cost + N*inner_scan.cost,
                cardinality(rels, preds, combo), None,
                f"NL({left.desc}, {inner.name})")

def join_cols(preds, left_set, r):
    return [p for p in preds if p.kind == "join" and r in p.rels
            and (p.rels - {r}) <= left_set]

def merge_scan(left, inner, preds, combo, rels):
    jc = join_cols(preds, left.rel_set, inner.name)
    if not jc: return None
    col = jc[0].column; N = left.card
    sort_inner = 0.0 if col in inner.indexes else sort_cost(inner.ncard, inner.tcard)
    sort_outer = 0.0 if left.order == col else sort_cost(left.card, left.card)
    rsicard = cardinality({inner.name: inner}, preds, frozenset([inner.name]))
    inner_cost = inner.tcard/max(1.0, N) + W*rsicard
    cost = left.cost + sort_outer + sort_inner + N*inner_cost
    return Plan(combo, cost, cardinality(rels, preds, combo), col,
                f"merge({left.desc}, {inner.name} on {col})")

def connected(preds, left_set, r):
    return any(p.kind == "join" and r in p.rels and (p.rels - {r}) & left_set
               for p in preds)

def keep(table, plan):
    key = (plan.rel_set, plan.order)
    old_cost = table[key].cost if key in table else math.inf
    if plan.cost < old_cost:
        table[key] = plan

def optimize(rels, preds, order_by=None):
    names = list(rels); n = len(names); plans = {}
    for nm in names:                                   # size 1: access paths
        for p in access_paths(rels[nm], preds): keep(plans, p)
    for size in range(2, n+1):                         # sizes 2..n: grow subsets
        for subset in combinations(names, size):
            S = frozenset(subset)
            for inner in subset:
                left_set = S - {inner}
                if not connected(preds, left_set, inner) and \
                   any(connected(preds, left_set, r) for r in subset if r != inner):
                    continue                           # defer cross products
                for left in [pl for (rs, _), pl in plans.items() if rs == left_set]:
                    keep(plans, nested_loop(left, rels[inner], preds, S, rels))
                    ms = merge_scan(left, rels[inner], preds, S, rels)
                    if ms is not None: keep(plans, ms)
    full = frozenset(names)
    cands = [pl for (rs, _), pl in plans.items() if rs == full]
    if order_by is None:
        return min(cands, key=lambda p: p.cost)
    in_order = [p for p in cands if p.order == order_by]
    cheapest = min(cands, key=lambda p: p.cost)
    sort_then = cheapest.cost + sort_cost(cheapest.card, cheapest.card)
    if in_order and min(in_order, key=lambda p: p.cost).cost <= sort_then:
        return min(in_order, key=lambda p: p.cost)
    return Plan(full, sort_then, cheapest.card, order_by, f"sort({cheapest.desc})")
```

On the three-table EMP/DEPT/JOB query (clerks in Denver), the subset lattice is what lets the optimizer
compare the JOB-first, DEPT-first, and EMP-first alternatives from catalog costs instead of inheriting the
FROM-list order.
