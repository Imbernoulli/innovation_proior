# Selinger-style cost-based join-order optimization

**Problem.** Given a non-procedural relational query — a FROM-list of n tables plus a WHERE-clause of
predicates — choose an access path for each table, a method for each join, and the *order* in which to join
the tables, so as to minimize total execution cost. The result of joining n tables is order-independent, but
the cost is not: intermediate cardinalities can swing by orders of magnitude. There are n! left-deep orders
before access paths and join methods are considered; broader ordering-plus-nesting enumeration is often
summarized as n!·(n−1)!, which is 20!·19! ≈ 2.96×10³⁵ for n=20. Exhaustive enumeration is not viable.

## Key idea

Three moves collapse the search:

1. **Restrict to left-deep trees.** Always join one *base* table (the inner) onto the running composite (the
   outer): (((A⋈B)⋈C)⋈D). This removes the extra nesting-choice factor, lets the inner use its own indexes,
   and pipelines the composite without materializing intermediates unless a sort is wanted.

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
   (≈ 4.1×10⁷ for n=20, m=2), with the DP table updated whenever a candidate is cheaper in the same
   subset/order slot.

3. **Keep interesting orders.** An output order named by ORDER BY/GROUP BY, or usable as a merge-join column,
   is *interesting*: a plan that arrives already sorted can save a downstream sort even if it is costlier on
   its own. So the DP keeps, per subset, the cheapest *unordered* plan **and** the cheapest plan per
   interesting order; a plan is only ever compared within its own order-slot. At the end, for an ORDER BY,
   compare (cheapest unordered + sort) vs (cheapest already in order) and take the min. Orders equated by
   join predicates collapse into one equivalence class, bounding the bookkeeping.

Two prunings sharpen it: **defer cross products** until no remaining relation can join to the current
composite by a predicate, and use only the two join methods that matter in practice: nested-loop and
sort-merge.

## Cost and selectivity model

Cost is `COST = PAGE FETCHES + W·(tuple-interface calls)`, W tuning I/O vs CPU. A per-boolean-factor
**selectivity** F estimates the surviving fraction: equality on an indexed column F = 1/ICARD (else 1/10);
column = column join F = 1/MAX(ICARD₁,ICARD₂), or 1/ICARD when only one side has an index (else 1/10); range
column > value F = (high−value)/(high−low) (else 1/3); BETWEEN uses (value₂−value₁)/(high−low) (else 1/4);
IN-list uses list length times equality selectivity capped at 1/2; IN-subquery uses expected subquery-result
cardinality divided by the product of subquery input cardinalities; conjunction F₁·F₂, disjunction
F₁+F₂−F₁·F₂, negation 1−F. QCARD is the product of table cardinalities and all applicable selectivities;
RSICARD uses the sargable factors that filter inside the storage scan. Single-table access-path costs include
the unique-index equality case `1+1+W`, segment scan `TCARD/P + W·RSICARD`, clustered index
`F·(NINDX+TCARD) + W·RSICARD`, and non-clustered index `F·(NINDX+NCARD) + W·RSICARD`, using `TCARD` instead
of `NCARD` for the data-page term when the qualifying tuples fit in the buffer. The model is crude, so it is
used as a ranking device, not as a runtime oracle.

For joins, nested-loop costs `C-outer(path1) + N·C-inner(path2)`, where N is the estimated outer composite
cardinality and the inner scan can use join predicates as search arguments. Merge scan has the same
merge-work shape plus any required sorting; if the inner side has been sorted into a temporary list,
`C-inner(sorted list) = TEMPPAGES/N + W·RSICARD` for the expected matching group per outer tuple, so the inner
pages are fetched once across the merge rather than once per outer tuple.

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
    unique: bool = False

@dataclass
class Relation:
    name: str; ncard: int; tcard: int; npages: int
    indexes: dict = field(default_factory=dict)
    buffer_pages: Optional[int] = None
    @property
    def p(self): return self.tcard / max(1, self.npages)

@dataclass
class Predicate:
    kind: str                 # 'eq_val' | 'range' | 'between' | 'in_list' | 'subquery_in' | 'join'
    rels: frozenset
    column: Optional[str] = None
    index_icard: Optional[int] = None
    other_icard: Optional[int] = None
    frac: Optional[float] = None
    count: int = 1
    subquery_card: Optional[float] = None
    subquery_base_card: Optional[float] = None
    sargable: bool = True
    negated: bool = False

    def selectivity(self):
        if self.kind == "eq_val":
            f = 1.0 / self.index_icard if self.index_icard else 0.1
        elif self.kind == "range":
            f = self.frac if self.frac is not None else 1.0 / 3.0
        elif self.kind == "between":
            f = self.frac if self.frac is not None else 1.0 / 4.0
        elif self.kind == "in_list":
            base = 1.0 / self.index_icard if self.index_icard else 0.1
            f = min(0.5, self.count * base)
        elif self.kind == "subquery_in":
            if self.subquery_card is None or not self.subquery_base_card:
                f = 0.1
            else:
                f = min(1.0, self.subquery_card / self.subquery_base_card)
        elif self.kind == "join":
            if self.index_icard and self.other_icard:
                f = 1.0 / max(self.index_icard, self.other_icard)
            elif self.index_icard:
                f = 1.0 / self.index_icard
            else:
                f = 0.1
        else:
            raise ValueError(self.kind)
        return 1.0 - f if self.negated else f

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

def rel_selectivity(rel, preds, available, only_sargable=False, column=None):
    f = 1.0
    for p in preds:
        if rel.name not in p.rels or not p.rels <= available:
            continue
        if only_sargable and not p.sargable:
            continue
        if column is not None and p.column != column:
            continue
        f *= p.selectivity()
    return f

def rel_card(rel, preds, available):
    return rel.ncard * rel_selectivity(rel, preds, available)

def rsicard(rel, preds, available):
    return rel.ncard * rel_selectivity(rel, preds, available, only_sargable=True)

def fits_buffer(rel, pages):
    return rel.buffer_pages is not None and pages <= rel.buffer_pages

def access_paths(rel, preds, available_rels=None):
    available = frozenset([rel.name]) if available_rels is None else frozenset(available_rels)
    out = rel_card(rel, preds, available)
    rsi = rsicard(rel, preds, available)
    plans = [Plan(frozenset([rel.name]), rel.tcard / rel.p + W * rsi, out, None,
                  f"segscan({rel.name})")]
    for col, idx in rel.indexes.items():
        matching = [p for p in preds if rel.name in p.rels and p.rels <= available
                    and p.sargable and p.column == col]
        if idx.unique and any(p.kind in ("eq_val", "join") for p in matching):
            plans.append(Plan(frozenset([rel.name]), 1 + 1 + W, out, col,
                              f"unique_index({rel.name}.{col})"))
            continue
        f = 1.0
        for p in matching: f *= p.selectivity()
        if idx.clustered:
            data_pages = rel.tcard
        else:
            needed_pages = f * rel.tcard if matching else rel.tcard
            data_pages = rel.tcard if fits_buffer(rel, needed_pages) else rel.ncard
        cost = (f * (idx.nindx + data_pages) if matching else (idx.nindx + data_pages)) + W * rsi
        plans.append(Plan(frozenset([rel.name]), cost, out, col,
                          f"index({rel.name}.{col})"))
    return plans

def sort_cost(card, pages):
    pages = max(1.0, pages)
    return pages * max(1.0, math.log2(max(2.0, pages))) + W * card

def nested_loop(left, inner, preds, combo, rels):
    available = left.rel_set | {inner.name}
    inner_scan = min(access_paths(inner, preds, available), key=lambda p: p.cost)
    return Plan(combo, left.cost + left.card * inner_scan.cost,
                cardinality(rels, preds, combo), None,
                f"NL({left.desc}, {inner.name})")

def join_cols(preds, left_set, r):
    return [p for p in preds if p.kind == "join" and r in p.rels
            and (p.rels - {r}) <= left_set]

def merge_scan(left, inner, preds, combo, rels):
    jc = join_cols(preds, left.rel_set, inner.name)
    if not jc: return None
    col = jc[0].column
    sort_outer = 0.0 if left.order == col else sort_cost(left.card, left.card)

    local_paths = access_paths(inner, preds, frozenset([inner.name]))
    ordered = [p for p in local_paths if p.order == col]
    ordered_cost = min((p.cost for p in ordered), default=math.inf)
    base = min(local_paths, key=lambda p: p.cost)
    sorted_cost = base.cost + sort_cost(base.card, base.card)
    inner_once = min(ordered_cost, sorted_cost)

    cost = left.cost + sort_outer + inner_once
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
    names = list(rels); all_names = set(names); n = len(names); plans = {}
    for nm in names:                                   # size 1: access paths
        for p in access_paths(rels[nm], preds): keep(plans, p)
    for size in range(2, n+1):                         # sizes 2..n: grow subsets
        for subset in combinations(names, size):
            S = frozenset(subset)
            for inner in subset:
                left_set = S - {inner}
                remaining = all_names - set(S)
                if not connected(preds, left_set, inner):
                    if any(connected(preds, left_set, r) for r in remaining):
                        continue                       # defer cross products
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
