## Research question

A high-level relational query language lets a user say *what* data they want — a SELECT-list,
a FROM-list of tables, and a WHERE-clause that is a boolean combination of predicates — without
saying *how* to get it. The user names no indexes, no scan methods, and crucially no order in which
to join the tables. The system is left holding a hard optimization problem: for a single query, pick
an **access path** for each table (which index, or a sequential scan), pick a **join method** for each
join (there are several), and pick the **order** in which the tables are joined — so as to minimize the
total cost of running the query.

The order matters enormously. The *result* of joining n tables is the same set of tuples no matter how
the joins are nested — that is a theorem of the relational algebra. But the *cost* of producing it varies
by orders of magnitude, because different orders produce wildly different intermediate cardinalities: join
the two tables whose join is most selective first and the running result stays small; join the wrong pair
first and you carry a near-Cartesian blow-up through every later step. So a chosen order is, in effect, a
chosen cost, and the spread between the best and the worst is large.

The catch is the size of the search space. With n tables in the FROM-list there are n! left-deep table
orders before access paths and join methods are considered. If bushier binary combinations are also admitted,
a common broad enumeration count adds a factorial-scale nesting factor, often summarized as n!·(n−1)!; for a
20-way join that shorthand is 20!·19! ≈ 2.96×10³⁵. The exact tree-counting convention is less important than
the conclusion: enumerating all plans is hopeless. What a solution must achieve is to search this space
implicitly, costing only a tractable number of plans, yet still land on (or very near) the cheapest one, and
do it fast enough that the optimization itself is cheap relative to running the query, especially when a query
is compiled once and run many times.

## Background

**Relational algebra and order-independence.** Codd's relational model (Codd 1970; textbook treatment in
Date 1975) gives queries as expressions over relations. The join ⋈ is associative and commutative on the
result: (A⋈B)⋈C, A⋈(B⋈C), (B⋈A)⋈C all yield the same relation. This is the load-bearing fact — it is *why*
there is freedom to reorder, and it is *why* the only thing distinguishing orders is cost, not correctness.

**The combinatorial explosion.** Two multiplicities compound. First the **order** of the n tables:
n! linear orders, which already defeats exhaustive search for large n. Second, if plans are not restricted
to a left-deep shape, there are additional binary-combination choices; the common shorthand n!·(n−1)! puts
the 20-way ordering-and-nesting space at 20!·19! ≈ 2.96×10³⁵ before join-method and access-path choices are
added. Even the smaller left-deep n! space is too large to scan directly.

**Access paths on a single table.** A table is stored as tuples on data pages, accessed through a
tuple-at-a-time scan interface. Two scan types exist: a **segment scan** touches every page of the segment
once; an **index scan** walks a B-tree's leaf pages in key order, following each leaf entry to its data
tuple, and can start/stop at key bounds. An index is **clustered** if the data pages are physically kept in
the index's key order, so one data page fetch yields many wanted tuples; a non-clustered index may fetch a
fresh data page per tuple. A predicate is **sargable** ("column comparison-operator value") if it can be
pushed down into the scan as a search argument and evaluated inside the storage system, so non-matching
tuples are filtered before crossing the (relatively expensive) call interface. Every tuple returned by the
query must satisfy every top-level conjunct of the WHERE-clause; each such conjunct is a **boolean factor**.

**Selectivity and cardinality estimation.** To compare access paths and join orders by cost you need to
predict how many tuples survive each predicate. The estimate per boolean factor is a **selectivity factor**
F ∈ (0,1], the expected fraction of tuples it passes. Under a uniform-distribution assumption an equality on
an indexed column with ICARD distinct keys passes 1/ICARD of the tuples; a column = column join with both
sides indexed passes 1/MAX(ICARD₁,ICARD₂), or 1/ICARD for the indexed side when only one side has an index.
For an arithmetic range with known bounds, interpolate within the observed key range: column > value uses
(high−value)/(high−low), and BETWEEN value₁ AND value₂ uses (value₂−value₁)/(high−low). A column IN a list
uses the list length times the equality selectivity, capped at 1/2; a column IN a subquery uses the expected
subquery-result cardinality divided by the product of the subquery input cardinalities. When statistics are
missing, fixed defaults are used (1/10 for an equality, 1/3 for an open range, 1/4 for a BETWEEN).
Conjunctions multiply (F₁·F₂, assuming independence), disjunctions combine as F₁+F₂−F₁·F₂, negation is
1−F. QCARD, the estimated output cardinality of a query or joined set, is the product of the input
cardinalities times the product of all applicable selectivity factors. RSICARD, the estimated count of
tuples crossing the storage interface, multiplies only the sargable factors that can be applied inside the
scan. The statistics this needs — per table the tuple cardinality and page count, per index the number of
distinct keys and the number of index pages — are cheap to keep in a catalog and refresh periodically.

**Join methods.** For joining two inputs (each possibly the result of earlier joins), two methods are known
to suffice in practice. **Nested-loop**: scan the outer input; for each outer tuple, scan the inner input
for tuples matching the join predicate. Any access path may serve either side. **Sort-merge** (merging
scans): require both inputs sorted on the join column, then sweep them in lockstep, advancing whichever
cursor is behind and emitting matches — each input is read essentially once. The merge variant pays a
sorting cost up front unless an input already arrives in join-column order (from an index, or from being the
output of an earlier sort). An empirical study of two-way joins (Blasgen & Eswaran 1976) found that for any
but the very smallest relations one of these two methods is always optimal or near-optimal — so a system
need implement only these two and lose almost nothing.

**Join cost formulas.** For a nested-loop join, the cost is C-outer(path1) + N·C-inner(path2), where N is the
estimated cardinality of the outer composite and the inner access path may use join predicates as search
arguments parameterized by the current outer tuple. The merge scan has the same merge-work shape,
C-outer(path1) + N·C-inner(path2), plus any required sort costs. When the inner side is a sorted temporary
list, C-inner(sorted list) = TEMPPAGES/N + W·RSICARD, where RSICARD is the expected matching group per outer
tuple; multiplying by N fetches the inner pages once across the merge rather than once per outer tuple.

**Interesting orders.** A scan or a merge can deliver its output already sorted on some column. That sorted
output is *worth* something to a later operator: an ORDER BY or GROUP BY can skip a final sort, and a later
sort-merge join can skip sorting that input. So an output order specified by an ORDER BY/GROUP BY clause, or
used as a join column by a later merge join, is an **interesting order**. The consequence for cost: the
cheapest plan for a set of tables, judged on its own join cost, is not always the best choice — a slightly
costlier plan that happens to emerge in an interesting order can win on the whole query by saving a
downstream sort.

**Diagnostic that licenses cost-based search.** The selectivity model is crude — uniform distributions,
independence, fixed default fractions. The justification for trusting it cannot be absolute runtime accuracy.
The optimizer uses the estimates only to compare plans, so the relevant diagnostic is whether the estimated
ordering of candidate costs tracks the real ordering. A coarse-but-monotone model can be useful even when
every absolute runtime prediction is off.

## Baselines

**INGRES query decomposition (Wong & Youssefi 1976; Stonebraker, Wong, Kreps & Held 1976).** The
contemporary alternative attacks a multi-relation query by *decomposition*: repeatedly detach a one-variable
subquery (a single-relation restriction that can be run immediately) and, when only multi-variable pieces
remain, perform **tuple substitution** — pick one relation, and for each of its tuples substitute the
constant values into the rest of the query, recursively solving the smaller query that results. It is a
dynamic, runtime, heuristic strategy: the order in which variables are substituted is chosen by rules of
thumb (e.g. substitute the smallest relation), decisions are made tuple-by-tuple at execution time, and
nothing is compiled or globally costed. Gap: there is no global, quantitative minimization over the whole
join order — the heuristic can pick a poor substitution order, and because the work is interpreted at
runtime there is no compiled plan to amortize over repeated executions of the same query.

**Exhaustive enumeration.** The conceptually obvious baseline — generate every join order, every tree shape,
every method/access-path assignment, cost each, take the min — is correct but the ordering-and-nesting space is
factorial-scale and computable only for tiny n. It is the thing a real method must approximate without paying
its price.

**Single-table access-path selection in isolation.** Choosing the cheapest index-or-scan for *one* table
against a set of local predicates is the easy, already-solved sub-problem (cost each available path from the
catalog statistics, take the minimum; keep one per interesting order). Gap: it says nothing about how to
combine tables — the join-order problem is exactly what it leaves open, and naively optimizing each table's
access in isolation and then joining ignores that the right access path for a table depends on its role
(outer vs inner) in the join.

## Evaluation settings

The natural yardstick is a relational engine with a stored-catalog of statistics (per-table cardinality and
page counts; per-index distinct-key and page counts), B-tree indexes (clustered and non-clustered) plus
sequential segment scans as the access paths, and nested-loop and sort-merge as the join methods. Workloads
range from single-table selections to multi-way joins; a standard illustrative query joins three tables —
employees, departments, jobs — with two local equality predicates and two join predicates, e.g. "retrieve
the name, salary, job title and department name of employees who are clerks and work in Denver". The metrics
of interest are the resource cost of the chosen plan (page fetches plus a weighted count of tuple-interface
calls, the I/O-plus-CPU cost the optimizer itself minimizes) and the cost of optimization (CPU time and
working storage to choose the plan), the latter mattering because a compiled query is optimized once and run
many times, so optimization cost is amortized.

## Code framework

The pieces below already exist: a catalog of statistics, a selectivity rule-set, single-table access-path
costing, and a tuple-at-a-time scan interface. The empty slot is the strategy that, given a FROM-list of
several tables and the join predicates among them, picks the join order, the join methods, and the per-table
access paths together to minimize total cost without enumerating all plans.

```python
from dataclasses import dataclass, field
from typing import Optional

W = 0.1  # I/O-vs-CPU weight in COST = PAGE FETCHES + W * (RSI CALLS)

@dataclass
class Index:
    name: str; column: str
    icard: int        # distinct keys
    nindx: int        # index pages
    clustered: bool
    unique: bool = False

@dataclass
class Relation:
    name: str
    ncard: int        # tuple cardinality
    tcard: int        # data pages
    npages: int       # nonempty pages in the containing segment
    indexes: dict = field(default_factory=dict)
    buffer_pages: Optional[int] = None
    @property
    def p(self): return self.tcard / max(1, self.npages)

@dataclass
class Predicate:
    kind: str         # 'eq_val' | 'range' | 'between' | 'in_list' | 'subquery_in' | 'join'
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
    def selectivity(self) -> float:
        pass

@dataclass
class Plan:
    rel_set: frozenset
    cost: float
    card: float
    order: Optional[str]
    desc: str

def applicable(preds, rel_set) -> list:
    pass

def cardinality(relations, preds, rel_set) -> float:
    """Product of input cardinalities * product of applicable selectivities."""
    pass

def rel_selectivity(rel, preds, available_rels, only_sargable=False, column=None) -> float:
    pass

def rel_card(rel, preds, available_rels) -> float:
    pass

def rsicard(rel, preds, available_rels) -> float:
    pass

def fits_buffer(rel, pages) -> bool:
    pass

def access_paths(rel, preds, available_rels=None) -> list:
    """Every reasonable single-table scan (segment + each index), costed; one per output order.
    available_rels lets an inner scan use join predicates once the outer tuple supplies the other side.
    """
    pass

def sort_cost(card, pages) -> float:
    pass

def nested_loop(outer_plan, inner_relation, preds, rel_set, relations):
    """Cost one nested-loop candidate."""
    pass

def join_cols(preds, left_set, rel_name) -> list:
    pass

def merge_scan(outer_plan, inner_relation, preds, rel_set, relations):
    """Cost one merge-scan candidate when a join-column order is available or can be produced."""
    pass

def connected(preds, left_set, rel_name) -> bool:
    pass

def keep(plan_table, plan):
    pass

def optimize(relations: dict, preds, order_by: Optional[str] = None):
    """Given several tables and the join predicates among them, choose the join order,
    the join method per join, and the access path per table to minimize total cost,
    without enumerating the factorial join-order space.
    """
    pass
```
