A user of a high-level relational language writes only *what* data they want — a SELECT-list, a FROM-list of tables, and a WHERE-clause of predicates — and says nothing about *how* to get it: no index, no scan method, and crucially no order in which to join the tables. That leaves the system holding a hard optimization problem. For one query it must pick an access path for each table (an index or a sequential scan), a method for each join, and the order in which the tables are joined, all to minimize total execution cost. The order is the part that bites. By the relational algebra the join is associative and commutative *on the result* — $(A\bowtie B)\bowtie C$ and $A\bowtie(B\bowtie C)$ return the same set of tuples — so order can never change correctness. But it changes cost by orders of magnitude, because different orders produce wildly different intermediate cardinalities: join the most selective pair first and the running result stays small; join the wrong pair first and a near-Cartesian blow-up is carried through every later step. Picking an order *is* picking a cost, so the problem is not to find a correct plan but the cheapest among many correct ones.

The obvious answer — enumerate every plan and take the minimum — dies on the size of the space. With $n$ tables there are $n!$ left-deep orders before access paths or join methods are chosen, and admitting bushier binary nestings adds a further factorial-scale factor often written $n!\cdot(n-1)!$; for a 20-way join that shorthand is $20!\cdot 19!\approx 2.96\times10^{35}$. Even the smaller left-deep $n!$ alone defeats exhaustive search. The contemporary alternative, INGRES-style query decomposition, sidesteps global costing entirely: it detaches one-variable subqueries and, when only multi-variable pieces remain, does tuple substitution, choosing the substitution order by runtime rules of thumb (substitute the smallest relation, etc.). It is dynamic, interpreted, and heuristic — there is no global quantitative minimization over the whole join order, so a poor substitution order is never ruled out, and because the work is done at runtime there is no compiled plan to amortize over the many executions of a query that is compiled once and run often. And solving the easy sub-problem in isolation — choosing the cheapest index-or-scan for a single table — says nothing about how to combine tables, and worse, the right access path for a table depends on whether it sits outer or inner in the join, so per-table optimization done independently is not even locally right. We need something that searches the factorial space *implicitly*, costing only a tractable number of plans, yet lands on or very near the cheapest.

I propose Selinger-style cost-based join-order optimization. It rests on three moves that collapse the search, sitting on top of a quantitative cost-and-selectivity model that lets every candidate be compared by number. The cost of any access path or plan is $$\text{COST} = \text{PAGE FETCHES} + W\cdot(\text{tuple-interface calls})$$ where $W$ is a machine-dependent knob trading I/O against CPU (the interface calls, RSICARD, proxy the CPU because most CPU goes into them). Every formula needs a *selectivity* $F\in(0,1]$, the fraction of tuples a predicate passes, estimated from cheap catalog counts under a uniform-and-independent model: an equality on an indexed column is $F=1/\text{ICARD}$ (else $1/10$); a column = column join is $F=1/\max(\text{ICARD}_1,\text{ICARD}_2)$, or $1/\text{ICARD}$ when only one side is indexed (else $1/10$); a range $\text{col}>v$ interpolates $F=(\text{high}-v)/(\text{high}-\text{low})$ (else $1/3$); BETWEEN gives $(v_2-v_1)/(\text{high}-\text{low})$ (else $1/4$); an IN-list is the list length times the equality selectivity capped at $1/2$ so a long list cannot silently become "everything"; an IN-subquery is the expected subquery cardinality over the product of its input cardinalities; conjunctions multiply $F_1 F_2$, disjunctions combine $F_1+F_2-F_1 F_2$, negation is $1-F$. The composite cardinality of a set of tables, QCARD, is the product of their cardinalities times the product of applicable selectivities; RSICARD multiplies only the *sargable* factors, those pushable into the storage scan. The single-table costs then follow mechanically: a unique index matching an equality is the cheap special case $1+1+W$; a segment scan is $\text{TCARD}/P + W\cdot\text{RSICARD}$; a clustered matching index is $F\cdot(\text{NINDX}+\text{TCARD}) + W\cdot\text{RSICARD}$; a non-clustered one is $F\cdot(\text{NINDX}+\text{NCARD}) + W\cdot\text{RSICARD}$, with the data-page term dropping from $\text{NCARD}$ to $\text{TCARD}$ when the qualifying tuples fit in the buffer. I should be honest that this model is crude — uniform distributions, independent columns, invented defaults — but I never *report* a runtime; I only ever *compare* two plans and keep the cheaper. So I need the model to rank correctly, not to be accurate in absolute value, and a coarse-but-monotone estimate gets the ranking right far more reliably than it gets the magnitude right. That weaker bet is exactly enough, and it is what licenses everything downstream.

The first move is to restrict the plan shape to left-deep trees: the running result is always the outer, and each join brings in one *base* table as the inner, as in $(((A\bowtie B)\bowtie C)\bowtie D)$, never $(A\bowtie B)\bowtie(C\bowtie D)$. This forfeits the occasional cheaper bushy plan, but it buys two things. A left-deep tree is fixed by the *sequence* of tables, so the extra nesting factor vanishes; and because the inner is always a base table I can use that table's own indexes directly as the inner access path, while the outer is a single growing pipeline that never has to be materialized to disk unless I deliberately sort it — bushy plans would force both sub-results to be materialized before combining. Left-deep is smaller *and* operationally nicer.

That still leaves $n!$ sequences, and the second move kills the factorial. Sitting at the point where some set $\{A,B,C\}$ is built and I am about to join $D$, the cost of that join depends only on the composite's cardinality (a product of cardinalities and applicable selectivities — order-independent by the algebra), the predicates between $D$ and the *set* $\{A,B,C\}$, the access paths on $D$, and the output order the composite happens to be in. None of those, except possibly the order, cares *how* $\{A,B,C\}$ was assembled. So to finish the query cheaply from "$\{A,B,C\}$ is built" I only need the single cheapest way to *have built* $\{A,B,C\}$; every other plan producing that same set is dead weight. The minimum-cost plan for a *set* is a sufficient summary of the set — the principle of optimality — and it is exactly the hook for dynamic programming. Rather than enumerate sequences I memoize per subset and build bottom-up by size, with the recurrence $$\text{optjoin}(S) = \min_{a\in S}\Big[\text{cost}\big(\text{optjoin}(S-\{a\})\big) + \text{join-cost}\big(\text{optjoin}(S-\{a\}), a\big) + \text{access-cost}(a)\Big].$$ Size-1 entries are the best access path per table; size $k$ subsets are grown from cached size-$(k-1)$ answers by peeling off one inner table. The subsets number $\sum_k\binom{n}{k}=2^n$, and each tries up to $n$ peels times $m$ join methods, so the work is $O(n\cdot m\cdot 2^n)$ plan evaluations — about $4.1\times10^7$ for $n=20,\,m=2$, a tractable exponential in place of an unbounded factorial. It is satisfying that the very associativity that made order irrelevant *to the result* is what makes the subset, not its history, the right thing to memoize.

Two physical choices keep the per-node work small. For the join itself only two methods are carried: nested-loop, where the outer is scanned and the inner re-scanned for each of the $N$ outer tuples at cost $\text{C-outer} + N\cdot\text{C-inner}$; and sort-merge, where both inputs are sorted on the join column and swept in lockstep, each read essentially once, at the same $\text{C-outer}+N\cdot\text{C-inner}$ shape plus any sort cost. When the inner is a sorted temporary list, $\text{C-inner} = \text{TEMPPAGES}/N + W\cdot\text{RSICARD}$ for the expected matching group per outer tuple, so multiplying by $N$ fetches those inner pages once across the whole merge rather than once per outer tuple; merge wins precisely when existing or cheaply-bought order makes the inner scan cheaper than repeated rescans. An empirical study of two-way joins (Blasgen & Eswaran 1976) found one of these two is always optimal or near-optimal for any but the tiniest relations, so a third method would double the search work at every node for almost nothing. The other pruning defers cross products: extending a composite by a table with no join predicate to it is a Cartesian blow-up, and since the algebra lets a cross product happen at any point, I do it as late as possible — when adding a table along a join edge is possible I do that and skip the cross-product extension, allowing it only when no remaining table can join the current composite by a predicate. Disconnected components are still handled, but only after every selective edge-connected extension is exhausted, which for a sparse join graph prunes the lattice toward connected subsets.

The third move handles the one thing the subset summary cannot suppress: output *order*. A plan for $S$ that emerges already sorted on a column saves a later sort — for an ORDER BY or GROUP BY on that column, or for a downstream merge join that wants its input in that order. So a plan that is costlier on its own but arrives pre-sorted can win the whole query by sparing that sort, and pruning purely on cost would throw it away. An order is therefore *interesting* if some later operator can exploit it. For each subset I keep not one plan but the best *unordered* plan and the best plan per interesting order, and a plan is only ever compared against others in its *same* order-slot, never knocked out by a cheaper unordered one. The bookkeeping stays bounded because there are few interesting orders and several collapse into one equivalence class — if predicates equate $D.x=E.x=F.x$, then "sorted on $x$" via any of them is one useful order. At the top, with the full set assembled and an ORDER BY present, I compare (cheapest unordered plan + the cost of sorting it into the required order) against (cheapest plan already in that order) and take the minimum; the pre-sorted plan wins exactly when its extra join cost is less than the sort it lets me skip. With no ORDER BY and no downstream merge there are no interesting orders and the cheapest full-set plan wins outright. On the canonical three-table query — EMP, DEPT, JOB, "name, salary, job title, department name of employees who are clerks in Denver", with local predicates TITLE='CLERK' and LOC='DENVER' and join predicates EMP.DNO=DEPT.DNO and EMP.JOB=JOB.JOB — the subset lattice is precisely what lets the optimizer weigh JOB-first, DEPT-first, and EMP-first from catalog costs instead of inheriting the FROM-list order.

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
