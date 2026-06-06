OK, let me think this through from scratch. Someone hands me a query in a high-level relational
language. They wrote a SELECT-list, a FROM-list of a few tables, and a WHERE-clause that's a boolean
combination of predicates. What they did *not* write is any instruction about how to actually run it: no
index to use, no scan method, and — the part that's nagging at me — no order in which to join the tables.
That's the whole point of the language, the user shouldn't have to know. But it means I have to choose all
of that, and choose it well, because the choices are not cosmetic.

Start with the part that scares me most: the join order. Why does it even matter? The relational algebra
says the join is associative and commutative *on the result* — join three tables in any nesting and you get
the same set of tuples back. So order can't affect correctness. Good. But then why am I worried? Because it
affects *cost*, and not by a little. Picture joining A, B, C where A⋈B is highly selective (the join keys
match for almost nothing) but A⋈C is nearly a cross product. If I join A and B first, the intermediate
result is tiny and I carry almost nothing into the join with C. If I join A and C first, I carry a near-
Cartesian blow-up through the rest of the plan, touching enormous intermediate results that I'll mostly
throw away. Same answer, costs that differ by orders of magnitude. So picking the order *is* picking the
cost. That reframes the whole problem: I'm not searching for a correct plan, I'm searching for a cheap one
among many correct ones.

So just enumerate them and take the cheapest? Let me count. With n tables there are n! ways to order them.
And after the order is fixed there are still many binary choices about which partial result to join next, so
this order-and-nesting space is n!·(n−1)!. Plus at every join node I get to pick a
method, and at every leaf a scan. Forget the method/scan factors for a second; just the order × nesting count
for n=20 is 20!·19! ≈ 2.9×10³⁵. That's not "slow", that's "never finishes". Exhaustive enumeration is
correct and useless. I need to search this space without visiting most of it.

Before I attack the hard combinatorial part, let me nail the easy sub-problem so I know what a "plan for a
table" even costs. One table, some local predicates on it. How do I scan it? Either a sequential scan of the
whole segment — touch every data page once — or, if there's an index on a column that a predicate mentions,
an index scan that walks the B-tree leaves in key order and chases each entry to its tuple, possibly starting
and stopping at key bounds. The index can be *clustered* (data physically in key order, so one page fetch
yields many wanted tuples) or not (worst case a fresh page fetch per tuple). I want a cost number. The
machine spends time on two things: fetching pages (I/O) and making calls across the tuple interface (a good
proxy for CPU, since most of the CPU goes into those calls). Those interface calls are RSICARD: the tuples
that survive the search arguments inside the storage system and have to cross the RSI boundary. So let my
cost be

    COST = PAGE FETCHES + W · (number of tuple-interface calls)

with W a knob that tunes the I/O-vs-CPU balance for the machine. Now I can write down the page-fetch count
for each access path. A segment scan fetches TCARD/P pages (TCARD data pages, P = TCARD divided by the
number of non-empty segment pages) plus W·RSICARD. A clustered index matching predicates with combined
selectivity F fetches about F·(NINDX + TCARD) plus W·RSICARD — a fraction F of the index pages and of the
(co-ordered) data pages. A non-clustered index matching costs F·(NINDX + NCARD) + W·RSICARD, because in the
worst case each of the F·NCARD qualifying tuples drags in its own data page. An index that matches *no*
predicate still costs the full NINDX + TCARD if clustered, or NINDX + NCARD if not, again plus W·RSICARD, if
I scan it only for its order. Fine — these are mechanical once I have the statistics, which I keep in a
catalog: per table NCARD (tuples), TCARD (data pages), P; per index ICARD (distinct keys), NINDX (index
pages).

But every one of those formulas needs F, the selectivity — what fraction of tuples a predicate passes. And
the join cost will need the same thing, scaled up: how many tuples survive a *set* of predicates over a set
of tables. I don't have full distributions, only the catalog counts. So I'll model it. For an equality on an
indexed column with ICARD distinct keys, assume the values are spread evenly: F = 1/ICARD. For a column =
column join with both columns indexed, each key of the smaller-cardinality index has a match, so
F = 1/MAX(ICARD₁, ICARD₂); if only one side is indexed, I use 1/ICARD for that side. For a range column >
value where the column is arithmetic and I know the value, linearly interpolate:
F = (high − value)/(high − low). For BETWEEN value₁ AND value₂, the same interpolation gives
F = (value₂ − value₁)/(high − low). When I have no statistic at all I fall back to fixed guesses — 1/10 for
an equality, 1/3 for an open-ended comparison, 1/4 for a BETWEEN — chosen only so an equality looks more
selective than a range and a range looks more selective than nothing. Conjunctions
multiply, F₁·F₂, assuming independence; a disjunction is F₁ + F₂ − F₁·F₂; a negation is 1 − F. Then the
estimated cardinality of joining a set of tables is just the product of their cardinalities times the
product of the selectivities of the predicates that apply within that set.

Now I should be honest with myself: this model is crude. Uniform distributions, independent columns, made-up
default fractions. Am I about to build a careful optimizer on sand? Let me think about what I actually need
from the estimates. I'm never going to *report* a predicted runtime to anyone. All I do with a cost is
*compare* two plans and keep the cheaper. So I don't need the costs to be accurate in absolute value — I need
the *ranking* they induce to match the ranking of the true costs. And a coarse-but-monotone model can get
the ranking right even when every absolute number is off: if plan X really does touch ten times the tuples
of plan Y, even a sloppy estimate will usually put X above Y. That's the bet, and it's a much weaker, much
safer bet than "predict the runtime". It's exactly enough for what an optimizer does. Good — I'll proceed,
and treat the model as a ranking device, not an oracle.

Back to the joins themselves: how do I physically join two inputs, each possibly the result of earlier
joins? Two methods, and I'll convince myself two is enough. **Nested-loop**: scan the outer input, and for
each outer tuple scan the inner for matching tuples. Cost is C-outer plus, for each of the N outer tuples,
one inner scan: C-nested-loop = C-outer(outer) + N · C-inner(inner), where N is the cardinality of the outer
composite — the product of the cardinalities of the relations joined so far times the selectivities of the
predicates already applied. **Sort-merge**: require both inputs sorted on the join column, then sweep them in
lockstep, advancing whichever side is behind and emitting matches — each input read essentially once. Why
bother with two? Because the inner scan in nested-loop can be brutal if the inner is large and unindexed on
the join column — you rescan it per outer tuple. Sort-merge pays a sort up front but then reads each side
once, and the payoff is huge when an input is already sorted (from an index, or from being the output of an
earlier sort), so the sort is free. Should I add a third method? An empirical study of two-way joins
(Blasgen & Eswaran 1976) found that for anything but the tiniest relations one of these two is always optimal
or near-optimal. So a third method would buy me almost nothing while doubling the search work at every node.
Two methods it is.

Putting the two cost formulas side by side makes the merge case less mysterious. The cost of doing the merge
is still C-outer(path1) + N · C-inner(path2), with sorting costs added if either input has to be sorted first.
The difference is what C-inner means. In nested-loop, the inner scan can be a full scan or an index probe for
each outer tuple. In merge scan, because the inner is sorted on the join column, the inner scan for one outer
join value is only the contiguous matching group. If the inner had to be sorted into a temporary list, a clean
estimate is C-inner(sorted list) = TEMPPAGES/N + W·RSICARD, so across N outer tuples the inner pages are
fetched once rather than once per outer tuple. Merge wins exactly when sorting or existing order makes that
inner scan much cheaper than repeated rescans.

Now the real problem, the join order, with the easy pieces in hand. n!·(n−1)! is out. Let me stare at *why*
it's so big and whether the structure has slack. The (n−1)! factor is the nesting freedom — all the ways to
choose which partial result is extended next. What if I allow only one shape? Restrict to **left-deep** trees:
the running result is always the *outer*, and each join brings in one *base* table as the inner, as in
(((A⋈B)⋈C)⋈D), never (A⋈B)⋈(C⋈D). What do I lose and what do I gain? I lose the bushy shapes — sometimes a
bushy plan really is cheaper. But I gain two big things. First, I kill the entire (n−1)! factor: a left-deep
tree is determined by the *sequence* of tables, there's only one shape. Second — and this is the one I like —
because the inner is always a base table, I can use that table's *own indexes* directly as the inner access
path; and because
the outer is a single growing pipeline, I never have to materialize an intermediate result to disk unless I
deliberately sort it. The composite flows one tuple at a time into the next join. Bushy plans would force me
to materialize both sub-results before combining. So left-deep isn't just smaller, it's *operationally*
nicer. I'll commit to left-deep.

That removes (n−1)! and leaves n! — the order of the tables. Still factorial, still hopeless for n=20. I need
to crack the n! itself. Let me think about what's actually redundant in enumerating all n! sequences. Take a
sequence that joins A, B, C in that order, then continues with D, then E. And another that joins B, A, C —
same first three tables {A,B,C}, different internal order — then also continues with D, then E. When I'm
sitting at the point where {A,B,C} is built and I'm about to join D, does the *internal history* of how
{A,B,C} got assembled matter to the D-join? Let me check carefully, because the whole thing hinges on this.
The cost of joining D onto the composite depends on: the cardinality of the composite (which is just the
product of A,B,C cardinalities times the applicable selectivities — *order-independent*, by the algebra), the
predicates that apply between D and {A,B,C} (which depend only on the *set* {A,B,C}, not its order), the
output order the composite happens to be in (hmm — flag this, come back to it), and the access paths
available on D. None of those — except possibly the output order — care *how* {A,B,C} was built.

So if I want the cheapest way to finish the query from the state "{A,B,C} is built", I only need the
cheapest way to *have built* {A,B,C} — its minimum cost — and I can throw away every other plan that produces
{A,B,C}, no matter what internal order it used. The minimum-cost plan for a *set* of tables is a sufficient
summary of that set; the rest is dead weight. That's the principle of optimality, and it's exactly the hook
for dynamic programming. I don't enumerate sequences; I compute, for each *subset* S of the tables, the best
plan that produces S, building larger subsets from smaller ones.

Let me write the recurrence. For a set S, the best left-deep plan ends by joining some single table a ∈ S as
the inner onto the best plan for the rest, S − {a}:

    optjoin(S) = min over a ∈ S of [ cost( optjoin(S − {a}) )
                                     + join-cost( optjoin(S − {a}), a )
                                     + access-cost(a) ]

and optjoin(S − {a}) is already computed and cached, because S − {a} is smaller than S. So I build bottom-up
by subset *size*: size 1 is just the best access path for each single table; size 2 builds every pair from
the size-1 answers; size k builds every k-subset by taking each (k−1)-subset's cached best plan and joining
one more table; up to size n. Each subset is computed once and reused. In a compact implementation of the
same recurrence, the cost of a candidate join can be represented as the output cardinality of the joined set
plus the already cached cost of the left side plus the base access cost of the right side, and the DP table
entry is replaced exactly when that new cost is lower than the old one.

Now count it, because the whole point was to beat n!·(n−1)!. The number of subsets is Σ over k of C(n,k) =
2ⁿ. For each subset I try up to n ways to peel off the last table, times m join methods. So O(n·m·2ⁿ) plan
evaluations. For n=20, m=2 that's about 4.1×10⁷ — seven orders of magnitude per table count, down from
2.9×10³⁵. That's the difference between "never" and "a few seconds". Storage is at most ~2ⁿ table entries.
This is the move: trade the factorial of *sequences* for the exponential of *subsets*, by memoizing per
subset. I find it satisfying that the exact same associativity that made order *not matter for the result* is
what makes the subset, not its history, the right thing to memoize.

I can prune the subset lattice further. When I extend a composite S by a table a, what if a has *no* join
predicate to anything in S? Then a⋈S is a Cartesian product — N multiplies by a's full cardinality with no
selectivity to tame it. That's almost always a disaster, and the algebra lets me defer it: a cross product
can be done at any point, so do it as *late* as possible, after all the selective joins have shrunk the
result. Concretely: when building S, only consider peeling off a table a that *is* joined by a predicate to
the rest of S — only grow the composite along an edge of the join graph — and skip the cross-product
extensions as long as an edge-connected extension exists. This means I only ever materialize the *connected*
subsets of the join graph, which for a query whose predicates form a sparse graph is far fewer than 2ⁿ.

Now back to the thing I flagged: the output *order*. I waved my hand and said the cost of finishing from "S
is built" depends only on S, but I caught myself — it might also depend on the *order* the composite for S
arrives in. Let me see why that's not a nuisance to suppress but something I actually have to keep. Suppose
the query ends with ORDER BY on some column, or suppose a later join is a merge that wants its input sorted
on the join column. A plan for S that delivers its tuples *already sorted* on that column saves me a sort
later — maybe a big sort. So consider two plans for S: plan U is cheaper on its own but unordered; plan V is
costlier on its own but emerges sorted on the column I'll need next (because, say, it finished with a merge
join on that column, or accessed a clustered index in that order). If I prune V just because it lost to U on
join cost, I might be throwing away the globally cheapest plan — U will have to pay for a sort that V got for
free. So pruning purely on cost is wrong when order has downstream value.

The fix: an output order is **interesting** if some later operator can exploit it — it's named in an ORDER BY
or GROUP BY, or it's a join column a later merge could use. For each subset S I don't keep just one best plan;
I keep the best *unordered* plan **and** the best plan for each interesting order. A plan in an interesting
order is only ever compared against other plans in that *same* order; it's never knocked out by a cheaper
unordered plan. The bookkeeping is bounded because there aren't many interesting orders, and several physical
orders collapse into one: if predicates equate D.x = E.x = F.x, then "sorted on x" via any of those columns
is the same useful order — one equivalence class, one slot to track. So I keep one best plan per
(subset, interesting-order-class), plus the unordered slot. The DP-table update is just: a new plan replaces
the stored one for its (subset, order) only if it's cheaper *in that slot*.

How does this resolve at the end? When the whole query is assembled and there's an ORDER BY, I have, for the
full set of tables, a cheapest unordered plan and (maybe) a cheapest plan already in the required order. I
compare two finishing options: (cheapest unordered plan + the cost of sorting its output into the required
order) versus (cheapest plan that's already in that order). Take the min. The pre-sorted plan wins exactly
when its extra join cost is less than the sort it lets me skip. If there's no ORDER BY and no downstream
merge, there are no interesting orders and I just take the cheapest plan, done. Keeping interesting orders
multiplies the work by roughly (k+1) where k is the number of interesting orders — a small, bounded price for
not throwing away plans that pay off downstream.

Let me also pin down the running cardinality N that drives the join costs, because it shows up everywhere. N
for a composite S is the product of the cardinalities of the tables in S times the product of the
selectivities of every predicate applicable within S — the same QCARD-style estimate I built earlier,
order-independent, which is *why* it's safe to attach to the set S rather than to a particular plan. The
nested-loop cost for joining inner a onto composite S is C(plan for S) + N(S) · C-inner(a), with C-inner(a)
computed from the selected access path on a after applying the local and join predicates available at that
step. The merge cost has the same C-outer + N·C-inner shape, plus any required sort cost; for a sorted
temporary inner, C-inner is TEMPPAGES/N + W·RSICARD. The merge's output is itself in the join-column order, so
it lands in the corresponding interesting-order slot.

Let me assemble this into the algorithm. Build the DP table bottom-up. Size 1: for each table, every access
path (segment scan plus each index), costed by the single-table formulas; keep the cheapest per output order
(an index scan produces an interesting order; the segment scan is unordered). Sizes 2 through n: for each
connected subset S, for each table a in S whose removal leaves S − {a} still joined to a by a predicate
(skip cross-product peels while an edge peel exists), take each cached plan for S − {a} across its kept
orders, and cost the legal physical alternatives; keep the cheapest result per (S, order) slot. At the top,
finalize against the ORDER BY as above.

Let me make it concrete on the canonical three-table query: employees EMP, departments DEPT, jobs JOB —
"name, salary, job title, department name of employees who are clerks and work in Denver." Local predicates
TITLE='CLERK' (on JOB) and LOC='DENVER' (on DEPT); join predicates EMP.DNO = DEPT.DNO and EMP.JOB = JOB.JOB.
Size 1: for EMP there's the DNO index, the JOB index, and a segment scan — the index scans give interesting
orders DNO and JOB; for DEPT a DNO index and a scan; for JOB a JOB index and a scan. Size 2: pairs that share
a predicate — {EMP,DEPT} via DNO, {EMP,JOB} via JOB — costed both orderings (EMP outer or the small table
outer) and both methods; {DEPT,JOB} share no predicate so I defer that cross product. Size 3: extend each
best pair with the third table along its join edge, comparing nested-loop and merge, and keep the cheapest.
The search can now compare starting with the tiny restricted tables against starting with EMP, and it does so
from the same cost equations instead of trusting the FROM-list order.

Here's the code.

```python
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
import math

W = 0.1  # COST = PAGE FETCHES + W * (RSI CALLS)

# --- catalog statistics (NCARD, TCARD, P; per index ICARD, NINDX) ---
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
        if self.kind == "eq_val":                         # = value: 1/ICARD, else 1/10
            return 1.0/self.index_icard if self.index_icard else 0.1
        if self.kind == "range":                          # interpolate, else 1/3
            return self.frac if self.frac is not None else 1.0/3.0
        if self.kind == "between":                        # interpolate, else 1/4
            return self.frac if self.frac is not None else 1.0/4.0
        if self.kind == "join":                           # 1/MAX(ICARD1,ICARD2)
            if self.index_icard and self.other_icard:
                return 1.0/max(self.index_icard, self.other_icard)
            if self.index_icard:
                return 1.0/self.index_icard
            return 0.1
        raise ValueError(self.kind)

def applicable(preds, S):                                  # predicates wholly inside S
    return [p for p in preds if p.rels <= S]

def cardinality(rels, preds, S):                           # product of cards * applicable F's
    c = 1.0
    for n in S: c *= rels[n].ncard
    for p in applicable(preds, S): c *= p.selectivity()
    return c

@dataclass
class Plan:
    rel_set: frozenset; cost: float; card: float
    order: Optional[str]; desc: str

# --- single-table access paths (segment scan + each index) ---
def access_paths(rel, preds):
    local = applicable(preds, frozenset([rel.name]))
    out = cardinality({rel.name: rel}, preds, frozenset([rel.name]))
    plans = [Plan(frozenset([rel.name]), rel.tcard/rel.p + W*out, out, None,
                  f"segscan({rel.name})")]                 # unordered
    for col, idx in rel.indexes.items():
        matching = [p for p in local if p.column == col]
        f = 1.0
        for p in matching:
            f *= p.selectivity()
        pages = rel.tcard if idx.clustered else rel.ncard
        cost = (f*(idx.nindx+pages) if matching else (idx.nindx+pages)) + W*out
        plans.append(Plan(frozenset([rel.name]), cost, out, col,   # index order = interesting
                          f"index({rel.name}.{col})"))
    return plans

def sort_cost(card, pages):
    return pages*max(1.0, math.log2(max(2.0, pages))) + W*card

# --- the two join methods: same C-outer + N*C-inner shape ---
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
    inner_cost = inner.tcard/max(1.0, N) + W*rsicard       # TEMPPAGES/N + W*RSICARD
    cost = left.cost + sort_outer + sort_inner + N*inner_cost
    return Plan(combo, cost, cardinality(rels, preds, combo), col,  # output in join order
                f"merge({left.desc}, {inner.name} on {col})")

# --- don't-consider-cross-products: only grow along a join edge ---
def connected(preds, left_set, r):
    return any(p.kind == "join" and r in p.rels and (p.rels - {r}) & left_set
               for p in preds)

def keep(table, plan):                                     # cheapest per (set, interesting order)
    key = (plan.rel_set, plan.order)
    old_cost = table[key].cost if key in table else math.inf
    if plan.cost < old_cost:
        table[key] = plan

# --- the dynamic program: optjoin(S) bottom-up over subset size ---
def optimize(rels, preds, order_by=None):
    names = list(rels); n = len(names); plans = {}
    for nm in names:                                       # size 1
        for p in access_paths(rels[nm], preds): keep(plans, p)
    for size in range(2, n+1):                             # sizes 2..n
        for subset in combinations(names, size):
            S = frozenset(subset)
            for inner in subset:
                left_set = S - {inner}
                if not connected(preds, left_set, inner) and \
                   any(connected(preds, left_set, r) for r in subset if r != inner):
                    continue                               # defer the cross product
                for left in [pl for (rs, _), pl in plans.items() if rs == left_set]:
                    keep(plans, nested_loop(left, rels[inner], preds, S, rels))
                    ms = merge_scan(left, rels[inner], preds, S, rels)
                    if ms is not None: keep(plans, ms)
    full = frozenset(names)
    cands = [pl for (rs, _), pl in plans.items() if rs == full]
    if order_by is None:                                   # no interesting order to honor
        return min(cands, key=lambda p: p.cost)
    in_order = [p for p in cands if p.order == order_by]   # compare: pre-sorted vs sort-at-end
    cheapest = min(cands, key=lambda p: p.cost)
    sort_then = cheapest.cost + sort_cost(cheapest.card, cheapest.card)
    if in_order and min(in_order, key=lambda p: p.cost).cost <= sort_then:
        return min(in_order, key=lambda p: p.cost)
    return Plan(full, sort_then, cheapest.card, order_by, f"sort({cheapest.desc})")
```

The causal chain, start to end: the join result is order-independent but its cost is not, and the number of
orders is n!·(n−1)! — unenumerable — so I restrict to left-deep trees (killing the nesting factor and
letting the inner be a base table with its own indexes), then observe that the cheapest plan for a *set* of
tables depends only on that set and not on how it was assembled, which lets me memoize per subset and build
bottom-up by size — turning n!·(n−1)! into O(n·m·2ⁿ) — costing each candidate with a coarse selectivity model
that I trust only for *ranking*, deferring cross products so the DP touches only connected subsets, and
keeping not one but one-plan-per-interesting-order so a pre-sorted plan that's costlier on its own can still
win the whole query by sparing a downstream sort.
