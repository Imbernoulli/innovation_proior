# Beam search

## Problem

Many problems build a solution as a sequence of decisions, forming a layered
decision graph: layer `t` holds the partial solutions that have made `t`
decisions, and each node fans out into a few children at layer `t+1`. We want
the sequence with the best final cumulative score. The exact frontier is
hopeless — with branching factor `b` and depth `D` it grows to `b^D` (millions
of decode paths, `2^n` knapsack subsets). We need a search that runs in time and
memory **linear in depth**, forward-only (no backtracking), and returns a
high-scoring solution even though we forgo a proof of optimality.

## Key idea

Sweep the graph forward layer by layer, but keep only a bounded *beam* of the
most promising partial solutions at each layer. Maintain a frontier of at most
`W` states (the **beam width**). To advance one layer: expand every state in the
frontier into its children, rank all the children by an evaluation that
estimates the **final** score of their best completion (`g + h`: score-so-far
plus an optimistic estimate of the rest), and **prune** the pool back to the top
`W`. Repeat to the last layer; the best feasible solution ever seen is the answer.

`W` is a dial. At `W = ∞` nothing is pruned: the frontier is the entire layer,
so the search becomes the exact full-breadth sweep — exhaustive (exponential
over a tree, or full-label dynamic programming once equal states are merged). At
`W = 1` it keeps the single best child each layer — greedy hill-climbing. In
between it trades a little optimality for a lot of speed: with a constant-time
evaluation and linear-time top-`W` selection the cost is `O(W · b · D)` — linear
in `W` and in depth, never exponential. Quality *tends* to improve as `W` grows
(and the exact optimum is recovered once the beam is wide enough never to prune
the eventual winner), but ordinary beam search carries **no monotonicity
guarantee** — a wider beam can occasionally return a worse solution; the
monotonic variants are separate algorithms. Beam search is deliberately
**incomplete and non-optimal** — a goal/optimum can be pruned — in exchange for a
predictable, tunable, linear-time forward search.

Two important refinements:

- **Filtered beam search.** When the accurate evaluation is expensive, prune in
  two stages: a cheap local score filters the children down to a *filter width*,
  then the expensive global evaluation ranks the survivors down to the *beam
  width* `W` — most of the ranking quality at a fraction of the evaluation cost.
- **The diversity trap and Chokudai search.** A fixed beam tends to fill all `W`
  slots with near-duplicates of one strong early lineage, so widening `W` gives
  diminishing returns. Fixes: dedup states (e.g. by hash), cap same-score
  states, or sample survivors stochastically (probability rising with score)
  instead of taking a hard top-`W`. The structural fix — **Chokudai search** —
  keeps a priority queue *per layer* of every state that ever reached it; one
  *pass* pops the single best un-expanded state at each layer in order and pushes
  its children forward (one pass ≈ a width-1 beam run to the end), then repeats,
  each pass tending to develop a different lineage. After `k` passes you have
  roughly `k` diverse solutions; it is anytime (loop until the time budget is
  spent, no `W` to tune) at the cost of retaining all states.

## Algorithm

```
beam_search(problem, W):
    frontier = [ initial_state ]
    best = initial_state
    repeat depth times:                         # forward, one layer per step
        pool = []
        for s in frontier:                      # expand every live state
            for c in children(s):               # one decision deeper
                pool.append(c)
        rank pool by  g(c) + h(c)               # score-so-far + optimistic est.
        frontier = top W of pool                # PRUNE -- the beam
        best = better of (best, best feasible state in frontier)
    return best
```

- `g(c)`: objective accumulated along the partial solution.
- `h(c)`: an *optimistic* estimate of the best remaining completion (for
  knapsack, the fractional-relaxation / LP bound). Ranking by `g + h` keeps a
  currently-behind-but-promising state alive instead of cutting it for an
  early-easy one.
- The single pruning line `frontier = top W of pool` is the whole method; no
  backtracking, memory bounded by `W`. Work is `O(W · b · D)` *given a
  constant-time evaluation and linear top-`W` selection*; an `O(n)`-per-state
  bound and a per-layer sort add their own factors (for this knapsack code,
  roughly `O(n · W · n)` from the bound plus `O(n · W log W)` from the sorts).

## Code

```python
from dataclasses import dataclass, field


@dataclass(order=True)
class State:
    key: float                              # rank by value-so-far + optimistic bound
    idx: int = field(compare=False)         # how many items decided (the layer)
    value: int = field(compare=False)       # value packed so far  (this is g)
    cap: int = field(compare=False)         # remaining capacity
    chosen: tuple = field(compare=False)    # item indices taken


def bound(value, cap, idx, items):
    """h: optimistic completion -- the LP (fractional-knapsack) relaxation of the
    remaining items: pack them in DESCENDING value/weight order, last item taken
    fractionally. The fractional optimum is the LP relaxation of the integer
    remainder, so it never underestimates the best integer completion -- exactly
    what we want to rank states by. (The descending-density order is what makes
    it an upper bound; input order is not.)"""
    b = value
    for w, v in sorted(items[idx:], key=lambda iv: iv[1] / iv[0], reverse=True):
        if w <= cap:
            cap -= w
            b += v
        else:
            b += v * (cap / w)
            break
    return b


def beam_search(items, capacity, W):
    n = len(items)
    layer = [State(0.0, 0, 0, capacity, ())]        # frontier at layer 0
    best = layer[0]
    for _ in range(n):                              # one layer per item, forward only
        pool = []
        for s in layer:                             # expand every live state
            w, v = items[s.idx]
            # skip item s.idx
            pool.append(State(bound(s.value, s.cap, s.idx + 1, items),
                              s.idx + 1, s.value, s.cap, s.chosen))
            # take item s.idx, if it fits
            if w <= s.cap:
                nv, nc = s.value + v, s.cap - w
                pool.append(State(bound(nv, nc, s.idx + 1, items),
                                  s.idx + 1, nv, nc, s.chosen + (s.idx,)))
        pool.sort(key=lambda s: s.key, reverse=True)   # rank by g + h
        layer = pool[:W]                               # PRUNE to beam width W
        for s in layer:
            if s.value > best.value:
                best = s
    return best.value, best.chosen


def brute_force(items, capacity):
    n = len(items)
    best = 0
    for mask in range(1 << n):
        w = sum(items[i][0] for i in range(n) if mask >> i & 1)
        if w <= capacity:
            best = max(best, sum(items[i][1] for i in range(n) if mask >> i & 1))
    return best


if __name__ == "__main__":
    import random
    random.seed(7)
    items = [(random.randint(1, 15), random.randint(1, 20)) for _ in range(20)]
    capacity = 60

    opt = brute_force(items, capacity)
    print(f"brute-force optimum: {opt}")
    for W in (1, 2, 5, 20, 100):
        val, chosen = beam_search(items, capacity, W)
        print(f"beam W={W:>3}: value={val:>3}  gap={opt - val:>2}  items={len(chosen)}")
```

Running it shows the signature: the achieved value climbs as the beam widens and
flattens once the beam is wide enough never to prune the eventual winner —

```
brute-force optimum: 185
beam W=  1: value=179  gap= 6  items=12
beam W=  2: value=183  gap= 2  items=13
beam W=  5: value=185  gap= 0  items=13
beam W= 20: value=185  gap= 0  items=13
beam W=100: value=185  gap= 0  items=13
```
