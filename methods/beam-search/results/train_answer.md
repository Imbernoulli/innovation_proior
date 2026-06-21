The problem is a layered decision graph: every node branches into several children, depth is large, and the exact frontier grows exponentially. We need a search that stays forward-only and linear in depth, without the exponential memory or the brittleness of committing to a single path. Exact methods like full-breadth dynamic programming or A* are correct, but they pay for optimality by touching the whole breadth at every step, which is unaffordable for speech decoding or large combinatorial instances. A single greedy path is cheap, but one locally attractive yet globally wrong decision ruins the solution, and recovering from that requires backtracking, which wastes all the work already done on the abandoned prefix.

The missing idea is neither keep-all nor keep-one, but keep the best few. At each layer we expand every currently live partial solution into its children, score each child by an estimate of the final score it could still achieve, and prune the pool down to a small fixed number of survivors. Memory is bounded by that small number, work is linear in depth, and there is no backtracking because discarded states are never revisited. The survivors act as a beam of near-miss alternatives around the most promising path, so a temporary ranking mistake has room to correct itself in later layers.

The method is beam search. We maintain a frontier of at most W states, called the beam width. To advance one layer, we generate every child of every state in the frontier, rank the children by g + h where g is the score accumulated so far and h is an optimistic estimate of the best remaining completion, and keep only the top W. The best feasible state ever seen during the sweep is returned as the answer. W is the central dial: at W = 1 the method is greedy hill-climbing, and at W = infinity nothing is pruned so it becomes the exact full-breadth sweep. Intermediate values trade a small amount of optimality for a large gain in speed. The cost is O(W · b · depth) given constant-time evaluation and linear top-W selection, so it scales linearly rather than exponentially.

Two refinements are worth keeping in mind. When the accurate global evaluation is expensive, we can prune in two stages: first use a cheap local score to filter down to a wider intermediate set, then apply the expensive evaluation only on those survivors before pruning to the final beam width. When the beam collapses into many near-duplicates of a single lucky lineage, diversity can be restored by deduplicating states, capping same-score states, or sampling survivors stochastically. A structural alternative called Chokudai search keeps a priority queue per layer and repeatedly runs width-one passes, each pass tending to develop a different lineage; this gives anytime behavior at the cost of retaining more states.

For concreteness, here is a working implementation for 0/1 knapsack. Each state records how many items have been decided, the value packed so far, the remaining capacity, and the chosen items. The ranking key is value-so-far plus the fractional-knapsack LP relaxation of the remaining items, which is an optimistic upper bound on any integer completion.

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
    """Optimistic completion: fractional-knapsack LP relaxation of remaining items.
    Pack remaining items in DESCENDING value/weight order, taking the last item
    fractionally. This never underestimates the best integer completion."""
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
    layer = [State(0.0, 0, 0, capacity, ())]
    best = layer[0]
    for _ in range(n):                          # one layer per item, forward only
        pool = []
        for s in layer:                         # expand every live state
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
        layer = pool[:W]                               # prune to beam width W
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
