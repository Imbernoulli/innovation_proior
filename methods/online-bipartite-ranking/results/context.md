# Context: beating one-half for online bipartite matching

## Research question

We are given a bipartite graph G = (L, R, E) that is promised to contain a perfect matching of size n. The vertices of L are known in advance. The vertices of R arrive one at a time; when a vertex j ∈ R arrives, the edges incident to j are revealed for the first time, and we must decide immediately and irrevocably which unmatched neighbor of j to match it to (or to leave it unmatched). We never see a future arrival before committing to the current one, and a match once made cannot be undone. The goal is to make the final matching as large as possible.

Performance is measured against the best offline matching by the competitive ratio: the worst case, over all graphs G and all arrival orders, of E[size of the matching the algorithm builds] / OPT, where OPT is the size of the maximum matching and the expectation is over the algorithm's internal randomness. The adversary is oblivious: it fixes the graph and the arrival order in advance, without seeing the algorithm's coin flips. The central question is how high this ratio can be pushed, and whether there is an algorithm that is provably optimal among all online algorithms.

The question matters because online bipartite matching is the combinatorial core of a family of assignment problems where supply is known but demand arrives over time and must be served on the spot: load balancing and online assignment, packet routing through switch queues, online market clearing, and the allocation of advertisement slots to bidders as search queries arrive. A tight bound here propagates into tight bounds for all of those.

## Background

The field state is competitive analysis of online algorithms: judge an online decision-maker by the worst-case ratio of its profit to that of an optimal offline algorithm on the same input. This framing had been applied to list update and paging (Sleator–Tarjan 1985), dynamic bin packing (Coffman–Garey–Johnson 1983), online graph coloring (Gyárfás–Lehel 1988), and the k-server problem (Manasse–McGeoch–Sleator 1988). What was missing was an application to matching, and in particular a clean separation between what deterministic and randomized online algorithms can achieve.

A second relevant strand is the adversary taxonomy for randomized online algorithms (Ben-David–Borodin–Karp–Tardos–Wigderson 1990): an oblivious adversary fixes the input before any coins are flipped, an adaptive-online adversary may build the input incrementally while watching the algorithm's responses. The distinction is sharp here. Against the adaptive-online adversary, no randomized matching algorithm can do better than n/2 + O(log n): the adversary specifies each column knowing the algorithm's past choices, and builds its own perfect matching column-by-column on rows the algorithm has not yet committed to. So any hope of beating one-half lives entirely in the oblivious model, where the random ordering of decisions is hidden from the graph that was already fixed.

The load-bearing combinatorial fact underneath everything is the structure of a maximal matching. If a matching M leaves some edge {u, v} with both endpoints unmatched, M was not maximal — you could add {u, v}. So any maximal matching is at least half the size of any other matching, in particular at least OPT/2. This is the reason the trivial algorithm already gets one-half, and it is the floor every more clever idea is trying to climb above.

A diagnostic observation about per-step randomization sets up the design pressure. The naive randomized algorithm flips a fresh coin at each arrival and matches to a uniformly random unmatched neighbor. Despite "being randomized," it performs essentially like the deterministic greedy — about n/2 + O(log n) in expectation on an adversarial instance. The instance that exposes this has ones at B_ij = 1 when i = j, and additionally a dense block (n/2 < j ≤ n, 1 ≤ i ≤ n/2): fresh per-arrival coins spend the algorithm's effort on the dense upper half during the first half of arrivals, leaving it without the specific rows the sparse lower half later requires. So even an explicitly randomized rule, applied afresh at each arrival, can be steered into wasting exactly the vertices that a later sparse arrival will require.

## Baselines

**Deterministic online matching.** When j arrives, the simplest rule matches it to any unmatched neighbor if one exists. Core property: the result is always a maximal matching, hence at least OPT/2. That lower bound is tight for every deterministic algorithm, including one that sometimes refuses available edges. The adversary works in disjoint two-row phases. Pick two unused offline vertices a and b, then present an online vertex adjacent to both. If the algorithm matches it to a, present a second vertex adjacent only to a; if it matches it to b, present a second vertex adjacent only to b; if it refuses the first vertex, present a second vertex adjacent only to a. The online algorithm gets at most one match in the phase, while the offline optimum matches both vertices by sending the first vertex to the other row and the singleton to its unique row. Repeating the phase gives ALG ≤ n/2 and OPT = n. So deterministic algorithms are pinned at 1/2 from both sides; the gap to be closed is entirely about whether randomization can break this.

**Naive randomized greedy (per-step coin).** When j arrives, match it to a uniformly random unmatched neighbor. Intuitively this should diffuse the adversary's ability to predict the algorithm's choices, but it does not help in the worst case: its expected matching is n/2 + O(log n), no better than deterministic up to lower-order terms. The gap it leaves open: the adversary can still arrange a graph where the algorithm's choices, though random, are systematically misallocated across time.

**Online fractional matching / water-filling (b-matching).** For the relaxed problem where each offline vertex can be matched up to b times (b large), a deterministic "water-level" strategy (Kalyanasundaram–Pruhs 2000) is natural: maintain a level y_i for each offline vertex and route an arriving vertex toward neighbors of lowest level. For fractional problems randomization buys nothing, and a deterministic water-filling rule attains 1 − 1/e. The gap relative to the integral problem: this is a fractional/divisible allocation, and there are simple instances (an 8-cycle with all edges at value 1/2, the first two arrivals diametrically opposite) showing that one cannot, in an online way, round such a fractional matching to an integral one of equal expected size by treating the fractional algorithm as a black box. So the integral problem genuinely needs its own randomized idea, not a rounding of the fractional one. Both problems do, however, share the same constant 1 − 1/e in their known bounds.

## Evaluation settings

The natural yardstick is the worst-case competitive ratio against an oblivious adversary: minimize, over all bipartite graphs admitting a perfect matching and all arrival orders, the ratio of expected algorithm size to OPT. The canonical hard family is the complete upper-triangular instance: rows and columns 1..n, with column j adjacent to rows 1..j, the unique perfect matching on the diagonal, and columns arriving in the order n, n−1, …, 1. This family is the stress test for any candidate algorithm because the first arrival sees every offline vertex while the last sees only one, so any misallocation of an early, widely-adjacent arrival is maximally costly. Two reference points bound the achievable range from prior analysis: deterministic and per-step-random algorithms sit at 1/2, and the fractional relaxation sits at 1 − 1/e. The metric is the limiting ratio as n → ∞; both an upper bound (no online algorithm can exceed it) and a matching algorithmic lower bound are required to call an algorithm optimal.

## Code framework

```python
import random

def stateful_match(L, arrivals, neighbors, state):
    """Generic one-pass online bipartite matching harness.

    L:          offline vertices, known in advance.
    arrivals:  online arrival order; each j revealed one at a time.
    neighbors:  j -> its neighbors in L, revealed only when j arrives.
    state:      auxiliary state available to the matching rule.
    """
    matched_to = {}                       # i in L -> j in R
    for j in arrivals:
        avail = [i for i in neighbors[j] if i not in matched_to]
        if avail:
            choice = choose_neighbor(avail, state)
            matched_to[choice] = j
    return matched_to


def choose_neighbor(avail, state):
    # TODO: choose one available neighbor.
    pass


def initialize_state(L, rng):
    # TODO: set up any state for the online rule.
    pass


def greedy_match(L, arrivals, neighbors):
    """Baseline first-neighbor greedy."""
    matched_to = {}
    for j in arrivals:
        for i in neighbors[j]:
            if i not in matched_to:
                matched_to[i] = j
                break
    return matched_to


def greedy_half_trap(n):
    """Instance where this fixed greedy rule gets exactly half of OPT."""
    if n % 2:
        raise ValueError("n must be even")
    half = n // 2
    L = list(range(1, n + 1))
    arrivals = list(range(1, n + 1))
    neighbors = {}
    for j in range(1, half + 1):
        neighbors[j] = L[:]
    for k in range(1, half + 1):
        neighbors[half + k] = [k]
    return L, arrivals, neighbors


def upper_triangular(n):
    """Rows and columns 1..n; column j adjacent to rows 1..j; arrivals n..1."""
    L = list(range(1, n + 1))
    neighbors = {j: list(range(1, j + 1)) for j in range(1, n + 1)}
    arrivals = list(range(n, 0, -1))
    return L, arrivals, neighbors


def competitive_ratio_on_upper_triangular(n, trials, match_fn, init_state, seed=0):
    rng = random.Random(seed)
    L, arrivals, neighbors = upper_triangular(n)
    total = 0
    for _ in range(trials):
        state = init_state(L, rng)
        total += len(match_fn(L, arrivals, neighbors, state))
    return (total / trials) / n
```
