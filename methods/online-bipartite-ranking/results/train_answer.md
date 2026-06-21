We are handed a bipartite graph $G = (L, R, E)$ promised to contain a perfect matching of size $n$. The offline side $L$ is known up front; the online side $R$ arrives one vertex at a time, and when $j \in R$ arrives its edges are revealed for the first time, at which point we must commit $j$ irrevocably to one of its currently-free neighbors in $L$ (or drop it) before we are allowed to see the next arrival. We want the final matching as large as possible, judged by the worst case over all graphs and all arrival orders of $\mathbb{E}[\text{ALG}]/\text{OPT}$ against an oblivious adversary — one who fixes the graph and the order in advance but cannot see our coins. The floor is easy and instructive. Plain greedy — when $j$ arrives, grab any free neighbor — always produces a maximal matching, and a maximal matching is at least half of any other: for any edge $\{u,v\}$ of a maximum matching $M^*$, if we left both endpoints free we could have added $\{u,v\}$, so we touch at least one endpoint of every $M^*$ edge, giving $|\text{ALG}| \ge \text{OPT}/2$. The trouble is that $1/2$ is also a ceiling for every deterministic rule. A two-row phase adversary defeats even an algorithm that refuses edges to save them: present a vertex adjacent to two unused offline vertices $a,b$; whichever way the algorithm commits (or if it refuses), follow with a vertex adjacent only to the "wrong" one, so the online algorithm gets at most one match per phase while the offline optimum gets two. Repeated on disjoint pairs this pins any deterministic algorithm to exactly $n/2$. So we must randomize — but the obvious randomization, picking a uniformly random free neighbor at each arrival, gains essentially nothing, $n/2 + O(\log n)$. The reason is sharp: the coin has no memory. On a graph that is the diagonal plus a dense block sending the second-half columns to first-half rows, fresh per-arrival coins pour the algorithm's matches into the dense block early, burning exactly the first-half rows that the sparse diagonal arrivals later require. Memoryless randomness can be steered into wasting the very vertices a future sparse arrival needs; independent local randomness is the wrong kind of randomness.

The failure is the clue: the defect is not that we are random but that we are random *anew* each step, with no consistent notion of which offline vertices we are protecting and which we are willing to spend early. So I propose RANKING: commit once, at initialization, to a single global uniformly random total order $\sigma$ over $L$, and then have every arrival defer to that one order. On the arrival of $j$, among $j$'s currently-unmatched neighbors, match $j$ to the one of highest priority — smallest $\sigma$ — and if none is free, drop $j$. The whole algorithm is one random object consulted greedily. What makes it beat $n$ fresh coins is that the decisions are now *correlated through a single object*: a high-ranked offline vertex is consistently preferred whenever available, so it is spent deliberately and early, while a low-ranked vertex is consistently held back, so it survives for a sparse late arrival. There is an implicit self-correction — among currently-eligible offline vertices the rule favors those that have been eligible least often, because the frequently-eligible ones were probably already grabbed by an earlier arrival that preferred them — so the algorithm naturally stops over-investing in the dense part of the graph, the exact failure mode that sank greedy and per-step random. It is genuinely distinct from per-step random, whose distribution over free neighbors is re-rolled every arrival; RANKING's choice among free neighbors is neither uniform nor independent across arrivals, it is whatever the one global order dictates.

The guarantee is that RANKING is $(1 - 1/e)$-competitive, and that no randomized online algorithm beats $1 - 1/e$, so it is optimal. The cleanest worst case to argue against is the complete upper-triangular instance: rows and columns $1..n$, column $j$ adjacent to rows $1..j$, the unique perfect matching on the diagonal, columns arriving $n, n-1, \dots, 1$. One can reduce to it: a perfect matching can be assumed (deleting a vertex outside a maximum matching changes the two runs by at most one alternating path, weakly worsening the ratio), and zeroing the subdiagonal can only hurt RANKING, because RANKING on the sparser upper-triangular graph is simulable as a refusal variant of RANKING on the original, and by induction the refusal variant's free set is always a superset of the full run's free set, so it matches a subset. On an upper-triangular instance, for each diagonal index $i$, at least one of $\{$row $i$, column $i\}$ is matched — if both were free, column $i$ would have matched to row $i$ on arrival — so with $D$ the set of indices where *both* are matched, the matching covers $n + |D|$ vertices and $|M| = (n + |D|)/2$, hence $\mathbb{E}|M| = n/2 + \tfrac{1}{2}\,\mathbb{E}|D|$. Everything above one-half is the both-matched set. Writing $x_t$ for the probability that the rank-$t$ offline vertex is matched, the engine is this: fix the rank-$t$ vertex $v$ with diagonal partner $u$; if $v$ is *unmatched* (probability $1 - x_t$), then when $u$ arrived it saw $v$ free, so $u$ is certainly matched, and to a neighbor of rank even smaller than $t$. The seductive count "$1 - x_t \le \tfrac{1}{n}\sum_{s \le t-1} x_s$" assumes $u$ is a uniformly random arriving vertex independent of the top-rank matched set $R_{t-1}$ — but $u = m^*(v)$ and $R_{t-1}$ are computed from the same permutation $\sigma$ and are correlated, so that recurrence is simply wrong. The independence must be manufactured, not wished for: start from a random $\sigma$, pick a uniformly random offline vertex, pull it out, and reinsert it at rank $t$ to form $\sigma'$. Now $u$ is uniform and independent of $\sigma$ because it depends only on which vertex was drawn. Removing and reinserting one vertex moves RANKING's matching by at most one alternating path along which ranks shift monotonically upward (each displaced vertex was already taking its best available option), so "$v$ unmatched under $\sigma'$" transports to "$u$ matched at rank $\le t$" in $\sigma$ — moving one vertex changes the partner's rank by at most one position, and integrality closes the gap. With $u$ uniform and independent of $R_t$, $\Pr[u \in R_t \mid \sigma] = |R_t|/n$, giving the corrected
$$1 - x_t \le \frac{1}{n}\sum_{s \le t} x_s,$$
the sum now running to $t$, honestly. Setting $S_t = \sum_{s \le t} x_s$, this is $S_t(1 + 1/n) \ge 1 + S_{t-1}$; the competitive ratio is $S_n/n$, and its smallest admissible value makes every inequality tight, $S_t(1 + 1/n) = 1 + S_{t-1}$ with $S_0 = 0$, whose minimizing sequence is $x_t = (1 - 1/(n+1))^t$ and whose ratio is
$$\frac{1}{n}\sum_{t=1}^n \Big(1 - \tfrac{1}{n+1}\Big)^t = 1 - \Big(1 - \tfrac{1}{n+1}\Big)^n \;\longrightarrow\; 1 - \frac{1}{e} \approx 0.632.$$
The $e$ is the geometric thinning of "still-available probability" at rate $1/n$ per rank, integrated into an exponential. That $1 - 1/e$ is also the ceiling follows from tracking RANDOM on the same instance — columns remaining $x$ and eligible rows $y$ obey $dy/dx = 1 + (y-1)/x$, which integrates to a matching of size $n(1 - 1/e) + o(n)$ — and Yao's lemma, since the best deterministic algorithm against the uniform distribution over row-permutations is greedy, matching RANDOM. No online algorithm exceeds $1 - 1/e$, and RANKING attains it.

The combinatorial route works but is heavy — a perturbed permutation, a monotone-alternating-path lemma, a refusal variant, and a recurrence that was wrong before it was right — and the appearance of the same $1 - 1/e$ that the *fractional* water-filling rule achieves with no randomness at all begs for one unifying analysis. Weak duality supplies it. The matching LP maximizes $\sum x_{ij}$ with each vertex's incident sum $\le 1$; its dual minimizes $\sum_i \alpha_i + \sum_j \beta_j$ subject to $\alpha_i + \beta_j \ge 1$ on every edge, so any feasible dual upper-bounds OPT. Deterministic online dual feasibility is too strong — it amounts to solving the fractional problem, and integral determinism gets only $1/2$ — so the right relaxation is feasibility *in expectation*, $\mathbb{E}[\alpha_i + \beta_j] \ge 1$ per edge over RANKING's randomness; that is exactly the crack that lets randomness in. Use the threshold form, where each $i$ draws $Y_i \sim U[0,1]$ and $j$ matches to its free neighbor of smallest $Y_i$ (sorted thresholds are a uniform permutation, so this *is* RANKING). On matching $i$ to $j$, split a per-match value of $1/F$ via a monotone non-decreasing $g:[0,1]\to[0,1]$ with $g(1)=1$:
$$\alpha_i = \frac{g(Y_i)}{F}, \qquad \beta_j = \frac{1 - g(Y_i)}{F},$$
with $\alpha_i = \beta_j = 0$ for the unmatched. Then $\alpha_i + \beta_j = 1/F$ on every match, so the dual value is exactly $(1/F)|\text{ALG}|$ deterministically — the primal-to-dual ratio is locked at $F$ regardless of the coins. There is a clean reading: $i$ keeps $g(Y_i)/F$ and *offers* $(1-g(Y_i))/F$ to $j$; since $g$ increases, a low-threshold (high-priority) $i$ offers more, and "$j$ takes the highest offer" is exactly "$j$ takes the smallest $Y_i$," reproducing RANKING. For expected feasibility on a fixed edge $(i,j)$, freeze all thresholds but $Y_i$ and run the algorithm with $i$ deleted; let $y^c$ be the threshold matched to $j$ there ($y^c = 1$ if $j$ goes unmatched). Two lemmas. The dominance lemma: $i$ is matched whenever $Y_i < y^c$, because if $i$ were free at $j$'s arrival the run would be identical to the $i$-deleted run, where $j$ takes threshold $y^c$ — but a lower-threshold free $i$ would have been taken instead, a contradiction — so $\mathbb{E}[\alpha_i] \ge \tfrac{1}{F}\int_0^{y^c} g(y)\,dy$. The monotonicity lemma: $\beta_j \ge (1 - g(y^c))/F$ for every $Y_i$, because by induction the real run's free set stays a superset of the $i$-deleted run's, so $j$'s real free neighbors include the threshold-$y^c$ one and $j$ matches to a threshold $\le y^c$; with $g$ increasing, $\beta_j$ is no smaller. Together, with $\theta = y^c$,
$$\mathbb{E}_{Y_i}[\alpha_i + \beta_j] \ge \frac{1}{F}\Big[\int_0^\theta g(y)\,dy + 1 - g(\theta)\Big],$$
so demanding $\int_0^\theta g(y)\,dy + 1 - g(\theta) \ge F$ for all $\theta$ makes the expected dual feasible, whence $\mathbb{E}[\text{ALG}] = F\cdot\mathbb{E}[\text{dual}] \ge F\cdot\text{OPT}$. Maximizing $F$ at equality and differentiating in $\theta$ gives $g(\theta) - g'(\theta) = 0$, so $g' = g$, $g(\theta) = Ce^\theta$, and $g(1) = 1$ forces $g(\theta) = e^{\theta - 1}$; then $\int_0^\theta e^{y-1}dy + 1 - e^{\theta-1} = 1 - e^{-1}$ identically, so $F = 1 - 1/e$. The exponential is forced by self-consistency across thresholds, with no perturbation and no path lemma. And the coincidence resolves: the fractional water-filling rule is the *same* dual scheme with $Y_i$ replaced by a deterministic capacity level, the same integral equation yielding the same constant — integral RANKING and fractional water-filling are one primal-dual algorithm differentiated only by whether the offline vertex's level is a continuous accumulation or a one-shot random threshold, with feasibility-in-expectation the exact price of integrality.

```python
"""
RANKING for online bipartite matching.

Offline side L is known up front; online side R arrives one vertex at a time,
revealing its edges only on arrival. Each arriving j must be matched (or dropped)
irrevocably before the next arrival.

Two equivalent forms are implemented:
  - permutation form: fix a uniformly random total order on L, match each arriving
    j to its unmatched neighbor of smallest rank.
  - threshold form: each i in L draws Y_i ~ U[0,1] independently, match each
    arriving j to the unmatched neighbor of smallest Y_i. (This is the form that
    makes the primal-dual analysis natural; g(Y_i)=e^{Y_i-1} sets the duals.)

Both are the SAME algorithm: the sorted Y-values induce a uniformly random
permutation of L.
"""

import random
import math
import heapq


def initialize_state(L, rng):
    """Draw the persistent random priorities used for the whole run."""
    return {i: rng.random() for i in L}      # Y_i ~ U[0,1]; ties have prob 0


def choose_neighbor(avail, state):
    """Choose the available neighbor with smallest persistent priority."""
    return min(avail, key=lambda v: state[v])


def stateful_match(L, arrivals, neighbors, state):
    """Generic one-pass harness filled with RANKING's priority rule."""
    matched_to = {}          # i in L  ->  j in R
    for j in arrivals:
        # On arrival we only get to see edges of j; among its UNMATCHED neighbors,
        # take the one earliest in the fixed order. No look-ahead, no revocation.
        avail = [i for i in neighbors[j] if i not in matched_to]
        if avail:
            matched_to[choose_neighbor(avail, state)] = j
    return matched_to


def ranking_match(L, arrivals, neighbors, rank):
    """Run RANKING.

    L:         iterable of offline vertices (known in advance).
    arrivals:  list giving the online arrival order (vertices of R).
    neighbors: dict j -> iterable of its neighbors in L (revealed on arrival of j).
    rank:      dict i -> comparable key; j is matched to the unmatched neighbor
               with the SMALLEST key. A uniformly random key gives RANKING.

    Returns the matching as a dict i -> j.
    """
    return stateful_match(L, arrivals, neighbors, rank)


def random_rank(L, rng):
    """Uniformly random total order on L (the random priority permutation)."""
    return initialize_state(L, rng)


def greedy_match(L, arrivals, neighbors):
    """Deterministic first-neighbor greedy.

    Always yields a maximal matching, hence >= OPT/2; an adversary forces = OPT/2.
    """
    matched_to = {}
    for j in arrivals:
        for i in neighbors[j]:
            if i not in matched_to:
                matched_to[i] = j
                break
    return matched_to


# --- complete upper-triangular graph ------------------------------------------
# Rows (offline L) and columns (online R) are 1..n. The matrix is upper-triangular
# with ones on and above the diagonal: column j is adjacent to rows 1..j. The
# unique perfect matching sits on the diagonal (row j matched to column j).
# Columns arrive in the order n, n-1, ..., 1: the first column
# to arrive sees every row, so an unlucky ranking wastes a high-priority row that a
# later, sparser column would have needed. RANKING attains 1 - 1/e here in the
# limit, matching the general upper bound.
def upper_triangular(n):
    L = list(range(1, n + 1))
    neighbors = {j: list(range(1, j + 1)) for j in range(1, n + 1)}  # rows 1..j
    arrivals = list(range(n, 0, -1))                                 # n, n-1, ..., 1
    return L, arrivals, neighbors


def competitive_ratio_on_upper_triangular(n, trials, match_fn, init_state, seed=0):
    rng = random.Random(seed)
    L, arrivals, neighbors = upper_triangular(n)
    total = 0
    for _ in range(trials):
        state = init_state(L, rng)
        total += len(match_fn(L, arrivals, neighbors, state))
    return (total / trials) / n


def ranking_size_upper_triangular(n, rng):
    """Fast simulation of RANKING on the upper-triangular graph."""
    rank = random_rank(range(1, n + 1), rng)
    heap = [(rank[i], i) for i in range(1, n + 1)]
    heapq.heapify(heap)

    matched = 0
    for j in range(n, 0, -1):
        while heap and heap[0][1] > j:
            heapq.heappop(heap)
        if heap:
            heapq.heappop(heap)
            matched += 1
    return matched


def ranking_ratio_on_upper_triangular(n, trials, seed=0):
    """Monte-Carlo estimate of E[|RANKING|]/OPT on the n x n upper-triangular graph.
    OPT = n (the diagonal perfect matching). Should approach 1 - 1/e for large n."""
    rng = random.Random(seed)
    total = 0
    for _ in range(trials):
        total += ranking_size_upper_triangular(n, rng)
    return (total / trials) / n


def greedy_half_trap(n):
    """Instance where first-neighbor greedy gets exactly half of OPT."""
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


def greedy_ratio_on_half_trap(n):
    """Deterministic greedy / OPT on its standard half lower-bound instance."""
    L, arrivals, neighbors = greedy_half_trap(n)
    return len(greedy_match(L, arrivals, neighbors)) / n


if __name__ == "__main__":
    target = 1 - 1 / math.e
    print("RANKING on the complete upper-triangular family")
    for n, trials in ((100, 2000), (1000, 2000), (10000, 500)):
        r = ranking_ratio_on_upper_triangular(n, trials=trials)
        print(f"n={n:5d}  E[RANKING]/OPT ~ {r:.4f}   (1-1/e = {target:.4f})")

    print("\nFirst-neighbor greedy on the deterministic half-trap")
    for n in (100, 1000, 10000):
        r = greedy_ratio_on_half_trap(n)
        print(f"n={n:5d}  GREEDY/OPT      = {r:.4f}   (1/2 = {0.5:.4f})")
```
