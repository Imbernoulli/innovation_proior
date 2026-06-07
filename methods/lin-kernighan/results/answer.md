# The Lin–Kernighan Algorithm for the Traveling-Salesman Problem

## Problem

Symmetric TSP: given an `n × n` symmetric distance matrix `c(i,j)`, find a tour (Hamiltonian cycle
visiting every city once) of minimum total length. Local-search heuristics improve a tour by
`k`-exchanges — deleting `k` links and reconnecting with `k` others — but fixing `k` in advance forces
a poor tradeoff: small `k` (2-opt, 3-opt) stalls at mediocre local optima, large `k` is `O(n^k)` and
unaffordable, and the right `k` cannot be known before running. Lin–Kernighan removes the fixed-`k`
drawback with a *variable-depth* sequential exchange.

## Key idea

Build the exchange one link at a time and let the depth `k` be decided dynamically. Maintain a city
chain `t_1, t_2, t_3, ...` where `x_i = (t_{2i-1}, t_{2i})` is a tour link to break and
`y_i = (t_{2i}, t_{2i+1})` is a new link to add; consecutive links share endpoints, so the sequence
`x_1, y_1, x_2, y_2, ...` is an adjoining chain. Three rules make it work:

- **Sequential exchange + feasibility (close-up).** Given the previously added `y_{i-1}` reaching
  `t_{2i-1}`, the broken link `x_i` is forced to be the one tour link whose removal leaves the chain
  *closeable*: adding `(t_{2i}, t_1)` yields a single tour. So at every depth there is a valid tour
  one link away, and the search may stop at any `i`.
- **Positive-gain criterion.** With `g_i = c(x_i) - c(y_i)` and running gain `G_i = g_1 + ... + g_i`,
  extend the chain only while `G_i > 0`. This loses no improving move: if a sequence of gains has a
  positive total sum, some cyclic permutation has all partial sums positive, so trying every start
  `t_1` recovers any improving sequential exchange. The proof: with prefix sums `S_j = g_1+...+g_j`
  (`S_0 = 0`, `S_n > 0`), let `k` be the largest index minimizing `S_{k-1}` over `S_0,...,S_{n-1}`;
  reading from `g_k`, a partial sum ending at `j ≥ k` is `S_j - S_{k-1} > 0`, and one wrapping to
  `j < k` is `(S_n - S_{k-1}) + S_j ≥ S_n > 0`.
- **Best close-up wins.** Track `G* = max_i (G_{i-1} + g_i*)` where `g_i* = c(x_i) - c(t_{2i}, t_1)`
  is the gain of closing up at depth `i`, with `k` the achieving depth. Stop when no admissible
  `x_i, y_i` remain or when `G_i ≤ G*`; if `G* > 0`, perform the depth-`k` exchange (new length
  `f(T) - G*`) and restart.

Supporting choices: prefer nearest neighbors for `y_i` (small `|y_i|` ⇒ large `g_i`, tight
branching); keep `X` and `Y` disjoint (no link broken then re-added, or vice versa, within an
iteration); allow the single infeasible alternate `x_2` (only at `i = 2`) to recover some
non-sequential moves; and backtrack only at levels 1 and 2 with ~5 candidates each, since the
improving move is almost always the first candidate (mean choice 1.2 and 1.8). The resulting local
optima are at least 3-opt, reached in time growing about as `n^{2.2}`.

## Algorithm

```
1.  Generate a random tour T.
2.  G* = 0; choose a start city t1; let x1 be a tour link at t1; i = 1.
3.  From the other end t2 of x1, choose y1 = (t2, t3) ∉ T with g1 = |x1| - |y1| > 0.
        If none exists, backtrack (Step 6).
4.  i = i + 1.
    (a) x_i = (t_{2i-1}, t_{2i}) is the unique tour link such that joining t_{2i} to t1 can close up.
    (b) Close-up check: with yi* = (t_{2i}, t1), gi* = |x_i| - |yi*|. If G_{i-1} + gi* > G*,
        set G* = G_{i-1} + gi*, k = i.
    (c) Choose y_i = (t_{2i}, t_{2i+1}) ∉ T, nearest first, with:
          - G_i = G_{i-1} + g_i > 0          (positive-gain criterion)
          - x_i, y_i disjoint from earlier links
          - an x_{i+1} can still be broken   (so close-up stays possible next step)
        If such y_i exists, go to 4; else go to 5.
5.  Stop the chain when no admissible x_i, y_i remain or G_i ≤ G*. If G* > 0, take the depth-k
    exchange: f(T') = f(T) - G*, T = T', go to 2.
6.  If G* = 0, limited backtracking: alternate y2 (increasing length, g1+g2 > 0); then alternate x2
    (infeasible close-up allowed only here); then alternate y1; then alternate x1; then new t1.
7.  When all n choices of t1 are exhausted without profit, T is a local optimum; optionally restart.
```

## Code

A faithful implementation of the Lin–Kernighan step: the `t1..t_{2i}` city chain, the broken/added
edge sets `X`/`Y`, the positive-gain and feasibility (close-up) checks, and backtracking limited to
levels 1 and 2.

```python
from copy import deepcopy


def make_pair(i, j):
    return (i, j) if i < j else (j, i)          # undirected edge as a sorted tuple


class TSP:
    edges = {}                                  # global symmetric cost matrix

    @staticmethod
    def dist(i, j):
        return TSP.edges[i][j]

    @staticmethod
    def path_cost(path):
        cost = TSP.dist(path[-1], path[0])      # close the loop
        for i in range(1, len(path)):
            cost += TSP.dist(path[i - 1], path[i])
        return cost


class Tour:
    """A tour as an ordered city list plus its edge set; `generate` is the close-up oracle."""

    def __init__(self, tour):
        self.tour = list(tour)
        self.size = len(self.tour)
        self.edges = {make_pair(self.tour[i - 1], self.tour[i]) for i in range(self.size)}

    def around(self, node):                     # the two tour-neighbours of `node`
        idx = self.tour.index(node)
        succ = idx + 1 if idx + 1 < self.size else 0
        return (self.tour[idx - 1], self.tour[succ])

    def contains(self, edge):
        return edge in self.edges

    def generate(self, broken, joined):
        """New edges = (tour - broken) + joined; rebuild and check it is ONE Hamiltonian cycle.
        Returns (is_tour, new_tour). This is the feasibility / close-up test."""
        edges = (self.edges - broken) | joined
        if len(edges) < self.size:
            return False, []
        successors, node = {}, 0
        while edges:                            # chain the edges into a successor map
            for i, j in edges:
                if i == node:
                    successors[node] = j; node = j; break
                if j == node:
                    successors[node] = i; node = i; break
            edges.remove((i, j))
        if len(successors) < self.size:
            return False, []
        succ = successors[0]
        new_tour, visited = [0], {0}
        while succ not in visited:              # premature repeat => disjoint subtours
            visited.add(succ); new_tour.append(succ); succ = successors[succ]
        return len(new_tour) == self.size, new_tour


class LinKernighan(TSP):
    def __init__(self, path):
        self.heuristic_path = list(path)
        self.heuristic_cost = TSP.path_cost(self.heuristic_path)

    def optimise(self):
        better = True
        self.solutions = set()
        # candidate neighbour lists (ordered by distance ascending => nearest first)
        self.neighbours = {}
        for i in self.heuristic_path:
            nbrs = [(TSP.dist(i, j), j) for j in self.heuristic_path
                    if j != i and TSP.dist(i, j) > 0]
            self.neighbours[i] = [j for _, j in sorted(nbrs)]
        while better:                           # restart at every improving move
            better = self.improve()
            self.solutions.add(str(self.heuristic_path))
        return self.heuristic_path, self.heuristic_cost

    def closest(self, t2i, tour, gain, broken, joined):
        """Candidate y_i = (t2i, node): keep only positive running gain G_i, not broken, not a tour
        edge; order by how good the next break looks. `gain` already holds G_{i-1} + |x_i|."""
        candidates = {}
        for node in self.neighbours[t2i]:
            yi = make_pair(t2i, node)
            Gi = gain - TSP.dist(t2i, node)             # running gain if we ADD y_i
            if Gi <= 0 or yi in broken or tour.contains(yi):
                continue                                # POSITIVE-GAIN CRITERION + disjointness
            for succ in tour.around(node):              # the x_{i+1} we could break next
                xi = make_pair(node, succ)
                if xi not in broken and xi not in joined:
                    diff = TSP.dist(node, succ) - TSP.dist(t2i, node)
                    if node not in candidates or diff > candidates[node][0]:
                        candidates[node] = [diff, Gi]
        return sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)

    def improve(self):
        tour = Tour(self.heuristic_path)
        for t1 in self.heuristic_path:                  # try every start city
            around = tour.around(t1)
            for t2 in around:                           # both choices of x1 (alternate x1)
                broken = {make_pair(t1, t2)}
                gain = TSP.dist(t1, t2)                 # |x1|
                close = self.closest(t2, tour, gain, broken, set())   # y1 with g1 > 0
                tries = 5                               # limited level-1 backtracking (~5)
                for t3, (_, Gi) in close:
                    if t3 in around:                    # y1 must not already be a tour link at t2
                        continue
                    joined = {make_pair(t2, t3)}
                    if self.choose_x(tour, t1, t3, Gi, broken, joined):
                        return True                     # improvement -> restart
                    tries -= 1
                    if tries == 0:
                        break
        return False

    def choose_x(self, tour, t1, last, gain, broken, joined):
        """Choose x_i to break from `last`; try to close up, else extend via choose_y."""
        if len(broken) == 4:                            # deep: commit to the longer x_i
            pred, succ = tour.around(last)
            around = [pred] if TSP.dist(pred, last) > TSP.dist(succ, last) else [succ]
        else:
            around = tour.around(last)                  # both tour links are candidate x_i
        for t2i in around:
            xi = make_pair(last, t2i)
            Gi = gain + TSP.dist(last, t2i)             # add |x_i| to the running gain
            if xi in joined or xi in broken:            # keep X and Y disjoint
                continue
            added = deepcopy(joined); added.add(make_pair(t2i, t1))    # close-up edge (t2i, t1)
            removed = deepcopy(broken); removed.add(xi)
            relink = Gi - TSP.dist(t2i, t1)             # improvement G* if we close up here
            is_tour, new_tour = tour.generate(removed, added)
            if not is_tour and len(added) > 2:          # infeasible close-up allowed only at i = 2
                continue
            if str(new_tour) in self.solutions:         # already-seen tour -> avoid cycling
                return False
            if is_tour and relink > 0:                  # strictly better tour: take it
                self.heuristic_path = new_tour
                self.heuristic_cost -= relink
                return True
            choice = self.choose_y(tour, t1, t2i, Gi, removed, joined)  # else extend the chain
            if len(broken) == 2:                        # full backtracking at level 2
                if choice:
                    return True
            else:
                return choice                           # single shot for i > 2
        return False

    def choose_y(self, tour, t1, t2i, gain, broken, joined):
        """Choose y_i from the close-up-ordered candidates: 5 at level 2, nearest only deeper."""
        ordered = self.closest(t2i, tour, gain, broken, joined)
        top = 5 if len(broken) == 2 else 1
        for node, (_, Gi) in ordered:
            added = deepcopy(joined); added.add(make_pair(t2i, node))  # y_i = (t2i, node)
            if self.choose_x(tour, t1, node, Gi, broken, added):
                return True
            top -= 1
            if top == 0:
                return False
        return False


if __name__ == "__main__":
    import random
    random.seed(0)
    n = 12
    pts = [(random.random(), random.random()) for _ in range(n)]
    TSP.edges = [[((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
                  for j in range(n)] for i in range(n)]
    lk = LinKernighan(list(range(n)))
    path, cost = lk.optimise()
    print("tour:", path, "length:", round(cost, 4))
```
