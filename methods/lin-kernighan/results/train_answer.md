We are given the symmetric traveling-salesman problem: an $n \times n$ symmetric matrix of distances $c(i,j)$, and we want the shortest tour — a single Hamiltonian cycle through all $n$ cities. With $(n-1)!/2$ distinct tours, enumeration is hopeless, and the exact methods of the day (Held–Karp, then branch and bound when an instance falls outside their tractable class) choke past a few dozen cities — the largest reported exact solution is 64 cities. So the realistic goal is not a certificate of optimality but reliably good tours, fast, with running time that grows gently with $n$. That puts us squarely in iterative improvement: start from a random tour $T$, find a transformation to a shorter tour $T'$, jump there, repeat until nothing improves (a local optimum), then restart from a fresh random tour and keep the best. The entire quality of this scheme lives in one place — the transformation in the middle. A weak step leaves many bad local optima; a strong step makes the local optima few and good, so a random start often lands on the global one.

The standard step is the $k$-opt interchange: delete $k$ tour links, reconnect the resulting paths with $k$ new links (reversing subpaths as needed) so the result is again a tour, and keep it if it is shorter. Croes's 2-opt fixes $k = 2$ — pull two links and reverse the one subsegment that reconnects them; cheap and always feasible, but a shallow neighborhood riddled with mediocre local optima. Lin's 3-opt fixes $k = 3$ — markedly better tours, but the neighborhood scan costs on the order of $n^3$, and one can keep climbing to $k = 4, 5$ for still-better quality at still-worse cost. The decisive frustration is that $k$ must be chosen *in advance*. The work grows like $O(n^k)$, there is no useful bound on how many improving exchanges a tour admits, and there is no way to know the right $k$ for a given instance before running: too small and we stall at a poor optimum, too large and every move is ruinous. The right depth surely varies from instance to instance and even from move to move. Fixing $k$ up front is exactly backwards — the data should tell us how deep to go.

I propose the Lin–Kernighan algorithm: a *variable-depth sequential exchange* in which $k$ is not an input but an outcome. Rather than commit to a $k$ and examine every $k$-subset, I build the set $X = \{x_1, x_2, \dots\}$ of links to remove and the set $Y = \{y_1, y_2, \dots\}$ of links to add one element at a time, choosing the most profitable pair available at each step, and let a stopping rule decide when to quit. The bookkeeping that makes this tractable is additivity of gains. Define the gain of swapping $x_i$ out for $y_i$ in as $g_i = |x_i| - |y_i|$ — length removed minus length added. If a whole group of swaps simply adds up, then for any reachable $T'$ we have $f(T) - f(T') = \sum_i g_i$, and $T'$ is shorter exactly when that sum is positive. This lets me reason about *partial* progress, not just finished exchanges.

To keep the construction one step away from a valid tour at every moment, I tie the broken and added links to a city chain $t_1, t_2, t_3, \dots$. Pick a start city $t_1$; let $x_1 = (t_1, t_2)$ be one of its two tour links — the first break. From $t_2$ add a new link $y_1 = (t_2, t_3)$ to some other city $t_3$. In general
$$x_i = (t_{2i-1}, t_{2i}) \in T, \qquad y_i = (t_{2i}, t_{2i+1}) \notin T,$$
so consecutive links share endpoints and the sequence $x_1, y_1, x_2, y_2, \dots$ zig-zags through the cities as an adjoining chain. This sequential structure is what lets me close up to a tour at any moment. After adding $y_{i-1}$ to reach $t_{2i-1}$ I am holding a Hamiltonian path; of the two tour links at $t_{2i-1}$, exactly one choice of $x_i$ leaves the chain *closeable* — adding $(t_{2i}, t_1)$ yields a single closed tour rather than two disjoint loops. So $x_i$ is forced, uniquely determined by the demand that I can always snap shut. At every depth $i$ there is a legal tour one link away; I never assemble a pile of swaps only to find they cannot be made into a tour.

This immediately gives a stop test. Before committing the open-ended $y_i$, tentatively close up with $y_i^\ast = (t_{2i}, t_1)$; the gain of closing here is $g_i^\ast = |x_i| - |y_i^\ast|$, and because feasibility held this really is a tour, improving on $T$ by $G_{i-1} + g_i^\ast$, where $G_{i-1} = g_1 + \dots + g_{i-1}$ is the running gain of the open chain. So I keep a running best
$$G^\ast = \max_i \, \bigl(G_{i-1} + g_i^\ast\bigr),$$
recording the depth $k$ that achieves it; $G^\ast$ starts at 0 and only rises, and at the end, if $G^\ast > 0$, I perform the depth-$k$ exchange, yielding $f(T') = f(T) - G^\ast$.

What keeps this from exploding is the choice of $y_i$ and the rule for when to stop extending. To make $g_i = |x_i| - |y_i|$ large I want $|y_i|$ small, so I try the nearest neighbors of $t_{2i}$ first — short new links are where the gain is — which bounds the branching to a few candidates. The stopping rule is the heart of the method. The naive answer, "stop as soon as some $g_i$ goes negative," is too greedy: a step that loses a little can set up a later step that wins a lot, the classic escape past a local minimum. The opposite extreme, "keep going while the total could still come out positive," is almost no constraint and lets the search wander forever. The right rule is the *positive-gain criterion*: extend the chain only while the running sum $G_i = g_1 + \dots + g_i$ stays strictly positive, i.e. insist every partial sum is positive, and abandon the chain the instant it hits zero.

It looks far too restrictive to demand positivity at every prefix — surely that throws away good moves whose gain dips negative in the middle. It does not, because of a fact about gains read cyclically: if a sequence $g_1, \dots, g_n$ has positive total sum, then some cyclic permutation of it has all partial sums positive. The proof is the load-bearing step. Write prefix sums $S_j = g_1 + \dots + g_j$ with $S_0 = 0$ and $S_n > 0$, and let $k$ be the *largest* index for which $S_{k-1}$ is the minimum among $S_0, S_1, \dots, S_{n-1}$. Read the sequence starting at $g_k$. A partial sum ending at index $j$ with $k \le j \le n$ equals $g_k + \dots + g_j = S_j - S_{k-1}$; since $S_{k-1}$ is the minimum and $k$ is the *largest* index attaining it, $S_j > S_{k-1}$ for every $j \ge k$, so this is strictly positive. A partial sum that wraps around, ending at $j$ with $1 \le j < k$, equals $(S_n - S_{k-1}) + S_j$, and since $S_j \ge S_{k-1}$ this is $\ge S_n > 0$. Every partial sum is positive. The consequence is exactly what I need: any improving sequential exchange — any chain with positive total gain — can be re-read, starting from the right city, so that every running partial sum stays positive. So if I enforce $G_i > 0$ at every step and am willing to try every starting city $t_1$, I miss no improving move; I only require starting the chain in the right place. In return, the criterion lets me prune away an enormous mass of fruitless deep searches, which is the only reason variable-depth search is affordable at all.

A few supporting choices make it honest and effective. The sets $X$ and $Y$ must stay disjoint within an iteration — a broken link is not re-added and an added link is not re-broken — which stops the chain from undoing its own work, simplifies the gain accounting, keeps running time down, and gives a clean finite stop. Each $y_i$ must also leave room for some $x_{i+1}$ to be broken next, so feasibility survives one more step. Purely sequential chains capture most improving moves but not all — the smallest non-sequential exchange touches four links and admits no adjoining closing order — so at level $i = 2$ only I also allow the *alternate* $x_2$, the infeasible-at-that-moment choice, to recover some of that lost power cheaply; I never permit such a feasibility violation deeper. Finally, backtracking is limited to levels 1 and 2 with about five candidates each. Full backtracking everywhere would find the optimum but is exhaustive search in disguise; empirically the improving move, when it exists, is almost always the very first candidate tried (mean choice number about 1.2 at level 1 and 1.8 at level 2), so capping to five contenders loses essentially no optima while nearly halving the running time. The resulting local optima are at least 3-opt — the first levels reproduce a 2- or 3-exchange — reached in time growing on average about as $n^{2.2}$, with roughly $n/4$ to $n/3$ improvements per local optimum: early moves are deep, large-$k$ chains, and as the tour sharpens they shrink to small $k$ around 2 to 7.

Concretely, a tour is an ordered city list plus its edge set; `generate(broken, joined)` forms $(\text{tour edges} - \text{broken}) + \text{joined}$, walks the successor map, and reports whether the result is one Hamiltonian cycle — the feasibility / close-up oracle. `closest` is where the nearest-neighbor preference and the positive-gain gate live: from the chain end $t_{2i}$ with running gain $G_{i-1} + |x_i|$ already accumulated, it scores each candidate $y_i = (t_{2i}, \text{node})$ by $G_i = \text{gain} - |y_i|$ and keeps only those with $G_i > 0$ that are neither broken nor a tour edge. `improve` runs level 1 over every start city and both choices of $x_1$, capped at five $y_1$ candidates; `choose_x` picks the forced (or, deep in the chain, the longer) $x_i$, tests the close-up, takes the move when `relink` $> 0$, and otherwise recurses; `choose_y` extends with five candidates at level 2 and only the top-ranked one deeper.

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
                    if t3 in around:                    # t3 can't be a tour-neighbour of t1
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
        """Choose y_i from the close-up-ordered candidates: 5 at level 2, top-ranked only deeper."""
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
