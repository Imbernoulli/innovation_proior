# Context

## Research question

Given $n$ cities with symmetric distances $c$ obeying the triangle inequality, find a minimum-cost tour visiting every city. Equivalently, find a minimum-cost connected Eulerian multigraph on the vertices (shortcut it to a Hamiltonian cycle using the triangle inequality). The problem is NP-hard, and in fact NP-hard to approximate within $123/122$, so the goal is the best possible polynomial-time approximation ratio.

For more than four decades, one algorithm has held the record: a tour of cost at most $\tfrac32$ times optimal. The central question is whether $\tfrac32$ is a real barrier or merely the limit of one particular idea — whether some polynomial-time algorithm can guarantee a ratio strictly below $\tfrac32$ for every metric instance. A solution would have to find structure that lets the matching step cost strictly less, in the worst case, than the half-of-optimal that the classical analysis charges it.

## Background

**The tree-plus-matching paradigm.** A spanning tree connects all the cities; the only obstruction to being Eulerian is that the odd-degree vertices must be paired up. The minimum spanning tree $T$ has cost at most that of the optimal tour (delete an edge from the tour to get a spanning path, a tree). Let $O$ be the odd-degree vertices of $T$; $|O|$ is even. Adding any perfect matching $M$ on $O$ makes every degree even, so $T\cup M$ is Eulerian. Shortcutting $T\cup M$ gives a tour of cost $\le c(T)+c(M)$.

**The matching bound and why it gives exactly $\tfrac32$.** The minimum perfect matching on $O$ costs at most $\tfrac12$ of optimal: the optimal tour, shortcut onto $O$, is a cycle through the vertices of $O$ whose cost is at most that of the full tour; this cycle splits into two perfect matchings on $O$, the cheaper of which costs at most half the tour. So $c(M)\le \tfrac12\mathrm{OPT}$, and $c(T)+c(M)\le \mathrm{OPT}+\tfrac12\mathrm{OPT}=\tfrac32\mathrm{OPT}$. This is tight: there are unit-weight gadgets in which the minimum spanning tree has *every* vertex odd and the cheapest matching on those odd vertices genuinely costs $(\tfrac12-o(1))\mathrm{OPT}$, so the analysis is an equality, not slack. The $\tfrac32$ is a real wall for any method that picks one tree and charges its matching against a single optimal tour.

**The linear-programming relaxation.** Replacing the integrality of a tour by fractional edge weights gives the subtour-elimination LP (the Held–Karp relaxation): minimize $\sum_e c_e x_e$ subject to $x(\delta(v))=2$ for every vertex (degree two), $x(\delta(S))\ge 2$ for every proper nonempty vertex set $S$ (every cut crossed at least twice, i.e. connectivity), and $x\ge 0$. Its optimum is a lower bound on the cost of the best tour. A useful normalization: pick a vertex, split it into $u_0,v_0$ joined by an edge $e_0$ with weight $1$ and cost $0$; then an optimal LP point $x^0$ has $x_{e_0}=1$, and $x$ restricted to the other edges $E$ lies in the spanning-tree polytope, so it is a convex combination of spanning trees.

**Distributions over spanning trees.** Given a point $x$ in the spanning-tree polytope, there are many distributions over spanning trees whose edge marginals equal $x$. A $\lambda$-uniform distribution assigns a tree $T$ probability proportional to $\prod_{e\in T}\lambda_e$ for nonnegative edge weights $\lambda$; among all marginal-respecting distributions, the one of this product form is the maximum-entropy distribution, and $\lambda$ achieving marginals $(1\pm 2^{-n})x$ is computable in polynomial time (multiplicative weights, interior point, or ellipsoid). Such distributions are *strongly Rayleigh*: their generating polynomial is real stable (nonvanishing when every variable has positive imaginary part). Strong Rayleigh-ness, studied by Borcea, Brändén and Liggett (2009), links probability to the geometry of polynomials and yields negative dependence: negative association, stochastic dominance, log-concave rank sequences, and closure under conditioning, projection, truncation and products. For a spanning-tree distribution, conditioning on a vertex set $S$ being internally a tree splits the sample into two *independent* trees, one inside $S$ and one on the contraction, each with the inherited marginals. A consequence: the degree of a vertex is distributed as a sum of independent Bernoullis with mean $2$, so a vertex is even with probability at least $\tfrac12(1-e^{-2})\approx0.43$.

**Antecedent: max-entropy rounding for special cases.** Asadpour, Goemans, Mądry, Oveis Gharan and Saberi (2010) introduced max-entropy spanning-tree rounding for asymmetric TSP; Oveis Gharan, Saberi and Singh (2011) used it for symmetric *graphic* TSP to get a ratio $\tfrac32-\varepsilon_0$. Their idea: sample the tree from the max-entropy distribution and observe that a constant fraction of edges lie exclusively on *even* near-minimum cuts, on which the O-join slack can be pushed below $x_e/2$. This works for $0/1$-cost graphic instances but stalls on general metrics, where a single high-cost edge can sit on many cuts and the chance that *all* of them are even at once is tiny.

**Antecedent: the half-integral case.** Karlin, Klein and Oveis Gharan (2019) handled metric instances whose LP optimum is half-integral (every $x_e\in\{0,\tfrac12,1\}$), getting $\tfrac32-\varepsilon$ there. Half-integral solutions are conjectured to be the hardest to round; the half-integral analysis worked out the degree-cut case and the local "reduce when nearby cuts are even, compensate when they are odd" accounting that the general analysis would have to globalize.

**Structure of near-minimum cuts.** For minimum cuts (value exactly $2$), the cactus representation (Dinits–Karzanov–Lomonosov) arranges crossing min-cuts of a connected component around a cycle of atoms $a_0,\dots,a_{m-1}$ with $x(E(a_i,a_{i+1}))=1$. Benczúr and Goemans extended this to near-minimum cuts ($x(\delta(S))\le 2+\eta$ for small $\eta$) via *polygon representations*, placing atoms on the sides of a polygon so that each cut is a diagonal.

**The O-join polytope.** Edmonds and Johnson (1973): for a set $O$ of even size, the minimum-cost $O$-join equals the optimum of $\min c(y)$ subject to $y(\delta(S))\ge 1$ for every $S$ with $|S\cap O|$ odd, and $y\ge 0$. Setting $y_e=x_e/2$ is feasible for any $O$ (since $x(\delta(S))\ge 2$) and certifies the $\tfrac12\mathrm{OPT}$ matching bound. Crucially, a cut's constraint only binds when $|S\cap O|$ is odd — equivalently when an odd number of tree edges cross $S$.

## Baselines

- **Christofides–Serdyukov (1976/78).** MST $T$, then minimum perfect matching $M$ on the odd-degree vertices of $T$, shortcut $T\cup M$. Ratio exactly $\tfrac32$; the matching bound $c(M)\le\tfrac12\mathrm{OPT}$ is tight on the all-odd unit-weight gadget. Gap it leaves: a single deterministic tree can be forced to have an expensive matching.
- **Best-in-the-middle LP bound (Wolsey; Shmoys–Williamson).** $y=x/2$ is a feasible O-join for any odd set, recovering the $\tfrac32$ ratio polyhedrally against the Held–Karp lower bound. Gap: charges every cut at $x_e/2$ with no use of which cuts are actually odd in a chosen tree.
- **Max-entropy rounding for graphic TSP (Oveis Gharan–Saberi–Singh 2011).** Sample the tree from the max-entropy distribution; reduce slack on edges that lie only on even near-min cuts. Ratio $\tfrac32-\varepsilon_0$ for graphic TSP. Gap: relies on edges being cheap and on cuts being even simultaneously, which fails for general high-cost edges on many cuts.
- **Half-integral metric TSP (Karlin–Klein–Oveis Gharan 2019).** $\tfrac32-\varepsilon$ when the LP optimum is half-integral. Gap: restricted to half-integral $x$; does not handle general fractional LP solutions or the full near-min-cut hierarchy.

## Evaluation settings

The yardstick is the worst-case approximation ratio over all metric instances: the (expected, for a randomized algorithm) cost of the output tour divided by the cost of the optimal tour, with the Held–Karp LP optimum as the computable lower bound on the optimal tour. Hardness sets the floor at $123/122$ and the incumbent ceiling is $\tfrac32$. Instances are weighted graphs with metric completion; the relevant special cases (graphic, half-integral, Euclidean) are the natural sub-yardsticks. A separate quantity of interest is the integrality gap of the Held–Karp polytope (worst-case ratio of optimal tour to LP optimum), known to be at least $\tfrac43$ and at most $\tfrac32$.

## Code framework

The primitives that already exist: an LP solver for the subtour relaxation (with a min-cut separation oracle), a routine to compute $\lambda$ achieving prescribed spanning-tree marginals, a sampler for $\lambda$-uniform spanning trees, a minimum-cost perfect-matching / minimum O-join solver, and Eulerian shortcutting. The open slot is *which tree to hand to the matching step* and *how to certify the matching is cheap*.

```python
def held_karp_lp(graph, cost):
    """Solve the subtour-elimination LP; return marginals x in the spanning-tree polytope
    (after splitting a vertex to expose the weight-1 cost-0 edge e0)."""
    ...

def lambda_for_marginals(graph, x):
    """Find lambda: E -> R>=0 so the lambda-uniform spanning-tree distribution has
    marginals ~ x (multiplicative weights / interior point)."""
    ...

def sample_spanning_tree(graph, lam):
    """Sample one spanning tree from the lambda-uniform distribution."""
    ...

def min_perfect_matching(odd_vertices, cost):
    """Minimum-cost perfect matching on the given vertex set (= minimum O-join)."""
    ...

def eulerian_shortcut(tree, matching):
    """Combine into an Eulerian multigraph and shortcut to a Hamiltonian tour."""
    ...

def choose_tree(graph, cost, x, lam):
    # TODO: which spanning tree do we feed to the matching step, and what property
    #       of that choice makes the odd-vertex matching provably cheaper than OPT/2?
    pass

def tsp_tour(graph, cost):
    x   = held_karp_lp(graph, cost)
    lam = lambda_for_marginals(graph, x)
    T   = choose_tree(graph, cost, x, lam)
    O   = odd_degree_vertices(T)
    M   = min_perfect_matching(O, cost)
    return eulerian_shortcut(T, M)
```
