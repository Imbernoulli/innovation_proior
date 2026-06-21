We are given an undirected graph on $2n$ nodes with symmetric edge costs $c(i,j) \ge 0$, and we must split the nodes into two sets $A$ and $B$ of exactly $n$ each so as to minimize the external cost — the total weight of edges that cross between the two sides, $T = \sum_{a\in A,\, b\in B} c(a,b)$. The balance is a hard constraint, not a soft preference: in the motivating application the two sides are physical cards or memory pages with a fixed capacity, so an unbalanced split is not a cheaper answer to the same problem, it is an answer to a different one. That constraint is the entire difficulty. Drop it and ask only for the cheapest cut separating the graph, and we are in friendly territory: Ford and Fulkerson's max-flow machinery solves it, treating costs as capacities so that maximum flow between two nodes equals the minimum cut separating them. But that minimum cut comes out at whatever sizes it likes — typically shearing off a tiny weakly attached lump — and there is no hook in the flow algorithm to demand equal sides, and no clean way to add one; post-processing an unbalanced cut into a balanced one throws away the very optimality the flow gave us. (The flow is not wasted: minimized over the choice of separated nodes it gives the global unconstrained min-cut, a valid lower bound on any balanced cut, since constraining can only raise the optimum.) Exhaustive search is hopeless too — there are $\tfrac{1}{2}\binom{2n}{n}$ balanced splits, about $7\cdot 10^{10}$ already at $n=20$, and far more for more than two blocks — so we abandon certifying optimality and aim instead for a heuristic that reliably reaches good cuts fast, with running time near $n^2$, judged statistically by how often a run lands on the known optimum.

That places us in iterative improvement: start from a balanced partition, find a transformation to a cheaper one, jump there, repeat until nothing improves, then restart and keep the best. The whole quality of such a scheme lives in the transformation. The smallest balance-preserving move is to exchange a pair: take $a\in A$ and $b\in B$ and swap their sides, leaving the sizes untouched. A partition that no single such exchange improves is "1-opt," but single exchanges stall — on the test matrices 1-opting finds the apparent optimum only about a tenth of the time and otherwise freezes a couple of units short, at a partition no single swap can improve even though the improving move exists, because reaching it requires swapping several pairs at once where some individual swaps make the cut momentarily worse before the group pays off. A one-swap-at-a-time greedy can never take that path, and fixing $\lambda$ pairs per move in advance is the wrong knob: small $\lambda$ is too weak, large $\lambda$ makes each move expensive, and there is no way to know the right $\lambda$ for an instance before running it. What we want is to escape the local minimum without committing to a depth in advance.

I propose the Kernighan–Lin algorithm: a variable-depth exchange procedure in which we greedily pre-select an entire sequence of pairwise exchanges and then apply only its best prefix. Everything is built on one piece of bookkeeping — the exact cost change of a single exchange. For a node $a\in A$, split its edges into the internal cost $I_a = \sum_{x\in A} c(a,x)$ (edges to its own side, holding it in place) and the external cost $E_a = \sum_{y\in B} c(a,y)$ (edges crossing the cut, pulling it across), and likewise $I_b, E_b$ for $b\in B$. Peel the cut cost into the part untouched by the swap, $z$, plus the rest. Before the swap, every edge from $a$ to $B$ and from $b$ to $A$ is cut, so $T = z + E_a + E_b - c(a,b)$, where the $-c(a,b)$ undoes a double count: the single $a$–$b$ edge sits inside both $E_a$ and $E_b$. After the swap $a$ is in $B$ and $b$ in $A$, so the edges that now cross are $a$'s edges to the old $A$ side ($I_a$) and $b$'s to the old $B$ side ($I_b$), giving $T' = z + I_a + I_b + c(a,b)$. The reduction is therefore

$$g = T - T' = (E_a - I_a) + (E_b - I_b) - 2\,c(a,b).$$

The factor of two is the punchline of the double count: the $a$–$b$ edge swings from $-c(a,b)$ to $+c(a,b)$, a net $2c(a,b)$ working against the swap, which is right — that edge stays cut whether or not we swap, so swapping can never relieve it and is charged twice for it. The quantity $E-I$ recurs, so define for every node its D-value

$$D_s = E_s - I_s,$$

the "how badly does this node want to be on the other side" number, and the gain becomes simply $g = D_a + D_b - 2\,c(a,b)$.

Now the variable-depth move. Suppose we perform a sequence of exchanges and want each one's gain to add honestly onto a running total; the requirement is that each gain be computed against the partition as it stands after the earlier swaps. So after tentatively swapping a pair we update everyone else's D-value. Having set aside $(a_i,b_i)$ — conceptually moving $a_i$ to $B$ and $b_i$ to $A$ — take a surviving $x$ still on the $A$ side. Its edge $(x,a_i)$ was internal (counted in $I_x$) and becomes external, moving weight $c(x,a_i)$ from the $I$ column to the $E$ column and thus raising $D_x$ by $2c(x,a_i)$; its edge $(x,b_i)$ was external and becomes internal, lowering $D_x$ by $2c(x,b_i)$. By the mirror argument for a surviving $y$ on the $B$ side,

$$D'_x = D_x + 2\,c(x,a_i) - 2\,c(x,b_i),\quad x\in A-\{a_i\};\qquad D'_y = D_y + 2\,c(y,b_i) - 2\,c(y,a_i),\quad y\in B-\{b_i\}.$$

The pass therefore runs as follows: compute all D-values; then $n$ times, pick the unlocked pair maximizing $g = D_a + D_b - 2c(a,b)$, lock it by removing both nodes from further contention this round, record its gain, and update the survivors' D-values by the rule above. Locking is not mere tidiness — because each chosen node is removed at once, every node moves at most once across the whole sequence, which is precisely what makes the gains additive and the bookkeeping a clean telescope (the swap of $(a_i,b_i)$ is computed against a partition where the earlier pairs have moved and the later ones have not, and since nothing is touched twice, summing the gains exactly reconstructs the cost change of any prefix). It also guarantees termination: there are only $n$ pairs to use up.

Why a prefix and not the whole thing? Exchanging the entire sequence merely relabels $A$ and $B$ into each other's complements, returning the same partition structure, so $\sum_{i=1}^n g_i = 0$. That zero is informative: unless every gain is zero, the positive gains must be offset by negative ones somewhere in the sequence. So we never swap the whole sequence — we swap a prefix. Form the partial sums $G_k = g_1 + \cdots + g_k$; exchanging the first $k$ pairs, $\{a_1,\dots,a_k\}$ for $\{b_1,\dots,b_k\}$, produces a balanced partition cheaper than the start by exactly $G_k$, so we choose the $k$ that maximizes $G_k$. The crucial design choice is that we do not stop the sequence at the first negative $g_i$ — a "halt when a swap stops paying" rule is exactly the 1-opt trap, refusing a step that loses a little even when it sets up a later step that wins a lot. By building the whole sequence first and only then picking the best prefix, we let the running sum dip into the red and climb back out; if the gains run $g_1 > 0$, $g_2 < 0$, $g_3 \gg 0$, the prefix through step 3 is taken and the temporary worsening is paid as the price of the deep later improvement. This is what breaks out of the local minimum, and the accepted depth $k$ is never fixed — it is whatever the data hands over, small near a good partition and large when there is much to fix. If $G_k > 0$ we actually perform that prefix exchange, recompute D-values from scratch, and run another pass; if $G_k \le 0$ no improving prefix exists and the partition is taken as locally optimal (or we restart from a fresh random partition and keep the best).

On cost: computing the initial D-values is an $n^2$ job, and the updates across a pass sum to $(n-1)+(n-2)+\cdots+1 \propto n^2$. The dominant naive cost is the selection — scanning all remaining $A\times B$ pairs $n$ times is $n^3$ per pass. We avoid that by sorting each side's D-values descending and scanning pairs in order of $D_a + D_b$: since $c(a,b)\ge 0$ we have $g \le D_a + D_b$, so the moment a pair's $D_a + D_b$ no longer exceeds the best gain seen this round we can stop, as no later pair can beat it. That brings a pass to near $n^2\log n$, and a small bounded number of passes makes the whole procedure near $n^2$ — gentle enough to afford many random restarts. Unequal target sizes are handled by padding with zero-cost dummy elements that freely absorb the slack and are discarded at the end (or by limiting the pairs exchanged per pass to the smaller side); unequal node weights by blowing a weight-$k$ node into $k$ unit nodes bound by very high-cost edges so the cluster is never cut; and $k$-way partitions by cycling the two-way exchange over all $\binom{k}{2}$ pairs of subsets until the partition is pairwise optimal.

```python
def kl_pass(cost, A, B):
    """One Kernighan-Lin pass on a balanced bipartition (A, B).
    Returns the improved (A, B) and the cost reduction G (0.0 if already locally optimal)."""
    n = len(cost)
    A, B = set(A), set(B)

    def compute_D(A, B):
        # D_s = E_s - I_s : external (edges crossing the cut) minus internal (edges to own side)
        D = {}
        for s in range(n):
            own, other = (A, B) if s in A else (B, A)
            I = sum(cost[s][x] for x in own if x != s)
            E = sum(cost[s][y] for y in other)
            D[s] = E - I
        return D

    D = compute_D(A, B)
    free_A, free_B = set(A), set(B)
    av, bv, gv = [], [], []

    for _ in range(len(A)):
        # select the unlocked pair maximizing the gain g = D[a] + D[b] - 2 c(a,b)
        best, ba, bb = None, None, None
        for a in free_A:
            for b in free_B:
                g = D[a] + D[b] - 2 * cost[a][b]
                if best is None or g > best:
                    best, ba, bb = g, a, b
        av.append(ba); bv.append(bb); gv.append(best)      # record and LOCK the pair
        free_A.discard(ba); free_B.discard(bb)

        # update survivors: ba moved A->B, bb moved B->A
        for x in free_A:
            D[x] += 2 * cost[x][ba] - 2 * cost[x][bb]
        for y in free_B:
            D[y] += 2 * cost[y][bb] - 2 * cost[y][ba]

    # best prefix: maximize the cumulative gain G_k = g_1 + ... + g_k (allowed to dip and recover)
    G, best_G, k = 0.0, 0.0, 0
    for i, g in enumerate(gv, start=1):
        G += g
        if G > best_G:
            best_G, k = G, i

    if best_G > 0:                                         # apply the improving prefix
        for i in range(k):
            A.discard(av[i]); A.add(bv[i])
            B.discard(bv[i]); B.add(av[i])
    return A, B, best_G


def kernighan_lin(cost, A, B):
    """Run passes until one certifies a local optimum (G <= 0)."""
    A, B = set(A), set(B)
    while True:
        A, B, G = kl_pass(cost, A, B)
        if G <= 0:
            return A, B


def external_cost(cost, A, B):
    return sum(cost[a][b] for a in A for b in B)


if __name__ == "__main__":
    import random
    random.seed(0)
    n2 = 12                                                # 2n = 12 nodes -> two sets of 6
    cost = [[0] * n2 for _ in range(n2)]
    for i in range(n2):
        for j in range(i + 1, n2):                         # random symmetric nonnegative costs
            w = random.randint(0, 5)
            cost[i][j] = cost[j][i] = w
    A0, B0 = set(range(0, n2 // 2)), set(range(n2 // 2, n2))
    print("start cut:", external_cost(cost, A0, B0))
    A, B = kernighan_lin(cost, A0, B0)
    print("A =", sorted(A), "B =", sorted(B), "cut:", external_cost(cost, A, B))
```
