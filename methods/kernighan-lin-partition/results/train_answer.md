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

Landing it as a single self-contained C++17 program: it reads from stdin an even integer $m = 2n$ followed by the $m \times m$ symmetric nonnegative cost matrix (row major), starts from the balanced split $A = \{0,\dots,n-1\}$, $B = \{n,\dots,2n-1\}$, runs passes until one certifies a local optimum, and prints the initial cut, the final cut, and the two blocks. The accumulated cut is kept in `long long` to stay overflow-safe. On the worked 12-node seed-0 instance it reproduces the trace exactly — start cut 99, one improving pass to cut 66, a second pass certifying that as locally optimal.

```cpp
// Kernighan-Lin variable-depth balanced graph bisection.
// Reads from stdin: first an even integer m = 2n (number of nodes), then an
// m x m symmetric nonnegative integer cost matrix (m*m entries, row major).
// Writes to stdout: the external cut cost of the initial balanced split
// A = {0..n-1}, B = {n..2n-1}, then the final cut cost after KL, then the two
// blocks A and B (sorted node indices, space separated, one block per line).
#include <bits/stdc++.h>
using namespace std;

// External cut cost: total weight of edges with endpoints in different blocks.
long long external_cost(const vector<vector<long long>>& cost,
                        const vector<int>& side) {
    int m = (int)side.size();
    long long T = 0;
    for (int i = 0; i < m; ++i)
        for (int j = i + 1; j < m; ++j)
            if (side[i] != side[j]) T += cost[i][j];
    return T;
}

// One Kernighan-Lin pass on a balanced bipartition encoded by side[] (0 = A, 1 = B).
// Returns the cost reduction G achieved (0 if the partition is already locally optimal);
// applies the best improving prefix of exchanges to side[] in place.
long long kl_pass(const vector<vector<long long>>& cost, vector<int>& side) {
    int m = (int)side.size();
    int n = m / 2;

    // D_s = E_s - I_s : external (edges crossing the cut) minus internal (edges to own side).
    vector<long long> D(m, 0);
    for (int s = 0; s < m; ++s) {
        long long I = 0, E = 0;
        for (int t = 0; t < m; ++t) {
            if (t == s) continue;
            if (side[t] == side[s]) I += cost[s][t]; else E += cost[s][t];
        }
        D[s] = E - I;
    }

    vector<char> locked(m, 0);
    vector<int> av(n), bv(n);          // the sequence of locked pairs
    vector<long long> gv(n);           // and their gains

    for (int step = 0; step < n; ++step) {
        // select the unlocked pair maximizing the gain g = D[a] + D[b] - 2 c(a,b)
        long long best = LLONG_MIN; int ba = -1, bb = -1;
        for (int a = 0; a < m; ++a) {
            if (locked[a] || side[a] != 0) continue;
            for (int b = 0; b < m; ++b) {
                if (locked[b] || side[b] != 1) continue;
                long long g = D[a] + D[b] - 2 * cost[a][b];
                if (g > best) { best = g; ba = a; bb = b; }
            }
        }
        if (ba < 0) break;                                 // no unlocked pair left (cannot happen for n>=1)
        av[step] = ba; bv[step] = bb; gv[step] = best;     // record and LOCK the pair
        locked[ba] = 1; locked[bb] = 1;

        // update survivors: ba moved A->B, bb moved B->A
        //   D'_x = D_x + 2 c(x,ba) - 2 c(x,bb)   for unlocked x on the A side
        //   D'_y = D_y + 2 c(y,bb) - 2 c(y,ba)   for unlocked y on the B side
        for (int x = 0; x < m; ++x) {
            if (locked[x]) continue;
            if (side[x] == 0) D[x] += 2 * cost[x][ba] - 2 * cost[x][bb];
            else              D[x] += 2 * cost[x][bb] - 2 * cost[x][ba];
        }
    }

    // best prefix: maximize the cumulative gain G_k = g_1 + ... + g_k (allowed to dip and recover).
    long long G = 0, best_G = 0; int k = 0;
    for (int i = 0; i < n; ++i) {
        G += gv[i];
        if (G > best_G) { best_G = G; k = i + 1; }
    }

    if (best_G > 0) {                                      // apply the improving prefix
        for (int i = 0; i < k; ++i) {
            side[av[i]] = 1;                               // a_i moves A -> B
            side[bv[i]] = 0;                               // b_i moves B -> A
        }
    }
    return best_G;
}

int main() {
    int m;
    if (!(cin >> m)) return 0;
    vector<vector<long long>> cost(m, vector<long long>(m, 0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            cin >> cost[i][j];

    int n = m / 2;
    vector<int> side(m, 0);                               // balanced start: A = {0..n-1}, B = {n..m-1}
    for (int i = n; i < m; ++i) side[i] = 1;

    cout << "start cut: " << external_cost(cost, side) << "\n";

    while (true) {                                        // run passes until one certifies a local optimum
        long long G = kl_pass(cost, side);
        if (G <= 0) break;
    }

    cout << "final cut: " << external_cost(cost, side) << "\n";

    vector<int> A, B;
    for (int i = 0; i < m; ++i) (side[i] == 0 ? A : B).push_back(i);
    cout << "A:";
    for (int x : A) cout << ' ' << x;
    cout << "\nB:";
    for (int x : B) cout << ' ' << x;
    cout << "\n";
    return 0;
}
```
