# The Kernighan–Lin Algorithm for Balanced Graph Partitioning

## Problem

Given an undirected graph on `2n` nodes with symmetric edge costs `c(i,j) ≥ 0`, split the nodes
into two sets `A`, `B` of exactly `n` each so that the **external cost** — the total cost of edges
cut, `T = Σ_{a∈A, b∈B} c(a,b)` — is minimized. The balance constraint is what makes the problem
hard: the unconstrained minimum cut is solvable by max-flow (Ford–Fulkerson) but comes out at
arbitrary sizes, and there is no natural way to force a balanced split onto it; exhaustive search
over the `½·C(2n,n)` balanced partitions is astronomically large. So we settle for a fast heuristic
that reliably reaches good cuts. This is the canonical min-cut bisection underneath top-down
min-cut VLSI placement.

## Key idea

The smallest balance-preserving move is to **exchange a pair** `a ∈ A`, `b ∈ B`. Writing the exact
cost change of one exchange gives the central quantity. For each node define the internal cost
`I_s` (edges to its own side) and external cost `E_s` (edges crossing the cut), and the **D-value**

    D_s = E_s − I_s.

**Gain of exchanging `a` and `b`:**  `g = D_a + D_b − 2·c(a,b)`  (reduction in `T`).
The `−2c(a,b)` is the double-count correction: the `a`–`b` edge stays cut either way and swings from
`−c(a,b)` to `+c(a,b)` across the swap.

A single best exchange (1-opt) stalls in poor local minima, and fixing a swap depth `λ` in advance
is the wrong knob. Instead, do a **variable-depth** move: greedily pre-select an entire sequence of
exchanges, locking each pair, then apply only the best prefix.

- **Lock + additive gains.** Pick the unlocked pair maximizing `g`, set it aside (each node moves at
  most once per pass), record its gain, and **update the surviving D-values** to reflect the
  tentative move:

      D'_x = D_x + 2·c(x, a_i) − 2·c(x, b_i),   x ∈ A − {a_i}
      D'_y = D_y + 2·c(y, b_i) − 2·c(y, a_i),   y ∈ B − {b_i}

  Locking keeps the gains additive, so swapping the first `k` pairs lowers `T` by exactly
  `G_k = g_1 + ⋯ + g_k`.

- **Best-prefix criterion.** Build the full sequence `(a_1,b_1), …, (a_n,b_n)` (its total gain is
  `Σ g_i = 0`, so unless every `g_i` is zero the positive gains must be offset by negative ones),
  then choose `k` maximizing `G_k`. Do **not** stop at the
  first negative gain: the running sum is allowed to dip and recover, which is exactly how the
  procedure escapes a local minimum, and the depth `k` is set by the data rather than fixed.

- **Apply and repeat.** If `G_k > 0`, exchange `{a_1,…,a_k}` with `{b_1,…,b_k}` (cost drops by
  `G_k`) and run another pass; if `G_k ≤ 0`, no improving prefix exists in this greedy pass, so the
  partition is taken as locally optimal. Each pass either improves the cut or certifies this local
  optimum, and the gains shrink as the partition improves, so few passes are expected to be needed.

Supporting facts: sorting the D-values per side and stopping the pair scan once `D_a + D_b` can no
longer beat the best gain (valid because `c ≥ 0`, so `g ≤ D_a + D_b`) keeps a pass near `n² log n`;
a small bounded number of passes makes the whole procedure near `n²`. Unequal target sizes are
handled by zero-cost **dummy** padding; unequal node weights by expanding a weight-`k` node into `k`
unit nodes bound by high-cost edges; `k`-way partitions by cycling the two-way exchange over all
`C(k,2)` pairs of subsets until pairwise-optimal.

## Algorithm

```
1.  Start from a balanced partition (A, B), |A| = |B| = n.
2.  Compute D_s = E_s - I_s for every node.
3.  free_A = A, free_B = B; sequences av, bv, gv empty.
4.  For step = 1 .. n:
       a) find unlocked a in free_A, b in free_B maximizing g = D[a] + D[b] - 2 c(a,b).
       b) append a -> av, b -> bv, g -> gv; remove a, b from free_A, free_B (LOCK).
       c) update D[x] += 2 c(x,a) - 2 c(x,b) for x in free_A;
          update D[y] += 2 c(y,b) - 2 c(y,a) for y in free_B.
5.  Find k maximizing G = sum(gv[1..k]).
6.  If G > 0: exchange av[1..k] with bv[1..k]; go to 2.
    Else: (A, B) is locally optimal; stop (or restart from a new partition, keep best).
```

## Code

A faithful, self-contained single-file C++17 program: D-values, the greedy max-gain locked-pair
selection, the D-value update, the cumulative-gain prefix choice, and multiple passes. It reads from
stdin an even integer `m = 2n` followed by an `m x m` symmetric nonnegative cost matrix (row major),
and writes the initial cut, the final cut, and the two blocks. (`long long` throughout for the
accumulated cut.)

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

### The Fiduccia–Mattheyses successor (linear-time, single-cell moves, gain buckets)

The pairwise exchange keeps balance exactly but pays an `n²`-per-selection cost and assumes graph
edges. A later refinement that became the workhorse of VLSI partitioning changes three things:

- **Single-cell moves.** Instead of swapping a pair, move *one* cell across the cut and enforce
  balance with an explicit ratio/size constraint. This handles **hypergraph nets** (a signal
  connecting many cells, the real netlist object — a net is *cut* if it touches both blocks) rather
  than only pairwise edges, and the **cell gain** `g(i)` is the number of nets by which the cutset
  would shrink if cell `i` were moved, an integer in `[-p(i), +p(i)]` where `p(i)` is the cell's pin
  count.

- **Gain buckets.** Because gains are small bounded integers, keep an array
  `BUCKET[-pmax .. +pmax]` whose entry `k` is a doubly-linked list of free cells with gain `k`. The
  highest-gain free cell is read off the top non-empty bucket; when a move changes a neighbor's gain,
  the neighbor is yanked from its list and re-inserted into the new bucket in O(1). Moving a cell
  only touches the cells it shares nets with, so total work per pass is **linear** in the network
  size.

- **Same outer loop.** As in KL: each cell is **locked** after its one move; the running
  cumulative gain over the pass is allowed to go negative (escaping local minima); the **best prefix**
  of moves seen during the pass is applied; passes repeat until no improvement. The KL escape-the-
  local-minimum scheme is reused — only the move, the gain object, and the selection data structure
  change.

```cpp
// Sketch of the FM selection/update primitives (hypergraph netlist).
// Cell gain = (# nets that would leave the cutset) - (# nets that would enter it) if `cell`
// moves to the other block. here == 1 means `cell` is the only cell on its side of the net;
// there == 0 means the net is entirely on `cell`'s side (would become cut).
int fm_gain(int cell, const vector<int>& side,
            const vector<vector<int>>& nets_of, const vector<vector<int>>& cells_of) {
    int g = 0;
    for (int net : nets_of[cell]) {
        int here = 0;
        for (int c : cells_of[net]) if (side[c] == side[cell]) ++here;
        int there = (int)cells_of[net].size() - here;
        if (here == 1)  ++g;          // cell is alone on its side -> moving it uncuts the net
        if (there == 0) --g;          // net is entirely on this side -> moving it cuts the net
    }
    return g;
}

// buckets[block][g] = doubly-linked list of free cells of gain g; pick the top non-empty bucket,
// move that cell, lock it, then for each free neighbor sharing a net recompute its gain and
// relocate it to its new bucket in O(1). Track the best cumulative-gain prefix over the pass and
// apply it; repeat passes until no improvement.
```
