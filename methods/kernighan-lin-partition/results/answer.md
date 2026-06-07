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
  `Σ g_i = 0`, so some `g_i` are negative), then choose `k` maximizing `G_k`. Do **not** stop at the
  first negative gain: the running sum is allowed to dip and recover, which is exactly how the
  procedure escapes a local minimum, and the depth `k` is set by the data rather than fixed.

- **Apply and repeat.** If `G_k > 0`, exchange `{a_1,…,a_k}` with `{b_1,…,b_k}` (cost drops by
  `G_k`) and run another pass; if `G_k ≤ 0`, the partition is locally optimal. Passes needed are few
  in practice.

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

A faithful, self-contained implementation: D-values, the greedy max-gain locked-pair selection, the
D-value update, the cumulative-gain prefix choice, and multiple passes.

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

```python
# Sketch of the FM selection/update primitives (hypergraph netlist).
def fm_gain(cell, side, nets_of, cells_of):
    """Cell gain = (# nets that would leave the cutset) - (# nets that would enter it) if `cell`
    moves to the other block. FS = nets with `cell` the only cell on its side; TE = nets entirely
    on `cell`'s side (would become cut)."""
    g = 0
    for net in nets_of[cell]:
        here = sum(1 for c in cells_of[net] if side[c] == side[cell])
        there = len(cells_of[net]) - here
        if here == 1:   g += 1        # cell is alone on its side -> moving it uncuts the net
        if there == 0:  g -= 1        # net is entirely on this side -> moving it cuts the net
    return g

# buckets[block][g] = doubly-linked list of free cells of gain g; pick the top non-empty bucket,
# move that cell, lock it, then for each free neighbor sharing a net recompute its gain and
# relocate it to its new bucket in O(1). Track the best cumulative-gain prefix over the pass and
# apply it; repeat passes until no improvement.
```
