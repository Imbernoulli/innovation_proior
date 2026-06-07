OK, let me think this through from scratch. I've got a graph with `2n` nodes and a symmetric cost
`c(i,j) ≥ 0` on every pair, and I want to split the nodes into two sets `A` and `B` of `n` each so
the total cost of edges running between the two sides — `T = Σ_{a∈A, b∈B} c(a,b)` — is as small as
possible. The balance is a hard constraint, not a soft preference: in the application I care about,
the two sides are physical cards with a fixed capacity, so an unbalanced split isn't a cheaper
answer to the same problem, it's an answer to a different problem.

That constraint is the whole difficulty, and it's worth being precise about why. If I drop the
balance requirement and just ask for the cheapest cut separating the graph into two parts, I'm in
friendly territory — that's a min-cut, and Ford and Fulkerson's max-flow machinery solves it: treat
the costs as capacities, the maximum flow between two nodes equals the minimum cut capacity
separating them. But that minimum cut comes out at *whatever sizes it likes*. Typically it shears
off a tiny weakly-attached lump on one side and leaves everything else on the other. There's no hook
in the flow algorithm to say "and make the two sides equal," and no obvious way to add one — if I
post-process the unbalanced cut into a balanced one I've thrown away the very optimality the flow
gave me. So the elegant polynomial method solves the wrong problem. (It's not useless: minimizing the
flow over every choice of the two separated nodes gives the *global* unconstrained min-cut, which is a
valid lower bound on any balanced cut's cost, since constraining can only
raise the optimum. But it doesn't give me the partition I need.)

And exhaustive search is hopeless. The number of balanced splits of `2n` nodes is `½·C(2n,n)`, which
blows up fast — for `n = 20` it is `½·C(40,20) ≈ 7·10^10`, and the moment I ask for more than two
blocks it explodes far faster still: splitting those same `40` nodes four ways into tens is
`40! / (10!^4 · 4!) ≈ 2·10^20` configurations. I could write it as an integer program with a pile of constraints to force the
uniformity, but any direct attack is going to demand an inordinate amount of computation at the
sizes I care about. So I'm not going to certify optimality. I'm going to build a heuristic that
reliably lands on good cuts, fast — running time near `n²`, not exponential — and I'll judge it
statistically, by how often a run reaches the known optimum.

So I'm in the iterative-improvement game: start from some balanced partition, find a transformation
that makes it cheaper, jump there, repeat until nothing improves — a local optimum — then restart and
keep the best. As always with local search, the entire quality of the scheme lives in the
transformation in the middle. If it's weak, I get stuck in lots of bad local optima; if it's strong,
the optima are few and good.

What's the simplest transformation that respects balance? I can't *move* a single node from `A` to
`B` — that breaks the `n/n` split, makes `A` too small and `B` too big. The smallest balance-
preserving move is to *exchange a pair*: take some `a ∈ A` and some `b ∈ B` and swap them. Sizes
stay put. So a partition is "1-opt" if no single such exchange reduces the cost, and I repeatedly do
improving exchanges until I hit a 1-opt partition. This is the partitioning analogue of the
single-link rearrangements Lin used on the traveling-salesman problem.

But I already know how this goes, because it's the same trap that fixed-`k` interchange heuristics
fall into everywhere. A single-pair exchange is a shallow neighborhood. On the test matrices, 1-opt
finds the apparent optimum only around a tenth of the time; most of the time it stalls at a 1-opt
partition that's a couple of units away from optimal but that *no single swap can improve*. The
improving move exists — it's just that getting to it requires swapping several pairs at once, and
some of those individual swaps make the cut momentarily *worse* before the group as a whole pays off.
A one-swap-at-a-time greedy procedure can never take that path, because it refuses the first
worsening step. I could fix this by exchanging `λ` pairs at a time for some larger `λ`, but then I'm
back to the fixed-knob disease: small `λ` is too weak, large `λ` makes each move expensive, and I
have no way to know the right `λ` for a given instance before I run. Fixing `λ` up front feels
exactly backwards. I'd like to escape the local minimum without committing to a depth in advance.

Before I can design anything clever I need the bookkeeping to be clean, and the right place to start
is: what does a *single* exchange actually cost me? Let me look hard at swapping one `a` for one `b`.
For a node `a ∈ A`, the edges from `a` split into two kinds. Some go to other nodes of `A` — call
their total the **internal** cost `I_a = Σ_{x∈A} c(a,x)`; these don't cross the cut, so they're
"holding `a` in place." The rest go to nodes of `B` — the **external** cost `E_a = Σ_{y∈B} c(a,y)`;
these *do* cross the cut, so they're "pulling `a` across." Symmetrically define `I_b`, `E_b` for a
node `b ∈ B`.

Now suppose I exchange `a` and `b`. I want the exact change in `T`. Let me peel the total cut cost
into the part that touches `a` or `b` and the part that doesn't. Let `z` be the cost of all
`A`–`B` connections that involve *neither* `a` nor `b`; that part is untouched by the swap. Every
edge from `a` to the `B` side is currently cut, and every edge from `b` to the `A` side is currently
cut, so before the swap

    T = z + E_a + E_b − c(a,b).

Why the `− c(a,b)`? Because the single edge between `a` and `b`, if it exists, has been counted
*twice* — once inside `E_a` (it's one of `a`'s edges to the `B` side, since `b ∈ B`) and once inside
`E_b` (it's one of `b`'s edges to the `A` side, since `a ∈ A`) — and it's only one cut edge, so I
subtract it once to undo the double count.

After the swap, `a` sits in `B` and `b` sits in `A`. Now `a`'s edges to the old `A` nodes are the
ones that cross — that's `I_a` — and `b`'s edges to the old `B` nodes cross — that's `I_b`. And the
edge between `a` and `b`: with `a` now in `B` and `b` now in `A`, it's still a cut edge, and again it
got counted once in the "`a`'s edges to the `A` side" bucket and once in the "`b`'s edges to the `B`
side" bucket, so

    T' = z + I_a + I_b + c(a,b).

The `z` is the same in both because those edges don't touch `a` or `b`. So the gain — the reduction
in cost from the exchange — is

    g = T − T' = (E_a − I_a) + (E_b − I_b) − 2·c(a,b).

There it is, and it's clean. The factor of two on `c(a,b)` is the punchline of the double-count: the
`a`–`b` edge swung from `−c(a,b)` to `+c(a,b)` as I went from `T` to `T'`, a net `2·c(a,b)` working
against the swap — which makes sense, an `a`–`b` edge stays cut whether or not I exchange them, so
swapping them can never relieve it and in fact charges me twice for it.

The quantity `E − I` keeps appearing, so let me name it. Define for every node `s` its **D-value**

    D_s = E_s − I_s,

the external cost minus the internal cost. It's exactly the "how badly does this node want to be on
the other side" number: positive `D` means more of `s`'s weight is reaching across the cut than is
holding it home, so moving it across would help. With this the gain of exchanging `a` and `b` is just

    g = D_a + D_b − 2·c(a,b).

This is the object I'll build everything on. To pick the single best exchange I compute all the
`D`-values, then look for the pair `a ∈ A`, `b ∈ B` maximizing `D_a + D_b − 2·c(a,b)`. That's the
1-opt step, now with a tidy formula instead of a brute re-evaluation of the whole cut for every
candidate pair.

But this still only gives me single exchanges, and I've already argued single exchanges stall. I
want to chain several, and I want the depth to come out of the data rather than be fixed in advance.
Suppose I do a *sequence* of exchanges. If I could compute each
one's gain as if it were the only one, and the total reduction were just the sum of those gains,
then I could reason about a *partial* chain of swaps — ask "would the first `k` of them, taken
together, lower the cost?" — instead of only being able to evaluate fully-formed exchanges. The
trick to making the sum honest is that each gain must be computed *against the partition as it stands
after the earlier swaps in the chain*. So after I tentatively swap a pair, I have to update the
`D`-values of everyone else to reflect that the two swapped nodes have changed sides — and then the
next gain, computed with the updated `D`'s, slots additively onto the running total.

Let me work out that update, because it's the engine of the whole thing. Say I've just set aside the
pair `(a_i, b_i)` — conceptually moved `a_i` to the `B` side and `b_i` to the `A` side. Take some
other node `x` still on the `A` side. How does its `D_x = E_x − I_x` change? Two of `x`'s edges are
affected. The edge `(x, a_i)`: before, `a_i` was in `A`, so this edge was *internal* to `x` (counted
in `I_x`); now `a_i` is in `B`, so the same edge is *external* (belongs in `E_x`). So it leaves `I_x`
and joins `E_x`. Since `D = E − I`, an edge of weight `c(x,a_i)` moving from the `I` column to the
`E` column raises `D_x` by `c(x,a_i)` from the `E` side *and* another `c(x,a_i)` from no longer being
subtracted in `I` — a total of `2·c(x,a_i)`. The edge `(x, b_i)`: before, `b_i` was in `B`, so this
edge was external (in `E_x`); now `b_i` is in `A`, so it's internal (in `I_x`). That moves a weight
`c(x,b_i)` from `E` to `I`, lowering `D_x` by `2·c(x,b_i)`. So

    D'_x = D_x + 2·c(x, a_i) − 2·c(x, b_i),    for x ∈ A − {a_i},

and by the mirror-image argument, for a node `y` still on the `B` side, the edge `(y, b_i)` goes
internal→external (its own-side partner `b_i` left for `A`, raising `D_y` by `2·c(y,b_i)`) and
`(y, a_i)` goes external→internal (the other-side `a_i` arrived in `B`, lowering `D_y` by
`2·c(y,a_i)`), giving

    D'_y = D_y + 2·c(y, b_i) − 2·c(y, a_i),    for y ∈ B − {b_i}.

Now I have a procedure forming. Compute all `D`'s. Pick the pair `(a_1, b_1)` maximizing
`g_1 = D_{a_1} + D_{b_1} − 2·c(a_1,b_1)` — the best single exchange available. Set that pair *aside*
— don't actually commit to swapping it yet, just remove `a_1` and `b_1` from further contention this
round and record `g_1`. Update every remaining node's `D`-value by the rule above. Now pick the best
pair among what's left, `(a_2, b_2)`, maximizing `g_2 = D'_{a_2} + D'_{b_2} − 2·c(a_2,b_2)`; note
`g_2` is the gain of exchanging `a_2, b_2` *given that `a_1, b_1` have already been exchanged*. Set
them aside, update again, and keep going until every node has been paired off: I get a full sequence
`(a_1,b_1), …, (a_n,b_n)` with gains `g_1, …, g_n`.

Here's why setting each pair aside — locking it — matters and isn't just tidiness. Because each node
is removed from contention the moment it's chosen, every node moves *at most once* across this whole
sequence. That's what makes the gains genuinely additive and the bookkeeping a clean telescope: the
swap of `(a_i, b_i)` is computed against a partition in which the earlier pairs have moved and the
later pairs haven't, and since no node is touched twice, summing the gains exactly reconstructs the
cost change of doing the whole prefix. It also guarantees the sequence terminates — there are only
`n` pairs to use up.

If I exchange the entire sequence — all `n` pairs — I've simply relabeled `A` and `B` into each
other's complements in a way that returns the same partition structure, so the total `Σ_1^n g_i = 0`.
That zero is actually informative: it tells me the gains can't all be positive (unless they're all
zero); the sequence necessarily has negative `g_i`'s in it somewhere. So I am *not* going to swap the
whole sequence. I'm going to swap a *prefix* of it.

This is the move that breaks out of the local minimum. Consider the partial sums
`G_k = g_1 + g_2 + ⋯ + g_k`. Exchanging exactly the first `k` pairs — `X = {a_1,…,a_k}` for
`Y = {b_1,…,b_k}` — produces a new balanced partition whose cost is lower than the start by exactly
`G_k`. So I should choose the `k` that maximizes `G_k`. Crucially I do **not** stop the sequence at
the first negative `g_i`. A greedy "halt when a swap stops paying" rule is exactly the 1-opt trap: it
would refuse a step that loses a little even when that step sets up a later step that wins a lot. By
building the *whole* sequence first and only then picking the best prefix, I let the running sum dip
into the red and climb back out. If `G_k` is maximized at some `k` where the gains went
`g_1 > 0`, `g_2 < 0`, `g_3 ≫ 0`, …, fine — I take that whole prefix. The temporary worsening at
step 2 is the price of reaching the deep improvement at step 3, and the prefix-sum criterion is what
lets me pay it.

Let me pin down the loop. After computing the full sequence and its gains, find `k` maximizing
`G = Σ_{i=1}^k g_i`. If `G > 0`, I've found a genuine improvement: actually perform the exchange of
`{a_1,…,a_k}` with `{b_1,…,b_k}`, dropping the cost by `G`, and treat the result as my new starting
partition — recompute `D`'s from scratch on it and run another pass. If `G ≤ 0`, then no prefix
improves the current partition; it's a local optimum with respect to this whole variable-depth
exchange, and I stop (or restart from a fresh random partition and keep the best). Notice the depth
`k` of the accepted move is never fixed — it's whatever the data hands me, small near a good
partition, large when there's a lot to fix. That's exactly the variable depth I wanted, and it falls
out of the prefix-maximization rather than being a parameter.

So the structure of one pass is: compute all `D`-values; then `n` times, select the maximum-gain
unlocked pair, lock it, record its gain, update the `D`-values of the survivors; then pick the
best-prefix `k`; if `G_k > 0` apply that prefix and start another pass, else declare the partition
locally optimal. I'd expect the number of passes needed to be small, because each pass either makes a
real dent or certifies a local optimum, and as the partition improves there is less left to gain, so
the per-pass improvement should shrink toward zero quickly — something to confirm once it runs.

Now, is this affordable? Let me count. Computing the initial `D`-values is an `n²` job: each of the
`2n` nodes is compared against all the others. The updates across a pass cost
`(n−1) + (n−2) + ⋯ + 1`, which is also `∝ n²` — after the `i`-th pair is locked I touch the `D`'s of
the remaining `~(n−i)` survivors. So far so good; the dominant cost is actually the *selection* of
each next pair, "find `a, b` maximizing `D_a + D_b − 2·c(a,b)`." Done naively that's a scan over all
remaining `A×B` pairs, `~n²` work, repeated `n` times — `n³` per pass, which is steep.

I can do much better by sorting. Sort the `A`-side `D`-values descending, `D_{a_1} ≥ D_{a_2} ≥ ⋯`,
and likewise the `B`-side. Now scan pairs in order of `D_a + D_b`. The key observation is that
because every `c(a,b) ≥ 0`, the gain `D_a + D_b − 2·c(a,b)` is *at most* `D_a + D_b`. So as I walk
down the sorted lists, the moment I reach a pair whose `D_a + D_b` no longer exceeds the best gain
I've already seen this round, I can stop — no later pair can beat it, since subtracting a nonnegative
`2c` can only make it smaller. So only a few likely contenders near the top of each sorted list ever
get examined. Sorting is `n log n`, done over the pass that's `~n² log n`, and the running time of a
pass comes out near `n² log n`.

If I want to shave that further, I can skip the full sort and just keep the largest `D_a` and largest
`D_b`, with the corresponding `a, b`, as the next exchange — essentially linear-time selection,
folded into the `D`-recomputation. That can miss the true max when the top pair happens to have a
large `c(a,b)` linking them (so its actual gain is small); a cheap fix is to also save the top two or
three `D`'s on each side and try those alternatives when the very top pair's `c(a,b)` is big. Keeping
a top-few from each side should recover the true best pair except in the rare case where all of them
are mutually linked by large costs, so I'd expect even two or three saved values to lose almost no
power while trading the sort for a linear scan — a speedup worth measuring against the sorted version.
Either way the per-pass cost is essentially
quadratic, and with a small bounded number of passes the whole procedure runs near `n²` — gentle
enough to take many random restarts and keep the best.

There's a loose end on the application side: the sizes are rarely conveniently `n` and `n`. Suppose I
want at least `n_1` and at most `n_2` elements per side, with `n_1 < n_2` and `n_1 + n_2 = 2n`. Two
clean ways to bend the equal-size procedure to this. First, just restrict the number of pairs
exchanged in one pass to `n_1` — I only ever swap as many pairs as the smaller side can give up,
which keeps every intermediate partition feasible. Second, and slicker: pad with **dummy** elements
that have *zero* cost to everything (all-zero rows and columns in `c`). Add `2n_2 − 2n` dummies so
the padded instance has `2n_2` elements, run the plain equal-`n_2`-and-`n_2` procedure on it, and at
the end discard the dummies. Because dummies cost nothing, the procedure freely parks them to absorb
the slack, and the real elements settle into whatever split between `n_1` and `n_2` minimizes the
true external cost. Same idea handles unequal *node* sizes: blow a node of size `k` into a cluster of
`k` size-1 nodes bound together by edges of very high cost, so the procedure won't ever cut the
cluster apart and effectively treats it as one heavy node.

And the two-way procedure is the building block for more sets: to partition into `k > 2` subsets,
start with `k` subsets of the right size and apply the two-way exchange to each pair of subsets,
cycling through all `C(k,2)` pairs repeatedly until the partition is *pairwise* optimal — no exchange
between any two subsets helps. Pairwise optimality is only a necessary condition for global
optimality (some genuinely multi-way rearrangement might still help), but it converges in a small
number of passes and gives good `k`-way partitions in practice.

Now let me land it on code. I'll represent the partition as two node sets `A` and `B`, keep the
symmetric cost matrix, and write the pass exactly as derived: compute `D`-values, then loop `n`
times selecting and locking the best unlocked pair while updating `D`'s, then choose the best prefix.

```python
def kl_pass(cost, A, B):
    """One Kernighan-Lin pass on a balanced bipartition (A, B). Returns the improved
    (A, B) and the cost reduction G achieved (0.0 if the partition is already locally optimal)."""
    n = len(cost)
    A, B = set(A), set(B)

    # D_s = E_s - I_s : external cost minus internal cost, for every node, on the CURRENT partition.
    def compute_D(A, B):
        D = {}
        for s in range(n):
            own, other = (A, B) if s in A else (B, A)
            I = sum(cost[s][x] for x in own if x != s)     # internal: edges to own side
            E = sum(cost[s][y] for y in other)             # external: edges crossing the cut
            D[s] = E - I
        return D

    D = compute_D(A, B)
    free_A, free_B = set(A), set(B)                        # unlocked nodes on each side
    av, bv, gv = [], [], []                                # the sequence of pairs and their gains

    # Build the full sequence: each step locks the max-gain unlocked pair and updates the D-values.
    for _ in range(len(A)):
        # select the unlocked pair maximizing g = D[a] + D[b] - 2 c(a,b)
        best, ba, bb = None, None, None
        for a in free_A:
            for b in free_B:
                g = D[a] + D[b] - 2 * cost[a][b]           # Lemma: gain of exchanging a and b
                if best is None or g > best:
                    best, ba, bb = g, a, b
        av.append(ba); bv.append(bb); gv.append(best)      # record and LOCK this pair
        free_A.discard(ba); free_B.discard(bb)

        # D-update for the survivors, reflecting that ba moved to B and bb moved to A:
        #   D'_x = D_x + 2 c(x,ba) - 2 c(x,bb)   for x still on the A side
        #   D'_y = D_y + 2 c(y,bb) - 2 c(y,ba)   for y still on the B side
        for x in free_A:
            D[x] += 2 * cost[x][ba] - 2 * cost[x][bb]
        for y in free_B:
            D[y] += 2 * cost[y][bb] - 2 * cost[y][ba]

    # Choose the prefix length k that maximizes the cumulative gain G_k = g_1 + ... + g_k.
    # The running sum may dip negative and recover -- that is how a local minimum is escaped.
    G, best_G, k = 0.0, 0.0, 0
    for i, g in enumerate(gv, start=1):
        G += g
        if G > best_G:
            best_G, k = G, i

    if best_G > 0:                                         # a real improvement: apply the prefix
        for i in range(k):
            A.discard(av[i]); A.add(bv[i])                 # exchange {a_1..a_k} with {b_1..b_k}
            B.discard(bv[i]); B.add(av[i])
    return A, B, best_G
```

The driver just runs passes until one certifies a local optimum (`G ≤ 0`):

```python
def kernighan_lin(cost, A, B):
    A, B = set(A), set(B)
    while True:
        A, B, G = kl_pass(cost, A, B)                      # one variable-depth pass
        if G <= 0:                                         # no improving prefix -> locally optimal
            return A, B
```

So the causal chain: the balance constraint is what makes the problem hard and what kills the elegant
max-flow min-cut, so I'm in local search, and the smallest balance-preserving move is a pairwise
exchange. Writing out the exact cost change of one exchange gives `g = D_a + D_b − 2·c(a,b)` with
`D = E − I`, the marginal "wants to cross" value of a node and a clean factor-of-two for the shared
edge. Single exchanges stall in 1-opt local minima, and fixing a depth `λ` in advance is the wrong
knob, so instead I greedily pre-select a *whole sequence* of exchanges — locking each pair so every
node moves at most once and the gains stay additive — recompute the `D`-values after each pick via
`D'_x = D_x + 2c(x,a_i) − 2c(x,b_i)`, and then pick the *prefix* `k` that maximizes the cumulative
sum `G_k`. Because I take the best prefix rather than stopping at the first negative gain, the running
total can dip and recover, which is exactly how the procedure escapes a local minimum, and because
`k` comes out of the data the depth is variable. Sorting the `D`-values and stopping the scan once
`D_a + D_b` can't beat the best-so-far (valid since `c ≥ 0`) keeps a pass near `n² log n`, and a small
number of passes makes the whole thing near `n²`. Dummy padding handles unequal sizes, high-cost
clustering handles unequal node weights, and cycling the two-way exchange over all pairs of subsets
handles `k`-way partitions.
