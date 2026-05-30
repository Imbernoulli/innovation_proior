OK, let me think this through from scratch. The thing I keep running into is the wall on sequence length. Self-attention is wonderful — every token looks at every token, in parallel, no recurrence — but the cost of "every looks at every" is exactly what bites. For a length-`n` sequence each query computes an inner product with each of the `n` keys, so I form an `n × n` matrix of scores. That matrix is `O(n²)` to compute and, worse, `O(n²)` to *store*, because I need it around for the softmax and the backward pass. With the model sizes and the accelerator memory I actually have, that quadratic term pins me to about 512 tokens. And 512 is not enough for the things I care about: a question whose answer is buried in a long passage, a document I want to classify or summarize whole, a stretch of DNA whose meaning depends on context thousands of bases away. So the goal is concrete: get attention down to linear in `n`, using far fewer than `n²` inner products.

But I want to be careful about what I'm allowed to lose. The reason I trust full attention isn't just that it works on benchmarks; it's that there are theorems. Yun et al. (2019) showed a transformer encoder can approximate any continuous sequence-to-sequence function on a compact domain to arbitrary accuracy. Pérez et al. (2019) showed the full encoder–decoder is Turing complete, given arbitrary precision. If I throw away inner products and end up with something strictly weaker — say, a model that provably can't represent certain functions full attention can — then I haven't found a cheaper transformer, I've found a different, lesser model. So the real target is: linear cost *and* the same expressive guarantees.

Let me get the right object in front of me before I start cutting. What is an attention mechanism, stripped down? It's a rule that says, for each query position `i`, which key positions it sums over. Call that set `N(i)`. Then

`Attn(X)_i = x_i + Σ_h σ( Q_h(x_i) K_h(X_{N(i)})^T ) · V_h(X_{N(i)})`,

with the residual `x_i`, heads `h`, and `σ` a softmax. In full attention `N(i)` is everything, `{1,…,n}`. So the set of inner products I evaluate is precisely the set of pairs `(i, j)` with `j ∈ N(i)`. That's a directed graph. Vertices are token positions; an arc `i → j` means "query `i` attends to key `j`." Full attention is the *complete* digraph — its adjacency matrix is the all-ones matrix, which is exactly why it's quadratic. BERT's attention is that all-ones matrix.

Once I write it that way the problem renames itself. I'm not "approximating a softmax" or "compressing memory." I'm deleting arcs from a complete graph and asking how few I can keep while the graph still *behaves* like the complete one. That's graph sparsification, and graph theory has a lot to say about when a sparse graph is a good stand-in for a dense one. So the question becomes: which subgraph on `n` nodes, with only `O(n)` arcs, still does the job of the complete graph?

What does "does the job" mean here, in graph terms? Let me reason about what the complete graph gives me that I'd hate to lose. First, any token can influence any other token in a single layer — there's a direct arc. If I sparsify, I lose most direct arcs, so influence has to travel along paths across layers. I don't need a direct arc between every pair, but I do need information to be able to *reach* from anywhere to anywhere in few hops; otherwise a fact at position 5 can't affect the representation at position 5000 without an absurd number of layers. So property one: the graph should have small diameter — short paths between any two nodes.

What graph with `O(n)` edges has short paths? This is exactly the regime where random graphs shine. Take an Erdős–Rényi graph — each possible edge included independently with some probability — tuned to have about `n` edges (up to log factors). It's a classical fact that the shortest path between any two nodes is `O(log n)`. And the spectral statement is even better for me: such a random graph approximates the complete graph spectrally — its adjacency matrix has a large gap between the first and second eigenvalue. A large spectral gap is the same thing as rapid mixing of a random walk, which is the precise sense in which "information flows fast between any pair of nodes." So a *random* set of arcs buys me the short-path / mixing property cheaply. Concretely: let each query attend to `r` randomly chosen keys, `A(i, ·) = 1` for `r` random `j`. That's `O(r·n) = O(n)` arcs and it gives me an expander.

Is random enough? Let me poke at it. A pure random graph has a known weakness: low clustering coefficient. The clustering coefficient measures how often two neighbors of a node are themselves neighbors — how many little cliques the graph contains. Random graphs are locally tree-like, almost no triangles. Why would I care about triangles in an attention graph? Because of what the data looks like. When people actually probed trained attention (Clark et al., 2019), the dominant inner products were with *nearby* tokens — neighboring-position attention carries a huge share of the signal. That matches the intuition from linguistics and from biology: a token's meaning is largely set by the tokens immediately around it, locality of reference. A purely random arc set ignores this completely — it would have me spend my scarce arcs on far-flung pairs and miss the cheap, high-value local ones.

So property two: locality. In graph terms, I want high clustering — local near-cliques. The plainest way to get that is a ring lattice: arrange the `n` tokens on a line (or ring) and connect each to its `w` nearest neighbors, `w/2` on each side. That's a sliding window: `A(i, i−w/2 : i+w/2) = 1`. Lots of overlapping local neighborhoods, high clustering, `O(w·n) = O(n)` arcs.

Now I have a tension. The ring lattice has high clustering but *long* paths — to get from one end to the other you crawl neighbor by neighbor, `O(n/w)` hops, terrible. The random graph has short paths but no clustering. I want both. And there's a famous construction that does exactly this: the Watts–Strogatz small-world model. Start from the ring lattice, then take a small fraction of the local edges and rewire each to a random target. Those few long-range shortcuts collapse the diameter to `O(log n)` while the bulk of the local structure (the clustering) survives. That's the recipe: a base of local window edges plus a sprinkling of random edges. In my setting I don't even need to *delete* the rewired local edges — keeping all of them is friendlier to the hardware and doesn't hurt the properties. So: window `+` random.

Let me sanity-check this before I get attached to it. I build a sparse attention with just window and random, pretrain it the way I'd pretrain a bidirectional encoder, and look at masked-LM accuracy and a couple of downstream tasks at length 512, against the dense baseline. And here's what I see: random alone trails the dense model badly, window alone trails it worse, and window+random together close some of the gap but still sit clearly below dense — a few points down on masked-LM, more on the harder tasks. Huh. So small-world intuition got me a graph that *should* mix fast and respect locality, and it's still not matching dense attention. Something is missing, and it's not a tuning issue — the structure itself is short of what dense attention does.

Let me stop guessing at the architecture and go back to the guarantee I refused to give up. The whole point was to keep the expressivity. So let me actually try to push the universal-approximation proof through *my* sparse graph and watch where it breaks — the failure point should tell me what arc I'm missing.

I can borrow the easy approximation step from dense attention. Any continuous `f` on the compact domain `[0,1]^{n×d}` can be approximated to `ε/3` by a piecewise-constant function `f̄` on a grid of granularity `δ`: chop the cube into little cells and let `f̄` be constant on each. The cleanest way to keep the columns from colliding is to add a positional embedding `E` whose `i`-th row shifts token `i` into its own numeric band; after the shift, a fixed vector `u` can be chosen so that the projection of column `i` lies in `[δ^{-(i-1)d}, δ^{-id})`. Different positions, disjoint buckets.

The cleanup at the end is also standard: the construction in the middle is allowed to use a hardmax and some convenient activation functions, and I can approximate those with the real softmax and ReLU at the cost of another `ε/3`. Fine.

The part that should expose the missing edge is the contextual mapping. I need attention to assign to each pair (whole input `X`, position `i`) a unique scalar code, such that no two distinct inputs and no two distinct positions ever share a code. If I can produce such unique codes, then the feed-forward layers can act as a giant lookup table — each code maps to the right output value of `f̄`. So everything rides on building unique, context-aware codes.

How does the dense proof build them? With a "selective shift" operator. Pick a direction `u`. Look at each column's projection `u^T x_j`. The operator shifts up the entries whose projection falls in a chosen range, and the amount it shifts is the *range* of the projections — the max minus the min over the columns it's looking at. By applying this with a carefully chosen sequence of ranges, you ratchet each column's value up in a way that records the entire input, and you can read off a unique code. The crucial thing the dense version uses, though, is that the max and min are taken *over all columns* — the operator at position `i` sees everything. That's where full attention is doing its work: `N(i)` is all of `[n]`, so `max_{j∈N(i)} u^T x_j` really is the global max.

Now I drop in my sparse graph and the construction breaks exactly there. With window+random, `N(i)` is a tiny set — a few neighbors and a few random keys. The max over `N(i)` is not the global max. Each query only ever sees a handful of columns, so there's no obvious way for any single position to corral information about the whole matrix and mint a code that depends on all of it. The proof doesn't limp; it stops. And that's the same wall the experiment hit: window+random can mix information *eventually*, over many hops, but a single contextual-mapping construction needs some position that can act on the entire sequence *now*.

So what would unstick it? I need at least one node that is connected to everything — a node whose `N` is all of `[n]`, so the global max/min is available — and, symmetrically, a node that everything can deposit information into and later read back from. That is: a token that attends to all tokens and is attended by all tokens. A *global* token.

Let me see if one global token is actually sufficient for the proof, because that would be the cheapest possible fix — one extra node, `O(n)` extra arcs. Put a special token at index `0`. Give it `N(0) = {1,…,n}` (it sees everyone) and put `0` into every other node's neighborhood, `N(i) ∋ 0` (everyone sees it). The minimal graph that does this is the *star*: `N(i) = {0, i}` for `i ≥ 1`, `N(0) = {1,…,n}`. If I can push the contextual-mapping construction through on this star, then any graph containing the star is a universal approximator; my window+random graph, once I add the global token, contains it.

With attention restricted to `N(i)`, I can implement, for a direction `u` and a band `[b_1, b_2]`,

`ψ_u(Z; b_1, b_2)_i = (max_{j∈N(i)} u^T Z_j − min_{j∈N(i)} u^T Z_j) e_1` if `u^T Z_i ∈ [b_1,b_2]`, else `0`,

where `e_1` is the first coordinate. The way I get the max and the min separately is the usual hardmax trick: an attention that, with the right query/key, returns the largest projected entry in `N(i)`, and a second one returning the smallest; the operator is their difference, and the band condition gates which positions are shifted. The amount of the shift is controlled by which arcs `N(i)` has — that's the "sparse" part — and the global token is what makes a relevant `N(i)` equal to the whole sequence.

Now thread the context through using the star. Choose `u = [1, δ^{-1}, δ^{-2}, …, δ^{-d+1}, δ^{-nd}]` and set the global token's value to `x_0 = (0,…,0,1)` so that `u^T x_0` starts at `δ^{-nd}`, just above the last ordinary bucket because `u^T x_n < δ^{-nd}`. So initially the projections are ordered `l_1 < l_2 < … < l_n < l_0`. I run through the ordinary columns in order; the recurring two-shift pattern is clean for columns `1` through `n-1`, and the last column gets the same low lift followed by a final transfer that does not pretend there is an `l_{n+1}` bucket.

For the low shift of phase `k`, I apply the operator with the band `[δ^{-(k-1)d}, δ^{-kd})`, narrow enough that only column `k`'s projection `l_k` lies in it. For that column, `N(k)` contains the global token (star edge), so the max over its neighborhood is `f̃_0^{k-1}` (the global token's current value) and the min is `l_k`. With shift scale `δ^{-d}` the new value of column `k` becomes

`f_k = δ^{-d}(f̃_0^{k-1} − l_k) + l_k`,

which, for small `δ`, is larger than `f̃_0^{k-1}` — column `k` has now jumped above the global token and, because the gap `f̃_0^{k-1} − l_k` encodes everything seen so far, `f_k` carries the running context.

For `k < n`, the high shift applies the operator in a band that catches only the global token (its value is in a known range `[S_{k-1}, T_{k-1})`). Now the global token attends over the whole sequence (star edge `N(0) = all`), so it sees the new global max — which is `f_k`, the column I just lifted — and the global min among ordinary columns, which is still `l_{k+1}`. With shift scale `δ^{-nd}` it updates to

`f̃_0^k = δ^{-nd}(f_k − l_{k+1}) + f̃_0^{k-1}`.

Substitute the expression for `f_k`:

`f̃_0^k = δ^{-nd}( δ^{-d}(f̃_0^{k-1} − l_k) + l_k − l_{k+1} ) + f̃_0^{k-1}`
` = δ^{-(n+1)d}(f̃_0^{k-1} − l_k) + δ^{-nd}(l_k − l_{k+1}) + f̃_0^{k-1}`
` = (δ^{-(n+1)d} + 1) f̃_0^{k-1} − (δ^{-(n+1)d} − δ^{-nd}) l_k − δ^{-nd} l_{k+1}.`

So the global token obeys a linear recursion. Let `α = δ^{-(n+1)d}+1`, `β = δ^{-(n+1)d} − δ^{-nd}`, and `γ = δ^{-nd}`. Then

`f̃_0^k = α f̃_0^{k-1} − β l_k − γ l_{k+1}`.

Unroll it from `f̃_0^0 = δ^{-nd}`:

`f̃_0^k = α^k δ^{-nd} − α^{k-1}β l_1 − Σ_{t=2}^{k} α^{k-t}(β+αγ) l_t − γ l_{k+1}`.

Each `l_t` enters through a different power of the huge multiplier `α`, so the value `f̃_0^k` is, in effect, the digits `l_1,…,l_k` written in a base so large they never interfere — a positional number system. That's why it's a unique encoding. Plug in the bounds `δ^{-(t-1)d} ≤ l_t < δ^{-td}` and I get an upper envelope `T_k` (using `l_t < δ^{-td}`) and a lower envelope `S_k` (using `l_t ≥ δ^{-(t-1)d}`), and by construction the global token stays sandwiched, `S_k ≤ f̃_0^k < T_k`. For small enough `δ` everything is dominated by the leading term `≈ δ^{-n(k+1)d - kd}`, so the lower-order corrections never threaten the ordering. The invariants I carry from phase to phase are: `S_k < f̃_0^k < T_k`; `T_{k-1} ≤ f_k < S_k` (the lifted column sits below the next global-token band, so the bands don't collide); and the total order `l_{k+1} < … < l_n < f_1 < … < f_k < f̃_0^k`. At `k=n` the low shift gives `f_n`, which already contains the entire prefix `l_1,…,l_n`; there is no unshifted `l_{n+1}` to use as the minimum, so I transfer this last code into the global token with a final selective shift that targets the known global band and uses `f_n` as the maximum and any fixed lower occupied band as the anchor. The anchor only changes a lower-order subtractive term; injectivity comes from the dominant `f_n` term. One more shift layer over the accumulated range then separates all the *columns'* codes too, so every entry is distinct — a genuine contextual mapping. And it only ever used star edges plus local moves: sparse attention.

That's the missing piece, and notice it arrived from two directions at once. The experiment said window+random falls short of dense; the proof said the contextual-mapping construction stalls because no sparse node can see the whole sequence. Both are cured by the same object — a token wired to everyone, in and out. The global token isn't a heuristic add-on; it's the thing that recovers the universal-approximation theorem for a sparse graph, because the star is the minimal graph the proof needs and the global token is what plants the star inside my window+random pattern.

There are two natural ways to install global tokens. I can promote a subset `G` of the *existing* tokens to global — set `A(i,:) = 1` and `A(:,i) = 1` for `i ∈ G` — which costs no extra positions; call this the internal construction. Or I can *append* new global tokens, CLS-like, that exist only to attend and be attended: build a bigger matrix `B` with `g` new rows/columns set to all-ones and the old pattern in the corner. That adds dedicated scratch capacity to store context, which tends to help in practice; call this the extended construction. The theory only needs one global token (one star); a practical block layout can afford a couple of global blocks.

Now I have the full pattern: each query attends to a window of `w` neighbors, to `r` random keys, and to (or as) `g` global tokens. The cost per query is `g + w + r`, all constants in `n`, so the whole mechanism is `O((g+w+r)·n·d) = O(n)` in compute and `O(n)` in memory. That's the linear scaling I wanted, and it makes much longer contexts feasible under the same memory pressure.

Before I move on I should be honest about whether I've gotten something for nothing — is sparse attention just strictly better? It can't be, and I want to pin down what it costs. Consider a clean task: given `n` unit vectors `u_1,…,u_n`, output for each `j` its *furthest* vector `u_{j^*}`, `j^* = argmax_k ‖u_k − u_j‖²`. For unit vectors `‖u_k − u_j‖² = 2 − 2⟨u_k, u_j⟩`, so the furthest vector is the one with the *smallest* inner product, `j^* = argmin_k ⟨u_k, u_j⟩`. Full attention nails this in a single layer: embed `x_i = [u_i; 0]`, set `Q([a;b]) = −a`, `K([a;b]) = a`, `V([a;b]) = [0;a]`. Then the score `⟨Q(x_i), K(x_j)⟩ = ⟨−u_i, u_j⟩`, and a hardmax over `j` picks `argmax_j ⟨−u_i,u_j⟩ = argmin_j ⟨u_i,u_j⟩ = j^*`; the value brings back `[0; u_{j^*}]`, and with the residual `a_i = [u_i; u_{j^*}]`. One layer, `O(1)`.

Can any sparse pattern match that? The reduction is direct. Suppose a sparse network with `Õ(n)` arcs and `l` layers solves the task; with head/hidden sizes `O(d)` and `d = Θ(log² n)`, each layer costs `Õ(n d³)`, so the whole thing is `Õ(n l d³)`. But solving this task lets me solve Orthogonal Vectors: for each `j` I now know its minimum inner product (via `u_{j^*}`), so with `O(n)` more work I can check whether any pair is orthogonal — whether the minimum inner product is `0`. The Orthogonal Vectors Conjecture says deciding that for `n` boolean vectors in dimension `d ≥ c log n` cannot be done in `O(n^{2-ε})` time. If my sparse network had `l = O(n^{1-ε})` layers, the total would be `Õ(n^{2-ε})`, contradicting the conjecture. So any sparse pattern with `Õ(n)` edges needs `Ω̃(n^{1-o(1)})` layers for this task — there is a genuine cost to sparsity. Good: the mechanism is as expressive in the limit, but not free; on problems that truly need all-pairs comparisons, depth pays for the missing width.

One more guarantee to chase down: Turing completeness. The encoder result above is about a single forward pass; the stronger claim is that an encoder–decoder can simulate any Turing machine. Pérez et al. (2019) showed this for full attention under arbitrary precision (without unbounded precision a transformer is just a finite-state machine, so that assumption is unavoidable). Their decoder keeps the machine's state `q` and head symbol `s` in the sequence, and at each step uses *full* attention to do one tricky retrieval: find the symbol last written at the cell the head is about to move to. That's an argmin over the entire history in one shot — exactly the all-pairs move my sparse decoder can't do directly. Same wall as before.

But the wall has the same kind of crack. An argmin over `{0,…,t}` doesn't have to be computed in one attention; min is associative, so `min_{0≤m≤t} χ_m = min(…min(min(χ_0, χ_1), χ_2)…, χ_t)`. I can fold the min one step at a time. So I let one Turing-machine step occupy *several* transformer decoder steps: a "compute" step that actually advances the machine, plus a run of "intermediate" steps that each absorb one more term into the running min. I lay out a causal sparse decoder graph where node `j(j+1)/2 + k` attends back to compute node `k(k+1)/2` (the candidate term) and also has the local/self edge needed to carry the current accumulator; at the first position in a block that local source is the previous compute node, and after that the accumulator is already stored at the current intermediate position. Walking these intermediate nodes accumulates the min over past machine steps `0,1,…` correctly, and by the time I reach the next compute node I have the full argmin — the last-written symbol — without any single node attending to all of history. The map from transformer step `i` to machine step is `g(i) = ⌊(−1 + √(1+8i))/2⌋`, since `k` runs `1..j+1` inside block `j`; roughly `O(√i)` intermediate steps per machine step. To keep the machine state frozen across the intermediate steps while the min aggregates, I add one extra decoder layer — a switching layer keyed on the indicator `h(i) = g(i+1) − g(i)` (one at a compute step, zero at an intermediate step) that either advances the state or copies the previous one. The rest of the construction — cross-attending to the encoder to fire the transition `δ(q,s) → (q',v,m)`, updating the head position `c ← c + m`, writing the new symbol — carries over from the full-attention proof essentially unchanged. The conclusion: a sparse decoder using `O(n)` inner products per step is still Turing complete. So both theorems survive the sparsification, which is the whole reason the global token earned its place.

Now I have to make this run fast, and here's the practical wall: even though my attention matrix is sparse, GPUs and TPUs are terrible at fine-grained sparse matrix multiplication. They want big coalesced loads of contiguous memory, and a sliding window or a random-key lookup is a scatter of tiny reads. A literally sparse adjacency matrix would not actually speed anything up. So I can't implement the graph arc-by-arc; I have to reshape it into dense block operations.

The fix is to blockify. Pick a block size `b` and group the tokens into `⌈n/b⌉` contiguous blocks. Define the whole pattern at the granularity of blocks: a query *block* attends to a window of neighbor key *blocks*, to a few random key *blocks*, and to the global *blocks*. Reshape `Q` and `K` to `⌈n/b⌉ × b × d`. The block-diagonal scores — each query block against the *same-index* key block — are one batched matmul, `A_{jst} = Σ_u Q'_{jsu} K'_{jtu}`, at cost `O(nbd)`. To get the window, I don't gather: I make `w` copies of the blocked key tensor and *roll* the `j`-th copy by `j` blocks along the block axis, so stacking them lines up, for every query block, its `w` neighbor key blocks — pure reshape and shift, no scatter. The global component is even simpler: just always concatenate the first (and last) key blocks, since they're global, fixed for every query block. Only the random component genuinely needs a gather — but `r` is tiny (a handful of blocks), so that cost is negligible. Stack global, window, and random key blocks into one compact tensor `K''` of size `⌈n/b⌉ × (g+w+r)b × d`, and the final scores are one dense product `Q' · K''` at cost `O(n(g+w+r)bd)` — dense, contiguous, exactly what the hardware likes. Scatter the result back into the sparse pattern and finish the softmax and the value aggregation the same way.

So the code splits the query blocks into a few cases. The first block is global — it attends to the whole sequence, so I just do a dense row of attention for it; same for the last block. Every middle query block gets the packed treatment: build its key set as [global end-blocks] ++ [its three window neighbor blocks, via stacking the rolled copies] ++ [its `r` gathered random blocks], one softmax, one weighted sum of the corresponding values. The random-block plan per query block is drawn once, excluding the global blocks, the block's own window neighbors, and itself, so the random arcs add genuinely new reach rather than duplicating what window and global already cover. Concretely:

```python
import math
import numpy as np
import torch
import torch.nn as nn


class BigBirdBlockSparseAttention(nn.Module):
    # attention = window + random + global, with the same five block cases
    # used by the block-sparse implementation.
    def __init__(self, hidden_size, num_heads, block_size=64,
                 num_random_blocks=3, seed=None):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.seed = seed
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x):
        b, n, _ = x.size()
        return x.view(b, n, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    @staticmethod
    def _matmul(a, b):
        return torch.matmul(a, b)

    @staticmethod
    def _matmul_t(a, b):
        return torch.matmul(a, b.transpose(-1, -2))

    def _rand_blocks(self, num_blocks):
        # One random plan for every non-global query block. Boundary query
        # blocks need the same extra exclusions used in the model code so the
        # random blocks do not duplicate their special window/global blocks.
        rng = np.random.RandomState(self.seed)
        r = self.num_random_blocks
        plan = np.zeros((num_blocks - 2, r), dtype=np.int64)
        for block in range(1, num_blocks - 1):
            forbidden = {0, num_blocks - 1, block - 1, block, block + 1}
            if block == 1:
                forbidden.add(num_blocks - 2)
            if block == num_blocks - 2:
                forbidden.add(1)
            choices = [k for k in range(1, num_blocks - 1) if k not in forbidden]
            if len(choices) < r:
                raise ValueError("sequence has too few blocks for this random plan")
            plan[block - 1] = rng.permutation(choices)[:r]
        return torch.tensor(plan, dtype=torch.long)

    def _gather_random(self, blocked, plan):
        b, h, nb, B, d = blocked.shape
        r = plan.shape[-1]
        idx = plan.to(blocked.device).view(1, 1, nb - 2, r, 1, 1)
        idx = idx.expand(b, h, nb - 2, r, B, d)
        src = blocked.unsqueeze(2).expand(b, h, nb - 2, nb, B, d)
        return torch.gather(src, 3, idx).reshape(b, h, nb - 2, r * B, d)

    def _attend(self, q, k, v, scale):
        scores = self._matmul_t(q, k) * scale
        return self._matmul(torch.softmax(scores, dim=-1), v)

    def forward(self, hidden_states):
        b, n, _ = hidden_states.size()
        B = self.block_size
        assert n % B == 0
        nb = n // B
        assert nb >= 5
        scale = 1.0 / math.sqrt(self.head_size)

        q = self._split_heads(self.query(hidden_states))
        k = self._split_heads(self.key(hidden_states))
        v = self._split_heads(self.value(hidden_states))
        h, d = self.num_heads, self.head_size
        q_blk = q.view(b, h, nb, B, d)
        k_blk = k.view(b, h, nb, B, d)
        v_blk = v.view(b, h, nb, B, d)
        plan = self._rand_blocks(nb)
        rand_k = self._gather_random(k_blk, plan)
        rand_v = self._gather_random(v_blk, plan)

        # 1st and last query blocks are global rows: they attend to all keys.
        first = self._attend(q_blk[:, :, 0], k, v, scale).unsqueeze(2)
        last = self._attend(q_blk[:, :, -1], k, v, scale).unsqueeze(2)

        # 2nd block: [first global, local blocks 1 and 2, last global, random].
        second_k = torch.cat(
            [k_blk[:, :, 0], k_blk[:, :, 1], k_blk[:, :, 2],
             k_blk[:, :, -1], rand_k[:, :, 0]],
            dim=2,
        )
        second_v = torch.cat(
            [v_blk[:, :, 0], v_blk[:, :, 1], v_blk[:, :, 2],
             v_blk[:, :, -1], rand_v[:, :, 0]],
            dim=2,
        )
        second = self._attend(q_blk[:, :, 1], second_k, second_v, scale).unsqueeze(2)

        # True middle blocks: first global + three-block window + random + last global.
        mid_q = q_blk[:, :, 2:-2]
        band_k = torch.cat([k_blk[:, :, 1:-3], k_blk[:, :, 2:-2], k_blk[:, :, 3:-1]], dim=3)
        band_v = torch.cat([v_blk[:, :, 1:-3], v_blk[:, :, 2:-2], v_blk[:, :, 3:-1]], dim=3)
        first_g_k = k_blk[:, :, 0].unsqueeze(2).expand(b, h, nb - 4, B, d)
        first_g_v = v_blk[:, :, 0].unsqueeze(2).expand(b, h, nb - 4, B, d)
        last_g_k = k_blk[:, :, -1].unsqueeze(2).expand(b, h, nb - 4, B, d)
        last_g_v = v_blk[:, :, -1].unsqueeze(2).expand(b, h, nb - 4, B, d)
        mid_k = torch.cat([first_g_k, band_k, rand_k[:, :, 1:-1], last_g_k], dim=3)
        mid_v = torch.cat([first_g_v, band_v, rand_v[:, :, 1:-1], last_g_v], dim=3)
        middle = self._attend(mid_q, mid_k, mid_v, scale)

        # 2nd-last block mirrors the 2nd block near the right boundary.
        second_last_k = torch.cat(
            [k_blk[:, :, 0], k_blk[:, :, -3], k_blk[:, :, -2],
             k_blk[:, :, -1], rand_k[:, :, -1]],
            dim=2,
        )
        second_last_v = torch.cat(
            [v_blk[:, :, 0], v_blk[:, :, -3], v_blk[:, :, -2],
             v_blk[:, :, -1], rand_v[:, :, -1]],
            dim=2,
        )
        second_last = self._attend(
            q_blk[:, :, -2], second_last_k, second_last_v, scale
        ).unsqueeze(2)

        out = torch.cat([first, second, middle, second_last, last], dim=2)
        out = out.reshape(b, h, n, d)
        return out.permute(0, 2, 1, 3).reshape(b, n, h * d)
```

Stepping back through the chain: attention is a fully-connected graph and that's exactly why it's `O(n²)`; reading it as a graph turns "be cheaper" into "delete arcs." Random arcs buy short paths and rapid mixing (an expander, like an Erdős–Rényi graph); window arcs buy locality and clustering (the ring-lattice / small-world structure); together they're `O(n)` but, both in the experiment and in the universal-approximation proof, they fall short — the contextual-mapping construction stalls because no sparse node can see the whole sequence. The fix that both the data and the theorem point to is a global token wired to everyone, planting the star graph that the proof needs; with it the mechanism is again a universal approximator, and by folding the decoder's one-shot retrieval into an associative running-min over extra steps it stays Turing complete — while a furthest-vector task and the Orthogonal Vectors Conjecture show sparsity is not free, costing extra depth on genuinely all-pairs problems. Finally, because hardware hates fine-grained sparsity, the graph is realized in blocks: blockify, roll for the window, fix the global blocks, gather the few random ones, and the whole sparse pattern collapses into dense block matmuls of cost `O(n(g+w+r)bd)`.
