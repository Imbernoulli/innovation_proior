The oracle gave me three numbers to chase under a quarter of the attention matrix, and they split exactly
the way I feared. NIAH came in at `1.0` — full attention retrieves the needle perfectly, every query
routing directly to the one key that holds it. The two QA tasks came in modest: Qasper F1 `0.1406`,
MultiFieldQA-EN F1 `0.3447`, capped by the 1.5B model's own competence rather than by attention coverage.
Reading those as a loss budget: NIAH has `0.8` of headroom before it hits the `~0.2` a model scores when,
missing the needle, it guesses among a few answer forms — so the NIAH gap is enormous and a mask that
misses the needle will show it starkly. The QA numbers are already low, so there is little absolute F1 to
lose and whatever I drop comes off a shallow base. That asymmetry tells me where the danger is. On QA the
evidence is spread across the document, so a mask covering *some* of it still answers partially and loses
F1 gracefully; on NIAH the answer lives in one position, so a mask that fails to cover the needle's block
falls off a cliff from `1.0` toward chance. Whatever I build first, NIAH is the discriminator that will
punish me hardest, and a content-blind pattern is exactly the kind of thing that can miss the needle by
construction.

The design problem is sharp: out of the full causal lower triangle keep at most a fraction `0.25` (the
harness aborts above `0.25 + 0.02` averaged over the 24 layers), with no retraining and nothing but the
`q, k, v` in front of me this forward — the loop runs `use_cache=False`, so there is no accumulating decode
cache to lean on. The cleanest place to start is a *static* pattern: a mask depending only on `N`, built
once per prompt, cached across all 24 layers, paying nothing per layer to decide it. Starting static is the
right control, not timidity. If a well-constructed static pattern already recovers the oracle's NIAH, then
adaptivity is unnecessary and I have saved the complexity; if it collapses NIAH exactly where the
position-blind structure predicts, I will have *measured* the cost of staticness rather than assumed it,
which is the evidence that justifies going adaptive later. This rung answers which static pattern spends
the 25% budget best.

Read attention as a graph, because that reframes "spend the budget" as "which edges to keep." Put the `N`
positions as vertices; an edge `i → j` means query `i` attends key `j`. Full attention is the complete
causal graph, and that is why the oracle nailed NIAH — a direct edge from every query to the needle.
Sparsifying deletes most edges and asks whether the graph still behaves like the complete one, which needs
two properties. Locality: when people probe trained attention the dominant weight lands on nearby
positions, so I want a band connecting each query block to neighbor blocks — a ring lattice, high
clustering, a constant number of blocks per query block. And short paths: information has to reach from any
position to any other in few hops, or a fact at position 200 can never influence position 7000 within the
model's depth — and the depth is a hard number, 24 layers. A pure window fails the path property, and here
the depth bites. A radius-one window lets a query's representation absorb one block on either side per
layer — 64 tokens each way — growing linearly, so after all 24 layers the reach is `±24` blocks, about
`±1536` tokens. A needle planted 4000 tokens before the query is outside that multi-hop receptive field —
no path of length ≤ 24 in a pure-window graph connects them — so with a window alone the far needle is
unreachable *in principle*, not merely unlikely. Random edges fix exactly that: give each query block a few
randomly chosen non-window blocks and the graph becomes an expander — shortest paths drop to `O(log N)`,
a large spectral gap, rapid mixing. This is the Watts–Strogatz small-world recipe: a local ring plus a
sprinkle of long-range shortcuts collapses the diameter from linear to logarithmic while keeping the
clustering. With `log₂(128) ≈ 7`, a handful of random arcs per block puts every position within a few hops
of every other, comfortably inside the 24 layers of mixing I get.

Window plus random still falls short of dense, and I can see why from the contextual side. Some tasks need
one position to corral information about the *whole* sequence at once — a global summary, a CLS-like
aggregation, the construction that needs a single node seeing every column to mint a context code. With
only window and random edges no single node sees the whole sequence in one hop, so that stalls. The fix is
a *global* token — wired to everyone and attended by everyone — which plants a star inside the graph. So
window + random + global is the pattern whose graph provably approximates the complete one at linear
density.

Now I have to re-express all three in *this* task's vocabulary, because this is the frozen-inference
version, not the generic trained-from-scratch one. The full global+window+random attention learns its own
`W_Q, W_K, W_V`, appends dedicated CLS-like global tokens, and tiles into block matmuls that physically
skip the dropped blocks. None of that is available. The model is frozen — no projections to learn, and I
receive `q, k, v` already projected, RoPE'd, and GQA-replicated to 12 heads. I cannot *append* global
tokens, because the harness hands me a fixed `(B, H, N, D)` and reads density over the existing `N(N+1)/2`
pairs, so growing `N` breaks the contract. And I cannot write a block-sparse kernel — no Triton — so the
implementation is a masked-softmax over the full logits, which means the realized wall-clock speedup is not
the point; faithful *behavior* under the budget is. So the three ingredients install as *internal* roles on
the existing tokens: global = promote the first couple of blocks to attend-all / be-attended-by-all, the
natural sink position under causal masking since the earliest tokens are the only ones every later query
can see; window = a band of neighbor blocks; random = a fixed per-block sample of the rest.

Pick `BLOCK = 64` to match the harness, so at `N = 8K` there are 128 blocks. Global takes the first 2
blocks (128 tokens) as dual-role anchors; window takes a band of 3 blocks around each query block,
`|bi − bj| ≤ 1`. That leaves the random count, and the sizing has to be exact because the harness aborts on
density. A flat `round(0.25·128) = 32` kept blocks per query is `32/128 = 0.25`, right at the ceiling
before any causal or boundary effect — too close. The realized *token* density drifts *above* the flat
block fraction for two interacting reasons: the global rows — the first 2 blocks attending to everyone —
add pairs a per-query estimate never counts, and the random sample interacts with the causal AND and with
the half-filled diagonal block in ways `fraction × n_blocks` does not capture. My hand estimate of the
causal token density lands somewhere around `0.22–0.24`, but it systematically *under*-reads the
realization because it treats every row as keeping its full quota when the boundary rows and the diagonal
expansion do not. And this is a static pattern, so every one of the 24 layers sees the identical cached
mask and reports the identical density — there is no averaging cushion where heavy layers get diluted, the
single mask has to sit under `0.27` on its own. So I size against a conservatively discounted budget,
`target = round(0.25 · 0.88 · n_blocks)`, an `~12%` margin: `round(28.16) = 28`, so
`random = target − global − window = 28 − 5 = 23`. This deliberately under-spends — I leave roughly `12%`
of the budget unused as insurance against an abort I cannot precisely predict — and the density column will
settle whether the margin was earned or whether I was over-cautious.

Before I fix the random sample I estimate what it buys on NIAH, because that estimate is the reason to be
pessimistic and it is a computation, not a vibe. The retrieval that matters happens at the *end* of the
prompt — the question tokens, and the answer tokens as they generate, live in the last block or two — so
the query that has to reach the needle is the end block, and whether it covers the needle comes down to
that block's own kept set. Global will not help: the needle is planted mid-haystack, not in the first 2
blocks. Window will not help: the needle is thousands of tokens before the end, far outside `±1` block. So
the only edge that can cover it is one of the end block's 23 random draws, chosen from a pool of `~123`
blocks — probability `23/123 ≈ 0.19`. There is a two-hop expander route in principle, but a single needle
token propagated through softmax-averaging is diluted at every hop, so the multi-hop route is unreliable
and the direct-coverage `~0.19` dominates. That lands NIAH near chance, and it is a *structural* ceiling:
no amount of retuning the window or the sink count moves a fixed random sample onto a position it did not
anticipate.

The random pattern must be *fixed*, not resampled per forward, because the same module instance replays at
every generation step and a drifting mask would make attention non-deterministic across steps and could
spike density on some forward. So I sample once, keyed by `(n_blocks, device, g, w, r)` with a
deterministic seed derived from `n_blocks` and the process seed, cache the `(n_blocks, n_blocks)` boolean
keep-matrix, and reuse it; each query block draws `r = 23` blocks from the pool that excludes the global,
window, and its own block so the arcs add genuinely new reach. Then I AND with the block causal mask
`bi ≥ bj`, expand to a token-level `(N, N)` mask by mapping each token to its block, AND again with the
strict token triangle `i ≥ j`, and report `last_density` as the *measured* fraction of the realized token
mask over `N(N+1)/2` — not the block algebra, because the diagonal trimming and any final-block padding
make the two disagree exactly at the boundary I cannot predict. The token triangle is not redundant on top
of the block one: the kept diagonal block `bi = bj` would, at block granularity, let an early query token
attend a *later* key token in the same block — a token from its own future — so the strict AND trims each
kept diagonal block down to its lower-left half, and that trimming (roughly half of each `64×64` diagonal
block) is one of the boundary effects the block algebra cannot price.

The attention is the masked-softmax form: compute `QKᵀ · scale` in float32 (the inputs are fp16/bf16 and I
am about to `masked_fill` with `−∞`, where a masked row keeping only a few keys can have a large logit gap
I do not want fp16 to mangle), set the non-kept entries to `−∞`, softmax along the keys, `nan_to_num` any
row that kept nothing, multiply by `V`, cast back to the input dtype.

Now the falsifiable part, against the oracle's measured numbers. On the QA tasks, where evidence is
distributed, the window covers the local context and the random expander edges reach scattered spans, so I
expect bigbird to recover a meaningful share of the oracle's F1 — the global+window+random graph is built
to keep distributed information reachable, and the QA base was low to begin with. If the QA numbers land
far *below* the oracle while the densities sit at budget, then the random blocks are actively hurting
rather than merely reaching, which would itself be a finding about where the budget is being wasted. On
NIAH, the discriminator, the `~0.19` coverage estimate says it should collapse from `1.0` toward chance —
the cliff the oracle's `1.0` warned about, and a *structural* failure rather than a tuning miss, since the
random blocks are a fixed sample chosen blind to the question. So the bar against rung 1: stay under the
`0.25 + 0.02` density ceiling on every layer, recover a usable share of the QA F1 against `0.1406` /
`0.3447`, and — the prediction I most expect to *fail* — keep NIAH anywhere near `1.0`. If NIAH craters
while the QA F1 holds and the densities sit right at budget, that is not a tuning miss; it is the diagnosis
that a static content-blind mask cannot route to a query-specific position, and it points the next rung at
spending the static budget more honestly before I pay for adaptivity. The distilled module — the scaffold
fill with the deterministic random cache and the `0.88` budget margin — is in the answer.
