The oracle gave me the three numbers I now have to chase down under a quarter of the attention matrix,
and they split exactly the way I feared. NIAH at 8K came in at `1.0` — full attention retrieves the
needle perfectly, because every query can route directly to the one key that holds it. The two QA tasks
came in modest: Qasper F1 `0.1406`, MultiFieldQA-EN F1 `0.3447`. So the ceiling is not "perfect
everywhere" — the QA scores are capped by the 1.5B model's own competence, while NIAH is a clean
position-coverage test that full attention aces. That tells me where the danger is. On QA the evidence is
spread across the document, so a sparse mask that covers *some* of it can still answer partially and lose
F1 gracefully; on NIAH the answer lives in one position, so a mask that fails to cover the needle's block
falls off a cliff from `1.0` to near chance. Whatever I build first, NIAH is the discriminator that will
punish me hardest, and a content-blind pattern is exactly the kind of thing that can miss the needle by
construction.

So the design problem is sharp: out of the full causal lower triangle of `(q, k)` pairs, keep at most a
fraction `0.25` (the harness aborts above `0.25 + 0.02` averaged over the 24 layers), with no retraining
and no access to anything but the `q, k, v` in front of me this forward — the loop runs
`use_cache=False`, so there is no accumulating decode cache to lean on. The cleanest place to start is a
*static* pattern: a mask that depends only on `N`, not on the content, so I can build it once per prompt,
cache it across all 24 layers, and pay nothing per layer to decide it. The question is which static
pattern spends the 25% budget best.

Let me reason about attention as a graph, because that reframes "spend the budget" as "which edges to
keep." Put the `N` token positions as vertices; an edge `i → j` means query `i` attends to key `j`.
Full attention is the complete causal graph, and that is why the oracle nailed NIAH — there is a direct
edge from every query to the needle. Sparsifying means deleting most edges and asking whether the graph
still behaves like the complete one. Two properties of the complete graph are load-bearing. First, short
paths: information has to be able to *reach* from any position to any other in few hops, or a fact at
position 200 can never influence the representation at position 7000 within the model's depth. Second,
locality: when people actually probe trained attention, the dominant weight lands on *nearby* positions
— a token's meaning is largely set by its neighbors. A good static pattern has to supply both.

The window edges supply locality directly: arrange the tokens on a line and connect each query block to
a band of neighbor blocks. That is a ring-lattice — high clustering, lots of overlapping local
neighborhoods — and at the block level it costs a constant number of blocks per query block. But a pure
window has terrible paths: to cross the sequence you crawl block by block, `O(N/window)` hops, far more
than the model's depth, so the needle at the far end is unreachable. Random edges fix exactly that: give
each query block a few randomly chosen non-window blocks, and the graph becomes an expander — shortest
paths drop to `O(log N)`, a large spectral gap, rapid mixing. This is the Watts–Strogatz small-world
recipe: a local ring plus a sprinkle of long-range shortcuts collapses the diameter while keeping the
clustering. Window plus random, then, gives me both properties at linear cost.

But window plus random alone is known to fall short of dense — and I can see why from the contextual
side, not just the graph side. There are tasks where some position has to corral information about the
*whole* sequence at once: a global summary, a CLS-like aggregation, the universal-approximation
construction that needs a single node seeing every column to mint a context code. With only window and
random edges, every query's neighborhood is a tiny handful of blocks and no single node sees the whole
sequence in one hop; the construction stalls. The fix is a *global* token — a position wired to
everyone, attended by everyone — which plants a star inside the window+random graph and recovers the
missing reach. So the third ingredient is global anchors.

Now I have to re-express all three in *this* task's vocabulary, and here is where this rung is not the
generic trained-from-scratch version. The full global+window+random attention learns its own `W_Q, W_K, W_V`
projections, can append dedicated CLS-like global tokens, and tiles into block matmuls that physically
skip the dropped blocks for a real speedup. None of that is available here. The model is frozen at
inference — I have no projections to learn, I receive `q, k, v` already projected and RoPE'd and
GQA-replicated to 12 heads. I cannot *append* extended global tokens, because the harness hands me a
fixed `(B, H, N, D)` and reads density over the existing `N(N+1)/2` causal pairs; growing `N` would
break the contract. And I cannot write a block-sparse kernel — no Triton — so the implementation is a
masked-softmax over the full logits, which means the realized speedup is not the point; faithful
*behavior* under the budget is. So the three ingredients have to be installed as *internal* roles on the
existing tokens: global = promote the first couple of token-blocks to attend-all / be-attended-by-all
(the natural sink position under causal masking, since the earliest tokens are the only ones visible to
every later query); window = a band of neighbor blocks; random = a fixed per-block sample of the rest.

Pick the block size to match the harness: `BLOCK = 64`, so at `N=8K` there are 128 blocks. Budget the
blocks. Global takes the first 2 blocks (128 tokens) as dual-role anchors — every query block attends to
them, and they attend to everyone. Window takes a band of 3 blocks around each query block, `|bi − bj| ≤
1`. That leaves the random count, and this is where I have to be careful about the budget, because the
harness aborts on density, not on my intent. If I naively set `random = round(0.25 · 128) − global −
window`, the measured density overshoots. Why? Two reasons. The global *rows* — the first 2 blocks
attending to everyone — add causal pairs the linear per-query estimate does not count; and the random
sampling plus the causal AND interact so the realized causal-triangle fraction lands above the flat
`fraction × n_blocks` estimate, especially because early query blocks have few admissible blocks and the
fixed window/global already saturate them. So I size the random count against a *conservatively
discounted* budget: `target = round(0.25 · 0.88 · n_blocks)`, an ~12% margin, then `random = target −
global − window`, clamped non-negative and below the admissible pool. The margin is not a borrowed constant;
it is calibrated so the measured density sits clear of the `0.25 + 0.02` ceiling at every context length
the harness evaluates, because a single layer crossing the line aborts the entire run and scores me
nothing.

The random pattern itself has to be *fixed*, not resampled per forward, because the same module instance
replays at every generation step and a drifting mask would make the attention non-deterministic across
steps and could spike density on some forward. So I sample the random blocks once, keyed by
`(n_blocks, device, g, w, r)`, with a deterministic seed derived from `n_blocks` and the process seed,
cache the `(n_blocks, n_blocks)` boolean keep-matrix, and reuse it. For each query block I draw `r`
blocks from the pool that excludes the global and window blocks and the block itself, so the random arcs
add genuinely new reach rather than re-covering what global and window already cover. Then I AND the
whole block-keep matrix with the causal block mask `bi ≥ bj`, expand it to a token-level `(N, N)` mask by
mapping each token to its block, AND again with the strict token-level causal triangle so the diagonal
block is properly lower-triangular, and report `last_density` as the *measured* fraction of the realized
token mask over `N(N+1)/2` — not the block algebra, because the contract wants the true kept fraction and
the diagonal-block trimming and any final-block padding make the algebra and the realization disagree at
the boundary.

The attention then is the masked-softmax form: compute `QKᵀ · scale` in float32 (the inputs are
fp16/bf16 and I am about to `masked_fill` with `−∞` and softmax, which is far more stable upcast), set
the non-kept entries to `−∞`, softmax along the keys, `nan_to_num` any row that somehow kept nothing,
multiply by `V`, cast back to the input dtype. Float32 here is a quality decision, not a kernel one — the
budget cares about the mask, not the arithmetic precision, and stable softmax is worth the transient
memory.

Now the falsifiable part, against the oracle's measured numbers. On the QA tasks, where evidence is
distributed, the window covers the local context and the random expander edges reach scattered spans, so
I expect bigbird to recover a meaningful fraction of the oracle's F1 — somewhere in the band of "most of
it," because the global+window+random graph is specifically built to keep distributed information
reachable. On NIAH, the discriminator, I am much more worried, and the worry is structural: the needle
sits in one block, and whether bigbird covers it depends entirely on whether that block happens to fall
in some query's window, be one of the 2 global blocks, or be among the handful of *fixed random* blocks a
query drew. The window will not reach it unless the needle is near the query; the global blocks only
cover the first 128 tokens; the random blocks are a fixed sample chosen blind to the question. So on a
needle planted in the middle of the haystack, bigbird should hit it only by luck, and I expect NIAH to
collapse from the oracle's `1.0` toward chance — the very cliff the NIAH-vs-QA split warned me about.
Concretely, the bar I am clearing against rung 1: stay under the `0.25 + 0.02` density ceiling on every
layer (or the run aborts), recover a usable share of the QA F1 (oracle `0.1406` / `0.3447`), and — this
is the prediction I most expect to *fail* — keep NIAH anywhere near `1.0`. If NIAH craters while the
window/random densities sit right at budget, that is not a tuning miss; it is the diagnosis that a static
content-blind mask cannot route to a query-specific position, and it points the next rung at a pattern
that at least gets the *anchors and window* right before I make the selection adaptive. The distilled
module — the literal scaffold fill, with the deterministic random cache and the 0.88 budget margin — is
in the answer.
