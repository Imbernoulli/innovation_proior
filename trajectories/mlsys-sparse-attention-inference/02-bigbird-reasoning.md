The oracle gave me the three numbers I now have to chase down under a quarter of the attention matrix,
and they split exactly the way I feared. NIAH at 8K came in at `1.0` — full attention retrieves the
needle perfectly, because every query can route directly to the one key that holds it. The two QA tasks
came in modest: Qasper F1 `0.1406`, MultiFieldQA-EN F1 `0.3447`. So the ceiling is not "perfect
everywhere" — the QA scores are capped by the 1.5B model's own competence, while NIAH is a clean
position-coverage test that full attention aces. Let me read those numbers as a budget for how much I can
afford to lose. NIAH has `0.8` of headroom to fall before it hits the `0.2` I would expect from a
retrieval task where the model, missing the needle, is left guessing among a few answer forms — so the
NIAH gap is enormous and a mask that misses the needle will show it starkly. The QA numbers are the
opposite: `0.1406` and `0.3447` are already low, so there is little absolute F1 to lose, and whatever I
drop I drop off a shallow base. That asymmetry tells me where the danger is. On QA the evidence is spread
across the document, so a sparse mask that covers *some* of it can still answer partially and loses F1
gracefully; on NIAH the answer lives in one position, so a mask that fails to cover the needle's block
falls off a cliff from `1.0` toward chance. Whatever I build first, NIAH is the discriminator that will
punish me hardest, and a content-blind pattern is exactly the kind of thing that can miss the needle by
construction.

So the design problem is sharp: out of the full causal lower triangle of `(q, k)` pairs, keep at most a
fraction `0.25` (the harness aborts above `0.25 + 0.02` averaged over the 24 layers), with no retraining
and no access to anything but the `q, k, v` in front of me this forward — the loop runs `use_cache=False`,
so there is no accumulating decode cache to lean on. The cleanest place to start is a *static* pattern: a
mask that depends only on `N`, not on the content, so I can build it once per prompt, cache it across all
24 layers, and pay nothing per layer to decide it. Starting static is not timidity; it is the right
control. If a well-constructed static pattern already recovers the oracle's NIAH, then adaptivity is
unnecessary and I have saved myself the complexity; if it collapses NIAH exactly where the position-blind
structure predicts, I will have *measured* the cost of staticness rather than assumed it, which is the
evidence that justifies going adaptive later. The question this rung answers is which static pattern spends
the 25% budget best.

Three static patterns are genuinely on the table, and I want to weigh them before committing. The first is
to pour the whole budget into the widest possible sliding window — no anchors, no random, just the last
`~2000` keys per query, since `0.25·8000 ≈ 2000`. It is maximally local and captures the recent context the
model leans on hardest, but I will show in a moment that it strands anything beyond a couple thousand
tokens, so it cannot retrieve a mid-haystack needle and it drops distant QA evidence entirely. The second
is a window plus a few always-on anchors at the start — cheaper to reason about, but it only adds coverage
of the *first* few tokens, which does nothing for evidence in the middle. The third is to hold back some of
the window budget and spend it on random long-range links plus global anchors, the pattern whose graph is a
provable approximation to the complete one. At this point in the ladder the graph argument is the strongest
reason I have to believe *any* static pattern can approximate dense, and it says plainly that reachability
requires long-range edges, so I take the third option seriously and reason it out fully rather than
defaulting to the comfortable local window. Whether that theoretical reachability actually cashes out into
retrieval at inference is exactly what the measured numbers will decide.

Let me reason about attention as a graph, because that reframes "spend the budget" as "which edges to
keep." Put the `N` token positions as vertices; an edge `i → j` means query `i` attends to key `j`.
Full attention is the complete causal graph, and that is why the oracle nailed NIAH — there is a direct
edge from every query to the needle. Sparsifying means deleting most edges and asking whether the graph
still behaves like the complete one. Two properties of the complete graph are load-bearing. First, short
paths: information has to be able to *reach* from any position to any other in few hops, or a fact at
position 200 can never influence the representation at position 7000 within the model's depth. Second,
locality: when people probe trained attention the dominant weight lands on *nearby* positions — a token's
meaning is largely set by its neighbors. A good static pattern has to supply both, and I want to check the
"few hops" property against the actual depth of this model rather than wave at it, because the depth is a
hard number: 24 layers.

The window edges supply locality directly: arrange the tokens on a line and connect each query block to a
band of neighbor blocks. That is a ring-lattice — high clustering, lots of overlapping local neighborhoods
— and at the block level it costs a constant number of blocks per query block. But a pure window has
terrible paths, and here is where the depth number bites. If the window is a band of radius one block,
then in a single attention layer a query's representation can absorb information from one block on either
side — 64 tokens each way. Stacking layers grows that receptive field linearly: after `L` layers the
representation at a position depends on tokens within `L` blocks, so after all 24 layers the reach is
`±24` blocks, about `±1536` tokens. A needle planted 4000 tokens before the query is simply outside that
multi-hop receptive field — no path of length ≤ 24 in a pure-window graph connects them — so with a window
alone the far needle is unreachable *in principle*, not merely unlikely. Random edges fix exactly that:
give each query block a few randomly chosen non-window blocks, and the graph becomes an expander —
shortest paths drop to `O(log N)`, a large spectral gap, rapid mixing. This is the Watts–Strogatz
small-world recipe: a local ring plus a sprinkle of long-range shortcuts collapses the diameter from
linear to logarithmic while keeping the clustering. With `log₂(128) ≈ 7`, a handful of random arcs per
block is enough to put every position within a few hops of every other, comfortably inside the 24-layer
depth. There is a spectral way to say the same thing that makes the requirement precise: a pure ring
lattice has a tiny spectral gap, so information mixes across it only as fast as a random walk crawls the
ring — mixing time on the order of the diameter, `~N/window` — whereas adding random edges lifts the
spectral gap toward a constant, and the mixing time drops to `O(log N)`. The number of layers is the
budget of mixing steps I get, so I need the graph's mixing time to be below 24; the ring alone blows
through that budget by an order of magnitude and the expander lands well under it. Window plus random,
then, gives me both properties — locality and reachability — at linear cost, and it does so for a reason I
can point at rather than by analogy.

But window plus random alone is known to fall short of dense, and I can see why from the contextual side,
not just the graph side. There are tasks where some position has to corral information about the *whole*
sequence at once: a global summary, a CLS-like aggregation, the universal-approximation construction that
needs a single node seeing every column to mint a context code. With only window and random edges every
query's neighborhood is a tiny handful of blocks and no single node sees the whole sequence in one hop, so
that construction stalls. The fix is a *global* token — a position wired to everyone and attended by
everyone — which plants a star inside the window+random graph and recovers the missing reach. So the third
ingredient is global anchors, and window + random + global is the pattern whose graph is provably a good
approximation to the complete one at linear density.

Now I have to re-express all three in *this* task's vocabulary, and here is where this rung is not the
generic trained-from-scratch version. The full global+window+random attention learns its own
`W_Q, W_K, W_V` projections, can append dedicated CLS-like global tokens, and tiles into block matmuls
that physically skip the dropped blocks for a real speedup. None of that is available here. The model is
frozen at inference — I have no projections to learn, I receive `q, k, v` already projected and RoPE'd and
GQA-replicated to 12 heads. I cannot *append* extended global tokens, because the harness hands me a fixed
`(B, H, N, D)` and reads density over the existing `N(N+1)/2` causal pairs; growing `N` would break the
contract. And I cannot write a block-sparse kernel — no Triton — so the implementation is a masked-softmax
over the full logits, which means the realized wall-clock speedup is not the point; faithful *behavior*
under the budget is. So the three ingredients have to be installed as *internal* roles on the existing
tokens: global = promote the first couple of token-blocks to attend-all / be-attended-by-all, which is the
natural sink position under causal masking since the earliest tokens are the only ones visible to every
later query; window = a band of neighbor blocks; random = a fixed per-block sample of the rest.

Pick the block size to match the harness: `BLOCK = 64`, so at `N = 8K` there are 128 blocks. Now budget
the blocks, and this is arithmetic I have to get exactly right because the harness aborts on density, not
on intent. Global takes the first 2 blocks (128 tokens) as dual-role anchors. Window takes a band of 3
blocks around each query block, `|bi − bj| ≤ 1` (radius one, which is the `w // 2` above). That leaves the
random count. The naive sizing is `random = round(0.25 · 128) − global − window`: with `round(0.25·128) =
32`, that would be `32 − 2 − 3 = 27` random blocks, and 32 kept blocks per query is a flat block density of
`32/128 = 0.25` — right at the ceiling before any causal or boundary effect. That is too close. The
measured *token* density will not equal the flat block fraction, and it tends to land *above* it for two
interacting reasons. The global *rows* — the first 2 blocks attending to everyone — add pairs a flat
per-query estimate never counts; and the random sample interacts with the causal AND and with the
block→token expansion of the half-filled diagonal in ways the clean `fraction × n_blocks` figure does not
capture, so the realized fraction over the causal triangle drifts off the block algebra at the boundary. I
cannot predict the drift precisely from a hand count — my back-of-envelope for the causal token density of
a 28-block pattern lands somewhere around `0.22–0.24`, but that estimate systematically *under*-reads the
realization because it treats every row as keeping its full quota when the boundary rows and the diagonal
expansion do not behave that cleanly. Since I would rather leave a little budget on the table than have one
layer cross `0.27` and abort the whole run for nothing, I size against a conservatively discounted budget,
`target = round(0.25 · 0.88 · n_blocks)`, an `~12%` margin. At 128 blocks that is `round(28.16) = 28`, so
`random = target − global − window = 28 − 5 = 23`. Whether even the `0.88` discount is conservative enough
against the causal boundary is genuinely something I cannot settle on paper — the density column will
settle it, and if it grazes `0.25` I will have learned the margin was thin. One structural point sharpens
the stakes: the harness aborts on the *mean* density over the 24 layers, but this is a static pattern, so
every layer sees the identical cached mask and reports the identical density — the 24-layer mean is just
that one mask's value. There is no averaging cushion where a few heavy layers get diluted by light ones;
the single mask has to sit under `0.27` on its own, which is precisely why I would rather discount the
budget than trust an exact size. I should be honest that the `0.88` discount is a deliberate under-spend: I
am leaving roughly `12%` of the budget unused as insurance against an abort I cannot precisely predict, and
that unused budget is attention I could in principle have given to more window or more random reach. It is a
trade — a bit of recovered attention surrendered for safety — and if the density comes back well under
`0.25` it will mean I was over-cautious and could have spent more, while if it grazes `0.25` it will mean the
margin was earned. Either reading is useful, which is another reason to run this rung rather than reason
about it in the abstract.

Before I fix the random sample I should stop and estimate what it buys on NIAH, because that estimate is
the whole reason to be pessimistic here, and it is a computation, not a vibe. The retrieval that matters
happens at the *end* of the prompt — the question tokens, and the answer tokens as they generate, live in
the last block or two. So the query that has to reach the needle is the end block, and whether it covers
the needle's block comes down to that end block's own kept set. Global will not help: the needle is planted
somewhere in the middle, not in the first 2 blocks. Window will not help: the needle is thousands of tokens
before the end, far outside `±1` block. So the only edge that can cover it is one of the end block's 23
random draws, chosen from a pool of roughly `123` blocks. The probability that the needle's single block is
among those 23 is about `23/123 ≈ 0.19`. There is a second route in principle — the expander could carry
the needle to the end over two hops through an intermediate block — but a single needle token propagated
through softmax-averaging attention is diluted at every hop, so the multi-hop route is unreliable and the
direct-coverage `~0.19` is the dominant term. That lands NIAH right around chance, and it is a *structural*
ceiling: no amount of retuning the window or the sink count moves a fixed random sample onto a position it
did not anticipate. This is the concrete form of the cliff I flagged from the oracle's `1.0`.

The random pattern itself has to be *fixed*, not resampled per forward, because the same module instance
replays at every generation step and a drifting mask would make the attention non-deterministic across
steps and could spike density on some forward. So I sample the random blocks once, keyed by
`(n_blocks, device, g, w, r)`, with a deterministic seed derived from `n_blocks` and the process seed,
cache the `(n_blocks, n_blocks)` boolean keep-matrix, and reuse it. For each query block I draw `r = 23`
blocks from the pool that excludes the global and window blocks and the block itself, so the random arcs
add genuinely new reach rather than re-covering what global and window already cover. Then I AND the whole
block-keep matrix with the causal block mask `bi ≥ bj`, expand it to a token-level `(N, N)` mask by mapping
each token to its block, AND again with the strict token-level causal triangle so the diagonal block is
properly lower-triangular, and report `last_density` as the *measured* fraction of the realized token mask
over `N(N+1)/2` — not the block algebra, because the diagonal-block trimming and any final-block padding
make the algebra and the realization disagree exactly at the boundary I just said I cannot predict.

The two-level causal masking is not redundant, and it is worth being explicit about why the token-level
triangle is needed on top of the block-level one. The block causal mask `bi ≥ bj` keeps a query block from
attending to any *strictly later* block, which is correct at block granularity. But the diagonal block
`bi = bj` is kept, and inside it the coarse "keep this block" would let an early query token in the block
attend to a *later* key token in the same block — a token from its own future — because block granularity
cannot see the ordering within a block. That would leak future information and corrupt the causal model. So
after expanding the block-keep to the token grid I AND with the strict token triangle `i ≥ j`, which trims
the kept diagonal block down to its lower-left half. This trimming is also exactly why the realized token
density falls below the block algebra: every kept diagonal block loses roughly half its `64×64` entries,
and that is one of the boundary effects my hand estimate could not price. The seed for the random draw is
derived from `n_blocks` and the process seed rather than left to the ambient RNG so the sampled pattern is
reproducible across the 24 layers and across generation steps; a fresh sample per layer would give each
layer a different graph and could push a single layer's density over the ceiling, and reproducibility is
what lets me cache one `(n_blocks, n_blocks)` matrix and reuse it everywhere.

The attention then is the masked-softmax form: compute `QKᵀ · scale` in float32 (the inputs are fp16/bf16
and I am about to `masked_fill` with `−∞` and softmax, which is far more stable upcast), set the non-kept
entries to `−∞`, softmax along the keys, `nan_to_num` any row that somehow kept nothing, multiply by `V`,
cast back to the input dtype. Float32 here is a quality decision, not a kernel one — the budget cares about
the mask, not the arithmetic precision, and a stable softmax is worth the transient memory, especially
since a masked row that keeps only a few keys can have a large logit gap and I do not want fp16 to mangle
it.

Now the falsifiable part, against the oracle's measured numbers. On the QA tasks, where evidence is
distributed, the window covers the local context and the random expander edges reach scattered spans, so I
expect bigbird to recover a meaningful fraction of the oracle's F1 — somewhere in the band of "most of it,"
because the global+window+random graph is specifically built to keep distributed information reachable, and
the QA base was low to begin with so there is not much to lose. If I had to pin a band: the window plus
`~22%` random block coverage should reach roughly half to two-thirds of the relevant spans on a typical
document, which against the oracle's `0.1406` and `0.3447` would put Qasper somewhere near `0.08–0.11` and
MultiFieldQA near `0.22–0.26`. That is a soft prediction — F1's formatting-cap component means the mapping
from span coverage to score is not linear — but it gives me a falsifiable range, and if the QA numbers land
far below it while the densities are at budget then the random blocks are actively hurting rather than
merely reaching, which would itself be a finding about where the budget is being wasted. On NIAH, the discriminator, I am much more
worried, and the worry is structural rather than a matter of tuning: the needle sits in one block, and
whether bigbird covers it depends entirely on whether that block happens to fall in some query's window, be
one of the 2 global blocks, or be among the 23 *fixed random* blocks that query drew. The window will not
reach it unless the needle is near the query; the global blocks only cover the first 128 tokens; the random
blocks are a fixed sample chosen blind to the question, so on a 128-block haystack a query covers roughly
`28/128 ≈ 22%` of blocks and the needle's block is in that set only by that same luck. So on a needle
planted in the middle of the haystack, bigbird should hit it only occasionally, and I expect NIAH to
collapse from the oracle's `1.0` toward chance — the very cliff the NIAH-vs-QA split warned about.
Concretely, the bar I am clearing against rung 1: stay under the `0.25 + 0.02` density ceiling on every
layer (or the run aborts), recover a usable share of the QA F1 against `0.1406` / `0.3447`, and — this is
the prediction I most expect to *fail* — keep NIAH anywhere near `1.0`. If NIAH craters while the QA F1
holds and the densities sit right at budget, that is not a tuning miss; it is the diagnosis that a static
content-blind mask cannot route to a query-specific position, and it points the next rung at spending the
static budget more honestly before I pay for adaptivity. The distilled module — the literal scaffold fill,
with the deterministic random cache and the `0.88` budget margin — is in the answer.
