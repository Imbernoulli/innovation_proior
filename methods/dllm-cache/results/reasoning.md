Let me start from what actually hurts. I have a diffusion language model — a bidirectional Transformer that generates by denoising. I hand it a prompt and a fully masked response of length `L`, and it runs `K` denoising steps: at each step it looks at the whole sequence, predicts a clean token at every position at once, commits a handful of the most confident ones, re-masks the rest, and repeats until nothing is masked. The output is good — bidirectional attention lets it plan holistically, and it doesn't suffer the reversal pathologies of left-to-right models. But it is painfully slow. To emit `L` tokens I run on the order of `L` steps, and *every single step is a full forward pass over the entire prompt-plus-response sequence*, recomputing attention and the feed-forward network in all `T` layers from nothing. Count it: per step the cost is dominated by `T·(8 n d² + 4 n² d + 6 n d m)` for sequence length `n`, hidden `d`, FFN intermediate `m`; multiply by `K ≈ n` steps and I'm at roughly `O(n³)`. An autoregressive model of the same size does the same job at `O(n²)`. That cubic-versus-quadratic gap is the whole problem, and I want to close it without retraining or changing the model's decoded behavior.

The obvious thing to reach for is the trick autoregressive models use to get their `O(n²)`: the Key-Value cache. Under a causal mask each token only ever attends to earlier positions, so once I've processed a token its Key and Value are *fixed* for the rest of generation — I stash them in a buffer, and at each new step I only project the one new position and attend it against the buffer. Beautiful, and exactly the redundancy-killer I want. So can I just bolt a KV cache onto the denoiser? Let me try. I process the prompt and the masked response at step `K`, cache everyone's K and V, and at step `K-1` I reuse them and only recompute... what, exactly? And here it falls apart. There is no causal mask. Every token attends to every other token, masked ones included. When the model unmasks a position between step `K` and step `K-1`, the context every *other* token sees has changed, so the Key and Value that a previously-committed token presents are *not* the same as last step — they drift as the surrounding sequence fills in. The K/V a position shows at step `m` simply aren't the K/V it shows at step `n`. And there's a second wall: the decoding order isn't fixed. In an autoregressive model I always know the next position is `i+1`, so I know whose states to compute. Here the transition rule picks whichever masked positions are most confident, anywhere in the response — I can't even pre-decide which token's states to refresh. The two assumptions the AR cache rests on, fixed states and fixed order, are both false here. So the exact-cache route is dead. Wall.

Fine — exact reuse is impossible, but maybe *approximate* reuse isn't. The AR cache is correct because the cached states are provably identical to recomputing them. Let me drop the demand for "identical" and use the adjacent-step diagnostic that actually matters: for each token, compare its feature at the current denoising step with its feature at the previous step — Key, Value, attention output, FFN output — by cosine similarity. If that similarity is high, a slightly stale copy is plausible; if it is low, reusing the cache is dangerous.

The pattern splits into two parts. First, the **prompt** region is uniformly stable, with adjacent-step similarity near one across its features. That makes sense the moment I say it out loud — the prompt tokens are never masked, never change, they're literally fixed input; the only reason their internal features move at all is that the *response* around them is filling in, and that's a slow, second-order drift. So the prompt's features are quasi-static across many steps, not just adjacent ones. Second, the **response** region is *not* uniform. Most response tokens are also highly similar to their previous version step-to-step — but a small minority are markedly dissimilar. At any given step only a handful of response positions have features that genuinely moved; the rest are nearly frozen. So the redundancy is real but *heterogeneous*: the prompt is redundant in bulk over long stretches, and the response is mostly redundant with a sparse set of exceptions.

This immediately tells me why the naive thing won't work, and it's worth being precise about the naive thing because it's the strawman I have to beat. The naive feature-cache, the one that works for image diffusion transformers, is: at the first step of a cache period compute *all* layer features and store them, then for the next several steps skip all that computation and just reuse the stored features wholesale; refresh every `N` steps. It buys roughly `N−1×` speedup and it's lossless-ish only while `N` is small, because the longer you coast on a frozen cache the further it drifts from the truth. Now apply one uniform `N` to my dLLM. If I pick `N` large enough to actually accelerate, I'll be reusing stale features for the few response tokens that *did* move — corrupting exactly the positions that carry the new information this step. If I pick `N` small enough to keep those movers fresh, I'm refreshing everything constantly, including the rock-stable prompt and the frozen majority of the response, and I save almost nothing. A single global interval is structurally wrong here for the same reason a single global learning rate is wrong across layers of different scale: the thing it's controlling has wildly different values in different regions. The fix has to be *differentiated* — treat the prompt and the response on different schedules, and within the response, treat the movers and the stayers differently.

So let me design the prompt side first, because it's the easy half. The prompt features are quasi-static, so I cache them and refresh them only on a *long* interval `K_p` — say recompute the prompt's Key, Value, attention output, and FFN output once, then reuse those cached copies for the next `K_p−1` steps, recompute, reuse again. Because the prompt is genuinely stable, `K_p` can be large — on the order of a hundred steps — and the error stays negligible. And the saving is exactly the cost of reprocessing the prompt, which I was paying `K` times for no reason; now I pay it `K/K_p` times. For a long prompt that alone is enormous, because the prompt can dominate the sequence length. Hold the prompt side at interval `K_p`.

Now the response, which is where the actual difficulty lives. The response evolves, so I can't just freeze it for a hundred steps. But most of it barely moves between adjacent steps, with a sparse set of movers. The structure I want is: refresh the *entire* response cache on a *short* interval `K_r` — a handful of steps, not a hundred — to bound how stale anything gets; and *between* those full refreshes, don't recompute the whole response, only recompute the small set of tokens that actually changed, and reuse the cache for everyone else. Two intervals, `K_p ≫ K_r`, and a sparse partial-update mode for the in-between steps. That's the skeleton. The cache itself, per layer `l`, holds the four intermediate features — `K`, `V`, attention output, FFN output — split into a prompt cache `C_p` and a response cache `C_r`. Why cache all four and not just K/V the way an AR cache does? Because the cost I'm trying to skip is the *whole* per-layer recompute — the attention output and the FFN are the heavy parts, the FFN especially with its `6 n d m`. If I only cached K and V I'd still have to run attention and the FFN over the full sequence every step; caching the attention output and FFN output too lets me skip those entirely for the tokens I'm reusing. So: four features, two caches, two intervals.

The hard part is the partial update. On an in-between step I need to recompute the few response tokens that moved and reuse the rest — but *which* tokens moved? I only know that after the fact: a token "moved" if its fresh attention/FFN output differs from the cached one. But computing the fresh attention output to find out it changed is exactly the expensive thing I'm trying to avoid — that's circular. I need a *cheap proxy*, computed before the expensive part, that predicts which tokens' expensive features changed. So I want some early, cheap feature whose adjacent-step change correlates with the adjacent-step change of the attention and FFN outputs.

The cheapest features in a Transformer block are the projections: I take the layer input, normalize it, and apply the linear `Q`, `K`, `V` projections — that's it, three matmuls, `O(n d²)`, no `n²` attention, no FFN. Of these, the candidate I'd reach for first is the Value, because the Value is *what the attention reads out*: a position's attention output is a weighted sum of Values across the sequence, so if a token's own Value barely changed and the sequence around it barely changed, the intuition says its attention output can't have changed much either, and the FFN sitting on top of a stable attention output should be stable too. That's the story. But I've been burned by plausible stories about attention before, so before I commit the whole method to it I want to actually watch the correlation happen on a concrete block rather than assert it.

Let me build the smallest honest test. Take `r = 12` response tokens of width `d`, fixed `Q/K/V` projections, run one softmax attention. Now mimic "between two denoising steps, a couple of tokens moved": perturb the *Value* of tokens 3 and 7 hard and leave the other ten nearly fixed, recompute the attention output, and measure per-token cosine similarity of old vs new for both `V` and the attention output. If the proxy is good, the two low-`V`-sim tokens should also be the two low-attention-sim tokens. Running it, the `V`-sims come out as expected — tokens 3 and 7 drop to ~0.47 and ~0.86 while the rest sit at 1.0 — but the attention-sims are a mess: token 3, the one whose Value I hammered, has attention-sim **0.983**, basically unmoved, while token 7 drops to 0.49. The Spearman between V-sim and attention-sim across the twelve tokens is **0.16**. That is not a usable proxy. The bottom-3-by-V-sim are tokens `{3,7,1}` but the bottom-3-by-attention-sim are `{7,1,0}` — token 3 is the worst offender by Value and yet its *own* attention output barely twitched.

Once I see it the reason is obvious, and it's a real correction, not a detail. In that test I perturbed a token's Value *in isolation*. But a token's attention output is the sum of *everyone's* Values weighted by *its* query — changing token 3's own Value mostly changes what *other* tokens read from token 3, not what token 3 reads from the sequence. A single token's Value change gets diluted across all the queries that look at it. So the isolated-V perturbation is the wrong model of the denoiser. In the actual rollout a token doesn't get its Value surgically edited — its whole *input representation* shifts when the surrounding sequence unmasks, and that one shift drives its query, its key, its value, and the FFN on top, all together. Let me redo the test that way: perturb the layer *input* of a sparse set of tokens (4, 9, 13) and leave the rest nearly fixed, then push everything through `V`-proj, attention, and a two-layer FFN, and measure V-sim, attention-sim, FFN-sim per token. Now it holds: the three moved tokens have the three lowest V-sims (0.69, 0.56, 0.62), and the Spearman of V-sim against attention-sim is **0.62** and against FFN-sim **0.65**. Bottom-3-by-V-sim is exactly `{4,9,13}` — the tokens I moved — and bottom-3-by-attention-sim and by-FFN-sim are `{4,13,15}`. So V-verify catches tokens 4 and 13 cleanly and flags 9 (which attention partially dilutes), and it misses token 15, which moved downstream a bit without a large Value change. Not a perfect selector — but it reliably catches the dominant movers, which is what I need, and the miss tells me something useful: I'll want a periodic full refresh as a backstop for exactly the tokens the Value proxy under-weights. So the proxy is: project the response Values cheaply, compare to the cached Values, recompute the tokens with the *lowest* Value similarity, and reuse the rest. Call it V-verify.

Let me pin down why "lowest similarity" is the right selection and not the other way round, because it's the crux. Reusing a token whose feature didn't move is nearly free of error — the cache *is* almost the right answer. Reusing a token whose feature moved a lot means injecting a stale, wrong feature, and that's where error comes from. So I want to *recompute the movers and reuse the stayers*: pick the tokens with the largest change — the lowest cosine similarity to cache — and refresh exactly those. Concretely, with an update ratio `ρ ∈ [0,1]`, I select the `⌊ρ·|y|⌋` response tokens with the lowest `V`-similarity, recompute their full features, scatter them back into the cache, and leave the other `(1−ρ)` fraction reading from cache. With `ρ` around a quarter I touch a quarter of the response and reuse three-quarters, but the quarter I touch is precisely the quarter that mattered.

Why cosine similarity and not, say, Euclidean distance on the Values? Because I care about *directional/semantic* change, and the Value vectors live at very different magnitudes across tokens and dimensions; an L2 distance would conflate "this token's representation rotated to mean something new" with "this token's representation got bigger," and it would let a few large-magnitude tokens dominate the selection. Cosine normalizes that away — it asks purely whether the direction changed — so cosine on the Values.

Now there's a subtlety in *how* I update the cached Values, and it's a small free win I almost miss. To run V-verify I have to compute the fresh Value for *every* response token — that's the whole point, I score all of them to find the bottom `ρ`. So after scoring, I'm holding a complete, fresh `V_response` in hand. The selected `ρ` tokens definitely get their `K`, attention output, and FFN output recomputed and scattered in. But what about the cached *Values* of the unselected `(1−ρ)` tokens? I already computed their fresh Values — they were free, a byproduct of scoring. So I should overwrite the *entire* cached `V_response` with the fresh one, not just the selected rows. It costs nothing extra (I have them), and it means next step's similarity check compares against a fully up-to-date Value baseline rather than a partly-stale one, which keeps the proxy honest. For `K`, the attention output, and the FFN output, though, I only recompute and scatter the selected rows — those *are* the expensive features, and computing them for the unselected tokens would defeat the purpose. So the asymmetry is deliberate: Values fully overwritten for the whole response because they are already available, everything else selectively scattered because it is expensive.

Let me also settle the selection granularity. The cache is stored per layer, and the V-verify score is computed per layer from that layer's Values, so the natural thing is to select tokens independently at each layer `l`. Is that wasteful — should I select once and apply the same set everywhere? It's only wasteful if the per-layer selection costs me, and it only costs me if the selected set differs a lot across layers. If the layer map and the Value projection are Lipschitz, then a token's V-verify score at layer `l` and at layer `l+1` differ by at most some small `ε`. As long as `ε` is smaller than the margin `Δ` between the `⌊ρ|y|⌋`-th and the `(⌊ρ|y|⌋+1)`-th scores at the selection boundary, the bottom-`ρ` membership carries from one layer to the next; it only flips for tokens bunched right at the cutoff. I haven't measured `ε` against `Δ` on a real run, so I won't claim the sets are identical — but the argument says they should overlap heavily, which means per-layer selection picks nearly the same tokens as a shared selection would and therefore costs almost nothing extra, while staying consistent with the fact that the cache itself is stored per layer. So I'll keep selection local. And I'll keep `ρ` the same at every layer rather than spending more updates in some layers than others — when I weigh giving lower layers a bigger budget versus upper layers, at a fixed total budget a flat `ρ` is the cleanest; lower-layer updates do propagate upward through the stack, but not strongly enough to justify a depth-dependent ratio under the same overall budget.

So now let me assemble the per-step, per-layer logic. At the first step `k=K` I have nothing cached, so I compute everything and initialize both caches, splitting each feature into its prompt part and response part. For every subsequent step, at each layer I check two booleans: is this a prompt-refresh step (`k` hits the `K_p` interval) and is this a response-refresh step (`k` hits the `K_r` interval)? That gives four cases. If both fire, full refresh — recompute prompt and response, restore both caches; this is the periodic from-scratch step that resets accumulated error. If only the response interval fires, recompute the whole response (all of it, fresh `K`/`V`/attn/FFN), reuse the cached prompt. If only the prompt interval fires, recompute the prompt and do the *adaptive partial update* on the response. And if neither fires — which is the vast majority of steps — reuse the cached prompt and do the adaptive partial update on the response. The adaptive partial update is the V-verify path: cheaply project all response Values, score against cache by cosine, pick the bottom-`ρ`, recompute and scatter `K`/attn/FFN for just those, overwrite all of `V_response`, reuse the rest. (One implementation nicety: I always recompute the very first layer fully, regardless of intervals — it's cheap relative to the stack and it keeps the inputs feeding the cached deeper layers honest.)

Let me make sure the attention bookkeeping in the partial-update case is actually correct, because mixing cached and fresh tokens in a bidirectional attention is where I could silently get it wrong. When I recompute the selected response tokens, their attention output is `softmax(q_selected · Kᵀ) · V` — and the `K` and `V` here must be the *full-sequence* keys and values: the cached prompt `K_p/V_p` concatenated with the response `K_r/V_r`, where `V_r` is the freshly overwritten Values and `K_r` has the selected rows updated. So the selected query tokens attend over everyone, with the freshest available K/V — that's right, a bidirectional attention needs the whole key/value set even when only a few queries are live. Then I take the resulting attention outputs for the selected tokens and scatter them into the cached response attention-output tensor at the selected indices, leaving the unselected rows as their cached values; the layer's attention output is then the cached prompt attention output concatenated with this updated response attention-output tensor. The FFN follows the same pattern on the post-attention residual: gather the selected response rows, run the FFN on just those, scatter back into the cached response FFN output, concatenate with the cached prompt FFN output. The unselected tokens never touch attention or the FFN — that's the whole saving — they just carry their cached outputs forward. And because the positions are non-contiguous, the rotary position embedding for the selected query rows has to be indexed by their actual positions, not by a contiguous range; I have to pass the selected indices (offset by the prompt length) into the rotary application so each recomputed query gets the right angle. Easy to get wrong, important to get right.

Now let me convince myself this is *safe* — that the approximation error stays bounded rather than compounding into garbage over `K` steps. Abstract a denoising step as `y_{k-1} ≈ α_k·y_k + (1−α_k)·F(y_k)`, where `F` is the Transformer map. Let `ỹ_k` be my cached/approximate trajectory and `δ_k = y_k − ỹ_k` the error. Then `‖δ_{k-1}‖ = ‖(α_k y_k + (1−α_k)F(y_k)) − (α_k ỹ_k + (1−α_k)F(ỹ_k))‖ ≤ α_k‖δ_k‖ + (1−α_k)‖F(y_k) − F(ỹ_k)‖`. If `F` is Lipschitz with constant `L_F`, that's `≤ (α_k + (1−α_k)L_F)‖δ_k‖`. Write `C_k = α_k + (1−α_k)L_F`. For an expressive model `L_F > 1`, so `C_k > 1` — the error tends to *amplify* geometrically step over step if I do nothing. So I do need something. And I have two somethings. The first is the periodic full response refresh: every `K_r` steps I recompute the response from scratch, which forces `δ ≈ 0` at that step. That alone caps the damage: between refreshes the error can grow by at most a product of `C_k` over a window of length at most `K_r`, i.e. `‖δ_{k_0−j}‖ ≤ (∏_{i=1}^{j} C_{k_0−i+1})‖δ_{k_0}‖` with `j < K_r`, so the peak is `O(C^{K_r})` and never runs away. The second something tightens it further: V-verify. On the in-between steps I recompute the `ρ` most-changed tokens, so only the *cached* `(1−ρ)` fraction's error gets amplified — the recomputed tokens re-enter at near-zero error. The recursion becomes `‖δ_{k-1}‖ ≤ C_k‖P_cache(δ_k)‖ + ε_step`, where `P_cache` projects onto the cached (un-recomputed) tokens and `ε_step` is the small residual error of the freshly computed ones. Since `‖P_cache(δ_k)‖ ≤ ‖δ_k‖`, and is strictly smaller when the selected tokens carry nonzero error, the effective amplification is some `C'_k < C_k` on the steps where V-verify removes the dominant error components.

Let me put numbers on this rather than leave it at `O(·)`, because the whole question is whether `K_r ≈ 5` is actually safe. Take a mildly expansive map, `L_F = 1.3`, and a midpoint schedule `α = 0.5`, so `C = 0.5 + 0.5·1.3 = 1.15 > 1` — the error does want to grow. With `K_r = 5` and refresh-only (no V-verify), the within-window peak is `C^{K_r−1} = 1.15^4 = 1.749`: the error not quite doubles before the periodic refresh zeros it. That's bounded, and it stays bounded for the run because every window starts from `δ ≈ 0` again — but a 1.75× swell on the worst step is not nothing, and at `L_F = 1.3` (`C^4 = 2.86`) it's getting uncomfortable. Now layer V-verify on top. If the recomputed `ρ` tokens carry the dominant error components, the cached fraction that survives is roughly `(1−ρ)` of the error, so the per-step factor drops to `C·(1−ρ) = 1.15·0.75 = 0.86 < 1`. A factor below 1 means the error *contracts* between V-verify steps instead of growing, and `0.86^4 ≈ 0.55`: the window peak collapses from 1.75× to below the starting error. Even at the harsher `L_F = 1.3`, `C·(1−ρ) = 1.30·0.75 = 0.975 < 1`, still contractive. So the two mechanisms compose, and the arithmetic shows *how*: the refresh interval gives a hard ceiling by zeroing the error every `K_r` steps (`1.75×` at `C=1.15`), and V-verify turns the within-window dynamics from amplifying (`C>1`) to contracting (`C(1−ρ)<1`) as long as it really is grabbing the worst offenders — which is the same "catch the dominant movers" property the toy test gave me, and the residual it leaves on the table (the token-15-style misses) is exactly what the periodic refresh is there to mop up. The two are not redundant; they cover each other's failure mode.

Let me also check the storage cost, because caching four features per layer could in principle blow up memory and undo the win. Per layer I store `K`, `V`, attention output, FFN output, each of shape `T × d` (sequence length `T`, hidden `d`). That's `4·T·d` elements per layer, `4·L·T·d` across `L` layers, i.e. `O(4·L·T·d)`; in bfloat16 that's `2·4·L·T·d` bytes. Crucially I keep only *one* version per layer — I overwrite in place, I don't accumulate a version per step — so the footprint is flat in `K`, not growing. Let me actually plug in the LLaDA-8B numbers rather than wave at "a few percent": `L = 32` layers, `d = 4096`, sequence `T = 512` (say 256 prompt + 256 response). That's `2·4·32·512·4096 = 5.37×10^8` bytes ≈ **537 MB**. The model itself in bf16 is `8×10^9 · 2 = 16 GB`. So the cache is `537 MB / 16 GB = 3.4%` over the model weights — under a gigabyte, exactly as I'd hoped, and it doesn't grow with the number of steps. So I'm trading a tiny, fixed memory overhead for a large, multiplicative compute saving. Fine trade.

And let me actually write down the compute saving to see where it comes from, splitting the `K` steps by mode. The full-refresh steps cost the same as a no-cache step but happen only `K/K_p` times — rare, since `K_p` is large. The response-only-refresh steps recompute just the response (length `r`) against the full sequence: the attention term drops from `4n²d` to `4rnd` (only `r` response queries, still attending over all `n`), giving `T·(8rd² + 4rnd + 6rdm)`, happening `(K/K_r − K/K_p)` times. The adaptive-partial-update steps — the bulk, `K·(1−1/K_r)` of them — cost a small fixed `2rd²` for the V-verify Value projection over all response tokens, plus the recompute of only `r̂ = ρr` selected tokens: `T·(2rd² + 8r̂d² + 4r̂nd + 6r̂dm)`. There it is in the algebra: the dominant quadratic attention `4n²d` collapses to `4r̂nd`, and the heavy FFN `6ndm` collapses to `6r̂dm`, with only the linear `2rd²` V-verify overhead added. With `ρ ≈ 0.25` and `K_p` large, the weighted sum is a fraction of the original `K·T·(8nd² + 4n²d + 6ndm)`. Let me total it with the same LLaDA-8B numbers (`L=32`, `d=4096`, `m=12288`, `p=r=256`, `n=512`, `K=256`, `K_p=100`, `K_r=5`, `ρ=0.25` so `r̂=64`). The baseline lands at `1.87×10^15` FLOPs. The three modes contribute: full refreshes `1.87×10^13` (only `K/K_p ≈ 2.6` of them, 4% of the new total), response-only refreshes `1.77×10^14` (40%), adaptive updates `2.43×10^14` (55%) — summing to `4.39×10^14`. That's a **4.25×** reduction. The split is informative: more than half the remaining cost is in the adaptive steps even though each is cheap, because there are so many of them (`K·(1−1/K_r) ≈ 205` of the 256), and a meaningful chunk is the response-only refreshes — so the two knobs that move the speedup most are `ρ` (how much each adaptive step costs) and `K_r` (how often I pay a whole-response refresh). And the prompt-only saving is real leverage: rerun the same formula with a long prompt `p=1024, r=256`, and caching the prompt on `K_p=100` instead of recomputing it every step removes ~80% of the baseline FLOPs on its own, before V-verify touches the response. The longer the prompt, the more the prompt cache alone buys. The `ρ ≈ 0.25` choice itself is a balance: too small a `ρ` and the fixed per-step overheads of *initiating* selective recompute — gather/scatter, kernel launches, data movement that doesn't scale with the token count — eat the savings, since there's a latency floor just to switch from "fully cached" to "partial update at all"; too large a `ρ` and I'm approaching full recompute. Around a quarter sits in the sweet spot between that fixed activation cost and the dynamic compute saved.

Let me now write it as the per-layer block forward, the thing that drops into the existing denoising rollout. The structure is the four cases, with the partial-update logic factored into helpers. The controller only has to expose a current step counter, the prompt length, the intervals `K_p`/`K_r` and ratio `ρ`, set/get on a per-layer per-segment cache, and refresh predicates that test `(current_step - 1) % interval == 0`.

```python
import torch
import torch.nn.functional as F


def select_low_similarity(v_new, v_cached, ratio):
    """V-verify: cosine similarity between current and cached Value per token;
    return the indices of the `ratio` fraction with the LOWEST similarity (most changed)."""
    num = int(v_new.size(1) * ratio)
    sim = F.cosine_similarity(v_new, v_cached, dim=-1)        # [B, r]
    return torch.topk(sim, k=num, largest=False).indices       # [B, num]


def cache_hook_block(self, x, cache, attention_bias=None):
    """One bidirectional Transformer block under dLLM-Cache. `cache` exposes:
    layer_id, prompt_length, current_step, kp, kr, rho, refresh_prompt(), refresh_gen(),
    get(seg, name), set(seg, name, feat)  with seg in {'prompt','gen'} and
    name in {'k','v','attn','mlp'}. Two caches per layer: prompt C_p and response C_r."""
    cache.update_step(self.layer_id)
    p = cache.prompt_length
    x_prompt, x_gen = x[:, :p, :], x[:, p:, :]
    B, n, d = x.shape

    # always refresh the first layer; otherwise obey the two intervals
    refresh_prompt = cache.refresh_prompt(self.layer_id) or self.layer_id == 0
    refresh_gen = cache.refresh_gen(self.layer_id) or self.layer_id == 0
    rho = cache.rho
    transfer = 0.0 < rho <= 1.0                                # adaptive partial-update mode

    def project(z):                                            # cheap: norm + Q/K/V matmuls
        h = self.attn_norm(z)
        return self.q_proj(h), self.k_proj(h), self.v_proj(h)

    def attend(q, k, v, q_index=None):                         # full bidirectional attention
        att, _ = self.attention(q, k, v, attention_bias, q_index=q_index)
        return att

    def ffn(z):                                                # heavy: norm + SwiGLU FFN
        h = self.ff_norm(z)
        return self.ff_out(self.act(self.ff_proj(h)) * self.up_proj(h))

    # ---- attention sublayer: pick one of four cases ----
    if refresh_gen and refresh_prompt:
        # Case 1 — full refresh (also the k=K init): recompute everything, store both caches.
        q, k, v = project(x)
        cache.set('prompt', 'k', k[:, :p]);  cache.set('prompt', 'v', v[:, :p])
        cache.set('gen',    'k', k[:, p:]);  cache.set('gen',    'v', v[:, p:])
        att = attend(q, k, v)
        cache.set('prompt', 'attn', att[:, :p]);  cache.set('gen', 'attn', att[:, p:])

    elif refresh_gen and not refresh_prompt:
        # Case 2 — refresh the whole response, reuse the prompt cache.
        q, k_gen, v_gen = project(x_gen)
        cache.set('gen', 'k', k_gen);  cache.set('gen', 'v', v_gen)
        k = torch.cat([cache.get('prompt', 'k'), k_gen], dim=1)
        v = torch.cat([cache.get('prompt', 'v'), v_gen], dim=1)
        att_gen = attend(q, k, v)
        cache.set('gen', 'attn', att_gen)
        att = torch.cat([cache.get('prompt', 'attn'), att_gen], dim=1)

    elif refresh_prompt and not refresh_gen:
        # Case 3 — refresh the prompt; adaptive partial update of the response.
        q_p, k_p, v_p = project(x_prompt)
        cache.set('prompt', 'k', k_p);  cache.set('prompt', 'v', v_p)
        k_gen_c, v_gen_c = cache.get('gen', 'k'), cache.get('gen', 'v')
        att_gen_c = cache.get('gen', 'attn')
        if transfer:
            h_gen = self.attn_norm(x_gen)
            v_gen = self.v_proj(h_gen)                          # V-verify: all response Values
            idx = select_low_similarity(v_gen, v_gen_c, rho)    # bottom-rho, most-changed
            idx_e = idx.unsqueeze(-1).expand(-1, -1, d)
            sel = torch.gather(h_gen, 1, idx_e)                 # gather selected response rows
            q_sel, k_sel = self.q_proj(sel), self.k_proj(sel)   # recompute Q/K for movers only
            v_gen_c = v_gen                                     # overwrite ALL cached Values (free)
            k_gen_c = k_gen_c.scatter(1, idx_e, k_sel)          # scatter selected K back
            cache.set('gen', 'k', k_gen_c);  cache.set('gen', 'v', v_gen_c)
        k = torch.cat([k_p, k_gen_c], dim=1)
        v = torch.cat([v_p, v_gen_c], dim=1)
        if transfer:
            prompt_pos = torch.arange(p, device=x.device).unsqueeze(0).expand(B, -1)
            q_all = torch.cat([q_p, q_sel], dim=1)              # prompt + selected response queries
            pos = torch.cat([prompt_pos, idx + p], dim=1)       # their true positions for RoPE
            att_all = attend(q_all, k, v, q_index=pos)
            att_p = att_all[:, :p]
            att_gen_c = att_gen_c.scatter(1, idx_e, att_all[:, p:])   # scatter movers' attn out
            cache.set('gen', 'attn', att_gen_c)
        else:
            att_p = attend(q_p, k, v, q_index=torch.arange(p, device=x.device)
                           .unsqueeze(0).expand(B, -1))
        cache.set('prompt', 'attn', att_p)
        att = torch.cat([att_p, att_gen_c], dim=1)

    else:
        # Case 4 — neither interval fires (the common step): reuse prompt; adaptive update response.
        att_gen_c = cache.get('gen', 'attn')
        if transfer:
            h_gen = self.attn_norm(x_gen)
            v_gen = self.v_proj(h_gen)                          # V-verify
            k_gen_c, v_gen_c = cache.get('gen', 'k'), cache.get('gen', 'v')
            k_p_c, v_p_c = cache.get('prompt', 'k'), cache.get('prompt', 'v')
            idx = select_low_similarity(v_gen, v_gen_c, rho)
            idx_e = idx.unsqueeze(-1).expand(-1, -1, d)
            sel = torch.gather(h_gen, 1, idx_e)
            q_sel, k_sel = self.q_proj(sel), self.k_proj(sel)
            v_gen_c = v_gen                                     # full Value overwrite
            k_gen_c = k_gen_c.scatter(1, idx_e, k_sel)          # selected K scatter
            cache.set('gen', 'k', k_gen_c);  cache.set('gen', 'v', v_gen_c)
            k = torch.cat([k_p_c, k_gen_c], dim=1)
            v = torch.cat([v_p_c, v_gen_c], dim=1)
            att_sel = attend(q_sel, k, v, q_index=idx + p)      # selected queries over full seq
            att_gen_c = att_gen_c.scatter(1, idx_e, att_sel)
            cache.set('gen', 'attn', att_gen_c)
        att = torch.cat([cache.get('prompt', 'attn'), att_gen_c], dim=1)

    x = x + self.dropout(att)                                   # residual after attention
    og_x = x
    x_prompt, x_gen = x[:, :p, :], x[:, p:, :]

    # ---- FFN sublayer: same four cases, scatter movers, reuse the rest ----
    if refresh_gen and refresh_prompt:
        x = ffn(x)
        cache.set('gen', 'mlp', x[:, p:]);  cache.set('prompt', 'mlp', x[:, :p])
    elif refresh_gen and not refresh_prompt:
        x_gen = ffn(x_gen);  cache.set('gen', 'mlp', x_gen)
        x = torch.cat([cache.get('prompt', 'mlp'), x_gen], dim=1)
    elif refresh_prompt and not refresh_gen:
        mlp_gen_c = cache.get('gen', 'mlp')
        if transfer:
            sel = torch.gather(x_gen, 1, idx_e)
            both = ffn(torch.cat([x_prompt, sel], dim=1))      # prompt + movers through FFN
            x_prompt = both[:, :p]
            mlp_gen_c = mlp_gen_c.scatter(1, idx_e, both[:, p:])
            cache.set('gen', 'mlp', mlp_gen_c)
        else:
            x_prompt = ffn(x_prompt)
        cache.set('prompt', 'mlp', x_prompt)
        x = torch.cat([x_prompt, mlp_gen_c], dim=1)
    else:
        mlp_gen_c = cache.get('gen', 'mlp')
        if transfer:
            sel = torch.gather(x_gen, 1, idx_e)
            mlp_gen_c = mlp_gen_c.scatter(1, idx_e, ffn(sel))  # only movers through FFN
            cache.set('gen', 'mlp', mlp_gen_c)
        x = torch.cat([cache.get('prompt', 'mlp'), mlp_gen_c], dim=1)

    x = self.dropout(x)
    return og_x + x, None                                      # residual after FFN
```

So the causal chain, end to end. I started cubic because a bidirectional denoiser recomputes the entire sequence from scratch at every one of `K` steps. The autoregressive KV cache that would fix this is structurally inapplicable — no causal mask means a committed token's K/V keep drifting, and no fixed decoding order means I can't even pre-decide whose states to refresh. So I dropped exact reuse for approximate reuse and followed the adjacent-step feature-similarity diagnostic, which revealed two redundancies: the prompt is quasi-static in bulk, and the response is mostly static with a sparse set of movers. A single uniform cache interval can't serve both, so I split the cache into a prompt cache on a long interval `K_p` and a response cache on a short interval `K_r`, storing all four heavy features (`K`, `V`, attention output, FFN output) so I skip attention and FFN, not just KV. Between full response refreshes I do a partial update, but I can't afford to compute the expensive features just to learn which tokens changed — so I needed a cheap proxy, and a small block-level test settled which one: an *isolated* Value perturbation predicts the attention change poorly (Spearman 0.16, because a token's own Value mostly feeds *other* tokens' readouts), but the realistic *input-level* shift — the way a token actually moves when the sequence unmasks — makes the Value's adjacent-step similarity track the downstream attention/FFN similarity (Spearman ~0.62), catching the dominant movers. That gives the V-verify proxy: recompute the `ρ` lowest-Value-similarity (most-changed) tokens and reuse the rest, with cosine (not L2) to capture directional change, the full `V_response` overwritten for free since I computed it for scoring, and only `K`/attention/FFN scattered for the movers. The proxy isn't perfect — it misses tokens that move downstream without a large Value change — which is precisely why the error analysis pairs it with the periodic refresh: instantiating the recursion (`C=1.15` at `K_r=5`) the refresh alone caps the within-window error swell at `1.75×` by zeroing it every five steps, and V-verify turns the in-window factor from `C=1.15` (amplifying) to `C(1−ρ)=0.86` (contracting), the two covering each other's failure mode. Storage stays a flat 3.4% over the 8B weights (537 MB) because I keep one cache version per layer. The compute collapses where it counts: quadratic attention `4n²d → 4r̂nd` and FFN `6ndm → 6r̂dm`, with only a linear `2rd²` V-verify overhead — totaling a 4.25× FLOPs reduction on a 256/256 split, more on a long prompt where the prompt cache alone removes ~80% — and `ρ ≈ 0.25` sits between the fixed cost of initiating selective recompute and the dynamic compute it saves. The whole thing is training-free — it only reorders *which* features get recomputed — and drops straight into the existing block-wise low-confidence denoising rollout.
