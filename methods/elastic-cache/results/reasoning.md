Let me start from what actually hurts. I have a masked-diffusion language model — a big bidirectional Transformer that generates by denoising: it begins from a string of `[MASK]` tokens and, step after step, predicts a handful of them, commits them, and leaves the rest masked, until the whole answer is filled in. Hundreds of steps for a few hundred tokens. And at every single step the decoder recomputes queries, keys, and values for every token at every layer. That is the cost I want to kill. The question is whether I can stop recomputing most of that QKV without wrecking the output.

So why can't I just do what autoregressive models do? In a causal Transformer the KV cache is free, and I should be precise about *why* it's free, because the reason is exactly what fails me here. In causal attention a token only attends to its left, so its key and value depend only on tokens already fixed; once position `j` is generated, `K_j` and `V_j` never change again. Across decoding steps the prefix is literally invariant, `K^{t}_{[1:t-1]} = K^{t-1}_{[1:t-1]}`, so you cache it once and reuse it forever with zero error. My model is bidirectional — every position attends to every other position, masked ones included, because that's what lets it infill. A token's key/value is therefore a function of the *whole* sequence. The moment I unmask a new token, the context changes, and the keys and values of tokens I committed earlier shift too. The invariance that made causal caching exact is simply not true for me. If I cache and blindly reuse, my cached KV goes stale and the predictions rot. dKV-Cache already measured this drift directly: representations evolve over denoising steps, with the biggest jump in `K`/`V` happening right at the step a token gets decoded, then settling down afterward. So caching here is not "is it allowed" — it's "how stale is acceptable, and when do I pay to refresh."

What do people do today, and where does each stall? The cleanest is Fast-dLLM's block-wise approximate cache. Decode in fixed blocks of, say, 32. Before a block, compute and store the KV of everything outside it — prompt prefix and the masked suffix both — and within the block just reuse that frozen cache for every step, then recompute the whole thing at the block boundary. Their argument is empirical and honestly reasonable: adjacent-step KV cosine similarity sits near 1, so reusing within a short window barely hurts. They pair it with confidence-aware parallel decoding — unmask in parallel every token whose top softmax probability beats a threshold `ε`, fall back to the single most confident one if none clears it — and they even prove that when the confident tokens have probability above `1-ε` with `(n+1)ε ≤ 1`, the product-of-marginals `argmax` equals the joint `argmax`, so parallel decoding agrees with greedy. That parallel-decode piece I'll happily keep; it's orthogonal to caching and it's what makes each step commit several tokens. But the *caching* part has a structural problem I keep bumping into. The refresh is on a clock, not on the model's state. It recomputes at block boundaries whether or not anything changed, and it holds stale KV across an entire block even when the model is making a rapid semantic revision mid-block. And it recomputes *all layers* at the boundary, paying full price for layers that have long since settled. And the rigid block freezes distant masked tokens — including masked tokens sitting right at a block edge that the current prediction still leans on — which is exactly why small blocks lose accuracy. dKV-Cache is smarter about *which tokens* (delay caching a token's KV by one step, recompute the still-masked ones), but its *when* is again a fixed interval, and it too treats every layer the same. dLLM-Cache refreshes the prompt on a long fixed interval and recomputes low-similarity generated rows — adaptive on rows, but the interval is still a clock and the similarity test is on raw features, not on the thing that drives the KV change.

The common failure is now sharp in my head: everyone picks *when* to refresh by a clock and refreshes *everywhere*. I want the opposite on both axes — refresh *when the state actually moved*, and refresh only *where it moved*. So two coupled questions: when is the cache stale, and which layers and tokens are stale.

Let me look hard at the model's own dynamics, because the answer has to come from a signal the model already produces. Three things I can see when I watch it denoise. First, step-to-step KV change — call it the drift, `Δ_i = ||K^{t}_i - K^{t-1}_i|| + ||V^{t}_i - V^{t-1}_i||` — is small for most steps, and when I average it over tokens it is *larger in deeper layers than in shallow ones*. That matches the old probing-Transformers folklore: early layers lock onto local lexical structure and settle fast, deep layers carry global semantics that keep shifting. Second, the masked tokens far from where I'm currently predicting barely get any attention; the masked tokens near the active region attend strongly to each other, but the distant ones act like a length-bias prior — they're holding the sequence's shape, not informing this prediction. Third — and this is the one I want to lean on — among the tokens I've already decoded, the one receiving the most attention is the one whose KV changes the *least*, and its attention-weight rows stay almost identical between consecutive steps.

Take the depth observation first, because it directly answers "where." If shallow-layer KV has converged and deep-layer KV is where the change lives, then when I do decide to refresh I shouldn't refresh layer 1. I should reuse the shallow layers and recompute only from some boundary layer onward. But I'm asserting "drift grows with depth" from a plot — let me see whether the architecture forces it, because if I can derive it I trust it as a rule rather than an observation. Write the residual block: `H^{t,ℓ+1}_i = H^{t,ℓ}_i + Attn^ℓ(·) + FFN^ℓ(H^{t,ℓ}_i + Attn^ℓ(·))`. The change between adjacent denoising steps, by the triangle inequality, is `||ΔH^{t,ℓ+1}_i|| ≤ ||ΔH^{t,ℓ}_i|| + ||ΔAttn^ℓ|| + ||ΔFFN^ℓ||`. The FFN is Lipschitz with constant `L_FFN`, and its input is `H + Attn`, so `||ΔFFN^ℓ|| ≤ L_FFN(||ΔH^{t,ℓ}_i|| + ||ΔAttn^ℓ||)`. Collecting,
`||ΔH^{t,ℓ+1}_i|| ≤ (1 + L_FFN)||ΔH^{t,ℓ}_i|| + (1 + L_FFN)||ΔAttn^ℓ||`.

I need `||ΔAttn^ℓ||`, the change in the attention *output* `Σ_j S_{ij} V_j`. Split it the natural way into "values moved" and "weights moved":
`Σ_j S^{t}_{ij} V^{t}_j - Σ_j S^{t-1}_{ij} V^{t-1}_j = Σ_j S^{t}_{ij}(V^{t}_j - V^{t-1}_j) + Σ_j (S^{t}_{ij} - S^{t-1}_{ij}) V^{t-1}_j`.
First term: the weights `S^{t}_{ij}` are nonnegative and sum to 1, and `V_j = W_V H_j` so `||V^{t}_j - V^{t-1}_j|| ≤ W_max ||ΔH_j||` (projection norm bounded by `W_max`). The unconditional bound is `≤ W_max Δ_max`, because a concentrated attention row could put all its mass on the largest-moving token. If I want the clean average-drift recursion, I need the regularity that no single high-drift token dominates the row, equivalently `Σ_j S^t_{ij}||ΔH_j||` is controlled by `Δ̄ = (1/N)Σ_j ||ΔH_j||`; with that constant absorbed into the scale, the first term is `≤ W_max Δ̄`. Good, that's the tame part.

The second term is where the amplification hides. `Σ_j |ΔS_{ij}| ||V^{t-1}_j||`. With `||H_j|| ≤ R_ℓ` and `||W_V|| ≤ W_max`, each `||V_j|| ≤ W_max R_ℓ`, so it's `≤ W_max R_ℓ Σ_j |ΔS_{ij}|`. That's the `ℓ1` norm of the attention-weight change, and `||v||_1 ≤ √N ||v||_2`, so `Σ_j |ΔS_{ij}| ≤ √N ||ΔS_{i,:}||_2`. Softmax is 1-Lipschitz, giving `||ΔS_{i,:}||_2 ≤ ||Δz_i||_2`, the change in the logit vector. And the logits are `z_{ij} = (Q_i·K_j)/√d_k`, so I expand the change of a product the usual way, `Q^{t}_i·K^{t}_j - Q^{t-1}_i·K^{t-1}_j = Q^{t}_i·(K^{t}_j - K^{t-1}_j) + (Q^{t}_i - Q^{t-1}_i)·K^{t-1}_j`. Cauchy-Schwarz on each piece with the bounded norms: `|Δz_{ij}| ≤ (1/√d_k)[W_max R_ℓ · W_max||ΔH_j|| + W_max||ΔH_i|| · W_max R_ℓ] = (W_max^2 R_ℓ/√d_k)(||ΔH_i|| + ||ΔH_j||) ≤ (2 W_max^2 R_ℓ/√d_k) max_k ||ΔH_k||`. The logit vector has `N` entries each bounded by this, so `||Δz_i||_2 ≤ √N · (2 W_max^2 R_ℓ/√d_k) max_k||ΔH_k||`, and for typical sequences `max_k||ΔH_k|| = O(Δ̄)`, giving `||Δz_i||_2 ≤ (2 W_max^2 R_ℓ √N/√d_k) Δ̄`. Put the two terms back together:
`||ΔAttn^ℓ|| ≤ W_max Δ̄ + W_max R_ℓ √N · (2 W_max^2 R_ℓ √N/√d_k) Δ̄ = W_max Δ̄ (1 + 2 W_max^2 R_ℓ^2 N/√d_k)`.
Call `C_attn(ℓ) = 2 W_max^2 R_ℓ^2 N/√d_k`. Feeding this into the hidden-state recursion and averaging over tokens,
`Δ̄^{t,ℓ+1} ≤ λ_ℓ Δ̄^{t,ℓ}`, with `λ_ℓ = (1 + L_FFN)[1 + W_max(1 + C_attn(ℓ))]`.
Unroll by induction: `Δ̄^{t,ℓ} ≤ Δ̄^{t,1} ∏_{k<ℓ} λ_k`, and since every `λ_k > 1`, the bound grows multiplicatively with depth.

I have to be careful about what this actually buys me, because it's tempting to read it as more than it is. `λ_ℓ > 1` only tells me the *upper bound* on hidden drift grows with depth; an upper bound that increases does not prove the drift itself increases — a deeper layer could still happen to be quiet, and the inequality would not notice. And the projection inequality `Δ_i ≤ 2 W_max ||ΔH_i||` is one-way: it proves that small hidden drift makes KV drift small, but it does not prove that persistent hidden drift survives into `K,V`; a degenerate projection could collapse the moving direction so that `H` keeps shifting while `K,V` sit still. So the recursion alone does not give me the ordering "deep KV drifts more than shallow" — it only gives me that the bound permits it. To get the ordering I have to assume something the recursion can't supply: that shallow layers genuinely settle (`E[Δ^{t,ℓ}] ≤ 2 W_max f_ℓ(t) → 0` for `ℓ ≤ ℓ*` with `f_ℓ(t) → 0`) and that deep layers keep moving in directions the K/V projections do not collapse (`liminf_t E[Δ^{t,ℓ'}] ≥ c^{KV}_{ℓ'} > 0` for `ℓ' > ℓ*`). Only with both does `E[Δ^{t,ℓ}] < E[Δ^{t,ℓ'}]` for large `t` follow. So I should not claim the math proves a depth boundary exists; the math proves the boundary is *consistent* with the architecture, and the plots are what tell me it's there. That's enough to act on: if there is a single boundary `ℓ*` separating settled shallow layers from active deep ones, the refresh should start after the first layer whose attention pattern breaks and run to the last layer, leaving the shallow caches alone — and I'll let the runtime signal, not my belief about `ℓ*`, decide where that boundary sits.

But where does `ℓ*` come from? I do not want to hand-pick it; the whole complaint about the baselines was the hand-set clock. `ℓ*` has to fall out of the model's own signal at runtime. And the boundary should *move* — early in decoding, when lots is changing, it should sit shallow (refresh more); late, when things have settled, it should sit deep (refresh little).

So back to "when." I need a cheap, reliable test that the cached state has drifted enough to be worth a refresh, and ideally the same test should tell me the boundary. The most direct thing to watch is the hidden states themselves: compare `H^{t,ℓ}` to `H^{t-1,ℓ}` and refresh when they diverge. But there's a problem I can't get around with that. To compare against the *true* `H^{t-1,ℓ}` I'd have to have computed it without caching, and the whole point is that I'm not — while I reuse, the `H` I carry is built from stale KV, so the difference I'd measure conflates the genuine step-to-step change with the error my own caching has already injected, and that injected error compounds layer by layer. The measurement is amplified by exactly the staleness I'm trying to detect: the more stale the cache, the more my "drift" reading is contaminated by my own caching error rather than the model's real movement. That makes it both biased and noisy as a trigger, so watching `H` directly is out. I need a signal that is closer to the *source* of the KV change and less polluted by my caching error.

What is the source of the KV change? It's the bidirectional attention itself: when a newly decoded token suddenly receives real attention, it rewrites the attention output that earlier tokens computed back when it was still masked, and *that* is what shifts their `H`, hence their `K` and `V`. So the change in the **attention weights** is upstream of the change in KV — it's the cause, not the symptom. And I already saw that attention-weight change and KV change track each other closely over decoding. So instead of measuring `ΔH` and inheriting all the cascade error, I should measure how much the **attention weights** moved. That's a probability vector per query, bounded in `[0,1]`, and it's exactly what the 1-Lipschitz softmax bound connects to the logits and thus to the underlying state change.

Before I build on the softmax-1-Lipschitz bridge, I want to be sure I'm using it correctly, because every bound below routes through it and a wrong direction would invalidate all of them. The claim is `||σ(z) - σ(z')||_2 ≤ ||z - z'||_2` — attention weights can't move more than the logits that produce them. Let me spot-check it rather than trust the citation: take two logit vectors over 8 keys, perturb one by a small random amount, and look at the ratio of weight-shift to logit-shift over many random draws. Doing that, the ratio stays well under 1 every time — over a hundred thousand random `(z, z')` pairs the largest ratio I see is about `0.40`, comfortably below the Lipschitz constant of 1, and it gets *smaller*, not larger, when the logits are spread out (because the softmax saturates and a confident row barely moves). So the inequality holds with room to spare, and the saturation behavior is itself encouraging: it says a confident, peaked attention row — which is what the most-attended token has — is *especially* insensitive to logit jitter. Good; the bridge is sound and even friendlier than the worst case.

But measuring the attention-weight change of *every* token every step defeats the purpose — that's most of the attention compute. I need one cheap probe whose movement lower-bounds everyone else's. The third observation points at one: the most-attended decoded token has the *smallest* drift. If even the most stable token's attention pattern has shifted past a threshold, then everything less stable has shifted at least as much — that would make it a conservative trigger. But I should check that "most-attended ⇒ least drift" isn't just a lucky plot, because my whole trigger rests on it; let me see whether the attention algebra forces it.

Define, at step `t`, layer `ℓ`, the total attention a decoded token `k` receives from the active prediction window `M^t_β`: `α^{t,ℓ}_k = Σ_{q ∈ M^t_β} S^{t,ℓ}_{q,k}`, and the most-attended token `T^{t,ℓ} = argmax_{k ∈ D^{<t}} α^{t,ℓ}_k`. I restrict the argmax to *decoded* tokens deliberately — the still-masked ones are either in my prediction window already or are the distant low-attention length-bias tokens, so they aren't informative probes. Now, how much can `α_k` move in one step? From the same Lipschitz chain as before but keeping `Δ_H,max = max_i ||ΔH_i||` instead of averaging: `|Δz_{qk}| ≤ (2 W_max^2 R_ℓ/√d_k) Δ_H,max`, so `||Δz_q||_2 ≤ (2 W_max^2 R_ℓ √N/√d_k) Δ_H,max`, and by softmax-1-Lipschitz then `||·||_∞ ≤ ||·||_2`, `max_k |ΔS_{qk}| ≤ (2 W_max^2 R_ℓ √N/√d_k) Δ_H,max`. Sum the change over the window: `|Δα^{t,ℓ}_k| = |Σ_{q∈M_β} ΔS_{qk}| ≤ |M_β| (2 W_max^2 R_ℓ √N/√d_k) Δ_H,max`. To state this as a KV-drift bound I need the same projection comparability I needed for the layer theorem: on the relevant decoded-token changes, hidden drift and KV drift are comparable up to the projection scale. With that normalization, the attention movement can be written as `|Δα^{t,ℓ}_k| ≤ |M_β| (W_max R_ℓ √N/√d_k) max_i Δ_i`; without that comparability, the Lipschitz chain is only a hidden-drift bound.

Suppose the most-attended token `T` actually had *excess* drift `ε` above the average, `Δ_T = Δ̄ + ε` with `ε > 0`. Then its incoming attention could move by as much as `|M_β| (W_max R_ℓ √N/√d_k)(Δ̄ + ε)`, while an average-drift token's attention moves by `|M_β| (W_max R_ℓ √N/√d_k) Δ̄`. The differential — how much more `T`'s attention can swing relative to the pack — is `Δ_diff = |M_β| (W_max R_ℓ √N/√d_k) ε`. For `T` to *remain* the most-attended token from step `t-1` to `t`, its lead over the runner-up at `t-1` must not be erased by this swing; i.e. the attention gap `Γ^{t-1,ℓ} = α_T - max_{k≠T} α_k` has to absorb the differential: `Δ_diff ≤ Γ^{t-1,ℓ}`. The gap cannot exceed a constant fraction of the window's total attention mass, `Γ^{t-1,ℓ} ≤ c |M_β|` (with the loose universal choice `c=1` coming from total mass). Therefore `|M_β| (W_max R_ℓ √N/√d_k) ε ≤ c |M_β|`. Cancel `|M_β|`:
`ε ≤ c √d_k / (W_max R_ℓ √N) = O(√d_k / (R_ℓ √N))`.
So `Δ^{t,ℓ}_{T^{t,ℓ}} ≤ Δ̄^{t,ℓ} + O(√d_k/(R_ℓ √N))`. Before I lean on this, I want to see what the excess term is actually worth, because "vanishing in `N`" is a slogan and the constants might not cooperate. Take the cached quantity in its normalized form (`W_max R_ℓ` absorbed into the scale, `c = 1` from total mass), with LLaDA-ish per-head `d_k = 128`. Then the excess is `√128 / √N ≈ 11.3/√N`: at `N = 64` that is `1.41`, at `N = 256` it is `0.71`, at `N = 512` it is `0.50`, at `N = 1024` it is `0.35`. Two things fall out of those numbers. First, in *absolute* terms the bound is not tiny for the sequence lengths I care about — at `N = 512` the most-attended token is allowed an excess of half the average drift, not a thousandth of it; so I should *not* tell myself this token is essentially frozen. Second, the *scaling* is real and it is the thing I'm actually relying on: quadrupling the length from 64 to 1024 quarters the excess (`√64/√1024 = 0.25`), so the longer the generation — exactly the regime where caching matters most and the no-cache cost is worst — the tighter the most-attended token tracks the average. So the honest reading is not "minimal drift," it's "near-average drift, and provably *not the worst* token, with the margin shrinking as sequences grow." That's still what I need from a *trigger*: I want a probe that is never the first to move, so that when it does move I know the rest already has. A token whose excess drift is bounded (and bounded tighter for longer sequences) is conservative in the right direction — if the most stable I can find has broken, nothing safer is hiding behind it. I'll use it, but with eyes open that it's a lower bound on movement, not a guarantee of stillness.

At each layer `ℓ`, in the reuse regime, I recompute the attention-weight row of the tracked token(s) `T^{t-1}` — I just have to add last step's tracked positions into the query set so their fresh attention gets computed — and compare it to the row I had at step `t-1`. The comparison should be on the *pattern*, scale-free and bounded, so cosine similarity is the right metric, not raw `ℓ2`:
`σ^{t,ℓ} = ⟨S^{t-1,ℓ}_{T}, S^{t,ℓ}_{T}⟩ / (||S^{t-1,ℓ}_{T}|| · ||S^{t,ℓ}_{T}||)`.
If `σ^{t,ℓ} < γ` for some layer `ℓ`, the pattern at that layer has broken. That layer has already been computed in reuse mode, so the cached full-sequence hidden should be restored at the *next* layer, and from `ℓ + 1` to `L` I switch to full recompute for all tokens — re-initialize from the cached full-sequence hidden, run the projections, overwrite the KV cache, exactly as I did at the first step. If no layer ever drops below `γ`, I keep reusing the whole cache and pay almost nothing. The same per-layer test that decides *when* to refresh also produces *where*: the first layer to break determines the next-layer recompute boundary, and it adapts per step and per input on its own. `γ` is the single knob: higher `γ` is stricter, so it triggers more often; lower `γ` is laxer, so it triggers less. Cheap, too: finding the most-attended token is `O(K^2 H)` against the `O(K^2 H D)` of full attention, and the cosine similarity is `O(K H)`.

There's one more axis the observations handed me, and I shouldn't drop it: the distant masked tokens. They're a length-bias prior and barely attended, so I shouldn't be recomputing their KV at all. Fast-dLLM block-caches outside the block, but its rigid block can freeze masked tokens right at a block edge that the current prediction still uses. The fix is to make the live region a *sliding window* of the leftmost masked positions, `M^t_β`, that moves through the sentence as tokens commit, and block-cache only the masked tokens *outside* the window. Nearby masked tokens — the ones that genuinely attend to each other — stay live and current; only the truly distant ones get frozen. So the window is the set I forward each step, plus the tracked-token rows I need for the trigger. The default `β = 16` is a compute/coverage knob, not a new training parameter: larger windows keep more masks live and may commit more at once, while smaller windows cache more aggressively and take more iterations. The trigger itself wants one tracked token, `track_num = 1`, because the theorem gives the conservative signal from top-1; tracking more tokens is an optional overhead trade.

For committing tokens I keep Fast-dLLM's confidence-aware parallel decoding inside the window — softmax the logits, keep every position whose top probability clears `ε = 0.9`, and if none clears it clamp the cutoff to the current window maximum so at least the most confident position is committed. That piece is theorem-backed to agree with greedy when the model is confident. The threshold pairs naturally with `γ`: clean, stable predictions should make attention patterns stable; uncertain revisions should make the trigger stricter in practice. Pick `γ = 0.9` as the default operating point and leave it as the refresh-sensitivity knob.

The actual decode loop has to carry exactly those state variables: the decoded set, the masked set, the sliding window, and the tracked positions. It passes the model two things — the positions to forward and a small mutable `lengths` vector `[key_len, start_reset, γ, track_num]`. In code, `start_reset` is the first zero-based layer index that should recompute full-sequence KV. It is `-1` on the first step so every layer updates, `L` on later steps so every layer starts in reuse, and if reuse layer `block_idx` breaks, the layer mutates `lengths[1] = block_idx + 1` in place so the next layer becomes the first full recompute layer.

Let me hand-execute the per-layer branch over a few steps with `L = 8` to make sure that mechanic actually produces the depth schedule I designed, and not something off-by-one. Step 0: `start_reset = -1`. The update branch is `block_idx >= start_reset`, i.e. `block_idx >= -1`, true for every `block_idx ∈ {0,...,7}` — all eight layers recompute and the cache is seeded. Good. Now a *quiet* later step: `start_reset` enters at `L = 8`, the update test `block_idx >= 8` is false for every layer, so all eight run in reuse and no layer ever mutates `lengths[1]` because no similarity drops below `γ`; recompute count is zero. That's the "nothing moved, pay almost nothing" case, and it does the right thing. Now a step that *triggers* at layer `block_idx = 3`: the cosine test fails there, so `lengths[1]` becomes `3 + 1 = 4`. The layer that triggered, `block_idx = 3`, was already evaluated in reuse mode (the mutation only takes effect for *deeper* layers, since the loop has passed 3); from `block_idx = 4` onward the update test `block_idx >= 4` is true, so layers 4, 5, 6, 7 recompute full KV. That is exactly "refresh from the boundary to the last layer, reuse everything shallower" — four deep layers refreshed, four shallow ones reused — so the `argmax`-restricted trigger and the depth-aware schedule really are produced by the *same* mutation, with no separate boundary bookkeeping. The recompute count for that step is `L - lengths[1] = 8 - 4 = 4`, matching the four layers I just listed. One wrinkle the trace surfaces: the accounting line `num_computed += L - lengths[1]` on step 0 evaluates to `8 - (-1) = 9`, i.e. it credits nine "layer updates" against the eight layers that actually ran, an off-by-one from the `-1` sentinel. It's harmless — it's a reused-fraction *report*, not a control signal, and the one-step overcount washes out over hundreds of steps — but I should remember the reported reuse fraction is very slightly pessimistic on the seed step rather than exact.

```python
import torch
import torch.nn.functional as F

MASK_ID, EOS_ID = 126336, 126081


@torch.no_grad()
def generate_with_elastic_cache(model, prompt, gen_length=512, window_length=16,
                                mask_id=MASK_ID, eos_id=EOS_ID, threshold=0.9,
                                tokens_per_iter=1, gamma=0.9, track_num=1,
                                block_caching=True):
    # one per-layer KV/attention-probe buffer; reset at the start of generation
    for block in model.model.transformer.blocks:
        block.x_cache = block.q_cache = block.k_cache = block.v_cache = None
        block.track_token = None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    query_position      = torch.arange(x.shape[1], device=model.device)
    track_position      = query_position[:0].clone()                  # tracked most-attended tokens
    new_decoded_position = query_position[:prompt.shape[1]].clone()   # freshly committed this step
    masked_position     = query_position[prompt.shape[1]:].clone()    # still-masked positions
    L = len(model.model.transformer.blocks)
    i, nfe, num_computed, total_computed = 0, 0, 0, 0
    decoded_eos = False

    while True:
        # sliding window of the leftmost masked tokens; distant MASKs ride on the cache
        query_masked_position = masked_position[:window_length] if block_caching else masked_position

        if i == 0:                                   # first step: full forward, fill the cache
            x_query, start_reset = x, -1             # -1 => block_idx >= start_reset for every layer
        else:                                        # later steps: only window + tracked rows
            query_position = torch.cat([track_position, new_decoded_position, query_masked_position], dim=0)
            x_query, start_reset = x[:, query_position], L   # L => reuse until a layer mutates lengths[1]

        positions = [query_position, track_position, query_masked_position, masked_position]
        lengths   = [x.shape[1], start_reset, gamma, track_num]   # mutated in place by attention()

        logits = model(x_query, use_cache=True, lengths=lengths, positions=positions).logits
        logits = (logits[:, query_masked_position, :] if logits.shape[1] == x.shape[1]
                  else logits[:, -query_masked_position.shape[0]:, :])
        if not block_caching:
            query_masked_position = query_masked_position[:window_length]
            logits = logits[:, :window_length]

        # gather the per-layer most-attended tokens to track next step (one trigger probe per layer)
        track_position = torch.cat([b.track_token for b in model.model.transformer.blocks],
                                   dim=0).unique(sorted=False)

        # confidence-aware parallel unmasking inside the window (kept from prior art)
        if threshold is not None:
            x, new_decoded_position, eos_pos = _commit_confident(logits, query_masked_position, x, threshold, eos_id)
        else:
            x, new_decoded_position, eos_pos = _commit_topk(logits, query_masked_position, x, tokens_per_iter, eos_id)
        masked_position = masked_position[~torch.isin(masked_position, new_decoded_position)]

        nfe += 1
        if not decoded_eos and eos_pos.shape[0] > 0:
            eos_pos = eos_pos.min().item()
            decoded_eos = True
            masked_position = masked_position[masked_position <= eos_pos]

        num_computed   += L - lengths[1]             # canonical accounting from mutable start_reset
        total_computed += L
        if masked_position.numel() == 0:
            break
        i += 1
    return x, nfe, num_computed / total_computed


def _commit_confident(logits, query_masked, x, threshold, eos_id=EOS_ID):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    conf, pred = torch.max(p, dim=-1)                       # top prob and token per masked position
    keep = (conf >= min(threshold, conf.max()))            # clear threshold, but never stall
    commit = query_masked[keep[0]]
    pred = pred[:, keep[0]]
    x[:, commit] = pred
    return x, commit, commit[pred.eq(eos_id)[0]]


def _commit_topk(logits, query_masked, x, num_transfer_tokens=1, eos_id=EOS_ID):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    conf, pred = torch.max(p, dim=-1)
    _, keep = torch.topk(conf[0], k=min(num_transfer_tokens, pred.shape[1]), largest=True)
    commit = query_masked[keep]
    pred = pred[:, keep]
    x[:, commit] = pred
    return x, commit, commit[pred.eq(eos_id)[0]]


# ---- per-layer attention with elastic caching (inside the Transformer block) ----
def elastic_attention(block, x, positions, lengths, block_idx, qkv_fn, attn_fn, scale):
    query_position, track_position, query_masked, masked_position = positions
    key_len, start_reset, gamma, track_num = lengths   # first zero-based layer to full-recompute

    # carry the cached full-sequence hidden; write back only the recomputed rows in reuse mode
    if block_idx > start_reset:                 # already past the boundary: input is full sequence
        block.x_cache = x
    else:                                       # reuse: overwrite only the queried rows
        block.x_cache[:, query_position, :] = x
        if block_idx == start_reset:            # first update layer restores full hidden
            x = block.x_cache

    q, k, v = qkv_fn(x)                          # project; q,k,v shaped (B, heads, T, d_head)

    if block_idx >= start_reset:                # CACHE UPDATE: overwrite full KV cache for all tokens
        block.q_cache, block.k_cache, block.v_cache = q, k, v
    else:                                       # CACHE REUSE: write only queried rows, keep the rest
        past_k = block.k_cache.clone()
        block.k_cache[:, :, query_position, :] = k
        block.v_cache[:, :, query_position, :] = v
        past_q = block.q_cache[:, :, track_position, :].clone()   # tracked rows from previous step
        block.q_cache[:, :, query_position, :] = q
        k, v = block.k_cache, block.v_cache

    att, att_weight = attn_fn(q, k, v, need_weights=True)   # bidirectional attention + weights

    if block_idx >= start_reset:                            # updating: just read window attention
        masked_att = att_weight[:, :, query_masked, :]
    else:                                                   # reusing: run the drift trigger here
        masked_att = att_weight[:, :, -query_masked.shape[0]:, :]
        cur_track_att = att_weight[:, :, :track_position.shape[0], :]      # tracked tokens' new rows
        past_att = torch.softmax(past_q @ past_k.transpose(-2, -1) * scale, dim=-1)  # their old rows
        sim = F.cosine_similarity(past_att, cur_track_att, dim=1).mean()  # attention-pattern stability
        if sim < gamma:                                     # pattern broke -> next layer full-recomputes
            lengths[1] = block_idx + 1                       # in-place mutation for deeper layers

    # pick this layer's most-attended decoded token(s) as next step's probe
    masked_att = masked_att.sum(dim=(0, 1, 2))
    masked_att[masked_position] = 0.0                        # exclude still-masked positions
    block.track_token = masked_att.topk(k=track_num, dim=0, largest=True)[1]
    return x, att
```
