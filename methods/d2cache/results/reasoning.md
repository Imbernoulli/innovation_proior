Let me start from what actually hurts. I have a diffusion language model — LLaDA, an 8B model — that generates by denoising. The response is a fixed block of `n` mask tokens, and over `T` steps the model looks at the whole sequence, predicts every still-masked position at once, and I commit a few of the most confident ones; repeat until no masks remain. The thing that makes this work, and the thing that makes it slow, are the same thing: bidirectional attention. Every token attends to every other token, left and right, masked or not. That's what gives me right-context and any-order decoding. But it also means that when I flip one mask to a real token, the context of *every* token changes, so the keys and values of every token shift, and I have to recompute the entire sequence's KV at every layer, every step. An autoregressive model of the same size generates one token per forward and reuses essentially all of its KV from a prefix cache. I do `T` full-sequence forwards. That's the gap I want to close: I want a KV cache for a diffusion model.

The first thing I have to be honest about is *why* the ordinary prefix cache is simply illegal here, not just inconvenient. In an autoregressive model, token `i` attends only to tokens `≤ i`, and I append new tokens at the end. So once I've computed `i`'s key and value, nothing I do later can change them — no future token feeds into a past token's representation — and the cache is exact and append-only. Both of those facts fail for me. My attention is bidirectional: token `i`'s representation depends on tokens on both sides, including the masked ones. And I don't append; I fill in slots in a fixed-length array, anywhere. So the moment a mask at position `j` becomes a concrete token, every position that attends to `j` — which is everyone — sees a new context, and its K and V move. There's no prefix that's frozen. Exact reuse is off the table. So whatever I build has to be an *approximate* cache: reuse KV that's *close enough* and recompute the rest. Fine. The whole game becomes: which tokens' KV states are stale enough that I have to recompute them, and which are stable enough that I can reuse last step's?

Stated that way, this isn't new — people are already doing approximate KV caching for diffusion LMs, and the way they all do it is to cut the sequence into a static segment and a dynamic segment: cache the static part for many steps, refresh the dynamic part often. dKV-Cache stores a token's KV one step late and refreshes on a schedule. Fast-dLLM rides on block-wise semi-autoregressive decoding and caches everything except the current block (or, more aggressively, recomputes only the current block and caches everything else). dLLM-Cache splits prompt versus response and refreshes prompt every `K_p` steps and response every `K_r` steps, and on the in-between steps it does something cleverer for the response: it recomputes the value vectors, takes the cosine similarity of each response token's current `V` to its cached `V`, and refreshes only the `ρ`-fraction whose `V` drifted most. That `V`-drift criterion is genuinely good — it's the first hint that drift is token-specific. But look at what's common to all three: the *decision* is made at the level of a segment or a block, on a *fixed schedule*. dLLM-Cache's intervals `K_p`, `K_r` and ratio `ρ` have to be re-tuned for every model and every dataset — the published tables literally list different `(K_p, K_r)` for GSM8K versus HumanEval versus MMLU. Fast-dLLM is chained to the rigid block boundary. And in every case, one rule covers a whole segment of tokens at once.

Why is that the limitation? Because a segment is not a unit of *KV dynamics*. Inside one response segment at one step, some tokens are dead stable and some are about to change a lot — and a single interval has to treat them the same. So you inevitably do both wrong things at once: you refresh tokens whose KV has already settled (wasted compute) and you reuse tokens whose KV is actively moving (lost quality). To do better I'd need to know the dynamics at the granularity they actually live at, which is per token, per step. The diagnostic traces give me exactly that granularity.

The trace of one masked token's layer-averaged key state, projected down with PCA across decoding steps, is not a smooth drift and not a constant. It's three phases. Early on, the key barely moves — call it the gradual-change phase. Then, in the handful of steps right before this token itself gets decoded, the key state lurches — a rapid-change phase. Then, once the token is decoded, it goes essentially still for the rest of generation — a stable phase. The value state of the same token does the same thing, same shape, same magnitude. And I can see *why* the lurch happens right before decoding: that's exactly when this token's local context is filling in. If a neighbor mask gets decoded, the neighbor flips from a `[MASK]` embedding to a concrete-token embedding, and that's a big change to the context this token attends to — and the closer the neighbor, the bigger the change to how the model represents my token. So the KV moves most when the neighborhood is resolving.

This is the lever. If a masked token's KV only really moves during its rapid-change phase, then I only need to recompute its KV during that phase. In the gradual phase and the stable phase I can cache it and reuse it, and doing so doesn't degrade the final generation, because I'm not skipping any *informative* update, only the ones where the state wasn't going anywhere. That's a token-level statement. A segment can never express it, because at any given step the tokens in a segment are scattered across all three phases.

But now I've walked straight into a wall, and it's a sharp one. To update a masked token *during* its rapid-change phase, I have to know that it's about to be decoded — I have to identify the rapid-change tokens *before* they're decoded. But "about to be decoded" is exactly the thing the decoding process is in the middle of figuring out. It's circular: I need to know which tokens are imminent to give them fresh KV, but knowing which are imminent is what fresh KV would help me decide. Chicken and egg. If I can't break this, the three-phase observation is just a nice picture with no algorithm attached.

So I need a *predictor* of imminence that doesn't require having already decoded the token — something structural, readable from the current state of the sequence. The decoding-order diagnostics have the structure I need: the model overwhelmingly decodes the next token *near* the one it just decoded — 90% of next-decoded positions are within a distance of 10 of the most recent one. The decoding front is spatially local. And it fits the lurch I just saw: a masked token surrounded by already-known tokens (prompt tokens or already-decoded tokens) has a tightly constrained context, so the model resolves it sooner. So imminence is predictable from the *density of known tokens in the local neighborhood* of a masked position. That's the structural signal that breaks the circularity — it depends only on which positions are currently known, not on having decoded the masked token.

Let me turn "density of known tokens nearby" into a number. The crudest version: count the known tokens within a fixed window around position `i`. But that throws away distance, and distance is the whole point — the lurch comes from *close* neighbors resolving, and the decoding-order diagnostics say closeness governs which masked position gets filled next. A known token sitting right next to my masked position should count for much more than one ten positions away. I want a weight that's high for near neighbors and smoothly fades for far ones. A Gaussian in the separation distance does exactly that and has no hard cutoff to tune sharply. So define, for masked position `i`,

  `D(i) = Σ_j φ(|i − j|) · 1{ j is known }`,   `φ(d) = exp( − d² / (2σ²) )`,

summing over all positions `j` that are currently known (prompt or already-decoded), each weighted by a Gaussian of its distance to `i`. This is a distance-aware aggregation of certainty flowing into `i` from its known neighbors — call it the certainty density.

The `σ` is the only knob — the width of the neighborhood the density looks at — and I don't want to just assert a value, so let me actually compute `D` on a small layout and watch what `σ` does. Take a length-20 sequence with prompt tokens known at positions 0–4 and two already-decoded tokens at 9 and 10; everything else masked. Ranking the masked positions by `D`:

  `σ = 2`:  top positions 5, 8, 6, 11   with `D ≈ 2.17, 1.68, 1.58, 1.49`
  `σ = 10`: top positions 5, 6, 7, 8    with `D ≈ 6.54, 6.46, 6.32, 6.13`
  `σ = 50`: top positions 5, 6, 7, 8    with `D ≈ 6.981, 6.977, 6.970, 6.961`

This tells me two concrete things. At `σ = 2` the density is jumpy — position 8 outranks 6 and 7 because it sits close to the decoded pair at 9–10, so a tiny local cluster dominates and the ranking is noisy. At `σ = 50` the opposite failure: the top four `D` values agree to the third decimal place, so the signal has flattened and the density can no longer tell apart positions that are five apart — every masked token looks the same, exactly the wash-out I worried about. At `σ = 10` the values are large, smoothly ordered, and pick out the positions adjacent to the prompt edge first, which is the behavior I want: a clear ranking that still respects distance. So a `σ` in the range of ten survives both failure modes, and it also matches the empirical scale of the decoding front (90% of next-decoded tokens within distance 10). I'll keep `σ = 10`.

At the implementation boundary I can compute this same distance-weighted known-token signal by convolution rather than by a quadratic loop: pad the known-token mask, convolve it with the Gaussian kernel by FFT, and divide by the available kernel mass near the boundaries so edge tokens are comparable. That normalization is an implementation detail for stable density values; the quantity I am deriving is still the Gaussian-weighted sum of known positions.

Now, density tells me a token is *structurally* about to be decoded — surrounded by known context. But there's a second kind of certainty I already have for free after each forward: the model's own prediction confidence `s^i`, the probability it assigns to its top guess for position `i`. Structural certainty and predictive certainty are different things — a token can sit in a dense neighborhood yet the model is still unsure of it, or be confident in a sparse one. I want both: to select tokens that are *about to be decoded* (high density) *and* that the model has a clear opinion about (high confidence). How should the two combine? I want a token to survive only if *neither* factor is small, so I need a combination where one small factor drags the score down regardless of the other. A sum won't do that — a token with `D` near zero but high confidence still scores middling under `D + s`. A product does: take a near-imminent confident token (`D=6, s=0.9`) against a confident-but-isolated one (`D=0.5, s=0.95`) and a dense-but-unsure one (`D=6, s=0.2`). The products are `5.4`, `0.475`, `1.2` — the first wins by a wide margin, while either weak factor pulls its product down by roughly its own ratio. That's the soft logical-AND behavior I'm after, so the calibrated score for a masked token is

  `D(i) · s^i`,

and I pick the top-`k` masked tokens by this product. Call their index set `M*` — the masked tokens whose KV I'll refresh this step. The candidate budget `k` is a small fixed count — a few dozen — because the rapid-change front is narrow; I don't need to refresh many masked tokens at once. I'll take `k = 32`.

And there's a bonus hiding in `D(i)·s^i` that I almost missed. I built it to choose *which masked tokens to recompute*. But it's also a *decoding criterion* — I could commit (transfer) masked tokens in order of `D·s` instead of by confidence alone. Why would I want to? Because plain confidence-based decoding has a known pathology in this model: near the tail of the response it gets prematurely overconfident about the end-of-sequence token and terminates too early, which is the whole reason block-wise semi-AR decoding exists — to force a roughly left-to-right order so the model doesn't commit the end before it's reasoned through the middle. But `D·s` already prefers tokens next to known tokens, and known tokens grow outward from the prompt, so decoding by `D·s` naturally produces a *quasi* left-to-right order without me hard-coding any block boundary. It defers an uncertain token in a sparse region until its neighborhood fills in, rather than committing it or committing a premature EOS past it. So the same score that solves the cache-selection problem also gives me a more reliable decoding order, for free. I'll keep that as an available decoding mode.

So Stage 1 handles masked tokens. But it can't be the whole story, and I can see exactly where it runs out. `D(i)·s^i` is defined for *masked* tokens — it uses prediction confidence, which only exists for positions the model is predicting. What about the prompt tokens and the already-decoded tokens? They have no `s^i`. Their KV moves little step to step, which is why caching helps, but "little" isn't "nothing," and if I never refresh them I'll accumulate error. I need a *second* criterion for the non-masked tokens, and it has to be a different kind of criterion, because the thing that makes a prompt/decoded token worth refreshing isn't that it's about to be decoded — it's already known. What makes it worth keeping fresh is that *other* tokens depend on it heavily. So the right notion for these tokens is **attention importance**: a token that many queries attend to strongly is one whose KV I can't afford to let drift.

This rhymes with something from the autoregressive world. People noticed long ago that transformer attention isn't spread evenly — it concentrates on a few salient tokens, the "attention sinks" — and they used that to prune or budget KV state by importance. The question is whether the same concentration holds in a *bidirectional* dLLM, where there's no causal sink at position zero and roles shift every step. The attention diagnostics say it does: queries pile their attention onto a small subset of key positions, and that subset is dominated by prompt and decoded tokens; masked tokens, consistent with everything above, receive almost no attention. Even better, adjacent-step rollout maps are nearly identical — the importance ranking is stable across adjacent steps. That stability is what makes this usable: I can compute importance from the *current* step's attention and trust it to tell me which tokens to keep fresh on the *next* step. Without that, an importance criterion would be useless, because I'd need next step's attention to decide next step's recompute set — another chicken-and-egg. The cross-step stability dissolves it.

Now, how do I actually measure "importance" faithfully? The naive thing is to read off the last layer's attention weights. That's a trap in a deep network: the per-layer attention gets diffuse as you go up, and the raw weights stop tracking which inputs actually drove the output. I need importance composed through all the layers, and I need to account for the residual connections, which carry a token's own information straight up and are a big part of how influence propagates. This is exactly what attention rollout was built for. Its logic: a transformer block computes `V_{l+1} = V_l + W_att V_l = (W_att + I) V_l`, so the effective per-layer mixing isn't `W_att` alone but `W_att + I` — add the identity for the residual path — and then renormalize each row to sum to one so it's still a proper distribution of influence. Compose those across layers by matrix multiplication, and you get an end-to-end attribution from inputs to outputs.

Let me write it for my setting, where there's a twist: I'm not querying the whole sequence this step. I only forward the tokens I selected — `M*` plus the importance set I'm about to define — so my attention matrix is *rectangular*: rows only for the queried positions, columns for the whole sequence. To compose layers I need a square `L × L` map, so I expand it. For a row `i` that *was* a query this step, its row is the (head-averaged) attention it produced. For a row `i` that *wasn't* queried, I have no fresh attention for it, so I set its row to the one-hot `e_i` — meaning "this token passes its own information through unchanged," which is the honest default for a token I didn't recompute. Average the heads to get `Ā^{(l)}`, build the expanded `E^{(l)}` that way, then form the per-layer transition

  `W^{(l)} = normalize_row-sum-to-1( E^{(l)} + I )`,

and roll it up,

  `C^{(0)} = I`,   `C^{(l)} = W^{(l)} · C^{(l−1)}`.

The final `C^{(N)}` holds the end-to-end influence between every pair of positions. To collapse it to a per-token importance, I want "how much total influence flows *into* token `j` across all queries," which is the column sum,

  `c_j = Σ_i C^{(N)}_{ij}`.

A token `j` with a big `c_j` is one that lots of positions are routing influence through. Before I trust that, let me check that the construction actually does what I claim on a case where I know the answer. Take three tokens, two layers, and make token 0 a sink: in both layers' head-averaged attention, the other two queries put about 0.7 of their weight on token 0 (`E1` rows `[.8,.1,.1], [.7,.2,.1], [.7,.1,.2]`, `E2` rows `[.9,.05,.05], [.6,.3,.1], [.6,.1,.3]`). Rolling up `W^{(l)} = normalize(E^{(l)} + I)` with `C^{(0)} = I` gives

  `C^{(2)} ≈ [[.872,.064,.064],[.515,.408,.078],[.515,.078,.408]]`,
  `c = (1.903, 0.549, 0.549)`,   `argmax_j c_j = 0`.

So the column sum hands token 0 about 3.5× the importance of the other two and the argmax lands on the sink — the construction recovers the heavily-attended token, which is what I needed. I also want to check the default I chose for unqueried rows. If only tokens 1 and 2 are queried and token 0's row is set to the one-hot `e_0 = [1,0,0]`, then after `+I` and renormalizing, row 0 of `W` is `[1,0,0]` exactly — token 0 passes its own information straight through and contributes nothing spurious to anyone else's influence. That's the honest "I didn't recompute this row" behavior I wanted, not an accidental zeroing or leak. Good — `c_j` is a token whose KV I must not let go stale.

Then I select from these. I could take a fixed top-`k` of `c_j`, but the influence distribution is heavy-tailed and its concentration varies — sometimes a handful of tokens own almost all the mass, sometimes it's spread wider. A fixed count would over- or under-select depending on the step. The adaptive alternative is nucleus selection: sort tokens by `c_j` descending, normalize to a distribution, and take the smallest prefix whose cumulative mass exceeds a threshold `p`. That way I keep "enough tokens to cover fraction `p` of the total influence," and the count flexes with how concentrated the step is. Call the selected set `U`.

I should sanity-check that a *small* `p` does the right thing in both regimes before committing to one. Take `L = 300` tokens. In a concentrated step — five sinks with influence `[40,25,15,10,8]` owning 77% of the mass, the remaining 295 tokens near zero — `p = 0.1` selects just **1** token: a tenth of the total influence is already covered by the single biggest sink. In a spread step — influence roughly uniform across all 300 — the same `p = 0.1` selects **19** tokens, because no small handful covers a tenth and the prefix has to grow. So one fixed `p` automatically refreshes almost nothing when attention is concentrated and broadens when it isn't, which is exactly the adaptivity I wanted from a *threshold* rather than a count. And because the diagnostics say dLLM attention really is concentrated step to step, the operating regime is the first one — a small `p = 0.1` pulls in the genuinely salient tokens and little else. I'll use `0.1`.

So now I have my two-stage recompute set: `M* ∪ U`. Stage 1 caught the masked tokens entering rapid change, by certainty density times confidence; Stage 2 caught the prompt/decoded tokens that everyone depends on, by attention rollout importance. The two stages exist because the two token populations have genuinely different dynamics — masked tokens change a lot but only briefly and locally, so a phase/density signal finds them; prompt and decoded tokens change little but are heavily attended, so an attention signal finds them — and neither signal can find the other population. Confidence-density says nothing about a prompt token (it has no prediction); attention salience under-selects a soon-to-decode masked token (it receives almost no attention). I need both, unioned. Every token *not* in `M* ∪ U` keeps last step's cached KV.

A couple of practical things before I trust this. First, whenever a masked token actually gets decoded this step — transferred from `[MASK]` to a real token — its KV *has* to be recomputed at least once next step, because its embedding just changed fundamentally; so the freshly transferred positions go into the active set unconditionally, on top of `M*` and `U`. Second, the active set as chosen is a scattering of isolated positions across the sequence, and the attention kernels run far better on a contiguous block of query rows than on a sparse gather of singletons. So I add a small inflation: if two selected positions are within a window `w` of each other, I select everything between them, closing the little gaps. It costs a few extra rows but makes the actual forward efficient, and the in-between tokens are usually about to matter anyway given the observed locality. There's a knob here — `w` — and it can be turned off (`w = 0`) when I'd rather keep the active set strictly minimal; the default in the implementation fills small gaps.

Let me also check the memory, because adding a cache had better not blow the budget I was trying to save. The KV cache itself is the same `2·L·N·d` floats a same-size autoregressive model would hold — that's just the price of caching at all. The only thing I add on top is the rollout matrix, which is `L × L` — for the sequence lengths in play that's negligible next to the `N·d`-scaled KV tensors. So the scheme is essentially free in memory. Good.

And none of this touches the model weights. Certainty density is computed from the mask layout; confidence comes from the forward I'm already doing; the rollout comes from attention weights I'm already producing; the selection and caching are pure bookkeeping around the forward. So the whole thing is training-free — a wrapper I can drop onto any pretrained dLLM. That was a hard requirement, since the only dense dLLMs I have are fixed checkpoints.

One implementation detail the rollout forces on me: to compute it I need the *explicit* per-head attention weights, the `softmax(QKᵀ)` matrix. The fused attention kernels — flash attention, fused SDPA — never materialize that matrix; they fuse the softmax into the value multiply for speed. So the attention layers have to run in *eager* mode that returns the weights. That's a real cost, but the rollout is not the bottleneck (the layer compute is), and it's the price of having a faithful, layer-composed importance signal rather than a noisy last-layer proxy.

Now let me put it down as code I'd actually ship, against the denoising loop's hooks. I'll keep a base cache that owns the generic plumbing — wrapping the model forward to narrow the input to the active rows and scatter logits back to full length, and wrapping each attention layer to manage the per-layer K/V cache — and put the selection in the subclass. The implementation has to preserve a few details I cannot hand-wave: confidence scores are only valid for rows actually queried on the previous forward, so I cache and update them selectively; the response block restricts where Stage 1 searches, with a bias so the current block does not starve; batch rows need the same query count, so a top-up pass pads smaller active sets; Dream-style AR-adapted models need the token before a selected mask; and newly transferred tokens are always forced into the next query mask. With those constraints, the cache lands like this.

```python
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from src.cache.base import dCache, AttentionContext
from src.utils import certainty_density, nucleus_select, top_up_mask_, is_adapted_from_ar

class d2Cache(dCache):
    """Dual aDaptive Cache: a training-free approximate KV cache for dLLMs.
    Only active query rows are recomputed; inactive rows reuse cached K/V."""

    def __init__(self, model_config, rollout_p=0.1, current_k=32, sigma=10.0, inflate_w=4):
        super().__init__(model_config)
        self.model_config = model_config
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._conf_cache = None
        self._full_q_mask = None
        self._density_score = None
        self._global_importance = None
        self.rollout_p, self.current_k = rollout_p, current_k
        self.sigma, self.inflate_w = sigma, inflate_w

    @contextmanager
    def model_forward(self, x):
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            if self._full_q_mask is not None:
                self.active_q_mask = self.top_up_mask(self._full_q_mask[self.active_seq_mask])
                ctx.x = x[self.active_q_mask].view(B, -1, C)
            yield ctx
            if self._full_q_mask is not None:
                assert ctx.logits is not None and self.active_q_mask is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)), dtype=ctx.logits.dtype, device=ctx.logits.device
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)

    @contextmanager
    def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
                  attention_mask=None, position_ids=None):
        with super().attention(
            layer_idx, x, attn_norm, q_proj, k_proj, v_proj, attention_mask, position_ids
        ) as ctx:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            else:
                if layer_idx == 0:
                    active_seq_idx = torch.where(self.active_seq_mask)[0]
                    nz = self.active_q_mask.nonzero(as_tuple=False)
                    self._active_q_indices = (active_seq_idx[nz[:, 0]], nz[:, 1])
                self.key_cache[layer_idx][self._active_q_indices] = ctx.k.flatten(0, 1)
                self.value_cache[layer_idx][self._active_q_indices] = ctx.v.flatten(0, 1)
                ctx.k = self.key_cache[layer_idx][self.active_seq_mask]
                ctx.v = self.value_cache[layer_idx][self.active_seq_mask]

            if layer_idx == 0:
                self._q_position_ids, self._kv_position_ids = AttentionContext.select_position_ids(
                    position_ids, self.active_q_mask
                )
                self._attention_mask = AttentionContext.convert_attention_mask(
                    attention_mask, dtype=ctx.k.dtype,
                    query_length=ctx.q.shape[1],
                    key_value_length=self.value_cache[layer_idx].shape[1],
                )
            ctx.q_position_ids = self._q_position_ids
            ctx.kv_position_ids = self._kv_position_ids
            ctx.attention_mask = self._attention_mask
            yield ctx

            assert ctx.attn_weight is not None, 'set attn_implementation="eager" for rollout'
            if layer_idx == 0:
                L = self.key_cache[layer_idx].size(1)
                self._attn_rollout = torch.eye(L, device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
            self.accumulate_attn_rollout(ctx.attn_weight)

    def top_up_mask(self, q_mask):
        q_mask = q_mask.clone()
        num_selected_per_seq = q_mask.sum(dim=-1)
        _, G = self._density_score.shape
        if torch.any(num_selected_per_seq != num_selected_per_seq.max()):
            combined_scores = torch.where(
                q_mask, -torch.inf, self._global_importance[self.active_seq_mask]
            )
            combined_scores[:, -G:] += (
                combined_scores.max() + self._density_score[self.active_seq_mask]
            )
            top_up_mask_(q_mask, int(num_selected_per_seq.max()), combined_scores)
        return q_mask

    def accumulate_attn_rollout(self, attn_scores):
        """One rollout step: W = normalize(E + I); C <- W @ C."""
        B, _, _, seq_len = attn_scores.shape
        device, dtype = attn_scores.device, attn_scores.dtype
        if self.active_q_mask is None:
            E = attn_scores.mean(dim=1)
        else:
            E = torch.eye(seq_len, device=device, dtype=dtype).repeat(B, 1, 1)
            E[self.active_q_mask] = attn_scores.mean(dim=1).reshape(-1, seq_len)
        W = E + torch.eye(seq_len, device=device, dtype=dtype)
        W = W / W.sum(dim=-1, keepdim=True)
        self._attn_rollout = W @ self._attn_rollout

    def on_step_end(self, block_mask, frame, delta):
        confidence = delta.confidence
        assert confidence is not None
        B, P = frame.prompts.shape
        B_active, G = confidence.shape
        T = P + G
        block_mask = block_mask[self.active_seq_mask]
        new_frame = frame.apply_delta(delta)
        device = confidence.device
        remaining_mask = new_frame.generated_tokens[self.active_seq_mask] == self.mask_token_id

        if self._conf_cache is None:
            self._conf_cache = confidence
        elif self.active_q_mask is not None:
            valid = self.active_q_mask[:, P:] & (
                frame.generated_tokens[self.active_seq_mask] == self.mask_token_id
            )
            active_conf = self._conf_cache[self.active_seq_mask]
            active_conf[valid] = confidence[valid]
            self._conf_cache[self.active_seq_mask] = active_conf

        block_size = block_mask.sum(dim=1, keepdim=True)
        meets_target = torch.cumsum(remaining_mask.int(), dim=1) >= self.current_k
        min_end = torch.argmax(meets_target.int(), dim=1, keepdim=True)
        min_end[~meets_target.any(dim=1, keepdim=True)] = G - 1
        search_end = (((min_end // block_size) + 1) * block_size) - 1
        block_start = torch.argmax(block_mask.int(), dim=1, keepdim=True)
        col = torch.arange(G, device=device)
        search_mask = (col >= block_start) & (col <= search_end)

        scores = self._conf_cache[self.active_seq_mask] * certainty_density(~remaining_mask, self.sigma)
        scores[block_mask] += scores.max()
        _, idx = torch.topk(
            torch.where(search_mask & remaining_mask, scores, -torch.inf),
            k=min(self.current_k, G), dim=-1,
        )
        selected_mask = torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(1, idx, True) & remaining_mask
        response_mask = F.pad(selected_mask[:, 1:], (0, 1), value=False) if is_adapted_from_ar(self.model_config) else selected_mask

        transfer_src_index = (
            delta.transfer_src_index
            if delta.transfer_src_index is not None
            else delta.transfer_index
        )
        lengths = torch.tensor(
            [ti.numel() for ti in transfer_src_index if ti.numel() > 0], device=device
        )
        row_indices = torch.repeat_interleave(
            torch.arange(B_active, device=device), lengths
        )
        col_indices = torch.cat(transfer_src_index)
        response_mask[row_indices, col_indices] = True

        q_mask = F.pad(response_mask, (P, 0), value=False)
        global_importance = self._attn_rollout.sum(dim=1)
        q_mask |= nucleus_select(global_importance, self.rollout_p, mask=~q_mask)
        if is_adapted_from_ar(self.model_config):
            q_mask[:, P - 1] = selected_mask[:, 0]

        if self.inflate_w > 0:
            arange_t = torch.arange(T, device=device).expand(B_active, -1)

            masked_indices_next = torch.where(q_mask, arange_t, T)
            next_selected_indices = torch.cummin(
                torch.flip(masked_indices_next, dims=[-1]), dim=-1
            ).values
            next_selected_indices = torch.flip(next_selected_indices, dims=[-1])
            dist_to_next_true = next_selected_indices - arange_t

            masked_indices_prev = torch.where(q_mask, arange_t, -1)
            prev_selected_indices = torch.cummax(masked_indices_prev, dim=-1).values
            dist_to_prev_true = arange_t - prev_selected_indices

            gap_len = dist_to_next_true + dist_to_prev_true
            q_mask |= (
                (gap_len <= self.inflate_w)
                & (prev_selected_indices >= 0)
                & (next_selected_indices < T)
            )

        if self._full_q_mask is None:
            self._full_q_mask = q_mask
            self._global_importance = global_importance
            self._density_score = scores
        else:
            self._full_q_mask[self.active_seq_mask] = q_mask
            self._global_importance[self.active_seq_mask] = global_importance
            self._density_score[self.active_seq_mask] = scores
```

So the causal chain. I started stuck: a diffusion LM recomputes the whole sequence's KV every step because bidirectional attention makes the exact prefix cache illegal — flipping one mask changes everyone's context. The existing fix, segment-level approximate caching on fixed schedules, is too coarse: one rule per segment both reuses KV that's moving and refreshes KV that's settled, and it needs per-dataset tuning. Looking at the actual dynamics, a masked token's KV moves in three phases and only really needs refreshing in the rapid-change phase right before it's decoded — a per-token statement no segment can capture. That created a chicken-and-egg: I must refresh rapid-change tokens before knowing which they are. The locality of decoding broke it — the next token is decoded near the last, so imminence is the density of known tokens nearby, which I turned into a Gaussian certainty density `D(i)` and combined with prediction confidence into `D·s` to pick the masked tokens to refresh (and, as a bonus, a quasi left-to-right decoding order that curbs premature EOS). That criterion is blind to prompt and decoded tokens, which need a different reason to refresh — they're heavily attended — so a second stage measures global importance with layer-composed attention rollout (`W = normalize(E + I)`, `C = W·C`, `c_j = Σ_i C_ij`) and nucleus-selects the salient ones, justified because dLLM attention is concentrated and stable across adjacent steps. The union is the recompute set; everything else reuses cache; freshly decoded tokens are forced in; small gaps are inflated for kernel efficiency; the only extra memory is an `L×L` matrix; and because it touches no weights it's a training-free wrapper that drops onto any dLLM.
