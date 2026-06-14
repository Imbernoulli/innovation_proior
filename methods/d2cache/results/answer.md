# d2Cache (Dual aDaptive Cache), distilled

d2Cache is a training-free approximate key-value (KV) cache for diffusion language models. At each denoising step it forwards only an adaptively chosen set of query rows, overwrites the cached K/V at those positions, and reuses cached K/V everywhere else. The recompute set is chosen in two stages: masked tokens are selected by certainty density times prediction confidence, and the remaining prompt/decoded tokens are selected by attention-rollout influence.

## Problem it solves

A diffusion language model denoises a fixed-length prompt-plus-response sequence with full bidirectional attention. When a `[MASK]` position becomes a real token, every other token's context can change, so an exact causal prefix cache is invalid. Recomputing the whole sequence at every step is expensive; blindly reusing stale K/V loses quality. The cache policy must therefore decide which token states are stale enough to refresh and which are stable enough to reuse, without retraining the model.

## Stage 1: certainty prior-guided masked-token selection

For a masked position `i`, with current masked set `M`, define the position-aware certainty density over known positions:

```text
D(i) = sum_j exp(-(i - j)^2 / (2 sigma^2)) * 1{j not in M}.
```

This is a Gaussian sum over prompt or already-decoded positions. Nearby known tokens count more because they constrain the masked token's local context more strongly; larger `sigma` broadens the neighborhood until per-token differences wash out. The masked-token score is the product of this structural certainty and the model's top-token confidence:

```text
score(i) = D(i) * s^i,      M* = arg top_k_{i in M} score(i).
```

The default implementation uses `current_k = 32` and `sigma = 10.0`. The mathematical score is the Gaussian sum above; the helper evaluates the same distance-weighted known-token signal with an FFT convolution and boundary-aware kernel-mass normalization so edge positions remain comparable. In the cache policy, Stage 1 is also restricted to a search window inside the active response block, and scores in the current block are biased upward so the block does not starve.

The same `D(i) * s^i` score can be used as a decoding score: it favors tokens near known context and therefore induces a quasi left-to-right order without a hard block boundary.

## Stage 2: attention-aware remaining-token selection

For prompt and already-decoded tokens, prediction confidence is not defined. Their refresh criterion is global influence. At each layer, average attention over heads and expand the queried-row attention map into a full `L x L` matrix `E`: queried rows receive fresh attention, and unqueried rows receive the one-hot row `e_i`. Add the residual path and row-normalize:

```text
W^(l) = normalize_row(E^(l) + I),
C^(0) = I,
C^(l) = W^(l) C^(l-1),
c_j = sum_i C^(N)_{ij}.
```

The column sum `c_j` is the total influence routed through token `j`. Select `U` by nucleus/top-p on `c`: normalize, sort by influence, and take the smallest high-influence prefix whose cumulative mass reaches or exceeds the threshold `p`. The implementation applies the threshold with a validity mask and `min_k = 1` (default `rollout_p = 0.1`). This adapts the number of refreshed tokens to how concentrated the rollout distribution is.

## Update rule

The next-step query mask is the union of Stage 1 and Stage 2, plus all freshly transferred tokens because their embeddings just changed. For Dream-style AR-adapted models, selecting a masked response position also selects the preceding token needed by the shifted attention layout. The implementation optionally inflates small gaps between selected positions (`inflate_w = 4` by default, `0` disables it) and tops up shorter batch rows so all active sequences forward the same number of query rows.

## Implementation

The cache is implemented as a subclass of the generic cache hooks used by the denoising loop. The important mechanics are: narrow the hidden states to `active_q_mask`, scatter logits back to full length, overwrite cached K/V only at active query indices, compute rollout from eager attention weights, and build the next `_full_q_mask` at `on_step_end`.

```python
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from src.cache.base import dCache, AttentionContext
from src.utils import certainty_density, nucleus_select, top_up_mask_, is_adapted_from_ar


class d2Cache(dCache):
    def __init__(self, model_config, rollout_p=0.1, current_k=32, sigma=10.0, inflate_w=4):
        super().__init__(model_config)
        self.key_cache = []
        self.value_cache = []
        self._conf_cache = None
        self._full_q_mask = None
        self._density_score = None
        self._global_importance = None
        self.rollout_p = rollout_p
        self.current_k = current_k
        self.sigma = sigma
        self.inflate_w = inflate_w

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
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
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
                    attention_mask,
                    dtype=ctx.k.dtype,
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
            k=min(self.current_k, G),
            dim=-1,
        )
        selected_mask = torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(1, idx, True)
        selected_mask &= remaining_mask

        if is_adapted_from_ar(self.model_config):
            response_mask = F.pad(selected_mask[:, 1:], (0, 1), value=False)
        else:
            response_mask = selected_mask

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

## Relation to prior methods

dLLM-Cache uses prompt/response segment refresh intervals plus a value-drift partial refresh; Fast-dLLM uses block-wise semi-autoregressive prefix/dual caching; dKV-Cache uses delayed scheduled caching. d2Cache replaces fixed segment schedules with per-token, per-step selection: certainty-prior selection for masked tokens and rollout-influence selection for prompt/decoded tokens.
