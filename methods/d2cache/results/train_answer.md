Diffusion language models generate text by iteratively denoising a fixed-length block of [MASK] tokens with full bidirectional attention over the entire prompt-plus-response sequence. This bidirectional structure is what lets them use future context and decode tokens in any order, but it also breaks the autoregressive prefix KV cache: when one masked position becomes a concrete token, every other position's context changes, so every key and value must in principle be recomputed at every layer on every step. Existing approximate caches cut the sequence into coarse segments—static prompt versus dynamic response, or whole blocks in semi-autoregressive decoding—and refresh them on fixed schedules. That granularity is wrong for the actual dynamics: inside any one segment, some tokens have already settled and are safe to reuse, while others are about to change rapidly and must be refreshed. A segment-level rule therefore both wastes compute on stable tokens and lets stale KV bleed quality from active ones.

The fix is to decide reuse and recompute at the resolution where the dynamics actually live, which is per token and per step. Tracing a masked token's KV state across decoding reveals three phases: a long gradual-change phase where it barely moves, a short rapid-change phase right before it is decoded, and a stable phase afterward where it is essentially frozen. A token only needs its KV refreshed during that rapid-change window. The challenge is identifying which masked tokens are entering that window before they are actually decoded. That seems circular, but decoding order in these models is spatially localized: the next token decoded is almost always near the one just decoded. So imminence can be read from the local context—specifically, from how densely a masked position is surrounded by already-known tokens. This structural signal, combined with the model's own prediction confidence, selects exactly the masked tokens whose KV is worth refreshing.

The method is called d2Cache, short for Dual aDaptive Cache. It is a training-free approximate KV cache that wraps any pretrained diffusion LM at inference time. At each denoising step it forwards only a small, adaptively chosen set of query rows, writes the fresh K/V projections back into a per-layer cache at those positions, and reuses cached K/V everywhere else. The active query set is built in two stages because the two token populations obey different rules. For masked tokens, d2Cache computes a certainty density D(i) that weights known positions near position i by a Gaussian in their separation distance, then multiplies by the model's top-token confidence s^i to score each masked position. The top-k positions by D(i)·s^i form the first part of the recompute set. For prompt and already-decoded tokens, which have no prediction confidence, d2Cache measures global importance through attention rollout: it composes per-layer attention maps with the residual identity path, producing an end-to-end influence matrix, and takes column sums to rank tokens by how much total influence flows through them. A nucleus-style top-p selection on these scores yields the second part of the recompute set. The union, plus any tokens freshly transferred from mask to concrete token, is forwarded; all other positions reuse their cached KV. Optionally, small gaps between selected positions are filled so the forward runs efficiently on near-contiguous rows.

The first stage captures the rapid-change masked tokens before they are decoded. The Gaussian density naturally prefers tokens surrounded by known context, and because decoding proceeds locally, those are exactly the tokens about to be resolved. Multiplying by confidence ensures the model has a clear opinion about them. The same D(i)·s^i score can also guide decoding order, producing a quasi-left-to-right progression without hard block boundaries and mitigating premature end-of-sequence overconfidence. The second stage captures attention sinks among the known tokens. Rollout is necessary because raw last-layer attention is diffuse and unfaithful; composing layers through the residual path gives a stable, interpretable importance ranking. Importance is also stable across adjacent steps, so current-step attention reliably tells us which known tokens to keep fresh on the next step. Because the method never touches model weights—density comes from the mask layout, confidence and attention from the forward already run—it is a drop-in inference wrapper.

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
