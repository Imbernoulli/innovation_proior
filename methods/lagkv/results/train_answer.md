When I serve a long-context decoder-only transformer, the KV cache hurts me on two axes simultaneously. Every layer and every key-value head stores the Key and Value vector of every token seen so far, so on a thirty-thousand-token prompt the cache rivals the model weights in size; and every decode step attends over the entire cache, so latency grows linearly in cache length while prefill attention is quadratic in the prompt. What I want is concrete: after prefill, rank the cached tokens by importance, keep only a small fixed-budget slice — say 20% — and decode from that reduced cache without the long-context task quality collapsing. The whole problem is to manufacture, out of the cached tensors alone, a per-token importance score, rank, keep the top fraction, evict the rest. But a usable solution must satisfy three constraints at once, and each constraint kills a different line of prior art. First, it must not need the attention matrix: production inference runs on FlashAttention, which is fast precisely because it never materializes the $n\times n$ matrix $A_i = \mathrm{softmax}(Q_i K_i^\top/\sqrt{d_h})$, so any score that is a function of attention weights simply cannot be computed in the stack I run. Second, it must be query-independent: I compress a context once and reuse it across turns and across different instructions, so the kept set cannot depend on where the question sits or what it asks. Third, it must actually cut compute, not just memory, which means dropping tokens rather than shrinking them.

These three rule out the dominant methods. The attention-based scorers — H2O, which accumulates the attention each token receives and keeps the power-law "heavy hitters," and SnapKV, which uses an observation window at the end of the prompt to see which prefix positions the query attends to — both work well but both need $A_i$ (so they are incompatible with FlashAttention) and both are instruction-dependent (H2O accumulates during decoding, SnapKV scores against the question window), so the kept set changes with the query and must be recomputed for every turn. Quantization (KIVI, CacheGen) stores the same tokens at 2 bits and genuinely shrinks memory, but it keeps every historical token, so the number of attention operations — the quadratic prefill, the linear-in-cache decode — is unchanged; it fixes memory and does nothing for compute. StreamingLLM is the cheap, robust baseline that fits all three constraints: keep the first few "sink" tokens plus a recent sliding window, evict the middle. It is query-free, attention-free, FlashAttention-compatible, and the sinks are principled — softmax forces a query's weights to sum to one, so a query with no strong match dumps the residual mass on the first few tokens, which are visible to every later position and become trained-in attention sinks. But its eviction is indiscriminate: anything between the sink and the recent window is dropped regardless of what it carries, so a needle buried in the middle of a long haystack is simply gone and retrieval collapses. StreamingLLM gives me the kernel-friendly, query-free skeleton — sink plus recent window — but not the part I actually need: a way to decide, among the middle tokens, which to keep. The one prior attempt at scoring from the cache alone, the L2-norm heuristic (keep keys with low L2 norm, since low key norm correlates with high later attention), is the right spirit but too thin: a single static scalar per token from the key alone, ignoring the value, with no notion of where the token sits or how it relates to its neighbors.

I propose LagKV, an attention-free, query-independent eviction score computed entirely by comparison among the cached K and V tensors. The starting raw material is what the quantization work already mapped about how these tensors are distributed. Token-wise locality (CacheGen): within a layer and channel, tokens close in position have nearly identical K/V values — the delta between consecutive tokens piles up tightly near zero — which falls straight out of the model being autoregressive, since the next representation does not jump abruptly from the previous one. Key/value asymmetry (KIVI): the key cache has a few fixed channels with persistently enormous magnitude (per-channel outliers), while the value cache has no such channel pattern and varies per token; keys carry structure along the channel axis, values along the token axis. The channel-outlier fact is a trap I must avoid first: if a handful of key channels are always giant, any naive magnitude- or variance-based score over the raw key vector is dominated by those same channels for every token, measuring nothing but the persistent channel norm leaking through. So before I can read importance off the spread of a token's vector, I have to strip the channel-specific scale by normalizing each channel — and the only honest statistics I have, with no labels and no query, come from locality. Because adjacent tokens barely move, a short contiguous chunk is a tight cloud whose per-channel min and max faithfully describe the local scale of each channel; min-max normalizing a chunk per channel maps each channel into roughly $[0,1]$ using neighborhood statistics, collapsing the giant channels onto the same footing as the rest and leaving each channel's relative position within the local cloud.

But which chunk's min and max? Normalizing a chunk by its own statistics is self-referential — I would be asking which token is unusual relative to itself and its chunk-mates, and worse, the min and max are achieved by tokens in the chunk, pinning those envelope-defining tokens to exactly 0 or 1 by construction. I need an external yardstick the scored tokens did not get to set, and locality hands it to me for free: the next chunk. Because the model is autoregressive and adjacent tokens barely move, the chunk immediately after the current one is a faithful sample of the same local distribution — the current chunk's tokens are "supposed to" look like the next chunk's. So I normalize the current chunk's K and V using the min and max of the next chunk, and the question I ask each token becomes sharp and query-free: how coherent is this token with what comes right after it? A token that sits comfortably inside the next chunk's envelope is predictable — the local flow was going to produce it anyway, so dropping it loses little — while a token that, even after normalizing by the next chunk's scale, is still strange is a discontinuity carrying information the next chunk does not explain. This is the "lag-relative" idea: score chunk $p$ against the reference statistics of chunk $p+1$, the lag.

Concretely, partition the cache after the first $S$ sink tokens into contiguous windows of size $L$. For window $p$, head $i$, and $Z \in \{K, V\}$ (a chunk of $L$ tokens by $d_h$ channels), take the next window $Z_i^{p+1}$ and collapse its $L$ tokens to one min and one max per channel along the token (seq) axis,
$$\min{}_i^{p,Z} = \min_{\mathrm{seq}}\big(Z_i^{p+1}\big),\qquad \max{}_i^{p,Z} = \max_{\mathrm{seq}}\big(Z_i^{p+1}\big),$$
then normalize the current chunk channel-wise into that reference frame,
$$\bar Z_i^{p} = \frac{Z_i^{p} - \min{}_i^{p,Z}}{\max{}_i^{p,Z} - \min{}_i^{p,Z}}.$$
The min over seq per channel is exactly what removes the persistent giant key channels — each channel is divided by its own local range — so the remaining spread is no longer a proxy for which raw channels are huge. Now I read importance off that spread: for a single token, the standard deviation over its $d_h$ channel entries is one number. A coherent token has all channels in a similar normalized position, hence small channel-wise std; an incoherent, surprising token has channels scattered across the range, hence large std. This per-token channel-wise std is the importance signal, well-defined only because the channel norms were stripped first. I do this for keys and for values, because KIVI says the two caches carry complementary structure — a token can be a discontinuity in the key flow, the value flow, or both — and I want to catch all three, so I produce a key score and a value score and add them. Before adding, a softmax over the $L$ tokens of the window does two useful things: it gives a common per-window scale, and because exp is convex it sharpens the tail, separating the genuinely high-std outliers from the merely above-average, which is exactly what I want when keeping only a small fraction:
$$\mathrm{score}(Z_i^p) = \mathrm{Softmax}_{\mathrm{tokens}}\big(\mathrm{Std}_{\mathrm{channel}}(\bar Z_i^p)\big),\qquad \mathrm{score}_i^p = \mathrm{score}(K_i^p) + \mathrm{score}(V_i^p).$$
Dividing this sum by a positive constant (the code uses $/2$) leaves the ranking unchanged.

Two pieces of bookkeeping make this safe and make it fit the harness. The bookkeeping first: the sinks — the first $S$ tokens — are never scored or evicted (the StreamingLLM anchor), and the very last chunk has no chunk after it to reference, so it cannot be scored — which is not a defect but precisely the recent sliding window I wanted, since recent tokens matter most and should never be compressed. So the always-kept tail is one full lag window plus whatever remainder is left when $q_{\text{len}}-S$ is not a multiple of $L$; if there is not even enough sequence for one scored chunk and one reference chunk ($q_{\text{len}} < S + 2L$), I skip compression entirely. The index arithmetic runs the scored region from $S$ to $\text{end\_idx} = S + \lfloor(q_{\text{len}}-S)/L\rfloor\,L$, with $\text{tail\_len} = L + (q_{\text{len}} - \text{end\_idx})$, and reshapes $[S:\text{end\_idx}]$ into partitions so chunk $p$ is scored by chunk $p+1$ in one vectorized pass. The harness asks for an overall retained fraction, not a per-window count, so I invert the retained-length budget for sequence length $L_s \ge S + 2L$ and per-window retention $r$,
$$L_R = S + rL\Big(\big\lfloor (L_s - S)/L\big\rfloor - 1\Big) + L + \mathrm{Mod}(L_s - S,\, L),\qquad C = 1 - L_R/L_s,$$
with $C = 0$ below $S + 2L$; the $+L + \mathrm{Mod}(\cdot)$ is the always-kept tail, and the $\lfloor(L_s-S)/L\rfloor - 1$ counts the post-sink windows minus the last one that is never compressed. The defaults are $S = 4$ (the sink mass concentrates on the first few tokens), $L = 128$, chosen short enough that locality holds yet large enough to estimate per-channel min/max and to make $rL$ cover the contiguous information span the task needs.

The second piece is what actually decides the implementation. What I want is top-$K$ per partition per head — keep $rL$ of each window's $L$ tokens so every region contributes its share — but the harness does the dead-simple thing: one global $\mathrm{topk}(n_{\text{kept}})$ over the whole cached sequence. A naive global top-K would not respect per-window quotas, because partitions at different layers/positions have different raw std scales, and a partition with systematically larger std could steal the whole budget. Rather than special-case the harness, I make the scores carry the per-partition structure so that one global top-K is a per-partition top-K. The fix is to replace each window's raw scores by their within-window rank: argsort the scores along the $L$ axis, argsort again — the double-argsort returns each token's rank position among its $L$ chunk-mates — and divide by $L$,
$$\mathrm{score} = \mathrm{argsort}(\mathrm{argsort}(\mathrm{score}))/L \in [0, 1).$$
Now every window has the identical set of values $\{0, 1/L, \dots, (L-1)/L\}$, just permuted by importance, so they are identically distributed across windows; once the sinks and tail have taken their guaranteed $1.0$ slots and the budget is set from the retained-length formula, the remaining global top-K is forced to take the same high-rank count $rL$ from every scored window. It also puts every scored token strictly in $[0,1)$, below the sinks and tail, which I fix at $1.0$ so they are always kept. The alternative — skip the rank step and let raw softmax-std scores compete globally (`cross_scoring=True`) — is a deliberate option for when I want budget to flow toward windows with larger raw outliers, but it loses the uniform-per-window guarantee, so it is off by default. Finally, the skip branch must degrade gracefully: when compression is skipped but the harness still forces a top-K, I hand back a ramp — ones for the sinks, and an increasing $\mathrm{arange}/(q_{\text{len}}-S)$ for the rest — so any forced top-K keeps the recent tail, which is exactly StreamingLLM behavior as the degenerate case. The result is an attention-free, query-free token score computed by straightforward comparison among the KV tensors themselves, dropped straight into the score-then-keep eviction harness as one `score` function.

```python
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ScorerPress:
    """Base: score tokens, keep the top n_kept globally. Subclass supplies score()."""

    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(self, module: nn.Module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values


@dataclass
class LagKVPress(ScorerPress):
    """LagKV: lag-relative, attention-free, query-free KV eviction score."""

    compression_ratio: float = 0.0
    n_sink: int = 4
    lag_size: int = 128
    cross_scoring: bool = False

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_key_value_heads, q_len, d = keys.shape

        if q_len < self.n_sink + 2 * self.lag_size:
            # too short for a scored chunk + a reference chunk: skip compression,
            # but ramp the post-sink scores so any forced top-K keeps the recent tail.
            score = torch.ones((bsz, num_key_value_heads, q_len), dtype=keys.dtype, device=keys.device)
            if q_len > self.n_sink:
                score[:, :, self.n_sink:] = (
                    torch.arange(q_len - self.n_sink, device=keys.device) / (q_len - self.n_sink)
                ).to(keys.dtype)
            return score

        end_idx = self.n_sink + ((q_len - self.n_sink) // self.lag_size) * self.lag_size
        tail_len = self.lag_size + q_len - end_idx   # last full window + remainder = sliding tail

        key_score = self._get_states_score(
            keys[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        value_score = self._get_states_score(
            values[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        score = (key_score + value_score) / 2        # same ordering as summed K+V

        if not self.cross_scoring:
            # within-window rank / L: under the aligned budget,
            # one global top-K becomes a per-window top-K at ratio r
            score = score.argsort(dim=-1).argsort(dim=-1) / self.lag_size
            score = score.to(keys.dtype)

        sink_score = torch.ones((bsz, num_key_value_heads, self.n_sink), dtype=score.dtype, device=score.device)
        tail_score = torch.ones((bsz, num_key_value_heads, tail_len), dtype=score.dtype, device=score.device)
        return torch.cat((sink_score, score.reshape(bsz, num_key_value_heads, -1), tail_score), dim=-1)

    def _get_states_score(self, target_v):
        # target_v: (b, h, num_partitions, L, d_h). Score chunk p by chunk p+1's stats (the lag).
        ref = target_v[:, :, 1:, :, :]    # next chunk = reference
        v = target_v[:, :, :-1, :, :]     # current chunk = scored
        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        # channel-wise normalize into the next chunk's frame; channel-wise std per token; softmax over tokens
        return ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)
```
