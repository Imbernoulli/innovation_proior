The KV cache is what makes autoregressive decoding cheap, but it grows linearly with sequence length, batch size, and depth, and for the long generations that matter most — multi-step reasoning chains, long-document QA, many concurrent decode paths — it becomes both the capacity and the bandwidth bottleneck. We want to store the cached keys and values at two or four bits instead of sixteen, and we want to do it the tuning-free way: take a group of cached numbers, find their min and max, and round each onto a $b$-bit grid via $\mathrm{qtz}(x) = \mathrm{round}((x-m)/\Delta)$ with zero-point $m=\min$ and scale $\Delta=(\max-\min)/(2^b-1)$, dequantizing as $\mathrm{deq}(\bar x)=\bar x\,\Delta + m$. No retraining, no calibration pass, streaming-friendly. The trouble is that every existing tuning-free quantizer — KIVI, GEAR, KVQuant, ZipCache — implicitly optimizes the same thing: reconstruction error. It picks groupings (per channel for keys, since a few channels carry persistent large magnitudes and per-channel grouping confines those outliers; per token for values, which have no such fixed-channel pattern) so that $\lVert k - \mathrm{deq}(k^{\mathrm{qtz}})\rVert_2$ is as small as possible. And at low bits, long reasoning chains still fall apart.

The reason they fall apart is that a cached key $k_i$ is never read as $k_i$. It is only ever consumed inside attention, where the new token's query $q_n$ dots it: the score is $q_n\!\cdot\!k_i^\top/\sqrt{d}$, the scores pass through a softmax, and the softmax weights average the values. So the quantity we should preserve is the attention *output*, not the key's Euclidean faithfulness. These are not the same thing. Two quantized keys with identical $\lVert k - \mathrm{deq}\rVert_2$ can disturb the next token's score by wildly different amounts: if the residual $r_i = k_i - \mathrm{deq}(k_i^{\mathrm{qtz}})$ points along $q_n$, the score shifts by $q_n\!\cdot\!r_i^\top$, which can be large; if $r_i$ is perpendicular to $q_n$, the score does not move at all. The compression criterion is blind to the *direction* of the residual, and direction is precisely what attention cares about. It spends bits making $k$ numerically faithful in every direction equally, when the model only needs it faithful in the directions the queries actually probe.

I propose SQuat — subspace-orthogonal KV cache quantization — which quantizes so the attention output stays close to full precision by making each key residual orthogonal to the directions the queries use, rather than small in every direction. The starting point is to make the intuition exact. Writing the pre-softmax score vector $w=[q_n\!\cdot\!k_i^\top/\sqrt d]_i$ and its dequantized counterpart $w^{\mathrm{deq}}$, stacking the values into $V$ and $V^{\mathrm{deq}}$, the attention output error is $\lVert \mathrm{softmax}(w)V - \mathrm{softmax}(w^{\mathrm{deq}})V^{\mathrm{deq}}\rVert_2$. Adding and subtracting the cross term $\mathrm{softmax}(w)V^{\mathrm{deq}}$ and applying the triangle inequality splits it into a value piece and a score piece. The value piece $\mathrm{softmax}(w)(V-V^{\mathrm{deq}})$ is a convex combination of the rows $v_i - v_i^{\mathrm{deq}}$, since softmax is a probability vector, so its norm is at most $\sum_i\lVert v_i - v_i^{\mathrm{deq}}\rVert_2$ — governed entirely by value reconstruction error, which justifies plain per-token value quantization. The score piece $(\mathrm{softmax}(w)-\mathrm{softmax}(w^{\mathrm{deq}}))V^{\mathrm{deq}}$ is bounded using the fact that softmax is $\tfrac12$-Lipschitz (its Jacobian $\mathrm{diag}(p)-pp^\top$ has operator norm at most $\tfrac12$) and $\lVert w - w^{\mathrm{deq}}\rVert_2 \le (1/\sqrt d)\sum_i|q_n\!\cdot\!r_i^\top|$. Together,

$$\lVert \mathrm{Attn}(\text{orig}) - \mathrm{Attn}(\text{quant})\rVert_2 \;\le\; \frac{\lVert V^{\mathrm{deq}}\rVert_F}{2\sqrt d}\sum_i\big|q_n(k_i-\mathrm{deq}(k_i^{\mathrm{qtz}}))^\top\big| \;+\; \sum_i\lVert v_i-\mathrm{deq}(v_i^{\mathrm{qtz}})\rVert_2.$$

The output is preserved when (i) value residuals are small — per-token min-max, settled — and (ii) the *inner products* $q_n\!\cdot\!r_i^\top$ are small, i.e. the key residual is orthogonal to the future query, not merely small in norm. That is a different objective from every compression baseline, and it is the lever.

The obvious objection is that at the moment we quantize token $i$'s key, the future queries $q_n$ for $n>i$ do not exist yet. The escape is structural: trained queries do not fill $d$-dimensional space — they cluster in a low-dimensional subspace, and the subspace spanned by the *prompt's* queries (available immediately after prefill) essentially coincides with the subspace spanned by prompt-plus-response queries. So I do not need the individual future query; I need the subspace future queries will live in, and the prompt gives it to me. After prefill I have $Q = X_0 W^Q$; take its SVD $Q = U\Sigma V^\top$ and define the subspace matrix $\hat Q = \mathrm{diag}(\sigma_1,\dots,\sigma_r)\,V[{:}r] \in \mathbb{R}^{r\times d}$ — the top-$r$ right singular vectors *scaled by their singular values*. The scaling is deliberate: I do not want to protect all $r$ directions equally, I want to protect the directions queries load onto most. Since the penalty involves $\hat Q^\top\hat Q = \sum_{j\le r}\sigma_j^2 v_j v_j^\top$, a big singular value gets a quadratically bigger weight, exactly encoding which directions matter to attention. For grouped-query attention, the query heads sharing a KV head are concatenated before the SVD, since the subspace must cover every query that will ever dot this head's keys.

With the subspace in hand, the honest key objective is $\min_{k^{\mathrm{qtz}}} \lVert k - \mathrm{deq}(k^{\mathrm{qtz}})\rVert_2$ subject to $\hat Q(k-\mathrm{deq}(k^{\mathrm{qtz}}))=0$ with $k^{\mathrm{qtz}}$ a vector of $b$-bit integers. That is combinatorial and integer-constrained, and a hard constraint is too rigid — the grid may contain no nearby point that kills every protected projection. So I relax orthogonality into a penalty with a knob $\lambda$:

$$\min \;\lVert k-\mathrm{deq}(k^{\mathrm{qtz}})\rVert_2^2 \;+\; \lambda\,\lVert \hat Q(k-\mathrm{deq}(k^{\mathrm{qtz}}))\rVert_2^2,$$

where $\lambda=0$ recovers pure reconstruction (the compression baseline) and $\lambda\to\infty$ drives the residual into the orthogonal complement of $\hat Q$. This defines the metric $P = I + \lambda\,\hat Q^\top\hat Q$, which is $I$ plus a PSD matrix, hence positive definite and invertible. To handle the integer grid inside it, I quantize coordinates in fixed order and *compensate*: after rounding a coordinate, the not-yet-quantized coordinates are still continuous and free to push the total residual back toward orthogonality. At step $t$, with $\delta = \hat k_t - \hat k_{t-1}$, the per-step problem is $\min_\delta \delta^\top P\delta$ subject to $T\delta = b$, where $T=[I,0]$ selects the first $t$ coordinates, frozen to zero, and $b$ pins coordinate $t$ to its rounding error $e_t$. The Lagrangian $L(\delta,\mu)=\delta^\top P\delta - \mu^\top(T\delta-b)$ gives stationarity $\delta = \tfrac12 P^{-1}T^\top\mu$ and feasibility $\mu = 2(TP^{-1}T^\top)^{-1}b$, so

$$\delta = P^{-1}T^\top (T P^{-1}T^\top)^{-1} b.$$

Blocking $P_{\mathrm{inv}} = P^{-1} = \begin{bmatrix}A_t & B_t^\top\\ B_t & C_t\end{bmatrix}$ aligned with the first-$t$/remaining split, we have $P^{-1}T^\top = [A_t; B_t]$ and $TP^{-1}T^\top = A_t$, so $P^{-1}T^\top(TP^{-1}T^\top)^{-1} = [I; B_t A_t^{-1}]$. With $b=[0;e_t]$, the first $t$ coordinates come out as $b$ itself (the frozen/pinned constraint) and the remaining coordinates update by $B_t h_t\, e_t$, where $h_t$ is the last column of $A_t^{-1}$. The rounding error is spread into the future coordinates in exactly the direction that minimizes the penalized residual.

This algebra is recognizable: $\min \delta^\top H\delta$ under a quadratic metric, quantize a coordinate, update the remaining coordinates by a rounding-error vector times inverse-metric blocks — it is the Optimal Brain Surgeon / OBQ error-feedback update (Hassibi et al.; Frantar & Alistarh's OBC; GPTQ), but with one substitution: the data Hessian $2XX^\top$ is replaced by the query-subspace metric $P = I + \lambda\hat Q^\top\hat Q$, which charges residual components along the query subspace by $(1+\lambda\sigma^2)$ and leaves the orthogonal complement at unit weight. That recognition also rescues efficiency. OBS in pure form quantizes greedily, with a per-row remaining set that forces a re-inversion per row — hopeless on-the-fly. But the GPTQ insight is that on large matrices, fixed-order quantization is nearly as good. In fixed order, $A_t$ and $A_t^{-1}$ depend only on $P$ — not on the data, not on which key — so they are identical for every key, token, and sample in the batch, computed once after prefill and reused for the whole decode.

For the inverse recursion, inverting each $A_t$ from scratch is $\sum_t t^3 = O(d^4)$, too slow. The $A_t$ are nested top-left blocks, so $A_t^{-1}$ downdates from $A_{t+1}^{-1}$ by one Schur step: writing $A_{t+1}^{-1} = \begin{bmatrix}M & N^\top\\ N & O\end{bmatrix}$ split off the trailing block, the block-inverse identity gives $A_t^{-1} = M - N^\top O^{-1} N$, which for $g=1$ is the single Gaussian-elimination step $A_t^{-1} = \bar A_{t+1} - (1/\bar a)\,a^\top a$. Starting at $A_T^{-1} = P = I + \lambda\hat Q^\top\hat Q$ (since $A_T$ is all of $P_{\mathrm{inv}}$, its inverse is $P$) and recursing down to $A_1^{-1}$, each downdate is $O(t^2)$ and the sweep is $O(d^3)$. To avoid coordinate-by-coordinate work entirely, I quantize a block of $g$ coordinates per iteration: $b$ carries a length-$g$ rounding-error vector $d_t$, $A_t \in \mathbb{R}^{tg\times tg}$, $H_t$ is the last $g$ columns of $A_t^{-1}$, $B_t$ is the bottom-left block of $P_{\mathrm{inv}}$, and the remaining-coordinate update is $+\,B_t H_t d_t$. With head dimension $128$ and $g=64$, that is $T=2$ blocks and a single compensation step: quantize block 1, compute its rounding error $d_t$, push $B_1 H_1 d_t$ into block 2, quantize block 2. Values stay on the plain per-token min-max path with the most-recent residual window kept FP16.

The knobs follow from the structure. $\lambda \approx 10^{-3}$: because $\hat Q$ carries *scaled* singular vectors, $\lambda\hat Q^\top\hat Q$ is already large at small $\lambda$, so this scale protects the dominant query directions without letting orthogonality swamp reconstruction. $r=60$: long-context prompts span a richer query subspace than short reasoning prompts (where a handful of directions suffice), so $r$ must be large enough to cover the span future queries will probably use, but not so large it over-constrains the residual. $g=64$ on a $128$-dim head gives $T=2$, hence one compensation step, for on-the-fly cost. $G=32$ and $R=32$ are inherited from the streaming compression setup: the freshest tokens stay FP16 until the residual buffer flushes, the rest quantized in groups of $G$. Sharing $\hat Q$ (and hence $A_t^{-1}$, $P_{\mathrm{inv}}$) across the batch is justified by the same low-dimensional fact — queries from different sequences of the same task land in nearly the same subspace — so one set of matrices serves the batch. The whole thing runs training-free and calibration-free, with the metric coming from the prompt's own queries at prefill.

```python
import math
import torch

FP_BITS = 16


class AdaptiveKVQuantizer:
    """SQuat: subspace-orthogonal K/V quantization. Build a query subspace from the
    prompt during prefill; quantize keys block-by-block, compensating the remaining
    coordinates so the residual stays orthogonal to that subspace. Values per token."""

    def __init__(self):
        self.bits = 4
        self.group_size = 32
        self.residual_length = 32
        self.subspace_dim = 60        # r
        self.squat_lambda = 0.001     # λ
        self.quant_group_size = 64    # g  (T = d/g)
        self.shared_svd = True
        self.query_subspaces = {}

    def reset_request(self, request_meta, budget_state):
        self.query_subspaces = {}

    def needs_prefill_qkv_observer(self) -> bool:
        return True

    def query_observation_position(self) -> str:
        return "post_rope"

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        # Q-hat = diag(sigma[:r]) * V[:r], from the prompt queries
        if query_states is None:
            return None
        batch, query_heads, _, head_dim = query_states.shape
        kv_heads = int(attention_meta.get("kv_heads", query_heads))
        if query_heads % kv_heads != 0:                       # GQA: concat shared query heads
            kv_heads = query_heads
        matrix = query_states.reshape(batch, kv_heads, -1, head_dim).float()
        rank = min(int(self.subspace_dim), matrix.shape[-2], matrix.shape[-1])
        if rank <= 0:
            return None
        _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        scaled_vh = torch.diag_embed(singular_values[:, :, :rank]).matmul(vh[:, :, :rank, :])
        self.query_subspaces[layer_id] = (scaled_vh[0:1] if self.shared_svd else scaled_vh).detach()
        return None

    def _residual_keep_length(self, seq_len: int) -> int:
        residual_length = max(0, min(seq_len, int(self.residual_length)))
        return seq_len % residual_length if residual_length else 0

    def _minmax_last_dim(self, data, group_size, bits):
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, 2 ** int(bits) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int        # Delta = (max-min)/(2^b-1)
        q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]

    def _generate_At_inv(self, query_subspace, tol: float = 1e-7):
        # A_T^{-1} = I + lambda Q^T Q ; downdate A_t^{-1} = M - N^T O^{-1} N
        batch, heads, _, head_dim = query_subspace.shape
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)                 # T
        matrices = [None] * groups
        eye = torch.eye(head_dim, device=query_subspace.device, dtype=torch.float32)
        A_t = eye.expand(batch, heads, head_dim, head_dim) + float(self.squat_lambda) * \
            query_subspace.float().transpose(-1, -2).matmul(query_subspace.float())
        matrices[groups - 1] = A_t
        for group_idx in range(groups - 1, 0, -1):
            current_dim = group_idx * q_group
            width = min(q_group, A_t.shape[-1] - current_dim)
            M_t1 = A_t[:, :, :current_dim, :current_dim]
            N_t1 = A_t[:, :, current_dim:current_dim + width, :current_dim]
            O_t1 = A_t[:, :, current_dim:current_dim + width, current_dim:current_dim + width]
            local_eye = torch.eye(width, device=query_subspace.device, dtype=torch.float32)
            O_inv = torch.inverse(O_t1 + tol * local_eye.expand(batch, heads, width, width))
            A_t = M_t1 - N_t1.transpose(-1, -2).matmul(O_inv.matmul(N_t1))
            matrices[group_idx - 1] = A_t[:, :, :, -q_group:]  # H_t = last g cols of A_t^{-1}
        return matrices

    def _squat_quantize_keys(self, key_states, query_subspace):
        batch, heads, _, head_dim = key_states.shape
        query_subspace = query_subspace.to(device=key_states.device)
        if query_subspace.shape[0] == 1 and batch > 1:
            query_subspace = query_subspace.expand(batch, -1, -1, -1)
        if query_subspace.shape[1] != heads or query_subspace.shape[-1] != head_dim:
            raise ValueError("query subspace shape does not match the key tensor")
        matrices = self._generate_At_inv(query_subspace)
        P_inv = torch.inverse(matrices[-1])                    # (I + lambda Q^T Q)^{-1}
        work = key_states.float().clone()
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)
        for group_idx in range(groups):
            start = group_idx * q_group
            end = min(head_dim, start + q_group)
            chunk = work[:, :, :, start:end]
            dequant = self._minmax_last_dim(chunk.transpose(2, 3).contiguous(),
                                            self.group_size, self.bits).transpose(2, 3)
            if group_idx < groups - 1:
                d_vec = (dequant - chunk).float()              # block rounding error d
                next_start = end
                H_t = matrices[group_idx]                       # last g cols of A_t^{-1}
                B_t = P_inv[:, :, next_start:, :next_start]     # bottom-left block of P_inv
                update = d_vec.matmul(H_t.transpose(-2, -1)).matmul(B_t.transpose(-2, -1))  # B_t H_t d
                work[:, :, :, next_start:] = work[:, :, :, next_start:] + update
            work[:, :, :, start:end] = dequant
        return work

    def _quantize_with_residual(self, tensor, quant_fn):
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = self._residual_keep_length(seq_len)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        work[:, :, :quant_end, :] = quant_fn(work[:, :, :quant_end, :])
        avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id, key_states, cache_meta):
        query_subspace = self.query_subspaces.get(layer_id)
        if query_subspace is None:
            raise RuntimeError("key quantization needs the prefill query observer")
        return self._quantize_with_residual(
            key_states, lambda data: self._squat_quantize_keys(data, query_subspace))

    def quantize_value(self, layer_id, value_states, cache_meta):
        return self._quantize_with_residual(
            value_states, lambda data: self._minmax_last_dim(data, self.group_size, self.bits))

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float:
        residual = self._residual_keep_length(seq_len)
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
```
