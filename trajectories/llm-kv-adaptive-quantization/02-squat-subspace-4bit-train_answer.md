The KIVI floor landed 37.68 on HotpotQA, 61.08 on passage retrieval, 47.71 on RepoBench, 65.31 on NIAH, and a soft 31.84 on GSM8K, all at 4.1875 effective bits ($\approx 3.82\times$). It holds quality where local-token noise is tolerable but cannot buy much compression, and its real flaw is conceptual: it optimizes *reconstruction* error, group by group it minimizes $\lVert k - \text{deq}(k) \rVert_2$, and that criterion is blind to the direction of the residual. A quantized key $k_i$ is never read *as* $k_i$ — it is only ever consumed inside attention, where the new token's query $q_n$ dots it, the score $q_n \cdot k_i^\top/\sqrt{d}$ goes through a softmax, and the softmax averages the values. Two quantized keys with identical reconstruction error can disturb the next token's score by wildly different amounts: if the residual $r_i = k_i - \text{deq}(k_i)$ points along $q_n$ the score shifts by $q_n \cdot r_i^\top$, possibly large; if $r_i$ is perpendicular to $q_n$ the score does not move at all. The floor spends bits making $k$ faithful in *every* direction equally, when the model only needs it faithful in the directions the queries probe — and that wasted precision is what shows up as the 4-bit GSM8K of 31.84 instead of something near full precision.

I propose SQuat: at the same 4-bit budget, make each key residual **orthogonal to the future query subspace** instead of small. The objective comes from bounding the actual quantity attention consumes. Write the pre-softmax score vector $w = [q_n \cdot k_i^\top/\sqrt{d}]_i$ and its quantized version $w^{\text{deq}}$, stack the values into $V$ and $V^{\text{deq}}$; I want to bound $\lVert \text{softmax}(w)V - \text{softmax}(w^{\text{deq}})V^{\text{deq}} \rVert_2$. Adding and subtracting the cross term $\text{softmax}(w)V^{\text{deq}}$ and applying the triangle inequality splits it into a value-error piece and a score-error piece. The value piece $\text{softmax}(w)(V - V^{\text{deq}})$ is a convex combination of value-row errors, so its norm is at most $\sum_i \lVert v_i - v_i^{\text{deq}} \rVert_2$ — governed entirely by value reconstruction error, which justifies the floor's per-token value quant unchanged. The score piece is the interesting one: $(\text{softmax}(w) - \text{softmax}(w^{\text{deq}}))V^{\text{deq}}$. The softmax map is Lipschitz with constant $1/2$ (its Jacobian $\text{diag}(p) - pp^\top$ has operator norm at most $1/2$), so $\lVert \text{softmax}(w) - \text{softmax}(w^{\text{deq}}) \rVert_2 \le \tfrac{1}{2}\lVert w - w^{\text{deq}} \rVert_2$, and componentwise $w_i - w_i^{\text{deq}} = q_n \cdot r_i^\top/\sqrt{d}$. Putting it together,
$$\lVert \text{out} - \text{out}^{\text{deq}} \rVert_2 \le \frac{\lVert V^{\text{deq}} \rVert_F}{2\sqrt{d}}\sum_i \lvert q_n \cdot r_i^\top \rvert + \sum_i \lVert v_i - v_i^{\text{deq}} \rVert_2.$$
The output is preserved when the value residuals are small *and* the inner products $q_n \cdot r_i^\top$ are small — not $\lVert r_i \rVert$ small. The key residual should be as orthogonal as possible to the future query.

The obvious wall is that at the moment I quantize token $i$'s key, the future queries $q_n$ for $n > i$ do not yet exist. The way out is structure: trained queries do not fill the $d$-dimensional space, they cluster in a low-dimensional subspace, and — crucially — the subspace spanned by the *prompt's* queries, which I have right after prefill, essentially coincides with the subspace spanned by prompt-plus-response queries. So I estimate the relevant directions before decoding a single token. This is where I need the harness machinery the floor ignored: `needs_prefill_qkv_observer()` now returns `True`, the observation position is post-RoPE (so the captured queries match the keys the model will dot), and `observe_prefill_qkv` builds a per-layer subspace. After prefill I have the prompt's $Q$ per layer; I take its SVD $Q = U\Sigma V^\top$ and define the subspace matrix $\hat{Q} = \text{diag}(\sigma_1, \dots, \sigma_r) \cdot V[:r]$ — the top-$r$ right singular vectors *scaled by their singular values*. I scale rather than take orthonormal $V[:r]$ because the penalty I am about to build involves $\hat{Q}^\top\hat{Q} = \sum_{j \le r} \sigma_j^2 v_j v_j^\top$, so a direction with a big singular value gets a quadratically bigger weight — the geometry of which directions matter to attention *is* the singular values, so I carry them. For grouped-query attention I concatenate the query tensors of the heads sharing a KV head before the SVD, and I share the subspace across the batch (taking sample 0), justified by the same low-dimensional fact.

With the directions in hand the objective is to minimize reconstruction error subject to the residual being orthogonal to $\hat{Q}$. A hard constraint $\hat{Q} \cdot r = 0$ over an integer grid is combinatorial and too rigid, so I relax it to a penalty with a knob $\lambda$: minimize $\lVert k - \text{deq}(k) \rVert^2 + \lambda \lVert \hat{Q}(k - \text{deq}(k)) \rVert^2$, i.e. minimize the residual under the metric $P = I + \lambda \hat{Q}^\top\hat{Q}$. At $\lambda = 0$ this is the floor; $\lambda \to \infty$ drives the residual into the orthogonal complement of $\hat{Q}$, and $\lambda$ continuously trades fidelity against orthogonality. To quantize the integer coordinates while respecting the penalty, I quantize one block of coordinates at a time and *compensate*: after I round a block, the already-quantized coordinates are locked on the grid but the not-yet-quantized ones are still continuous degrees of freedom I use to push the total residual back toward orthogonality. The per-step problem is $\min_\delta \delta^\top P \delta$ subject to the frozen coordinates fixed and the current block pinned to its rounding error — linear constraints $T\delta = b$ with $T = [I, 0]$. A Lagrangian gives $\delta = P^{-1}T^\top(TP^{-1}T^\top)^{-1}b$; blocking $P^{-1}$ as $[A_t; B_t]$ aligned to first-$t$/remaining gives $P^{-1}T^\top(TP^{-1}T^\top)^{-1} = [I; B_t A_t^{-1}]$, so the frozen coordinates come out as $b$ and the *remaining* coordinates get $B_t H_t \, d$ where $H_t$ is the relevant columns of $A_t^{-1}$ and $d$ is the block's rounding error.

This structure is exactly the Optimal Brain Surgeon / OBQ error-feedback update (Hassibi et al.; Frantar & Alistarh's OBC; GPTQ): quantize a coordinate, update the remaining coordinates by a rounding-error vector times inverse-metric blocks to cancel the induced error. In weight quantization the metric is the data Hessian $2XX^\top$; here I have replaced it with the *query-subspace metric* $P = I + \lambda \hat{Q}^\top\hat{Q}$, which charges residual components along the query subspace by $(1 + \lambda\sigma^2)$ and leaves the orthogonal complement at unit weight. That swap is the entire conceptual content — instead of measuring error under the data covariance, I measure it under the directions the queries actually probe, which the attention-output bound says is what matters. It also rescues me from OBS's per-row greedy ordering: GPTQ's observation that fixed-order quantization is nearly as good means $A_t$ and $A_t^{-1}$ depend only on $P$ — not on the data, not on which key — so I compute them once after prefill and reuse them for the whole decode, which is what makes this feasible in the decode loop at all.

The implementation is made cheap by two further moves. The nested $A_t$ are top-left blocks of the same $P^{-1}$, so I get $A_t^{-1}$ from $A_{t+1}^{-1}$ by a Schur-complement downdate $A_t^{-1} = M - N^\top O^{-1} N$ — one Gaussian-elimination step per block, $O(d^3)$ total instead of $O(d^4)$ from fresh inverses, recursing downward from $A_T^{-1} = P$. And I do not quantize coordinate-by-coordinate: I quantize a *block* of $g$ coordinates per iteration, and with head dimension 128 and $g = 64$ that is just two blocks — one compensation update. The block downdate uses the same Schur identity with a tiny $\text{tol} \cdot I$ dampening for numerical safety. Per-token values stay on the plain group-wise min-max path; the bound's first term needs nothing more. The knobs settle from structure: because $\hat{Q}$ carries the scaled singular vectors, $\lambda \hat{Q}^\top\hat{Q}$ becomes dominant at small $\lambda$, so $\lambda \approx 10^{-3}$ is the right scale; $r = 60$ is wide enough to cover the span the future queries probe without over-constraining the residual; $g = 64$ gives two blocks on a 128-dim head; and I inherit $G = 32$ quantization group and $R = 32$ residual window from the streaming setup, with the residual via $\text{seq\_len} \bmod 32$ so that at the 4096 reference span the residual is $0$ and `estimate_bits` reports exactly 4.0 bits — $4\times$ compression, a touch above the floor's 3.82×.

The bit width stays at 4 deliberately: this rung's whole bet is that at the *same* 4 bits, spending precision query-aware rather than reconstruction-blind recovers quality the floor wasted. The falsifiable expectation is that the value-error term does not move (values are quantized identically), so value-driven failures stay put, while the key-quant attention shift on retrieval/QA/code/NIAH should hold near or modestly improve on the floor at 4× efficiency. GSM8K is the open risk and the place I am least sure: the bound protects the attention *distribution*, but GSM8K's failure is *accumulation* across a long greedy chain, and a per-layer subspace built from the prompt may not protect the directions the generated arithmetic queries probe — so GSM8K could stay soft near 31.84. If that is what I see — the retrieval/QA/code/NIAH cluster holding at higher efficiency while GSM8K refuses to move — then the diagnosis for the next rung is already written: orthogonality fixes *which directions* within a layer, but it still spends the same bits on every layer, and the accumulation that kills GSM8K is a *layer-allocation* problem the within-layer trick cannot reach.

```python
class AdaptiveKVQuantizer:
    """SQuat-inspired subspace-orthogonal K/V 4-bit quantization."""

    def __init__(self):
        self.bits = 4
        self.group_size = 32
        self.residual_length = 32
        self.subspace_dim = 60
        self.squat_lambda = 0.001
        self.quant_group_size = 64
        self.shared_svd = True
        self.query_subspaces = {}

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.query_subspaces = {}

    def needs_prefill_qkv_observer(self) -> bool:
        return True

    def query_observation_position(self) -> str:
        return "post_rope"

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        if query_states is None:
            return None
        batch, query_heads, _, head_dim = query_states.shape
        kv_heads = int(attention_meta.get("kv_heads", query_heads))
        if query_heads % kv_heads != 0:
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

    def _minmax_last_dim(self, data: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, 2**int(bits) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int
        q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]

    def _generate_At_inv(self, query_subspace: torch.Tensor, tol: float = 1e-7):
        batch, heads, _, head_dim = query_subspace.shape
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)
        matrices = [None] * groups
        eye = torch.eye(head_dim, device=query_subspace.device, dtype=torch.float32)
        A_t = eye.expand(batch, heads, head_dim, head_dim) + float(self.squat_lambda) * query_subspace.float().transpose(
            -1, -2
        ).matmul(query_subspace.float())
        matrices[groups - 1] = A_t
        for group_idx in range(groups - 1, 0, -1):
            current_dim = group_idx * q_group
            width = min(q_group, A_t.shape[-1] - current_dim)
            M_t1 = A_t[:, :, :current_dim, :current_dim]
            N_t1 = A_t[:, :, current_dim : current_dim + width, :current_dim]
            O_t1 = A_t[:, :, current_dim : current_dim + width, current_dim : current_dim + width]
            local_eye = torch.eye(width, device=query_subspace.device, dtype=torch.float32)
            O_inv = torch.inverse(O_t1 + tol * local_eye.expand(batch, heads, width, width))
            A_t = M_t1 - N_t1.transpose(-1, -2).matmul(O_inv.matmul(N_t1))
            matrices[group_idx - 1] = A_t[:, :, :, -q_group:]
        return matrices

    def _squat_quantize_keys(self, key_states: torch.Tensor, query_subspace: torch.Tensor) -> torch.Tensor:
        batch, heads, _, head_dim = key_states.shape
        query_subspace = query_subspace.to(device=key_states.device)
        if query_subspace.shape[0] == 1 and batch > 1:
            query_subspace = query_subspace.expand(batch, -1, -1, -1)
        if query_subspace.shape[1] != heads or query_subspace.shape[-1] != head_dim:
            raise ValueError("SQuat query subspace shape does not match the key tensor")
        matrices = self._generate_At_inv(query_subspace)
        P_inv = torch.inverse(matrices[-1])
        work = key_states.float().clone()
        q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
        groups = math.ceil(head_dim / q_group)
        for group_idx in range(groups):
            start = group_idx * q_group
            end = min(head_dim, start + q_group)
            chunk = work[:, :, :, start:end]
            dequant = self._minmax_last_dim(chunk.transpose(2, 3).contiguous(), self.group_size, self.bits).transpose(2, 3)
            if group_idx < groups - 1:
                d_vec = (dequant - chunk).float()
                next_start = end
                H_t = matrices[group_idx]
                B_t = P_inv[:, :, next_start:, :next_start]
                update = d_vec.matmul(H_t.transpose(-2, -1)).matmul(B_t.transpose(-2, -1))
                work[:, :, :, next_start:] = work[:, :, :, next_start:] + update
            work[:, :, :, start:end] = dequant
        return work

    def _quantize_with_residual(self, tensor: torch.Tensor, quant_fn) -> tuple[torch.Tensor, float]:
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = self._residual_keep_length(seq_len)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        work[:, :, :quant_end, :] = quant_fn(work[:, :, :quant_end, :])
        avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        query_subspace = self.query_subspaces.get(layer_id)
        if query_subspace is None:
            raise RuntimeError("SQuat key quantization requires the prefill query observer")
        return self._quantize_with_residual(key_states, lambda data: self._squat_quantize_keys(data, query_subspace))

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize_with_residual(value_states, lambda data: self._minmax_last_dim(data, self.group_size, self.bits))

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        residual = self._residual_keep_length(seq_len)
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
```
