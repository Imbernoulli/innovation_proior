# SQuat, distilled

SQuat (Subspace-orthogonal KV cache quantization) quantizes the KV cache so that the *attention
output* — not the raw key/value tensor — stays close to full precision. It builds a
low-dimensional query subspace from the prompt during prefill, and when quantizing keys it
pushes each block's rounding error into the not-yet-quantized coordinates so the key residual
stays as orthogonal as possible to that subspace. It is training-free, calibration-free, and
runs on-the-fly inside the decode loop.

## Problem it solves

Low-bit KV-cache quantization for streaming LLM decoding. The cache is consumed only through
attention scores `q_n·k_iᵀ`, yet compression-based quantizers (KIVI, GEAR, KVQuant) minimize
the tensor reconstruction error `‖k − deq(k^qtz)‖₂`, which is blind to the *direction* of the
residual relative to the queries. The goal: drive the generated output toward the FP16 output
at a target bit-width, on-the-fly, no fine-tuning, no calibration data.

## Key idea

The attention-output error decomposes into a value term and a key term:

```
‖Attn(q_n,{k_i},{v_i}) − Attn(q_n,{k_i^qtz},{v_i^qtz})‖₂
  ≤ [ ‖V^deq‖_F / (2√d) ] · Σ_i |q_n (k_i − deq(k_i^qtz))ᵀ|
    + Σ_i ‖v_i − deq(v_i^qtz)‖₂.
```

(Proof: split into `softmax(w)(V−V^deq)` — bounded by `Σ‖v_i−deq‖` since softmax is
row-stochastic — and `(softmax(w)−softmax(w^deq))V^deq`, using softmax (1/2)-Lipschitz and
`‖w−w^deq‖₂ ≤ (1/√d)Σ|q_n(k_i−deq)ᵀ|`.) The output is preserved when (i) value residuals are
small and (ii) key residual projections onto future queries are small, scaled by
`‖V^deq‖_F`. So:

- **Values:** quantize per token to minimize `‖v − deq‖₂` (standard min-max).
- **Keys:** make the residual `k − deq(k^qtz)` orthogonal to the query subspace.

Future queries are unavailable at quantization time, but query tensors empirically lie in a
low-dimensional subspace that the *prompt's* queries already span (prompt-only span ≈
prompt+response span). So build the subspace from the prompt by SVD:

```
Q = X_prompt W^Q ;   Q = U Σ Vᵀ ;   Q̂ = diag(σ₁..σ_r) · V[:r] ∈ R^{r×d}.
```

Top-`r` right singular vectors **scaled by their singular values**, so the directions queries
use most are protected most. For grouped-query attention, concatenate the query heads sharing a
KV head before the SVD. Share `Q̂` across the batch (queries of same-task sequences span ≈ the
same subspace).

## Constrained problem and soft relaxation

```
min_{k^qtz} ‖k − deq(k^qtz)‖₂   s.t.  Q̂(k − deq(k^qtz)) = 0,  k^qtz b-bit ints.
```

Combinatorial + integer-constrained → relax orthogonality to a penalty:

```
min  ‖k − deq(k^qtz)‖₂² + λ ‖Q̂(k − deq(k^qtz))‖₂² .
```

`λ=0` recovers compression-based quantization; `λ→∞` enforces orthogonality. Define the metric
`P = I + λ Q̂ᵀQ̂` (PD), `P_inv = P⁻¹`.

## Iterative quantize-and-compensate (the closed-form update)

Quantize coordinate-blocks of size `g` in fixed order. At step `t`, after rounding the current
block, update the remaining coordinates to minimize the residual under `P`. With `δ = k̂_t −
k̂_{t−1}` and `d = deq(qtz(block)) − block`, the equality-constrained QP `min δᵀPδ s.t.
Tδ = b` (T selects the first `tg` coords, `b` is zero on earlier coords and `d` on the current
block) has Lagrangian solution

```
δ = P⁻¹ Tᵀ (T P⁻¹ Tᵀ)⁻¹ b .
```

Block `P_inv = [[A_t, B_tᵀ],[B_t, C_t]]` (A_t = top-left `tg×tg`). Then `P⁻¹Tᵀ = [A_t; B_t]`,
`TP⁻¹Tᵀ = A_t`, so `P⁻¹Tᵀ(TP⁻¹Tᵀ)⁻¹ = [I; B_t A_t⁻¹]`, and the not-yet-quantized coordinates
update by

```
[k̂_{t−1}]_{tg:} ← [k̂_{t−1}]_{tg:} + B_t H_t d ,     H_t = last g columns of A_t⁻¹ .
```

Element-wise (`g=1`): `+ (deq(qtz([k]_t)) − [k]_t) B_t h_t`. This is exactly the Optimal Brain
Surgeon / OBQ error-feedback update with the data Hessian `2XXᵀ` replaced by the query-subspace
metric `I + λ Q̂ᵀQ̂`. Quantizing in fixed order (the GPTQ insight) makes `A_t⁻¹`, `P_inv` the
same for every key/token/sample → compute once per prefill.

## Efficient inverse recursion (O(d⁴) → O(d³))

The `A_t` are nested top-left blocks, so downdate `A_t⁻¹` from `A_{t+1}⁻¹` by a Schur step.
Writing `A_{t+1}⁻¹ = [[M, Nᵀ],[N, O]]` (split off the trailing `g`-block):

```
A_t⁻¹ = M − Nᵀ O⁻¹ N .
```

(`g=1`: `A_t⁻¹ = Ā_{t+1} − (1/ā_{t+1}) a_{t+1}ᵀ a_{t+1}` — one Gaussian-elimination step.) Start
at `A_T⁻¹ = I + λ Q̂ᵀQ̂` (`T = d/g`) and recurse down to `A_1⁻¹`. Each downdate is `O(t²)` → the
sweep is `O(d³)`; computing `P_inv = (I + λ Q̂ᵀQ̂)⁻¹` can use Woodbury in `O(r d²)` when
`r ≪ d`. With `d=128`, `g=64`: `T=2`, so one quantize → one compensation → one quantize.

## Defaults (LongBench-style 4-bit configuration) and why

`bits=4`, `group_size G=32`, `residual_length R=32`, `subspace_dim r=60`, `λ=0.001`,
`quant_group_size g=64`, `shared_svd=True`.

- `λ=0.001`: `Q̂` uses *scaled* singular vectors, so `λQ̂ᵀQ̂` is large even at small `λ`;
  this scale protects the dominant query directions without letting the penalty swamp
  reconstruction.
- `r=60`: long-context prompts span a richer query subspace; small `r` (≈5) suffices for short
  reasoning prompts but under-covers long contexts.
- `g=64` on a 128-dim head → 2 blocks, 1 compensation step (fast, on-the-fly).
- `G=32`, `R=32`: standard streaming compression — the most-recent residual window of tokens
  stays FP16, the rest quantized in groups of `G`.

## Working code

The streaming-quantizer slot fills in as:

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

## Relation to prior methods

- **KIVI / GEAR / KVQuant** minimize tensor reconstruction error `‖k − deq‖₂`; SQuat instead
  preserves `q_n·k_iᵀ` by making the residual orthogonal to the query subspace. SQuat reuses
  KIVI's per-token value quantization and residual-buffer streaming structure.
- **OBS / OBQ / OBC / GPTQ** are the error-feedback ancestors: SQuat's update is their
  quantize-and-compensate rule with the data Hessian `2XXᵀ` replaced by `I + λ Q̂ᵀQ̂`, the
  fixed-order insight (GPTQ) and the `O(d³)` inverse downdate (OBC) carried over. The key
  difference: SQuat needs no calibration data and no offline pass — the metric comes from the
  prompt's own queries at prefill, computed once and reused across the stream and the batch.
