**Problem (from step 1).** The tokenizer gives ids; now pretrain a base LM from scratch that completes web text, and clear GPT-2's CORE = 0.256525 on one 8×H100 node in a few hours. The 2019 baseline (learned absolute positions, GELU, tied embeddings, Adam-on-everything) is far from compute-optimal at this scale. Every design choice is paid for in FLOPs, so each must earn its keep.

**Key idea.** A modded-nanogpt-lineage decoder-only transformer plus a matrix-aware optimizer:
- **Rotary positions** (relative offset falls out of the dot product, no position table) and **QK-norm** (RMS-norm q,k before attention → bounded logits, higher stable LR);
- **parameter-free RMS-norm**, **bias-free linears**, **ReLU² MLP** (beats GELU/SwiGLU at this scale), **untied** embedding/lm_head, **zero-init** block output projections (every block starts as identity);
- **value embeddings** gated into the attention `v` on alternating layers — large capacity at ~zero FLOPs (lookup + add); plus cheap per-layer residual scalars; GQA + sliding-window for cheap inference; logit softcap.
- **Muon** for the 2D weight matrices: take the SGD-momentum update G and replace it with its orthogonal factor UVᵀ (every singular direction gets a unit step), approximated by a 5-step Newton–Schulz / "Polar Express" quintic in bf16 — no SVD. **AdamW** for everything else (embeddings, lm_head, value-embeds, scalars), where the matrix-spectrum argument doesn't apply.

**Why it works.** Orthogonalizing the matrix update spreads each step evenly across the matrix's directions instead of letting a few large singular directions dominate, so the weights learn much faster per step — and the iteration is just a few matmuls, stably runnable in bf16. Rotary + QK-norm + zero-init + softcap make the deep stack trainable at a high learning rate; value embeddings add capacity the FLOP budget can't otherwise afford. Together they reach GPT-2 capability for ~600× less than the 2019 cost.

**Change / code.** The block (rotary + QK-norm + value-embed gate + ReLU² MLP), and the Muon orthogonalization step:

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))  # parameter-free RMS-norm

class CausalSelfAttention(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Value residual (ResFormer): mix value embedding into v with input-dependent per-head gate
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (0, 3)
            v = v + gate.unsqueeze(-1) * ve
        # Rotary (relative position) then QK-norm (bounded logits)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)   # QK norm
        q = q * 1.2; k = k * 1.2  # re-sharpen attention after norm
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))

class MLP(nn.Module):
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()   # ReLU^2
        return self.c_proj(x)

class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # pre-norm residual
        x = x + self.mlp(norm(x))
        return x
```

```python
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar Express: orthogonalize the update (approx nearest orthogonal matrix UV^T) in bf16
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):          # tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X; B = b * A + c * (A @ A); X = a * X + X @ B
    else:                                # wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT; B = b * A + c * (A @ A); X = a * X + B @ X
    g = X
    # ... per-column variance reduction (uniform per-neuron step) ...
    # Cautious weight decay + parameter update (decay only where update & weight agree in sign)
    lr = lr_t.to(g.dtype); wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

```python
# Optimizer split: Muon for transformer matrices, AdamW for embeddings / lm_head / scalars.
# AdamW LRs are scaled by ∝1/√(model_dim/768) so the recipe transfers across depths.
dmodel_lr_scale = (model_dim / 768) ** -0.5
param_groups = [
    dict(kind='adamw', params=lm_head_params,      lr=unembedding_lr * dmodel_lr_scale,       betas=(0.8, 0.96),  eps=1e-10, weight_decay=0.01),
    dict(kind='adamw', params=embedding_params,    lr=embedding_lr * dmodel_lr_scale,         betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
    dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5,   betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
    # ... resid_lambdas, x0_lambdas, smear scalars (all AdamW) ...
]
for shape in sorted({p.shape for p in matrix_params}):   # group matrices by shape, stack, orthogonalize
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr,
                             momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
```
