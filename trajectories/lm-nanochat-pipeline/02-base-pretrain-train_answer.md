The tokenizer commits the substrate — the `{1,2}` number grouping lands val_bpb at 0.965 at the sweep depth, the lowest of the candidates — and now I have ids to pretrain on. The model only has to do one thing here: complete web documents well enough to clear GPT-2's CORE = 0.256525 on a single 8×H100 node in a few hours. The 2019 baseline — learned absolute positions, GELU MLPs, tied input/output embeddings, Adam on everything — is correct but nowhere near compute-optimal at this scale, and every design choice is paid for in wall-clock. So the question is not "what is a transformer" but "which version of the decoder-only transformer, trained with which optimizer, reaches GPT-2 capability with the least compute." Each piece below has to earn its FLOPs.

I propose a **modded-nanogpt-lineage decoder-only transformer paired with a matrix-aware MuonAdamW optimizer**, and the architecture is best read from the residual stream outward. For positions I use **rotary position embedding** rather than a learned absolute table, because absolute index is the wrong invariant for language — what the model needs is *relative* offset, and an absolute scheme has to relearn the same relative relationship separately at every pair of positions and generalizes poorly past the trained length. Rotary instead *rotates* the query and key vectors by an angle proportional to their position, in 2D channel pairs, so the attention inner product $q\cdot k$ depends only on the position difference $i-j$ (a rotation by $\theta_i$ on $q$ and $\theta_j$ on $k$ leaves the dot product a function of $i-j$). Relative position falls out of the geometry for free, with no added parameters and no position table; the per-pair frequencies stride down the channels as $\text{inv\_freq}=\text{base}^{-2k/d}$, so low channels capture coarse long-range position and high channels fine local position.

For attention stability at this fragile scale I apply **QK-norm**: RMS-norm the queries and keys per head right after the rotary rotation. The logits $q\cdot k$ can otherwise grow large and spike the softmax, blowing up gradients; normalizing $q$ and $k$ bounds the magnitude of their dot product regardless of how the projections drift, which keeps the softmax well-conditioned and lets me run a higher learning rate without divergence. Because RMS-norm flattens the scale, I multiply $q$ and $k$ by a small fixed factor (1.2 on each) to re-sharpen the attention distribution afterward. The blocks are **pre-norm residual** — normalize the input to each sublayer, not the output — which keeps a clean identity path down the residual stream and makes deep stacks trainable, where post-norm deep transformers are notoriously hard to optimize. The norm itself is **parameter-free RMS-norm**, $x/\text{rms}(x)$, with no per-channel gain or bias: the learnable affine buys little here and is not worth the parameters or extra optimizer state, and it runs fine in bf16. In the same spirit, every linear projection is **bias-free**, since biases add parameters and a little instability for no measurable gain at this scale.

The MLP uses **ReLU²**, $\text{relu}(x)^2$, between its two projections rather than GELU. It is cheap and at this scale edges out GELU, and it edges out the fancier gated SwiGLU once I account fairly for SwiGLU's third projection and the wider hidden dimension it needs to match params and FLOPs — on step efficiency, wall-clock, and FLOPs, ReLU² comes out ahead. So the MLP is just `c_proj(relu(c_fc(x))²)` with a 4× hidden expansion and no bias. I **untie** the input embedding and output lm_head into separate matrices rather than sharing one as GPT-2 did: tying saves parameters but forces input and output representations to be transposes of each other, and at this scale the freedom is worth the cost. I normalize after the token embedding so the residual stream starts at unit scale, and I initialize the lm_head *very* small (std 0.001) so the model starts near-uniform over the vocabulary and the logits don't dominate early training. The two output projections inside each block — `attn.c_proj` and `mlp.c_proj` — I **initialize to zero**, so every block starts as the identity on the residual stream and only learns to deviate; the deep stack is then trainable from step zero, beginning as a clean residual highway that grows its computation.

One capacity trick is almost free in FLOPs. Matmuls are the expensive thing in a transformer; embedding *lookups* are nearly free. So I add **value embeddings** (ResFormer-style): a separate embedding table on alternating layers whose vector is added into the attention values $v$, gated per-head by an input-dependent sigmoid in $(0,3)$. This adds a large pile of parameters at almost zero FLOP cost — a lookup and an add, no matmul — and attempts to make it cheaper through low-rank or sharing all fail; the model wants the value embeddings at full capacity. Alongside them I add cheap per-layer learnable scalars on the residual stream — `resid_lambda` scaling the residual at each layer and `x0_lambda` blending the original normalized embedding back in — both nearly free and both helpful. For inference efficiency later, when the chat model generates token by token with a KV cache, I support **grouped-query attention** (fewer key/value heads than query heads, shrinking the KV cache) and **sliding-window attention** on most layers with the last layer always full-context, which cuts attention FLOPs without much quality cost at this scale.

The optimizer is what moves the needle most. AdamW on everything is fine, but the weight *matrices* — the big 2D projections that are most of the compute — have structure a per-coordinate optimizer ignores: Adam scales each coordinate by its own running second moment with no notion that these coordinates form a matrix whose *spectrum* matters. A momentum update $G$ on a matrix tends to be dominated by a few large singular directions, so it is effectively low-rank and lopsided. The idea is to **orthogonalize** the update — replace $G$ with its nearest orthogonal matrix, i.e. take the SVD $G=U\Sigma V^\top$ and use $UV^\top$ — so every singular direction gets a unit-scale step and the update spreads evenly across the matrix's directions. That is **Muon**: run ordinary SGD-with-momentum to get the raw update, then orthogonalize it before applying. A full SVD every step is far too slow, but I do not need the exact SVD, only the orthogonal (polar) factor $UV^\top$, which a short **Newton–Schulz / "Polar Express"** iteration approximates — a quintic polynomial in $G G^\top$ that pushes the singular values toward 1 while fixing the singular vectors. Five iterations of a well-chosen coefficient schedule get close enough, run entirely in bf16 as a few matmuls, and crucially do not even need to fully converge: pushing the singular values into a band around 1 works as well in practice and is cheaper. I pick the tall-vs-wide branch by which dimension is larger to keep the matmuls small, add a per-column variance-reduction rescale so the effective per-neuron learning rate is uniform, and apply a "cautious" weight decay that only decays a coordinate when the update and the weight agree in sign.

So the split is **Muon for the 2D weight matrices** — the transformer's projections, the bulk of the compute, where orthogonalized updates pay off — and **AdamW for everything else**: the embedding table, the lm_head, the value-embedding tables, and the per-layer scalars, which are 1D/embedding parameters where the matrix-spectrum argument does not apply and Adam's per-coordinate scaling is exactly right. Muon must *not* touch the embeddings or the final classifier. I group the matrix params by shape so same-shaped matrices stack and orthogonalize in one batched kernel, fuse the whole Muon step (momentum → polar-express → variance-reduction → cautious update) into a single compiled kernel and likewise the AdamW step to kill Python overhead, tune the learning rates per group (embeddings want a much higher LR than matrices), and scale them $\propto 1/\sqrt{\text{model\_dim}}$ so the same recipe transfers across depths. I softcap the logits before the loss as $\text{softcap}\cdot\tanh(\text{logits}/\text{softcap})$ with softcap $\approx 15$ to keep them bounded and stabilize the softmax/cross-entropy. Precision is explicit: master weights stored fp32 for the optimizer, cast to bf16 for the matmuls via a custom Linear (no autocast), with optional fp8 tensorwise-scaled matmuls on Hopper for extra throughput. The block is `x = x + attn(norm(x)); x = x + mlp(norm(x))`, the data is shuffled FineWeb-EDU-class web text streamed through a BOS-aligned dataloader so every row starts at a document boundary, the horizon is set by a tokens:params ratio a bit under Chinchilla, and the objective is plain next-token cross-entropy. The remaining gap is honest: this base model can only *complete* text — no conversation, no special-token format, no tool channel — and turning it into something you can talk to is the next stage's job.

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
