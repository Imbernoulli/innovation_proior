## Research question

Pretrain a GPT-2-class decoder language model. Inside each block the position-wise feed-forward network (FFN) is the largest consumer of training time: it projects each token's width-`d` vector up to `4d`, applies a pointwise nonlinearity, and projects back down to `d`. The only design choice here is the FFN's middle operation — the activation that sits between the two dense matmuls, and how it is executed on the GPU. Everything else about the model and training loop is fixed. Total cost is `steps × step-time`, so a method is judged on both levers at once: the activation can change *quality per step* (lower validation loss for the same tokens) and the execution can change *throughput per step* (wall-clock).

## Prior art / Background / Baselines

The FFN slot holds pointwise activations run as ordinary PyTorch ops.

- **ReLU FFN.** Hard zero threshold: `max(xW1, 0) W2`. Gap: for positive inputs it is exactly the identity, so a strongly-firing and a barely-firing unit pass through at the same unit slope; the positive half grows only linearly and adds no extra shaping.
- **GELU FFN.** Smooth ReLU-like curve `xΦ(x)`. Gap: still asymptotically linear; each element needs an `erf`/`tanh` evaluation (costlier than a rectifier), and on LM perplexity it is roughly on par with — not clearly better than — ReLU. This is the scaffold's default fill.
- **Swish FFN.** Sigmoid-gated self-product `xσ(βx)`. Gap: same asymptotically-linear shape and the same per-element sigmoid cost; it does not consistently beat ReLU/GELU on LM.
- **GLU-variant FFNs (ReGLU/GEGLU/SwiGLU).** Replace the activation with a Hadamard product of two projections, one squashed. Gap: they require a third weight matrix `V` and a `2/3` inner-width shrink to keep parameter count matched; the editable interface below only hands the FFN two weights, so a gated FFN with its own gate matrix cannot be expressed.
- **Vendor execution path.** The two matmuls run through cuBLAS near peak, while the activation runs as separate PyTorch elementwise ops. Gap: the wide `(tokens × 4d)` intermediate is streamed through HBM for each op in the activation chain, adding bandwidth cost and extra kernel launches.

## Fixed substrate / Code framework

A nanoGPT GPT-2-Medium training loop is frozen and must not be touched: 24 layers, 16 heads, width `n_embd=1024` (~355M params), block size 1024, no biases, weight-tied embeddings, bf16 autocast, `torch.compile`, 2-GPU DDP. AdamW (`β=(0.9, 0.95)`, `wd=0.1`, grad-clip 1.0), cosine LR with linear warmup. Flash attention in the attention block. The `MLP` module owns the two linear weights `c_fc` (up, `4·n_embd × n_embd`) and `c_proj` (down, `n_embd × 4·n_embd`) and the dropout, and calls a single function for the matmul→activation→matmul core, then applies dropout outside it. Pre-activations and outputs are produced under bf16 autocast; any in-activation arithmetic that can overflow in low precision must be done in fp32 and cast back.

## Editable interface

Exactly one function is editable — `fused_mlp_forward(x, w_fc, w_proj)` in `custom_pretrain.py`, plus an optional `CONFIG_OVERRIDES` dict limited to `{learning_rate, weight_decay, warmup_iters, min_lr, grad_clip}`. The contract is rigid:

- `x`: input rows `(B*T, n_embd)`;
- `w_fc`: up-projection weight `(4*n_embd, n_embd)`;
- `w_proj`: down-projection weight `(n_embd, 4*n_embd)`;
- returns: output rows `(B*T, n_embd)`.

Only these two weights are passed in, so every candidate is a two-matrix FFN: the design freedom is the pointwise activation between the two matmuls, the execution strategy for that activation, and optionally a hand-written autograd backward. The `MLP` module flattens to `(B*T, n_embd)` and re-applies dropout itself, so the function must be a pure FFN core. The starting fill is GELU through separate PyTorch ops.

```python
# EDITABLE region of custom_pretrain.py (lines 33-48) — default fill
def fused_mlp_forward(x, w_fc, w_proj):
    """MLP forward pass: linear -> activation -> linear.

    Args:
        x: input tensor (B*T, n_embd)
        w_fc: first linear weight (4*n_embd, n_embd)
        w_proj: second linear weight (n_embd, 4*n_embd)
    Returns:
        output tensor (B*T, n_embd)
    """
    h = F.gelu(x @ w_fc.t())
    return h @ w_proj.t()
```

Everything else — the two linear maps, the `4×` inner width, dropout, the surrounding block, the optimizer and the training loop — already exists.

## Evaluation settings

- **Model / data:** GPT-2 Medium (24L / 16H / 1024D, ~355M params) on FineWeb 10B with the GPT-2 tokenizer, ~7.1B tokens.
- **Training:** 13,535 iterations, batch size 64, gradient-accumulation 8, 2-GPU DDP; one seed (42).
- **Metrics.** Quality (lower is better): `val_loss` (held-out cross-entropy, primary), `wikitext2_ppl`, `lambada_ppl`. Throughput (lower is better): `elapsed` (training wall-clock). Downstream zero-shot accuracy (higher is better) is reported as a secondary read: `arc_easy`, `hellaswag` (and hidden `piqa`, `winogrande`). A change to the activation or its execution can move quality, not just speed.
