## Research question

Pretraining a GPT-2-class decoder language model, the bill is dominated by training compute, and
inside each block the position-wise feed-forward network (FFN) is the fatter consumer: project each
token's width-`d` vector up to `4d`, apply a pointwise nonlinearity, project back down to `d`. The
single thing being designed here is **the FFN's middle operation** — the activation that sits between
the two dense matmuls, *and how it is executed on the GPU* (which PyTorch ops, which custom kernel,
which hand-written backward). Everything else about the model and the training loop is fixed. Total
cost is `steps × step-time`, so there are two levers at once: the activation can change *quality per
step* (lower validation loss for the same tokens), and the execution can change *throughput per step*
(wall-clock), and a method is judged on both.

## Prior art before the first rung (FFN-activation lineage)

The first rung reacts to the stock pointwise activations the FFN slot has historically held. These are
the methods that precede the ladder; the substrate below is what the block converged to.

- **ReLU FFN (Vaswani et al. 2017).** `max(xW1, 0) W2` — hard sign-gate, no activation parameters, a
  0/1 mask in backward. Gap: for positive inputs it is *exactly* the identity, so a strongly-firing
  and a barely-firing unit pass through at the same unit slope; the activation adds no shaping, and its
  output grows only linearly.
- **GELU FFN (Hendrycks & Gimpel 2016).** `GELU(x) = xΦ(x)`, smoothing ReLU's corner and leaking a
  little negative signal through; the de-facto choice in BERT/GPT-style LMs. Gap: still
  *asymptotically linear* (its slope saturates near 1 as `x→∞`), needs an `erf`/`tanh` per element
  (costlier than a rectifier in both passes), and on LM perplexity it is roughly on par with — not
  clearly better than — plain ReLU. This is the scaffold's default fill.
- **Swish FFN (Ramachandran et al. 2017).** `xσ(βx)`, smooth and self-gated near the origin. Gap: same
  asymptotically-linear shape and the same per-element sigmoid cost; does not beat ReLU/GELU for LM.
- **GLU-variant FFNs (Shazeer 2020): ReGLU/GEGLU/SwiGLU.** Replace the activation-of-one-projection
  with a Hadamard product of two projections, one squashed, e.g. `(max(xW,0) ⊗ xV) W2` — a
  multiplicative gate that gives an un-attenuated gradient path (Dauphin et al. 2017) and the strongest
  LM perplexities in this family. Gap *here specifically*: they need a **third** weight matrix `V` and
  a `2/3` inner-width shrink to stay parameter-matched — and the edit surface below hands the FFN only
  **two** weights, so a gated FFN with its own `V` cannot be expressed in this slot at all.
- **GPU kernel execution: vendor libraries vs. custom kernels.** The FFN's two matmuls go to cuBLAS
  near peak, but the activation, run as separate PyTorch ops, is a *bandwidth-bound* elementwise pass
  that streams the wide `(tokens × 4d)` intermediate through HBM and back. A custom Triton kernel can
  in principle fuse the activation onto the matmul accumulator in registers to delete that traffic.
  Gap: hand-rolled tiled matmuls must out-run a heavily-tuned `torch.compile`/cuBLAS path to be worth
  it, which is far from guaranteed.

## The fixed substrate

A nanoGPT GPT-2-Medium training loop is frozen and must not be touched: 24 layers, 16 heads, width
`n_embd=1024` (~355M params), block size 1024, no biases, weight-tied embeddings, bf16 autocast,
`torch.compile`, 2-GPU DDP. AdamW (`β=(0.9, 0.95)`, `wd=0.1`, grad-clip 1.0), cosine LR with linear
warmup. Flash attention in the attention block. The FFN module (`MLP`) owns the two linear weights
`c_fc` (up, `4·n_embd × n_embd`) and `c_proj` (down, `n_embd × 4·n_embd`) and the dropout, and calls a
single function for the matmul→activation→matmul core, then applies dropout outside it. Pre-activations
and outputs are produced under bf16 autocast; any in-activation arithmetic that can overflow in low
precision must be done in fp32 and cast back.

## The editable interface

Exactly one function is editable — `fused_mlp_forward(x, w_fc, w_proj)` in `custom_pretrain.py` (the
template's lines 33–48), plus an optional `CONFIG_OVERRIDES` dict (lines 257–259) limited to
`{learning_rate, weight_decay, warmup_iters, min_lr, grad_clip}`. The contract is rigid and is the
reason the GLU variants above are out of reach:

- `x`: input rows `(B*T, n_embd)`;
- `w_fc`: up-projection weight `(4*n_embd, n_embd)`;
- `w_proj`: down-projection weight `(n_embd, 4*n_embd)`;
- returns: output rows `(B*T, n_embd)`.

Only these **two** weights are passed in — there is no third gate matrix — so every method on the
ladder is a fill of a *two-matrix* FFN: the design freedom is the pointwise activation between the two
matmuls, the kernel/fusion strategy that executes it, and (optionally) a hand-written autograd backward
for that activation. The `MLP` module flattens to `(B*T, n_embd)` and re-applies dropout itself, so the
function must be a pure FFN core. The starting point is the scaffold default: **GELU via separate
PyTorch ops**, letting autograd build the backward.

```python
# EDITABLE region of custom_pretrain.py (lines 33-48) — default fill
def fused_mlp_forward(x, w_fc, w_proj):
    """MLP forward pass: linear -> activation -> linear.

    Default implementation uses standard PyTorch ops.
    Can be replaced with a fused Triton kernel for better performance.

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

Everything else — the two linear maps, the `4×` inner width, dropout, the surrounding block, the
optimizer and the training loop — already exists. Each later method replaces exactly this function body
(and, where it helps, hand-writes the backward as a `torch.autograd.Function`).

## Evaluation settings

- **Model / data:** GPT-2 Medium (24L / 16H / 1024D, ~355M params) on FineWeb 10B with the GPT-2
  tokenizer, ~7.1B tokens (`D ≈ 20N`, Chinchilla-optimal).
- **Training:** 13535 iterations, batch size 64, gradient-accumulation 8, 2-GPU DDP; one seed (42).
- **Metrics, two axes.** Quality (lower is better): `val_loss` (held-out cross-entropy, primary),
  `wikitext2_ppl`, `lambada_ppl`. Throughput (lower is better): `elapsed` (training wall-clock).
  Downstream zero-shot accuracy (higher is better) is reported as a secondary read: `arc_easy`,
  `hellaswag` (and hidden `piqa`, `winogrande`). A kernel change that also changes the activation can
  move quality, not just speed.
