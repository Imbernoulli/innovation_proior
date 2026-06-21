# Context: information flow through depth in decoder-only Transformers (circa 2024)

## Research question

A decoder-only Transformer is a stack of identical blocks. The input tokens are embedded into
`H_0`, and each block reads the previous hidden state `H_{n-1}`, runs self-attention and an
MLP, and writes `H_n`. As the field pushes for capability through depth — scaling laws say
keep adding layers — a concrete question arises: **how well does the localized, token-level
information present in the initial embeddings actually survive to the deep layers that make the
final prediction?** The standard picture is "the residual stream carries it": every block adds
its update to the running `H`, so `H_0` is, in principle, always one identity-path away. But
that path is shared by everything — every block's output is summed into the same stream — so by
the time the signal reaches layer 24 the raw token content has been processed many times.

## Background

**The residual stream and why depth was supposed to be free.** ResNet (He et al. 2016)
established the identity shortcut `H_n = H_{n-1} + f(H_{n-1})`: routing an identity path around
each nonlinear block fixes the vanishing-gradient problem and lets very deep nets train at all.
DenseNet and Stochastic Depth (Huang et al. 2016, 2017) pushed further — let every layer read
the concatenated features of *all* preceding layers, so information and gradients flow directly
across many layers rather than only to the immediate neighbor. The lesson the Transformer
inherited: shortcuts that give later layers direct access to earlier representations improve
both optimization and information flow. The Transformer (Vaswani et al. 2017) wires one such
shortcut around each attention and each MLP sublayer.

**Over-smoothing in Transformers.** Empirically
the gains from depth flatten and can reverse: a 32-layer ViT underperforms a 24-layer ViT
(Zhou et al. 2021), and language-model gains shrink with further depth (Petty et al. 2024).
The mechanism is over-smoothing: self-attention acts as a low-pass filter. Wang et al. (2022)
show attention smooths token representations in ViTs; Shi et al. (2022) analyze the same
phenomenon in BERT from a graph-Laplacian view. Concretely, each attention layer replaces a
token's representation by a convex combination (the softmax row) of all tokens' value vectors,
which is a smoothing / averaging operation; iterate it through depth and token representations
drift toward each other. The consequence: in deep layers, **sequence-level features dominate
and token-level features are diluted**. A useful
formal handle (the variational view of attention): a single self-attention update is one
gradient-descent step that minimizes a nonlocal smoothing functional
`J(u) = (1/2) ∬ ||u(x) - u(y)||² k(x,y) dx dy`, whose minimizer is a *constant* function — so
repeated attention is a diffusion toward uniformity.

**Pathologies in trained Transformers.** Trained Transformers exhibit
"attention sinks" (Xiao et al. 2024): a large fraction of attention mass collapses onto a few
low-semantic tokens, typically the first token. The same sink tokens carry abnormally large
*value-state norms* — "value-state drains" (Guo et al. 2024b) — and abnormally large
*hidden-state norms*, the "massive activations" / residual-state peaks of Sun et al. (2024a).
Guo et al. (2024a) tie these together: a mutual-reinforcement loop between value-state drains
and attention sinks (active-vs-dormant attention heads). Measuring the entropy of token
importance across depth reveals a "concentration → dispersion → concentration" pattern, with
heavy concentration in deep layers of large trained models (Llama-8B, Mistral-7B). And
"Transformer layers as painters" (Sun et al. 2024b) reports that initial embeddings are highly
localized and become abstract within the first few layers, with **low similarity between `H_0`
and deep hidden states**.

**A first attempt to give deep layers more of the start: DenseFormer.** Pagliardini et al.
(2024) add Depth-Weighted-Averaging: after each block, replace `H_n` by a learnable static
weighted average of all previous outputs, `H_n = Σ_{i=0}^{n} λ_{n,i} H_i`, initialized to the
identity (`λ_{n,n}=1`, rest 0). Notably, the *learned* coefficients confirm the worry: deeper
layers assign larger weight to the initial embedding `H_0`.

## Baselines

**ResNet / DenseNet shortcuts (He 2016; Huang 2016, 2017).** Identity and dense skip
connections on the hidden state. Core idea: `H_n = H_{n-1} + f(H_{n-1})` (ResNet), or `H_n`
reads all previous `H_i` (DenseNet).

**DenseFormer (Pagliardini et al. 2024).** `H_n = Σ_{i=0}^{n} λ_{n,i} H_i`, learnable static
`λ`, identity init. Core idea: let each block read a learned mixture of all prior block
outputs, so `H_0` is reachable with its own learned weight rather than only through the summed
residual.

**NeuTRENO (Nguyen et al. 2023).** Starts from the variational view: since a self-attention
update is a descent step on the smoothing functional `J(u)` and `J`'s minimizer is constant,
pure attention diffuses to uniformity (over-smoothing). The fix is to descend on a
*regularized* objective instead — `E(u, f) = J(u) + (λ/2) ∫ ||u - f||² dx` — where the second,
convex *fidelity* term anchors the smoothed output `u` to a reference signal `f`. Its gradient
flow is `du/dt = -∇J(u) - λ(u-f)`. Euler-discretizing with `u(x,0)=v(x)` and choosing
`λ = λ̃/Δt(x)` gives the extra positive pull `λ̃(f(i)-v(i))`. Taking `f` to be the first
layer's value vectors `v^0` yields, per token `i`,
```
u(i) = Σ_j softmax(k_i·k_j/√d) v(j)  +  λ̃ (v^0(i) - v(i)),
```
i.e. add `λ(V_1 - V_n)` to each layer's attention output (default `λ=0.4`). Core idea:
re-supply the un-smoothed first-layer value to fight diffusion.

**KV-sharing methods (MQA, Shazeer 2019; GQA, Ainslie et al. 2023; CLA, Brandon et al. 2024).**
Reduce the inference KV cache by sharing keys and values across query heads (MQA/GQA) or across
layers (CLA). Core idea: store fewer K/V tensors.

## Evaluation settings

- **Pretraining corpus:** SlimPajama (Cerebras 2023), a ~20B-token subsample drawn by the
  original seven-domain proportions (CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia,
  StackExchange), tokenized with the RedPajama-INCITE-7B tokenizer (~50K vocab).
- **Architecture (held fixed across methods):** Llama-style decoder — pre-normalization,
  SwiGLU MLP, rotary position embeddings (RoPE, θ=10,000), no dropout. Sizes 82M / 180M / 320M
  / 468M (8 / 12 / – / 24 layers), plus a 1.6B run on ~200B internal tokens.
- **Optimization:** AdamW, `β=(0.9, 0.95)`, weight decay 0.1, max grad norm 1.0; ~2M-token
  batches, sequence length 2,048 (also 512 / 8,192 / 32,000 / 64,000 for length studies),
  10,000 steps, linear warmup (120 or 1,200) to peak lr 6e-4, cosine decay to 6e-5; FP16
  mixed precision, ZeRO-2, FlashAttention, 8×A100.
- **Metrics:** primary is average validation loss (cross-entropy) over the seven SlimPajama
  domains; perplexity on WikiText and LAMBADA; zero-shot accuracy on HellaSwag, OpenBookQA,
  WinoGrande, ARC-Easy, ARC-Challenge, PiQA. Matched-loss comparisons are read as
  parameter-efficiency or data-efficiency (how many fewer parameters / tokens reach a given
  loss). The natural diagnostics for the pathologies above: token-importance entropy by layer,
  value-state and hidden-state norms by token/layer, layer-to-layer hidden-state similarity,
  and PCA rank (principal components for 99% variance) of each layer's hidden state.

## Code framework

The intervention lives inside a standard decoder block. What already exists is the ordinary
pre-norm Transformer: a token + position embedding, a stack of blocks each doing causal
self-attention (project to Q, K, V; scaled-dot-product attention; output projection) and an
MLP, with residual connections, and an LM head (often weight-tied to the token embedding). The
block has one empty slot: the per-layer hook through which an extra source of information may be
introduced. Nothing about that hook is settled — what the extra source is, where it comes from,
where in the block it enters, and how it is combined — is exactly what is to be designed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, scale):
    """Standard causal scaled-dot-product attention. The softmax weights A
    are computed from q, k; v is what gets aggregated."""
    A = torch.softmax((q @ k.transpose(-2, -1)) * scale + causal_mask(q, k), dim=-1)
    return A @ v


class Block(nn.Module):
    """One pre-norm decoder block. layer_idx identifies this layer; aux is the
    optional extra information made available to this layer (see TODO)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.scale = (config.n_embd // config.n_head) ** -0.5

    def forward(self, x, aux=None):
        h = self.ln1(x)
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        # TODO: the per-layer intervention we will design.
        #       If an extra information source `aux` is supplied for this layer,
        #       decide whether/how it enters the block here.
        u = attention(q, k, v, self.scale)
        x = x + self.Wo(u)
        x = x + self.mlp(self.ln2(x))
        return x


class TokenEmbedding(nn.Module):
    """Maps token ids -> input embeddings, and owns whatever extra per-layer
    information sources the design may add."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # TODO: any additional embedding tables / state the intervention needs.

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(t, device=idx.device)
        return self.wte(idx) + self.wpe(pos)

    def get_aux(self, layer_idx):
        """Optional extra information to hand to block `layer_idx`, or None."""
        # TODO
        return None

    def get_lm_head_weight(self):
        return self.wte.weight


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = TokenEmbedding(config)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, idx):
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, aux=self.embed.get_aux(i))
        logits = F.linear(self.ln_f(x), self.embed.get_lm_head_weight())
        return logits
```

The block exposes `aux` and the embedding exposes `get_aux(layer_idx)`; the design fills in
what `aux` is and how the block uses it.
