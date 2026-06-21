# Context: stabilizing and scaling the residual stream of deep Transformer language models

## Research question

Modern language models are decoder-only Transformers built by stacking many identical
blocks, each adding the output of an attention or feed-forward sublayer back onto a shared
residual stream. Two coupled facts characterize this stack. First, training instability — loss
and gradient spikes — increases with depth, often forcing smaller learning rates, careful
initialization, or a learning-rate warmup. Second, deep Transformers frequently underperform
what their parameter count promises: past a certain depth, adding layers provides diminishing
returns, and diagnostic probing shows the deepest layers behave close to identity maps. The
question is how to modify how each block writes into the residual stream to address these
behaviors as depth grows.

## Background

**The residual stream and why depth is fragile.** A residual block computes
`x_{l+1} = x_l + F_l(x_l)`, where the identity skip `x_l` is the highway and `F_l` is the
sublayer's contribution (He et al. 2016). The skip is what makes deep nets trainable: it gives
gradients a direct path backward and lets each block start as a small perturbation of identity.
Where the normalization sits relative to the skip splits the field. **Post-LN**
(`x_{l+1} = Norm(x_l + F_l(x_l))`, Vaswani et al. 2017) normalizes *after* the residual add;
it is expressive but unstable at depth and needs a learning-rate warmup to train at all.
**Pre-LN** (`x_{l+1} = x_l + F_l(Norm(x_l))`, Xiong et al. 2020; Radford et al. 2019) normalizes
the sublayer *input*, leaving the skip untouched; this is far more stable and is the modern
default. Hybrids exist — **Sandwich-LN**
(Ding et al. 2021) adds a second norm on the branch output, and **DeepNorm** (Wang et al. 2022)
is a Post-LN that up-scales the skip and down-scales the init — each trading stability against
the bottom-vs-top gradient imbalance differently.

**Diagnostic finding 1 — Pre-LN's variance explosion makes deep layers vanish.** Unroll the
Pre-LN recursion: each block adds `F_l(Norm(x_l))` to `x_l`, and since the norm fixes the scale
of the *branch input* but not the *stream*, the variance of `x_l` accumulates with depth.
Sun et al. (2025, "The Curse of Depth") make this precise: under normal, zero-mean weights,
`σ²_{x_l} = σ²_{x_1}·Θ(∏_{k=1}^{l-1}(1 + 1/σ_{x_k}))`, so by depth `L` the stream variance is
bounded as `Θ(L) ≤ σ²_{x_L} ≤ Θ(exp(L))`. The block Jacobian is
`∂Pre-LN(x)/∂x = I + (∂f(LN(x))/∂LN(x))(∂LN(x)/∂x)`; because `Norm` divides by the (now large)
stream scale, the second term shrinks as `σ_{x_l}` grows, and the end-to-end norm
`‖∂y_L/∂x_1‖₂ ≤ ∏_{l}(1 + A/σ_{x_l} + B/σ²_{x_l})` *converges to a finite constant* when the
variance grows exponentially. As the block Jacobian collapses toward `I`, the block becomes
locally identity-like: the deepest layers stop transforming the input in a meaningful way. This
matches the probing evidence (Gromov et al.
2024; Men et al. 2024) that deep Pre-LN layers are redundant and prunable, and the layerwise
output-variance curves where Pre-LN layers blow up early in training while shallow layers stay
small.

**Diagnostic finding 2 — normalization implicitly biases blocks toward identity at init, and
that is *good*.** De & Smith (2020) show that in a normalized residual net the variance into the
ℓth block grows as `Var(x_ℓ) ≈ Var(x_{ℓ-1}) + 1 ≈ ℓ`; after the next residual addition, the newly
added branch contributes only a `1/(ℓ+1)` fraction of the output variance at initialization. In
other words, a well-behaved deep
net *starts out close to the identity*, dominated by its skip connections, and that is exactly
why its gradients are well-conditioned and it trains. The lesson the field drew: being near
identity at init is a feature to engineer for, not an accident.

**Diagnostic finding 3 — training is staged, and depth converges unevenly.** Optimization does
not proceed uniformly. Practitioners now use Warmup-Stable-Decay learning-rate schedules
(Hu et al. 2024) precisely because training has distinct phases: an early warmup where updates
are large and chaotic, a long stable phase of small gradual changes, and a final decay. And the
*layers* do not converge together: shallow layers settle into their final representation earlier
than deep ones (Erdogan et al. 2025, LayerLock; also reflected in representation-geometry
traces, Li et al. 2025). The dependency structure makes this intuitive — a deep layer's input is
the running output of all shallow layers, while a shallow layer's gradient is back-propagated
through all the deep layers above it — so the two ends of the stack are entangled, and early in
training the deep end is feeding on, and feeding back into, representations that have not
stabilized.

**The prevailing toolkit.** The accumulated wisdom for taming deep residual training is to
*control the magnitude of the model update at initialization* — through depth- or
width-aware initializations and through scalars multiplying the residual branch — so the network
starts in a benign regime and the optimizer takes over from there.

## Baselines

**ResNet identity skip (He et al. 2016).** `x_{l+1} = x_l + F_l(x_l)`. The skip is the enabling
trick for depth.

**Pre-LN vs. Post-LN (Vaswani et al. 2017; Xiong et al. 2020).** Pre-LN removes the need for a
warmup and trains stably; Post-LN is more expressive but unstable and warmup-dependent.

**ReZero (Bachlechner et al. 2020).** Put a single scalar on each residual branch and learn it:
`x_{i+1} = x_i + α_i·F[W_i](x_i)`, with every `α_i` initialized to 0. At init the net is exactly
identity (trivially dynamically isometric); the toy stack `x_L = (1+αw)^L x_0` has Jacobian
`(1+αw)^L`, so `α=0` preserves the input signal where `α=1` would blow it up. The `α_i` are then
trained by gradient descent like any other parameter.

**SkipInit (De & Smith 2020).** Replace normalization with a single learnable scalar `α` per
residual branch, initialized to `0` (or `1/√d`), reproducing normalization's
identity-at-init bias without the norm; this alone trains 1000-layer nets.

**Fixup / T-Fixup / DeepNorm (Zhang et al. 2019; Huang et al. 2020; Wang et al. 2022).** This
line identifies the *exploding model update* as the cause of deep-Transformer instability and
bounds it by a constant via depth-aware initialization and residual scaling. DeepNorm makes it
concrete and scalable: `x_{l+1} = Norm(α·x_l + F_β(x_l))` with a *constant* skip scale `α`
(decoder-only form `(2L)^{1/4}`) and matching branch-weight scale `β` (decoder-only form
`(8L)^{-1/4}`), which provably bounds the update at init
and trains past 1000 layers.

**LayerNorm Scaling (Sun et al. 2025).** To cure Diagnostic 1 directly, divide the normalized
branch input by `√l`: `x_{l+1} = x_l + F(Norm(x_l)/√l)`. This down-weights deeper residuals,
flattens the variance curve from exponential toward linear, and keeps deep Jacobians away from
identity.

**Progressive stacking / freezing (Gong et al. 2019; Yang et al. 2020; Erdogan et al. 2025).**
Train a shallow model and *add* layers in stages, or *freeze* layers once converged, exploiting
the shallow-converge-first phenomenon to save wall-clock.

**Residual-stream widening (Wang et al. 2019; Zhu et al. 2024).** DLCL and Hyper-Connections
maintain or aggregate multiple residual streams with learned transition/weighting to improve
representation capacity and gradient flow.

## Evaluation settings

- **Models:** decoder-only Transformers in the Llama style — RMSNorm, SwiGLU activations,
  Rotary Position Embeddings (RoPE). Weights initialized from a truncated normal
  `N(0, 0.02²)` (the OLMo-2 recipe), with depth-aware initializations (DS-Init, Scaled Init,
  DeepNorm) as variants. Scales from ~71M up to 7B parameters; depths swept from 12 to 120 layers.
- **Normalization variants under test:** Pre-LN, Post-LN, Sandwich-LN, DeepNorm, LayerNorm
  Scaling — to check generality of any residual modification across them.
- **Data:** C4-en (and an alternative corpus, ClimbMix), tokenized with the Llama tokenizer;
  sequences packed to length 1024; ~50B training tokens for main runs (6B for ablations).
- **Optimizer/schedule:** AdamW, `β=(0.9, 0.95)`, weight decay `0.1`, `ε=1e-8`, gradient
  clipping `1.0`; global batch 512; 100k steps; Warmup-Stable-Decay learning-rate schedule
  (2000-step warmup, linear decay to zero over the final 10%). Learning rates tuned per
  configuration.
- **Metrics:** validation/test perplexity on the pretraining corpus (primary); out-of-distribution
  perplexity on WikiText and LAMBADA; zero-shot accuracy on PIQA, SIQA, HellaSwag, WinoGrande,
  ARC-Easy, ARC-Challenge, OpenBookQA, RACE, LAMBADA, MMLU via the LM Evaluation Harness; and a
  *spike score* (fraction of points more than seven standard deviations from a rolling 1000-step
  mean of loss / gradient norm) as a stability measure.

## Code framework

The substrate is a standard decoder-only Transformer training stack (a nanoGPT-style harness).
The attention sublayer, the MLP/FFN sublayer, the normalization layer, and the config are fixed
primitives. A `Block` combines the two sublayers around the residual stream; the model embeds
tokens, runs the blocks in sequence, applies a final norm and an output projection, and computes
the cross-entropy loss. The optimizer is AdamW with the usual decay / no-decay parameter grouping.
What is *not* settled is how each block writes its sublayer outputs into the residual stream and
whether that write should change as a function of where the block sits and how far training has
progressed — that is the single empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# existing primitives: CausalSelfAttention, MLP, LayerNorm, GPTConfig


class Block(nn.Module):
    """One Transformer block: attention sublayer then MLP sublayer, around the
    residual stream. Pre-LN normalizes each sublayer's input; the skip is the identity
    highway. How the sublayer outputs are combined with the stream is the open design."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # attention sublayer contribution onto the stream
        x = x + self.mlp(self.ln_2(x))    # MLP sublayer contribution onto the stream
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # TODO: any state the residual-write design we will choose needs.

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        # TODO: how each block writes into the residual stream as it runs through the stack.
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```

The block loop and the `Block.forward` write are where the contribution will be filled in; any
state it needs is the empty slot in `GPT.__init__`.
