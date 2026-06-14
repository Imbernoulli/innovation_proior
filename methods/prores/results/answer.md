# ProRes (Progressive Residual Warmup), distilled

ProRes is a training-phase-aware modification of the Transformer residual stream. It multiplies
each layer's residual *branch* by a predefined, non-learnable scalar `α(l, t)` that starts at 0
(exact identity at initialization) and ramps to 1 over training, with deeper layers warming up
more slowly. Residual branches therefore switch on in a wave from shallow to deep: shallow layers
are prioritized early, and deeper layers engage at full strength only once the layers beneath them
have stabilized. It adds no learnable parameters and leaves the optimizer untouched.

## Problem it solves

Deep Transformer language models are (1) unstable to train — loss/gradient spikes worsen with
depth — and (2) wasteful, because in Pre-LN the deepest layers collapse toward identity maps and
contribute little. Existing fixes (depth-aware init, DeepNorm's constant skip-scaling, LayerNorm
Scaling's `1/√l`, ReZero/SkipInit's zero-init learnable scalar) act only *at initialization* and
are then frozen or left to the optimizer; none coordinates *when* and *in what depth order* each
layer should begin contributing across the staged trajectory of training.

## Key idea

Replace the residual write `x_{l+1} = x_l + F(Norm(x_l))` with

```
x_{l+1} = x_l + α(l, t) · F(Norm(x_l)),
```

where `α(l, t)` is a *predefined* scalar (not a parameter) depending on layer index `l` and step
`t`. The default **linear schedule**:

```
α(l, t) = min( t / (T · l), 1 ),   l = 1, …, L,
```

- `T` = warmup length of the first layer; layer `l` reaches `α = 1` at step `T·l`; the whole
  model finishes warming up at `T·L` and thereafter runs exactly as vanilla.
- **Identity at init** (`α(l, 0) = 0`): no early activation-variance blowup, well-conditioned
  gradients — the property ReZero/SkipInit engineer for, made exact and parameter-free.
- **Bounded update across depth *and* time**: early on only shallow layers have nonzero `α`, so
  the model update is throttled; the constraint relaxes itself layer-by-layer as training
  proceeds, tight during the chaotic warmup and fully released in the stable phase — unlike a
  constant init-time bound.
- **Shallow-before-deep ordering**: a deep branch stays near 0 while the shallow layers below it
  do their large early updates, so deep layers build on stabilized inputs instead of injecting
  randomly-initialized noise into the representations above and the gradients below.

The scalar rides on the *branch only*; the identity skip stays at weight 1, so `α=0` gives exactly
the identity and `α=1` gives exactly the unmodified block.

## Why a predefined schedule (not learned)

A learned per-layer scalar (ReZero, SkipInit) has no mechanism to enforce shallow-first order or
respect the training phase — the optimizer ramps branches in whatever order locally lowers the
loss. The shallow-before-deep order is a *prior* about how training should proceed (shallow layers
converge first; the stack's dependency structure couples deep-layer inputs to shallow outputs and
shallow gradients to deep layers), so it is imposed by a fixed schedule, not discovered. Cost:
zero learnable parameters.

## Why this shape

- **`equal`** (`min(t/T, 1)`, no `l`-dependence): all branches warm together — only delays init
  chaos and ignores the shallow-before-deep dependency order. (Equals BranchNorm as a Post-LN
  special case.)
- **`reverse`** (`τ_l = T·(L−l+1)`, deep-first): lets the last layers dominate early while shallow
  layers are still weak, matching the failure mode that warmup is meant to avoid.
- **static `fix-*`** (`1/L`, `1/√l` constant) vs. **`stagewise-*`** (same fraction at init, then
  relaxed to 1): tests the distinction between a permanent depth constraint and one that only
  protects the unstable early phase.
- Curvature variants share total warmup `T·L`: `linear` is the default; `linear-square`
  (`(min(t/(T l),1))²`) eases in more gently (better for touchier Post-LN); `linear-sqrt` faster.

## Generalization across normalization variants

Wherever a block has a residual branch, multiply that branch by `α(l, t)`:

| Variant | With ProRes |
|---|---|
| Pre-LN | `x_{l+1} = x_l + α(l,t)·F(Norm(x_l))` |
| Post-LN | `x_{l+1} = Norm(x_l + α(l,t)·F(x_l))` |
| Sandwich-LN | `x_{l+1} = x_l + α(l,t)·Norm(F(Norm(x_l)))` |
| DeepNorm | `x_{l+1} = Norm(α_const·x_l + α(l,t)·F_β(x_l))` |
| LayerNorm Scaling | `x_{l+1} = x_l + α(l,t)·F(Norm(x_l)/√l)` |

## Default hyperparameters

`T = 1000`, linear schedule, untuned. The only principled adjustment is at extreme depth: keep
`T·L ≤` total training steps so every layer reaches full strength (e.g. `T = 500` at 96–120
layers). The LR/optimizer setup is unchanged (AdamW, the usual decay/no-decay groups).

## Working code

Single-stream (nanoGPT-style) implementation. The branch scalar depends on the global step, which
the block does not see, so the schedule lives in the model's forward loop; `block_out = x + delta`
makes the per-block residual contribution `delta = block_out − x`, scaled by `α`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# existing primitives: CausalSelfAttention, MLP, LayerNorm, GPTConfig


class Block(nn.Module):
    """Vanilla Pre-LN block — unchanged. The residual schedule is applied in GPT.forward."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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

        # ProRes: T = first-layer warmup length; layer l warms up over T*l steps,
        # the whole model over T*L. The step counter is a buffer, not a parameter.
        self.prores_T = 1000
        self.register_buffer('_prores_step', torch.zeros(1, dtype=torch.long))

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        step = self._prores_step.item()
        T = self.prores_T

        for i, block in enumerate(self.transformer.h):
            block_out = block(x)                  # block_out = x + delta (Pre-LN residual)
            if self.training and step < T * (i + 1):
                layer_idx = i + 1                 # 1-indexed layer l
                alpha = min(step / (T * layer_idx), 1.0)   # alpha(l,t) = min(t/(T*l), 1)
                x = x + alpha * (block_out - x)   # scale only the branch: x <- x + alpha*delta
            else:
                x = block_out                     # alpha == 1: exact vanilla Pre-LN

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        if self.training:
            self._prores_step += 1               # first training forward uses t=0
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # unchanged: ProRes adds no learnable parameters
        decay = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```

Equivalent per-layer residual-branch form: give each decoder layer a plain float attribute
`self.alpha` (init 0.0), apply `x = residual + sublayer_out * self.alpha` for each sublayer, and
let the training loop set `alpha` per layer each step from
`α_list = clip(step / layerwise_warmup_steps, 0, 1)`, where `layerwise_warmup_steps =
linspace(T, T·L, L)` (equivalently `T·l`). Once `step > T·L` the schedule is finished and the
block runs exactly as vanilla.

## Schedule family (one knob `T`, total warmup `T·L` unless noted)

```
linear         min(t/(T l), 1)
linear-sqrt    (min(t/(T l), 1))^(1/2)
linear-square  (min(t/(T l), 1))^2
equal          min(t/T, 1)                                   [warmup T; all layers together]
reverse        min(t/(T(L-l+1)), 1)                          [deep-first]
stagewise-0    clip((t - T(l-1))/T, 0, 1)
stagewise-L    clip(...)·(1 - 1/L) + 1/L
stagewise-√l   clip(...)·(1 - 1/√l) + 1/√l
fix-L          1/L          fix-√L  1/√L          fix-√l  1/√l   [static, no warmup]
```

`linear` is the default. `linear-square` and `stagewise-L` are gentler early ramps for less stable
normalization choices.
