## Research question

A Transformer is a deep stack of layers, and each layer is built from two sub-layers: multi-head self-attention and a position-wise feed-forward network. Around those sub-layers sit a residual connection and layer normalization. The residual stream is the running state that every sub-layer reads from and writes back into, so the way the stream is updated controls both the forward signal scale and the gradient scale seen by the optimizer.

The optimization problem is practical and sharp. Deep Transformers trained from scratch are unusually dependent on learning-rate warm-up: instead of starting from a large learning rate and decaying it, the schedule starts near zero and ramps to a chosen maximum over a chosen number of iterations. That adds two sensitive knobs, slows the early part of training, and is expensive to tune for large models. The goal is to understand why the residual stream and layer normalization make warm-up necessary, then find a layer arrangement whose initial hidden-state and gradient scales are stable enough to train with a plain large initial learning rate and a normal decay schedule.

## Background

**Residual learning.** He, Zhang, Ren & Sun (2016) identified a degradation problem in deep plain networks: adding layers could raise training error even when the deeper network had enough capacity to represent the shallower one. The residual block changes the target from fitting `H(x)` directly to fitting `F(x) = H(x) - x`, so the block computes `x_{l+1} = x_l + F(x_l)`. If the identity map is best, the residual branch can be driven toward zero instead of forcing a stack of nonlinear layers to synthesize an identity from scratch.

**Clean identity shortcuts.** The later identity-mapping analysis makes the residual through-path precise. With an exact identity skip and no operation after the addition,

```text
x_L = x_l + sum_{i=l}^{L-1} F(x_i),
```

so depth composes additively rather than as a product of matrices. Differentiating the loss `E` gives

```text
dE/dx_l = dE/dx_L (1 + d/dx_l sum_{i=l}^{L-1} F(x_i)).
```

The leading `1` sends `dE/dx_L` directly to shallower layers. If the skip is scaled, `h(x_l) = lambda_l x_l`, the direct term becomes a product `prod_i lambda_i`, which explodes for `lambda > 1` and vanishes for `lambda < 1`; gates and projections on the skip introduce the same depth product.

**Layer normalization.** Ba, Kiros & Hinton (2016) normalize each example across its feature dimension:

```text
mu = (1/d) sum_k v_k,
sigma^2 = (1/d) sum_k (v_k - mu)^2,
LN(v) = gamma (v - mu) / sigma + beta.
```

With gain `gamma = 1` and bias `beta = 0`, layer normalization projects every nonconstant vector to radius `sqrt(d)`, so the expected squared norm is also `d`:

```text
||LN(v)||^2 = sum_k (v_k - mu)^2 / sigma^2 = d.
E||LN(v)||^2 = d.
```

Its Jacobian also depends on the input norm. If `y = x(I - 11^T/d)`, then

```text
J_LN(x) = (sqrt(d) / ||y||) (I - y y^T / ||y||^2) (I - 11^T/d),
```

and the two projection matrices have eigenvalues `0` or `1`, so `||J_LN(x)||_2 = O(sqrt(d) / ||x||_2)`.

**Mean-field tools at initialization.** With Xavier initialization, a `d x d` matrix has entries sampled as `N(0, 1/d)`, biases are zero, and layer-normalization gain starts at one. The simplified attention calculation sets query and key projections to zero at initialization, making attention a uniform average `(1/n) sum_j x_j W^V`. A basic feed-forward fact is

```text
X ~ N(0, sigma^2 I_d)  =>  E||ReLU(X)||^2 = (1/2) sigma^2 d.
```

These ingredients are enough to compute expected hidden-state norms and last-layer gradient scales before training begins.

## Baselines

**Post-LN Transformer layer.** Vaswani et al. (2017) arrange each sub-layer as

```text
x <- LayerNorm(x + Sublayer(x)).
```

The residual addition is immediately followed by layer normalization. This architecture works well when paired with warm-up, but from-scratch training is brittle without the ramp, and the sensitivity persists under both Adam and plain SGD. The relevant limitation is not merely optimizer bookkeeping: the architecture itself places a nonidentity operation on the repeated stream between layers, so any explanation has to account for how that operation changes hidden-state norms and backpropagated gradients across depth.

**Residual-branch scaling or gating.** A cheap way to calm a residual stack is to scale a branch contribution, as in `x <- x + a F(x)`, sometimes with `a` initialized small. That can damp how much a branch writes. The limitation is that it does not by itself explain a normalization-induced gradient scale, and putting the multiplier on the skip path, `x <- lambda x + F(x)`, reintroduces the `prod lambda` depth factor that clean identity shortcuts avoid.

**Optimizer-side warm-up remedies.** Another baseline explanation attributes warm-up to large early variance in adaptive optimizers. Rectifying the adaptive step can help in settings where Adam is the sole issue. The limitation is that the same warm-up dependence appears with non-adaptive SGD, so the residual stream's initial gradient scale remains a separate architectural problem.

**Simply deepening the same stack.** Increasing depth under the same residual/normalization arrangement gives a stronger model class but also makes the optimization path longer and the gradient path more sensitive to the per-layer Jacobians. The limitation is that added depth does not remove the warm-up dependence; it makes the scale accounting more important.

## Evaluation settings

Natural settings for this problem already include:

- **IWSLT14 German-to-English translation:** roughly 153K/7K/7K train/validation/test sentence pairs, joint BPE vocabulary, a 6-layer encoder and 6-layer decoder, hidden size 512, FFN size 1024, 4 heads, dropout, label-smoothed cross-entropy, validation loss, and BLEU.
- **WMT14 English-to-German translation:** roughly 4.5M sentence pairs, Transformer base scale, hidden size 512, 8 heads, label smoothing, validation loss, and BLEU on the standard validation/test split.
- **BERT-style pre-training:** 12 Transformer layers, hidden size 768, 12 heads, Wikipedia plus book text, validation loss during pre-training, then small downstream classification tasks such as MRPC and RTE.
- **Optimization diagnostics:** Adam with translation-standard beta values or plain SGD; schedules with and without linear warm-up before inverse-square-root, linear, or step decay; measurements of hidden-state norms and parameter-gradient norms at initialization.

## Code framework

The substrate is a standard Transformer/GPT-style language-model harness. Attention, the MLP, layer normalization, embeddings, the output projection, weight initialization, and cross-entropy loss are available primitives. The open design slot is the body of one block: it must update the running stream using the existing sub-layers and normalization modules, then the stack must decide whether anything is needed before logits.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def forward(self, x):
        pass


class MLP(nn.Module):
    def forward(self, x):
        pass


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # TODO: the architecture we design here updates and returns the stream.
        pass


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # TODO: a post-stack transform, if the design requires one.
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # TODO: any initialization rule implied by the block design.

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        # TODO: apply any post-stack transform chosen above.
        logits = self.lm_head(x)
        loss = (F.cross_entropy(logits.view(-1, logits.size(-1)),
                                targets.view(-1), ignore_index=-1)
                if targets is not None else None)
        return logits, loss
```
