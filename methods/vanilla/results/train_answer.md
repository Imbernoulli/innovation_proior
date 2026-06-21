The problem that started all of this is a nuisance, not a modeling question: a deep Transformer does not train like an ordinary deep network if I begin with a large learning rate and decay it. Instead I have to start near zero, ramp the learning rate over thousands of iterations, and only then switch to the usual decay. That warm-up phase is not a harmless convention. It adds two sensitive knobs — the peak rate and the ramp length — it slows the first part of training, it is expensive to tune for large models, and crucially the same dependence shows up when I swap Adam for plain SGD. That last fact is the decisive clue. If non-adaptive SGD still needs warm-up, then the cause is not Adam's noisy early second-moment estimates; the optimizer is being handed bad gradients that the architecture itself manufactures at initialization. The remedies on the table all miss this. Rectifying the adaptive step only helps when Adam is the sole culprit, and it isn't. Scaling or gating a residual branch can damp how much a branch writes, but it does not explain a normalization-induced gradient scale, and the moment the multiplier is moved onto the skip path, $x \leftarrow \lambda x + F(x)$, it reintroduces a $\prod_i \lambda_i$ depth factor that explodes or vanishes. Simply deepening the same stack gives a stronger model class but makes the gradient path more sensitive, not less, and does nothing to remove the warm-up dependence. So the real question is architectural: why do the residual stream and layer normalization, as the original Transformer arranges them, force warm-up, and what arrangement instead gives initial hidden-state and gradient scales stable enough to train with a plain large learning rate?

The method I land on is the Pre-LN Transformer block. I propose moving layer normalization out of the residual highway and into each sub-layer's branch input, so the stream is updated as
$$x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$$
for every attention and MLP sub-layer, with the skip path left as a pure $+\,x$, and then a single final $\mathrm{LayerNorm}$ applied once after all the blocks, just before the output head. The contrast baseline is the original Post-LN arrangement, $x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, which adds first and normalizes the sum. The whole design follows from understanding why that one normalization placement breaks the residual through-path. The clean residual identity is what makes depth tractable: if a block is $x_{l+1} = x_l + F(x_l)$, unrolling gives $x_L = x_l + \sum_{i=l}^{L-1} F(x_i)$, so depth composes additively, and differentiating a loss $E$ gives $\frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L}\big(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i)\big)$. That leading $1$ is the entire point — it carries the top gradient straight down to a shallow layer without multiplying it by every layer above. Post-LN destroys this by putting a non-identity operation on the repeated stream: between layers the update is really $x_{l+1} = \mathrm{LN}(x_l + F(x_l))$, so the backward highway accumulates a product of layer-normalization Jacobians, exactly the multiplicative structure the residual connection was invented to avoid.

To turn that qualitative diagnosis into the gradient scale that warm-up cares about, I work at initialization with the standard mean-field setup: a $d\times d$ matrix has entries $\mathcal{N}(0, 1/d)$, biases are zero, the layer-norm gain is one, and to simplify attention I take a single head with query and key projections zeroed, so attention is the uniform average $(1/n)\sum_j x_j W^V$. Two facts carry most of the calculation. First, with gain one and bias zero, layer normalization projects every vector onto the sphere of radius $\sqrt{d}$, so $\mathbb{E}\|\mathrm{LN}(v)\|^2 = d$. Second, if $X \sim \mathcal{N}(0, \sigma^2 I_d)$ then $\mathbb{E}\|\mathrm{ReLU}(X)\|^2 = \tfrac{1}{2}\sigma^2 d$, because half the Gaussian mass survives the ReLU. I also need the Jacobian norm of layer normalization itself: centering with $y = x(I - \mathbf{1}\mathbf{1}^\top/d)$ gives $J_{\mathrm{LN}}(x) = \frac{\sqrt{d}}{\|y\|}\big(I - \frac{yy^\top}{\|y\|^2}\big)\big(I - \frac{\mathbf{1}\mathbf{1}^\top}{d}\big)$, and since the two trailing matrices are projections with eigenvalues in $\{0,1\}$, $\|J_{\mathrm{LN}}(x)\|_2 = O(\sqrt{d}/\|x\|_2)$. The bigger the input to a layer norm, the smaller the gradient it passes back. In Post-LN, every layer normalization resets the stream to radius $\sqrt{d}$, and the input to the last normalization inside a layer settles at $\mathbb{E}\|x^{post,5}_{l,i}\|^2 = \tfrac{3}{2}d$ for every $l > 0$ — constant in depth. Pushing the last-layer FFN gradient through one such Jacobian (squared norm $O(d/(\tfrac{3}{2}d)) = O(1)$) and one ReLU coordinate bounded by $O(\ln d)$ via a Chernoff argument gives $\|\partial L/\partial W^{2,L}\|_F = O(d\sqrt{\ln d})$, with no $L$ in it: the parameters nearest the output get a huge gradient no matter how deep the model is, so a normal large learning rate makes a destabilizing first step, and warm-up "works" only by shrinking those steps. Worse, the same Post-LN Jacobian recurs across layers with norm about $\sqrt{2/3}$, so a lower layer $l$ sees its gradient decayed by $(2/3)^{(L-l)/2}$ — early and late layers want different learning rates, which a single global rate cannot satisfy.

The residual-network calculation tells me precisely what to change. I do not want to remove layer normalization, because I still need it to stabilize each branch's input; I just cannot leave it sitting on the highway. The only remaining place is before the sub-layer, on the branch input, which is exactly $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$. Now the thing being added to is raw $x$, the highway is identity-plus-addition again, the unrolling $x_L = x_l + \sum_{i=l}^{L-1} F(\mathrm{LN}(x_i))$ holds, and the leading $1$ in the gradient is restored. The price is that the stream is no longer reset between layers, so its norm grows: each branch reads a normalized vector, attention adds between $0$ and $d$ in expected squared norm and the FFN adds $d/2$, cross terms vanish by the zero-mean value projection, so one layer raises the expected squared norm by between $d/2$ and $3d/2$, and induction from the embedding scale $d$ gives $(1 + l/2)d \le \mathbb{E}\|x^{pre}_{l,i}\|^2 \le (1 + 3l/2)d$ — linear growth with depth. That looks like a new problem but is in fact the cure, because the Jacobian formula turns large input norms into small gradients. The final normalization before the head now receives a stream of squared norm $\Theta(Ld)$, so its Jacobian contributes $O(1/L)$, and repeating the last-layer bound — ReLU coordinate still $O(\ln d)$, but now an extra $1/L$ — yields
$$\Big\|\frac{\partial L}{\partial W^{2,L}}\Big\|_F = O\!\left(d\sqrt{\tfrac{\ln d}{L}}\right).$$
The large, depth-independent output gradient is gone; the top gradient shrinks like $1/\sqrt{L}$, which is precisely what removes the need for warm-up. The cross-layer product also tames itself: a Pre-LN layer Jacobian has the form $I +$ (branch Jacobian), and the branch's layer-norm factors scale as $O(1/\sqrt{j})$ with depth, so for deep layers the perturbation around $I$ is small, the eigenvalues stay near one, and gradients are balanced across the stack with no $(2/3)^{(L-l)/2}$ decay.

Two design choices close the loop. Because I stopped normalizing between layers, the vector after the last block has depth-dependent scale $\Theta(Ld)$, which would make the logits depend on depth; so I add exactly one cleanup $\mathrm{LayerNorm}$ after the stack, right before the head. It does not interrupt the residual highway between layers, and it is the very normalization whose $1/\sqrt{L}$ Jacobian appears in the gradient bound above — so it pays for itself. There is also a variance issue at the code level: the stream is an accumulator over two residual writes per layer (attention once, MLP once), and if every branch output projection starts with the same $0.02$ standard deviation, the accumulated residual variance grows with the number of additions. The conservative fix is to scale every residual output projection by $1/\sqrt{2L}$ — every parameter whose name ends in `c_proj.weight` is initialized with standard deviation $0.02/\sqrt{2\,n_{\text{layer}}}$ — so each branch writes a smaller initial increment and the stack accumulates them without letting depth dominate the variance. Token-embedding and output-head weights are tied as in the standard GPT harness. The end result is a block that reads a normalized copy of the stream and writes its sub-layer output back to the unnormalized stream, a single final layer norm before the head, and a depth-aware residual-projection initialization — an architecture whose initial hidden-state and gradient scales are stable enough to drop warm-up entirely.

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


class Block(nn.Module):
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

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

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
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
```
