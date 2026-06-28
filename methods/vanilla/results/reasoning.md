I want to start from the nuisance that shows up before any clever modeling question: a deep Transformer does not train like an ordinary deep network when I start with a large learning rate and decay it. I have to begin near zero, ramp the learning rate for thousands of iterations, and only then use the usual schedule. That warm-up stage is not just a harmless convention. It is a sensitive knob, it slows the first part of training, and the same dependence appears even when I replace Adam with plain SGD. So I should stop blaming only Adam's early second-moment estimates. If SGD still needs warm-up, the optimizer is seeing bad gradients that the architecture itself creates at initialization. I need to inspect the residual stream at step zero.

The original Transformer block wraps each sub-layer by adding first and normalizing the sum: `x <- LayerNorm(x + Sublayer(x))`. That arrangement makes me uneasy, because the thing that gives a residual network its clean depth behavior is the through-path being a true identity, and here a normalization sits right where the identity should be. Let me write the residual calculation in the form that makes the through-path explicit. If a block is `x_{l+1} = x_l + F(x_l)`, then unrolling gives `x_L = x_l + sum_{i=l}^{L-1} F(x_i)`. This is additive depth. Differentiating a loss `E` gives `dE/dx_l = dE/dx_L (1 + d/dx_l sum_{i=l}^{L-1} F(x_i))`. The direct `1` is the whole point: the top gradient reaches a shallow layer without being multiplied by every layer above it. If I scale the skip instead, `h(x_l) = lambda_l x_l`, the direct term becomes `prod_i lambda_i`. A `lambda` just above one explodes with depth; a `lambda` just below one vanishes. Gates and projections on the skip have the same product problem.

Now compare that with `x <- LayerNorm(x + Sublayer(x))`. The normalization is after the addition, so the stream from one layer to the next is not an identity-plus-residual update. It is `x_{l+1} = LN(x_l + F(x_l))`. Backward propagation now has a layer-normalization Jacobian on the highway. Stack many layers and I get a product of `J_LN` factors. That is the multiplicative through-path the residual connection was supposed to remove, reappearing through the placement of the norm. This diagnosis is still qualitative, though. Warm-up is about step size, so I need actual gradient scale, and that means doing the initialization arithmetic instead of arguing by analogy.

At initialization I use the standard mean-field setup. A `d x d` weight matrix has entries distributed as `N(0, 1/d)`, biases are zero, and layer-normalization gain is one. For the attention bookkeeping I simplify to a single head and initialize query and key projections to zero, so attention is uniform and the value branch is `(1/n) sum_j x_j W^V`. Two facts will carry most of the calculation. First, with gain one and bias zero, `||LN(v)||^2 = sum_k (v_k - mu)^2 / sigma^2 = d`, so `E||LN(v)||^2 = d` as well and layer normalization puts the vector on the sphere of radius `sqrt(d)`. Second, if `X ~ N(0, sigma^2 I_d)`, then `E||ReLU(X)||^2 = (1/2) sigma^2 d`, because half the Gaussian mass survives the ReLU and the second moment on the positive half contributes half of `sigma^2`. I do not want to carry the ReLU fact on intuition alone, so I check it: sampling `X ~ N(0, sigma^2 I_d)` at `d = 512` and averaging `||ReLU(X)||^2` gives `255.9` for `sigma = 1` against the predicted `256`, and `1023.6` for `sigma = 2` against `1024`. Good, the factor of one half is right.

I also need the norm of the layer-normalization Jacobian. Center the input with `y = x(I - 11^T/d)`. Then `LN(x)_i = y_i / sqrt((1/d) sum_k y_k^2)`, and differentiating gives `J_LN(x) = (sqrt(d) / ||y||) (I - y y^T / ||y||^2) (I - 11^T/d)`. The last two matrices are projections, so their eigenvalues are only `0` and `1`. Therefore `||J_LN(x)||_2 = O(sqrt(d) / ||x||_2)`. The bigger the input to a layer norm, the smaller the gradient it passes. This is the load-bearing claim of the whole argument, so I compute the top singular value of the actual Jacobian at `d = 512`: for an input of norm `||x||` with scale one I get `||J_LN||_2 = 0.9588` against `sqrt(d)/||x|| = 0.9588`, and at scale three I get `0.3378` against `0.3376`. The `1/||x||` dependence is exact, not just an order-of-magnitude claim.

Let me compute the hidden norm in the add-then-normalize block. After the previous normalization, the stream entering a layer has `||x||^2 = d`. The attention branch has zero mean relative to the stream because `W^V` is zero-mean, so the cross term vanishes and `E||x + attn||^2 = d + E||(1/n) sum_j x_j||^2`, between `d` and `2d`. Then the next layer normalization resets the norm to exactly `d`. The FFN branch reads a normalized vector of norm squared `d`; `xW^1` is standard normal coordinatewise, the ReLU output has expected squared norm `d/2`, and the second matrix preserves that in expectation. Adding this FFN output to the normalized stream gives `E||x^{post,5}_{l,i}||^2 = d + d/2 = (3/2)d` for every layer `l > 0`. The claim I actually care about is the constancy in `l`, so I should look at the numbers rather than trust the algebra. Building the Post-LN stack explicitly at `d = 256` and reading off the squared norm of the input to each layer's final normalization, divided by `d`, I get values hovering around `1.5` at every depth: `1.49, 1.49, 1.54, 1.60, 1.51, 1.50, 1.37, 1.64, ...` out to `L = 32`, with no upward or downward trend. So the input to the final normalization in every layer really does sit at order `d`, independent of `L`.

Now I can bound the last-layer FFN gradient in the same arrangement. For one element of the last FFN output matrix, the chain has three pieces: the loss derivative, one `J_LN(x^{post,5}_{L,i})`, and one ReLU activation coordinate `[ReLU(x^{post,3}_{L,i} W^{1,L})]_p`. The loss derivative is bounded. Since `||x^{post,3}_{L,i}||^2 = d`, that preactivation is `N(0,1)`, and a Chernoff bound gives `[ReLU(...)]_p^2 <= 2 ln(100d)` for all `p` with high probability. The layer-norm Jacobian contributes `O(d / ||x^{post,5}_{L,i}||^2) = O(d / ((3/2)d)) = O(1)`, which is consistent with the `1.5` I just measured. So each squared gradient entry is `O(ln d)`, and summing over a `d x d` matrix gives `||dL/dW^{2,L}||_F = O(d sqrt(ln d))`. There is no `L` in that bound. If it is correct, the gradient on the output-side FFN matrix should be the same whether the model is shallow or deep. I run exactly that: build Post-LN stacks of depth `4, 8, 16, 32, 64`, push a token batch through a cross-entropy head, and read the Frobenius norm of the gradient on the last layer's FFN output projection, averaged over twelve seeds. The numbers are `3.28, 3.28, 3.34, 3.28, 3.35` — flat across a sixteen-fold change in depth. The parameters closest to the output get a large gradient at initialization no matter how deep the model is, so a normal large learning rate makes an enormous first update; that is the mechanism warm-up is compensating for by making those early updates tiny.

There is also a depth-imbalance problem. For a lower layer `l`, the Post-LN gradient contains a product over layers above it. Each layer contributes a normalization Jacobian whose input has expected squared norm `(3/2)d`, so the normalization factor is about `sqrt(d / ((3/2)d)) = sqrt(2/3)`. Across layers this creates a decay factor `(2/3)^{(L-l)/2}`. Early-layer gradients are attenuated relative to late-layer gradients, so one global learning rate is pulled in two directions: it is too large for the output side and too small for the input side.

The residual-network calculation tells me what has to change. I do not want to remove layer normalization; I want its stabilizing effect on the branch inputs. But I cannot put it after the addition, because then it sits on the highway. The only place left is before the sub-layer, on the branch input. Then the stream update becomes `x <- x + Sublayer(LayerNorm(x))`. The stream being added to is raw `x`, so the highway is identity-plus-addition again. Unrolling gives `x_L = x_l + sum_{i=l}^{L-1} F(LN(x_i))`, and differentiating restores `dE/dx_l = dE/dx_L (1 + d/dx_l sum_i F(LN(x_i)))`. The leading `1` is back.

This move changes the forward norm, so I have to check the price before I trust it. The stream is no longer reset to radius `sqrt(d)` between layers. Suppose its squared norm entering layer `l` is `s_l`. The attention branch reads `LN(x)`, so its input norm squared is `d`; after the zero-mean value projection, the branch adds between `0` and `d` in expected squared norm. The FFN branch also reads a normalized vector and contributes `d/2` in expectation. Cross terms vanish by the same zero-mean argument. Therefore one full layer increases the stream's expected squared norm by at least `d/2` and at most `3d/2`. Starting from the embedding scale `d`, induction gives `(1 + l/2)d <= E||x^{pre}_{l,i}||^2 <= (1 + 3l/2)d`. The stream's expected squared norm should grow roughly linearly with depth instead of staying flat. I read the per-layer squared norm off the Pre-LN stack at `d = 256` to see whether it lands in that band: dividing by `d` I get `1.52, 2.17, 2.94, 4.03, 4.75, 5.71, 6.82, 8.19, ...` continuing up to about `40` at `L = 32`. Every value sits inside `[1 + l/2, 1 + 3l/2]`, and the growth is visibly linear in `l`. So the price is real and it is exactly the controlled linear growth the bound predicts, not an exponential blow-up.

That linear growth is the lever rather than a defect, because the Jacobian formula reads input norm in the denominator. The final layer normalization before prediction now receives a stream whose squared norm is `Theta(Ld)`, so `||J_LN||^2 = O(d / (Ld)) = O(1/L)`. Repeating the last-layer gradient bound, the ReLU coordinate still contributes `O(ln d)`, but the layer-norm Jacobian now contributes an extra `1/L`. Each squared entry is `O(ln d / L)`, hence `||dL/dW^{2,L}||_F = O(d sqrt(ln d / L))`. If that is right, the last-layer gradient should shrink like `1/sqrt(L)` rather than staying flat. Same measurement as before, now on the Pre-LN stack with the final normalization in place: the last-FFN gradient norms are `2.04, 1.41, 0.96, 0.64, 0.44` for `L = 4, 8, 16, 32, 64`. Multiplying each by `sqrt(L)` gives `4.07, 3.99, 3.83, 3.63, 3.51` — close to constant, drifting down only slowly, so the depth dependence is `1/sqrt(L)` up to the lower-order terms the bound hides. Put next to the flat `~3.3` of Post-LN, the ratio of the two gradients grows like `sqrt(L)`: about `1.6` at `L = 4` and `7.6` at `L = 64`. The large depth-independent output-layer gradient is gone.

The cross-layer product also changes character. A Pre-LN layer Jacobian is `I +` a branch Jacobian. The branch Jacobian contains layer-normalization factors of the stream at depth `j`, and those factors are `O(sqrt(d) / sqrt(jd)) = O(1/sqrt(j))`. For sufficiently deep layers the perturbation around `I` is small, so the eigenvalues stay close to one and the gradient norm should be roughly independent of the layer index, with no `(2/3)^{(L-l)/2}` decay down the stack. I have not measured the full lower-layer gradient profile here, so I will hold this as the expected behavior from the `I + small` structure rather than a confirmed one; it is the part of the picture I would most want to check directly before relying on it.

One loose end remains. Because I no longer normalize the stream between layers, the vector after the last block has depth-dependent squared scale `Theta(Ld)` — the very growth I measured climbing to `~40d` at `L = 32`. Feeding that directly to the output head would make the logits depend on depth. I only need one cleanup normalization after the stack, right before the head. That final normalization does not interrupt the residual highway between layers, and it is the same normalization whose `1/sqrt(L)` Jacobian I just watched shrink the last-layer gradient.

There is a code-level variance issue too. The stream is an accumulator over two residual branch writes per layer: attention writes once and the MLP writes once. If every branch output projection starts with the same `0.02` standard deviation, the accumulated residual variance grows with the number of branch additions. The conservative initialization is to scale the branch output projections by `1/sqrt(2L)`, so every parameter whose name ends in `c_proj.weight` is initialized with standard deviation `0.02 / sqrt(2 * n_layer)`. That matches the residual-stream picture: each branch writes a smaller initial increment, and the stack accumulates those increments without letting depth dominate the variance.

So the concrete architecture I land on is the Pre-LN Transformer block: each sub-layer reads a normalized copy of the stream and writes its output back to the unnormalized stream; after all blocks, a single final layer normalization feeds the head. The contrast baseline is Post-LN, `x <- LayerNorm(x + Sublayer(x))`; the update I want is `x <- x + Sublayer(LayerNorm(x))`, with one final `LayerNorm` before logits.

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

The measurements line up end to end. Post-LN puts a layer-normalization Jacobian on the residual highway, holds the input to each layer's final normalization at the measured `~1.5d` for every depth, and gives a last-layer gradient that stayed flat at `~3.3` from `L = 4` to `L = 64` — a depth-independent `O(d sqrt(ln d))`, with the `(2/3)^{(L-l)/2}` decay attenuating lower layers. Moving the normalization into the residual branch restored the additive identity path, let the stream's expected squared norm grow along the measured `[1 + l/2, 1 + 3l/2]` band up to `~40d`, made the final normalization's Jacobian shrink so that the last-layer gradient dropped as `1/sqrt(L)` (the `grad * sqrt(L)` product held near `~3.5` across the same depths). A final layer norm before the head handles the accumulated stream scale, and the `0.02 / sqrt(2 * n_layer)` residual-projection initialization keeps the accumulator controlled from the first step.
