A decoder-only Transformer is a stack of identical blocks, and the field keeps adding blocks because the scaling laws promise that depth buys capability. Yet past some point the deeper model is not better and is sometimes worse — a 32-layer vision Transformer loses to a 24-layer one, language-model gains flatten as layers pile on. This is strange precisely because depth was supposed to be nearly free: each block has a residual connection, $H_n = H_{n-1} + f(H_{n-1})$, so the initial token embedding $H_0$ is always one identity path away from every layer. ResNet taught us that identity shortcuts fix gradient flow and let arbitrarily deep nets train; DenseNet pushed it further, letting every layer read all earlier features. So in principle the early information is preserved. But "preserved in the residual stream" and "usable by layer 24" are not the same claim. The residual is one stream that every block writes into by addition, so $H_0$ sits buried under twenty-three blocks' worth of updates layered on top. And there is a specific reason the deep layers cannot recover it: self-attention is a smoothing operation. Each attention layer replaces a token's representation by a convex combination of all tokens' value vectors — the softmax row sums to one — which is an averaging, a low-pass filter over the sequence. Iterate it through depth and representations drift toward each other. This is over-smoothing, and it has been measured: in deep layers token representations grow similar, sequence-level features dominate, and the localized per-token information from $H_0$ is washed out. The hidden residual carries $H_0$ forward, but the very operation the deep layers run on it destroys the locality we want to keep.

The existing repairs each fall short in an instructive way. ResNet and DenseNet shortcuts operate on the hidden state $H$ — the thing that is already over-smoothed — so they re-inject representations that have themselves been through the smoothing filter, and summing dissimilar shallow and deep hidden states perturbs what the deep layer computes. DenseFormer averages whole hidden states, $H_n = \sum_{i=0}^{n} \lambda_{n,i} H_i$ with learnable static $\lambda$; because $H_0$ (localized) and a deep $H_n$ (abstract) have low similarity, summing them into the stream that feeds Q, K, and V alters the attention distribution the deep layer has learned, and it keeps $O(N^2)$ coefficients while retaining every prior hidden state. NeuTRENO is the most principled prior fix but injects its correction at the wrong place, as I will show. And KV-sharing methods are about cache size, not information flow; they do not touch over-smoothing at all. What we need is a cheap, modular intervention inside the block that restores un-smoothed early information to the deep layers without disturbing the abstract attention pattern depth bought us.

I propose Value Residual Learning, the architecture I call ResFormer, together with its practical value-embedding specialization. To see where to intervene I need the over-smoothing claim sharper than a slogan. Treat the sequence of token vectors as a function $u$ and ask what objective one self-attention update descends. Write the nonlocal smoothing functional
$$J(u) = \tfrac{1}{2}\iint \|u(x) - u(y)\|^2\, k(x,y)\, dx\, dy,$$
which penalizes any difference between token representations weighted by their affinity, so minimizing it makes tokens agree. Take its first variation: perturb $u_j \to u_j + \tau h_j$, differentiate at $\tau = 0$, change variables $(x,y)\to(y,x)$ in the second term, and the functional derivative comes out $\partial J/\partial u_j(x) = \int (u_j(x) - u_j(y))(k(x,y)+k(y,x))\, dy$. The gradient flow $du/dt = -\nabla J(u)$ therefore moves each $u(x)$ toward a $(k+k^\top)$-weighted average of the other positions. Euler-discretize with one step of size $\Delta t(x) = 1/\int (k(x,y)+k(y,x))\,dy$, initialize $u(x,0)=v(x)$ at the value vectors, choose the symmetric kernel $K(x,y) = \exp(k(x)^\top k(y)/\sqrt{d})$, and the single update is $u(x,\Delta t) = \sum_j \mathrm{softmax}(k_x^\top k_j/\sqrt{d})\,v(j)$ — exactly self-attention once the query breaks the symmetry. So self-attention literally is one gradient step on $J$, whose minimizer is a constant function: every token equal. Over-smoothing is not an implementation bug, it is the fixed point of the objective attention descends.

That diagnosis tells me how to fight it. If the trouble is descending $J$ alone, whose attractor is "all tokens identical," I should descend a regularized functional with a convex fidelity term that opposes collapse,
$$E(u, f) = J(u) + \tfrac{\lambda}{2}\int \|u(x) - f(x)\|^2\, dx,$$
which anchors $u$ to a reference signal $f$ — the same regularizer that keeps a denoised image from washing out to gray. Its gradient flow is $du/dt = -\nabla J(u) - \lambda(u-f)$; an Euler step from $u(x,0)=v(x)$ with the scaling $\lambda = \tilde\lambda/\Delta t(x)$ contributes $+\tilde\lambda(f(x)-v(x))$, so the per-token update becomes $u(i) = \sum_j \mathrm{softmax}(k_i^\top k_j/\sqrt{d})\,v(j) + \tilde\lambda(f(i)-v(i))$. The reference must be a representation that has not been smoothed, and the cleanest such signal in the network is the first layer's value vectors $V_1 = H_0 W^V_1$, computed straight from the token embedding before any attention smooths anything. Setting $f = V_1$ gives the layer-$n$ output correction $U_n = \mathrm{Attn}(Q_n,K_n,V_n) + \lambda(V_1 - V_n)$. This is NeuTRENO, and it is a real fix, but two things about where and how the term enters leave value on the table. First, $\lambda(V_1 - V_n)$ is added to the attention output, so $V_1$ is dropped on raw — it never passes through this layer's attention matrix $A_n$, even though which positions a query wants to read $V_1$ from is itself information. Second, the signed difference $V_1 - V_n$ both adds $V_1$ and subtracts the layer's own value, so the strength of the injection is entangled with a simultaneous suppression of $V_n$, and the net effect depends delicately on $\lambda$, helping only over a narrow window.

I fix both complaints by moving the injection into the value path before attention. Instead of correcting the output, build a new value and run ordinary attention on it,
$$V_n' = \lambda_{n,1}\, V_1 + \lambda_{n,2}\, V_n, \qquad V_1 = H_0 W^V_1,\quad V_n = H_{n-1} W^V_n,$$
$$U_n = \mathrm{Attn}(Q_n, K_n, V_n') = A_n V_n'.$$
Now $V_1$ and $V_n$ are aggregated by the identical learned weights $A_n$, so a query reads early-token information from the positions it actually attends to, at no extra compute. And because this is a positive weighted sum rather than a difference, nothing subtracts the layer's own value: with $\lambda_{n,1}=\lambda_{n,2}=0.5$ it is the plain average of "raw early value" and "this layer's value," and with $\lambda_{n,1}=2,\lambda_{n,2}=1$ it can weight the early value more heavily without ever turning $V_n$ negative — far more robust to $\lambda$, since there is no $-V_n$ term to overshoot.

What makes the value path the right and only safe channel is that it is the one place I can add early information without corrupting the learned attention distribution. $A_n$ is computed from $Q_n$ and $K_n$; the abstract, sequence-level mixing the depth bought me is exactly that distribution. Adding $V_1$ or $H_0$ into the query or key changes $Q_n$ or $K_n$ and hence $A_n$ itself; adding it to the post-softmax matrix corrupts $A_n$ more blatantly. Modifying $V$ leaves $A_n$ untouched and changes only what content gets aggregated under the existing weights. This is also why DenseFormer's averaging of whole hidden states is clumsier — an injected $H_i$ feeds all three of Q, K, V of the next layer and so perturbs the attention pattern; the value-only injection is the surgical version. The source is $V_1$ specifically because it is both the least-smoothed and the least-redundant signal: the ordinary hidden residual already carries $H_1$ forward and so already supplies $V_2 = H_1 W^V_2$ and everything after, so injecting $V_2$ would mostly re-deliver information the residual already provides, whereas $V_1 = H_0 W^V_1$ is a linear map of the raw token embedding, the purest token-level signal and the one the over-smoothing argument says is most diluted in the stream. Re-supplying all previous values $\sum_{i<n} V_i$ only dilutes that one clean signal with partly-smoothed, partly-redundant later values. And reusing $A_n$ rather than recomputing a separate cross-layer attention for $V_1$ avoids a second expensive attention pass and a new learned distribution that could itself over-smooth, at the cost of the mild prior that a query wants its early value from the same positions as its current value.

The deeper reason this helps points to a battery of side effects. The attention-sink pathology — deep layers dumping huge attention mass on a low-semantic token, usually the first — travels with "value-state drains," abnormally large value-state norms on those same sink tokens, in a mutual-reinforcement loop: a token with a giant value norm swamps the output $A_n V_n$ even when attended to lightly, so the model settles into sink-plus-drain. The first layer's value has no such drain; the drain is a learned deep-layer phenomenon, and $V_1$ is computed straight from the embedding. So injecting drain-free $V_1$ into the deep value $V_n'$ means the deep value no longer has to carry a pathological large norm on the sink token, which breaks the value-drain side of the loop and removes the model's reason to concentrate attention there — flattening the sink. It also means each deep layer, handed a good baseline value, only learns a small correction $\Delta V$ on top of it. The decisive test that this is a representational change and not merely an optimization shortcut from better gradient flow is whether boosting the first layer's learning rate — or specifically $W^V_1$'s — reproduces the behavior; the mechanism predicts it cannot. For the choice of coefficients, $\lambda_{n,1}=\lambda_{n,2}=0.5$ is the zero-parameter identity default; making $\lambda$ learnable per layer (init $0.5$) lets training decide, and since over-smoothing is worst deep, the learned weight on $V_1$ should grow with depth, which in turn justifies sparse variants that apply the $V_1$ residual only in the late layers.

One further move: reusing $V_1$ asks one tensor to do two jobs — be layer 1's own value and be the canonical early value re-injected everywhere. But $V_1 = H_0 W^V_1$ is functionally just a token-indexed lookup producing a value-space vector, and nothing requires that lookup to physically be the first layer's value projection. So I replace it with a dedicated embedding table $E_v$ mapping token id straight to a value-space residual with its own free parameters, gated and added into the value path exactly as before,
$$V_n' = V_n + \lambda_n\, E_{v,n}(\text{token}),$$
the same mechanism — a per-token, un-smoothed, drain-free signal injected into the value path under the layer's own attention matrix — but now decoupled from layer 1's responsibilities. The injection is gated by a learnable $\lambda$ per table (init $0.5$); the tables are initialized small ($\mathrm{std}\approx 0.01$) so the residual starts as a gentle perturbation; and rather than one table per layer, a handful of full-rank partitions suffice, placed where the mechanism wants them — two early, where raw token-value signals are still close to the input, and three in the last layers, where over-smoothing is most acute — so five gated token-to-value tables are injected at layers $1$, $2$, $N-3$, $N-2$, and $N-1$, each reading its own partition. This drops cleanly into a fixed pretraining harness as a self-contained embedding module.

```python
import torch
import torch.nn as nn


def attention(q, k, v, scale):
    A = torch.softmax((q @ k.transpose(-2, -1)) * scale + causal_mask(q, k), dim=-1)
    return A @ v                                   # V_1 rides the SAME attention matrix as V_n


class ResBlock(nn.Module):
    """Pre-norm decoder block with a value residual to the first layer's value V_1."""

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
        if layer_idx > 0:                              # init 0.5/0.5 -> Identity-ResFormer
            self.lam1 = nn.Parameter(torch.tensor(0.5))   # weight on V_1
            self.lam2 = nn.Parameter(torch.tensor(0.5))   # weight on this layer's value

    def forward(self, x, v_first):
        h = self.ln1(x)
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)      # V_n = H_{n-1} W^V_n
        if self.layer_idx == 0:
            v_first = v                                    # cache V_1 = H_0 W^V_1
        else:
            v = self.lam1 * v_first + self.lam2 * v        # V_n' = lam1*V_1 + lam2*V_n
        u = attention(q, k, v, self.scale)                # ordinary attention on mixed value
        x = x + self.Wo(u)
        x = x + self.mlp(self.ln2(x))
        return x, v_first
```

```python
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token + position embedding, plus gated value embeddings for selected layers.
    get_value_embed(i) -> lambda_i * E_v_i(token) (or None); the attention block does
    v = v + that, BEFORE attention, so the residual rides the layer's attention matrix."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        # A few dedicated token -> value-space tables (one partitioned table), small init.
        self.n_ve = 5
        self.vte = nn.Embedding(config.vocab_size * self.n_ve, config.n_embd)
        nn.init.normal_(self.vte.weight, mean=0.0, std=0.01)
        self.ve_lambda = nn.Parameter(torch.full((self.n_ve,), 0.5))   # learnable gate
        self._ve_layers = None
        self._cached_ve = None

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        if self._ve_layers is None:                       # first + last few layers
            self._ve_layers = [1, 2, self.n_layer - 3, self.n_layer - 2, self.n_layer - 1]
        vs = self.vocab_size
        self._cached_ve = {}
        for i, layer_idx in enumerate(self._ve_layers):
            offset_idx = idx + i * vs                      # partition i of the joint table
            self._cached_ve[layer_idx] = self.vte(offset_idx)
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        if self._cached_ve is None or layer_idx not in self._cached_ve:
            return None
        ve_idx = self._ve_layers.index(layer_idx)
        return self.ve_lambda[ve_idx] * self._cached_ve[layer_idx]    # lambda * E_v(token)

    def get_lm_head_weight(self):
        return self.wte.weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
```

```python
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        if value_embed is not None:                       # lambda * E_v(token), or None
            v = v + value_embed                           # value residual rides A as v does
        u = attention(q, k, v, self.scale)
```
