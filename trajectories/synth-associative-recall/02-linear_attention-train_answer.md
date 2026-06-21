The recurrent floor came back even sharper than the capacity argument predicted: $0.0265$ at `mqar-128`, $0.000469$ at `mqar-512`, $0.000266$ at `mqar-2048` — near-chance everywhere, including the easy 8-pair setting. That last number is the diagnostic one. Had the cell scored, say, $0.6$ at 128 and then fallen off with length, I would read it as a pure capacity story, the 64-dimensional state filling up. But near-zero even at 8 bindings says the problem is more basic than an overflowing bucket: the recurrent state never learned to do the content-addressed lookup at all. It has no operation that *compares* the current query token against the set of earlier keys; it only folds tokens into a running summary and hopes the read-out can project the right value back out. That is the failure I have to fix, and it names precisely what the next mixer must have — an explicit score between the query at position $t$ and every earlier key, so the lookup is content-addressed instead of compressed.

The mechanism with an explicit query-key score is attention, but full softmax attention forms an $N\times N$ matrix whose cost grows with the sequence, and the research question demands a *sub-quadratic* mixer. So I propose linear attention: keep the explicit query-key score the recurrent cell lacked, but collapse the over-keys part into a fixed-shape running state so the per-step cost stays bounded. Read attention as a kernel smoother — the output at position $i$ is a weighted average of values, weight proportional to a similarity between query $i$ and key $j$,
$$y_i = \frac{\sum_{j\le i}\mathrm{sim}(q_i,k_j)\,v_j}{\sum_{j\le i}\mathrm{sim}(q_i,k_j)}.$$
With $\mathrm{sim}(q,k)=\exp(q^\top k/\sqrt d)$ this is softmax attention exactly, but the only property $\mathrm{sim}$ truly needs is non-negativity, so the weights are a sensible average; the exponential is one choice among many. What forces softmax's growing matrix is that $\exp(q_i^\top k_j)$ tangles $q_i$ and $k_j$ inside a nonlinearity, so $q_i$ cannot be pulled out and reused across positions. Suppose instead the similarity *factored*, $\mathrm{sim}(q,k)=\varphi(q)^\top\varphi(k)$ for a non-negative feature map $\varphi$. Then the numerator becomes $\sum_{j\le i}\big(\varphi(q_i)^\top\varphi(k_j)\big)v_j$, and because $\varphi(q_i)$ does not depend on $j$ it pulls out: $\varphi(q_i)^\top\sum_{j\le i}\varphi(k_j)v_j^\top$. The inner sum $S_i=\sum_{j\le i}\varphi(k_j)v_j^\top$ is a matrix of fixed shape that accumulates, $S_i=S_{i-1}+\varphi(k_i)v_i^\top$, and the denominator carries $z_i=z_{i-1}+\varphi(k_i)$, so
$$y_i = \frac{\varphi(q_i)^\top S_i}{\varphi(q_i)^\top z_i}.$$
A fixed-shape running state, no growing cache — and crucially an *explicit* per-query, per-key score $\varphi(q_i)^\top\varphi(k_j)$, the comparison the recurrent cell never had. That is the bridge from step one: the LSTM also carried a fixed-shape state; what it lacked was the explicit content-addressed read, which linear attention supplies, so it should clear the $0.0265$ floor at `mqar-128` decisively.

The choice of $\varphi$ is where this rung's strength and its weakness both come from. The simplest positive feature map keeps the feature dimension equal to the head dimension and pushes each coordinate through something non-negative. A plain ReLU zeroes the gradient on the negative side, killing learning for any coordinate that goes negative; the exponential linear unit fixes that — $\mathrm{elu}(x)=x$ for $x\ge0$ and $e^x-1$ for $x<0$, bottoming out at $-1$ with a live gradient everywhere — so I shift it up to $\varphi(x)=\mathrm{elu}(x)+1\ge0$. Then $\varphi(q)^\top\varphi(k)$ is a sum of products of non-negatives, a valid attention, cheap, gradient-alive, with a small feature dimension that keeps the state compact. But I have to be honest about what this $\varphi$ does to the *shape* of the weights, because it is exactly what MQAR punishes. Recall wants a spiky, low-entropy distribution — when the query matches one key, almost all weight should jump onto that key's value. Softmax produces that because $\exp$ blows up the gap between a large dot product and a medium one. The $\mathrm{elu}+1$ map does not approximate $\exp$ at all; it is a smooth, roughly-linear positive function, and the kernel it induces is *gentle*, spreading weight broadly across keys. So the read $\varphi(q_i)^\top S_i$ returns the matching value diluted by a high-entropy smear of every other stored value: with few keys the matching term may still dominate the argmax, but with many keys the smear swamps the signal. This is structurally different from the LSTM's failure — the LSTM had no comparison at all; linear attention has the comparison, but the comparison is too blurry.

For the scaffold I use the training-time *quadratic view* rather than the sequential recurrence, and the two are identical math, so the choice is purely about GPU efficiency. The recurrent form accumulates $S_i,z_i$ with a scan down the sequence — perfect for generation, but a long sequential dependency stalls the tensor cores at training time. The quadratic view instead materializes the score matrix $A=\varphi(Q)\varphi(K)^\top$, applies a causal lower-triangular mask, and reads $y=(\text{masked }A)V$ normalized by the masked row-sums. Row $i$ of the masked product is $\sum_{j\le i}\big(\varphi(q_i)^\top\varphi(k_j)\big)v_j=\varphi(q_i)^\top S_i$ and the row-sum normalizer is $\sum_{j\le i}\varphi(q_i)^\top\varphi(k_j)=\varphi(q_i)^\top z_i$ — term for term the same function, computed in a different associativity order, but at these moderate lengths the $T\times T$ matmul is far faster than a scan and the memory is fine. The head dimension is a small projection (32) below $d_{\text{model}}=64$, so $\varphi$ acts in that 32-dim space and the kernel stays cheap; $q,k,v$ come from bias-free linear projections of the block-normed input and an `out_proj` maps back to $d_{\text{model}}$; the denominator is clamped away from zero. Causality is enforced by the explicit `tril` mask on the score matrix — not by a recurrence as in step one — which is the one place to be careful, since unlike the LSTM, attention is not causal by default. I add no short convolution inside the mixer: the substrate's fixed first-layer short causal conv already handles the local shifts, so this rung is purely the global linear-attention read. I expect it to genuinely solve `mqar-128` (well above $0.7$), confirming that step one's problem was the absence of a comparison and not capacity, but to sag as the key count climbs — well below the 128 number at `mqar-512`, back near the floor at `mqar-2048` — because a non-spiky kernel cannot single out the one matching key among many. Closing *that* gap with a spiky feature map is the natural next rung.

```python
# EDITABLE region of custom_strategy.py — step 2: linear attention (elu+1 feature map)
class CustomMixer(nn.Module):
    """Causal linear attention with phi(x) = elu(x) + 1 (Katharopoulos 2020)."""

    def __init__(self, d_model: int, seq_len: int, head_dim: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.phi(self.q_proj(x))                  # [B, T, F]
        k = self.phi(self.k_proj(x))                  # [B, T, F]
        v = self.v_proj(x)                            # [B, T, D]
        # Quadratic-view causal linear attention (training time): build the
        # T x T causal kernel matrix <q_t, k_s> and apply to v. This matches
        # the recurrent / prefix-sum view exactly but uses simple matmuls.
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, T, T]
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, 0.0)
        denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = scores / denom
        out = torch.matmul(weights, v)
        return self.out_proj(out)
```
