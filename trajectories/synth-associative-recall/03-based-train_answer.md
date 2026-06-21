Linear attention confirmed its diagnosis line for line. At `mqar-128` it jumped to $0.813$, a decisive break from the recurrent floor's $0.0265$, settling the step-one question: the LSTM failed for lack of a content-addressed comparison, because the moment the mixer had an explicit $\varphi(q)^\top\varphi(k)$ score it solved the easy setting. But then the predicted sag appeared, and it is steep — $0.368$ at `mqar-512`, $0.163$ at `mqar-2048`. This is not the LSTM's capacity collapse; linear attention's running state $S_i=\sum_{j\le i}\varphi(k_j)v_j^\top$ is a matrix, not a single vector, so it has room. It is a *selectivity* collapse: the read $\varphi(q_i)^\top S_i$ returns the matching value diluted by a high-entropy smear of every other stored value, and the more bindings there are, the larger that smear, until the matching term no longer wins the argmax. The $\mathrm{elu}+1$ kernel is too gentle. So the requirement is sharp — I need a feature map whose induced kernel is *spiky*, so that when $q_i$ matches one key $k_j$ the weight on it blows up past every non-matching key the way softmax's $\exp$ does — while keeping everything that worked: the factored $\varphi(q)^\top\varphi(k)$ that collapses the over-keys part into a fixed-shape state, the quadratic training view, and the sub-quadratic generation cost. I want to swap *only* the feature map for one that actually tracks $\exp(q^\top k)$.

I propose BASED: linear attention with a second-order Taylor feature map. The exponential has the series $\exp(x)=1+x+x^2/2+x^3/6+\cdots$, and a truncation is a polynomial in $x=q^\top k$, exactly the kind of similarity that factors through a finite explicit feature map. Truncated at second order, the kernel is
$$k(q,k)=1+q^\top k+\tfrac12(q^\top k)^2.$$
It is a legal kernel term by term: the constant $1$ is $\langle[1],[1]\rangle$, the linear term is the ordinary dot product, and the quadratic term $(q^\top k)^2=\sum_{a,b}(q_aq_b)(k_ak_b)$ is the dot product of the flattened outer products $q\otimes q$ and $k\otimes k$, so the explicit map $\varphi(x)=\big[\,1,\;x,\;(x\otimes x)/\sqrt2\,\big]$ reproduces $1+q^\top k+\tfrac12(q^\top k)^2$ exactly, with the $\sqrt2$ placing weight $\tfrac12$ on the squared block to match the Taylor coefficient. It is finite, deterministic, with no randomness and no variance injected into the attention weights — the last place I would want noise when I am trying to make a clean lookup spike. It has the two properties recall demands. Non-negativity: reading $g(x)=1+x+x^2/2$ as a function of the scalar $x=q^\top k$ and completing the square, $g(x)=\tfrac12\big((x+1)^2+1\big)\ge\tfrac12>0$ for every $q,k$, with no special-casing — stronger than $\mathrm{elu}+1$, which is only non-negative by construction. Spikiness: $g$ grows *quadratically* in $x$, so a large positive $q^\top k$ gets dramatically more weight than a medium one, the gap widening like the gap in $\exp$'s first three terms. That quadratic growth is the sharpening the roughly-linear $\mathrm{elu}+1$ map could not produce, and it is the missing ingredient behind the $0.368/0.163$ sag.

One scaling subtlety is load-bearing: a truncated Taylor series approximates $\exp$ only while its argument is modest, so if $q^\top k$ were huge the dropped $x^3/6$ and beyond would matter and the two-term polynomial would diverge from the true exponential. I keep the argument small by working in a *low* feature dimension — project $q$ and $k$ down to $\text{feature\_dim}=16$ with bias-free linears and scale by $\text{feature\_dim}^{-1/2}$ before forming the kernel, the same temperature softmax uses. This does double duty: it keeps $q^\top k$ in the regime where the second-order truncation is benign, and it keeps the quadratic feature expansion cheap (in the explicit-feature view the expansion is $1+\tilde d+\tilde d^2$ wide, dominated by $\tilde d^2$, so a small $\tilde d=16$ keeps it small). In this fill I never materialize $\varphi$ — I compute the kernel directly as $1+qk+0.5\,qk^2$ — but the same scaling argument applies to the raw $qk$. I also fold in the second lesson from linear attention: a global average is blunt at the fine token-to-token bookkeeping recall needs, lining a query up against the key that immediately preceded its value. Step two leaned entirely on the substrate's fixed first-layer short conv for that; this rung adds its *own* depthwise short causal convolution inside the mixer (kernel size 3, left-padded so it stays causal) and uses it twice — the convolved local signal is added to the input before forming $q,k,v$, so the comparison is built on locally-shifted features, and it is added back to the output as a residual, $\text{out\_proj}(\text{out})+\text{local}$, so the precise local shift survives the global average. This spiky-global-plus-cheap-local pairing is the genuine difference from the step-two fill.

Concretely in the scaffold: I form $q,k,v$ from the locally-mixed input $h=x+\text{local}$, scale $q$ and $k$ by $\text{feature\_dim}^{-1/2}$, compute the raw score matrix $qk=qk^\top$, then the Taylor-2 kernel $1+qk+0.5\,qk\cdot qk$ directly — the quadratic training view, mathematically identical to the recurrent state $S_i=\sum_{j\le i}\varphi(k_j)v_j^\top$ but expressed as one batched matmul, far faster than a scan at these lengths. The causal `tril` mask zeroes the kernel above the diagonal (attention is not causal by default — the same care step two needed). The normalizer is the masked row-sum of the kernel itself, $\text{denom}=\text{kernel.sum}(-1)$, clamped away from zero; note it includes the $1+\cdots$ constant in every entry, so it sums the *whole* Taylor-2 weights, not just the second-order part — the correct normalizer for this kernel, keeping the weights a proper average. Then $\text{weights}=\text{kernel}/\text{denom}$, $\text{out}=\text{weights}\cdot v$, and $\text{out\_proj}(\text{out})+\text{local}$. I expect the spiky kernel to erase most of the step-two sag at moderate scale — `mqar-128` from $0.813$ toward near-perfect, and the most dramatic gain at `mqar-512`, from $0.368$ up toward the $0.9$s, since 32 pairs is exactly where the gentle kernel's smear was eating accuracy. The harder prediction is `mqar-2048`: the spiky kernel sharpens the *read*, but the fixed-shape state still has to hold all 128 bindings, and the deeper recall-state floor — exact recall of an arbitrary one of $m$ bindings needs carried state that grows with $m$ — starts to bite, so I expect only modest improvement over $0.163$. Near-perfect at 128 and 512, still largely failing at 2048, would confirm that spikiness closes the *selectivity* gap completely at moderate scale but cannot, on its own, beat the state-capacity wall — leaving exactly one mechanism unbeaten: full softmax attention, whose $N\times N$ matrix pays the quadratic cost precisely so it never compresses the bindings into a fixed-shape state at all.

```python
# EDITABLE region of custom_strategy.py — step 3: BASED (Taylor-2 spiky linear attention)
class CustomMixer(nn.Module):
    """BASED-style short convolution + 2nd-order Taylor linear attention.

    Implementation uses BASED's `train_view="quadratic"` formulation: at
    training time we materialise the T x T attention matrix using the
    Taylor-2 kernel k(q,k) = 1 + q^T k + (q^T k)^2 / 2 (which equals
    <phi(q), phi(k)>) and apply a causal mask. This is mathematically
    identical to the recurrent / cumulative-sum view but uses memory
    O(B*T^2) instead of O(B*T*F*D), which is what the official BASED
    repo also does during training (see
    https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py
    `train_view`).
    """

    def __init__(self, d_model: int, seq_len: int, feature_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.q_proj = nn.Linear(d_model, feature_dim, bias=False)
        self.k_proj = nn.Linear(d_model, feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.conv_kernel_size = 3
        self.local_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.conv_kernel_size,
            groups=d_model,
            padding=0,
            bias=True,
        )
        # Scale q,k so q^T k stays in the radius of convergence of exp().
        self.scale = feature_dim ** -0.5

    def short_conv(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (self.conv_kernel_size - 1, 0))
        h = self.local_conv(h)
        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        local = self.short_conv(x)
        h = x + local
        q = self.q_proj(h) * self.scale
        k = self.k_proj(h) * self.scale
        v = self.v_proj(h)
        # Taylor-2 kernel score: 1 + q^T k + (q^T k)^2 / 2
        qk = torch.matmul(q, k.transpose(-1, -2))           # [B, T, T]
        kernel = 1.0 + qk + 0.5 * qk * qk                   # spiky, non-negative
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        kernel = kernel.masked_fill(~causal, 0.0)
        denom = kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = kernel / denom
        out = torch.matmul(weights, v)                      # [B, T, D]
        return self.out_proj(out) + local
```
