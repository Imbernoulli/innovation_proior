The starting point on this harness is APPNP made parametric: a monomial polynomial $\sum_k \texttt{temp}_k\, P^k x$ of the GCN-normalized adjacency $P = D^{-1/2} A D^{-1/2}$, with the coefficients *frozen* at the Personalized-PageRank pattern $\texttt{temp}_k = \alpha(1-\alpha)^k$. That is a fixed low-pass response by construction — geometrically decaying weight on each successive hop — and it is exactly the operator that cannot serve heterophilic graphs, where a hub whose neighbors carry *opposite* labels needs a filter that *contrasts* a node with its neighborhood (a high-frequency response) that a monotone-decaying PPR filter has no way to produce. So the minimal fix is to make those coefficients learnable. But unfreezing is not the whole design: with the symmetric normalized Laplacian $L = U \Lambda U^T$ a spectral filter applies a scalar $h$ to each frequency, $y = U\, h(\Lambda)\, U^T x$, and since the eigendecomposition is $O(n^3)$ I am forced to write $h$ as a degree-$K$ polynomial of $L$ and evaluate $\sum_k w_k L^k x$ with sparse mat-vecs. The shape of $h$ over the spectrum $[0,2]$ decides everything — an impulse at $\lambda=0$ is pure low-pass, an impulse at $\lambda=2$ is pure high-pass, anything between is band-pass — so the homophilic and heterophilic regimes do not want more or less propagation, they want *opposite* frequency responses. The real question is which parameterization makes learning into any of those shapes both expressive and well-behaved, and there is one thing I want beyond expressivity: the ability to *constrain* the response — in particular to keep it **non-negative** everywhere, since most of the filters one actually wants here (low-pass bumps, band-pass bumps) are non-negative functions of frequency, and a parameterization where "non-negative everywhere" is a *simple constraint on the parameters* would bias the optimizer toward valid, interpretable filters and away from the wild oscillatory responses that overfit. With monomial coefficients there is no transparent relation between the sign or magnitude of $w_k$ and the value $h(\lambda)$ at any frequency, so "keep $h \geq 0$ on $[0,2]$" is a hard global condition, not a box constraint a ReLU can impose.

I propose **BernNet**, a filter parameterized in the **Bernstein basis** on the spectrum. The degree-$K$ Bernstein basis polynomials on $[0,2]$ (rescaled from the textbook $[0,1]$) are

$$b_{k,K}(\lambda) = \frac{1}{2^K}\binom{K}{k}(2-\lambda)^{K-k}\lambda^k,\qquad k=0,\dots,K,$$

and they have exactly the three properties I want at once. They are **non-negative** on $[0,2]$, because each is a product of the non-negative factors $(2-\lambda)^{K-k}$ and $\lambda^k$ times a positive constant; they form a **partition of unity**, $\sum_k b_{k,K}(\lambda) = \frac{1}{2^K}\big((2-\lambda)+\lambda\big)^K = 1$ by the binomial theorem; and each $b_{k,K}$ peaks near frequency $\lambda \approx 2k/K$, so the coefficient on it controls a localized **bump** there. Writing the filter as $h(\lambda) = \sum_k \theta_k\, b_{k,K}(\lambda)$ with **all** $\theta_k \geq 0$ then makes $h$ a non-negative combination of non-negative functions, so $h(\lambda) \geq 0$ for *every* $\lambda$ in $[0,2]$ — a non-negativity certificate over the whole spectrum, not just at sample points, obtained purely from a box constraint on the coefficients. This is precisely what the monomial basis could not give me: "constrain the filter" becomes "ReLU the coefficients." And the bump structure makes the learned $\theta$ readable — large $\theta_0$ is low-pass, large $\theta_K$ is high-pass, a bump in the middle is band-pass — so the filter is expressive enough to reach any of the responses the four graphs need. The operator I apply is therefore

$$h(L) = \sum_{k=0}^{K} \mathrm{ReLU}(\theta_k)\,\frac{1}{2^K}\binom{K}{k}\,(2I-L)^{K-k}\,L^k,$$

with $L$ the symmetric normalized Laplacian.

Turning that into the sparse-mat-vec recurrence the harness supports needs both $L^j$ and $(2I-L)^j$ applied to a signal. Building $2I-L$ is one self-loop shift of $L$ — `add_self_loops` with the negated Laplacian weights and `fill_value=2.0`. I first compute the chain $\texttt{tmp}[i] = (2I-L)^i x$ for $i=0,\dots,K$ by propagating $K$ times through the $2I-L$ operator. Then each Bernstein term $k$ needs $L^k\,(2I-L)^{K-k} x$, which is $L$ applied $k$ times to the cached $\texttt{tmp}[K-k]$: the $k=0$ term is just $\frac{1}{2^K}\binom{K}{0}\theta_0\cdot\texttt{tmp}[K]$, and each subsequent term takes $\texttt{tmp}[K-i-1]$ and walks it through $L$ exactly $i+1$ times. The price is real and I want it on the record as this rung's one structural weakness: caching the $(2I-L)$ chain is $K$ propagations, but the second loop re-walks $L$ from scratch per term, so the total is $1+2+\dots+K \approx K^2/2$ extra sparse mat-vecs and the forward pass is $O(K^2 m d)$ — **quadratic in $K$** — where a monomial filter is linear. There is no linear-time rewrite of the Bernstein form on offer; I am buying the certificate and the controllability with a quadratic propagation cost. Approximation theory adds a second caveat: Bernstein approximation of a function with modulus of continuity $\omega$ has error scaling like $\omega(K^{-1/2})$, slower per degree than near-minimax options, so for a sharp target response Bernstein needs a larger $K$ for the same fidelity — and with $K=10$ fixed I cannot raise it to compensate.

The initialization falls out of the partition-of-unity property: I set every $\theta_k = 1$, which gives $h(\lambda) = \sum_k 1\cdot b_{k,K}(\lambda) = 1$ — the constant, **all-pass / identity** response that passes every frequency unchanged and bakes in no low- or high-pass bias. That is the neutral default I want; I start flat and let the labels pull the filter low- or high-pass, rather than starting from the homophily assumption that cripples the PPR default on heterophilic graphs. The ReLU on $\theta$ before forming the response keeps the non-negativity constraint live throughout training. There is no teleport probability here, so the harness's `ALPHA` is simply ignored, and I leave propagation dropout at the scaffold default (`dprate=0.0`) since propagation dropout tends to hurt spectral filters on heterophilic data. The decoupled APPNP-style encoder — dropout, `lin1`, ReLU, dropout, `lin2`, then propagation, then log-softmax — keeps the filter a single scalar function with $K+1$ parameters applied identically to every channel, so the parameter count stays tied to the feature dimension, independent of $K$. What I expect from this design is an asymmetry: the controllable non-negative basis should be a strong heterophilic performer — the constraint that keeps it inside the family of valid filters is exactly the regularizer the tiny 183-node WebKB graphs want — while on the homophilic citation graphs the slow $K^{-1/2}$ convergence at fixed $K$, paired with the quadratic cost that discourages raising $K$, should leave it the weakest of the learnable filters.

```python
class CustomProp(MessagePassing):
    """Bernstein polynomial propagation layer.

    Filter: h(L) = sum_{k=0}^{K} theta_k * C(K,k)/2^K * L^k * (2I-L)^{K-k}
    where theta_k = ReLU(learnable), C(K,k) is binomial coefficient,
    and L is the symmetric normalized Laplacian.
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # 2I - L
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0,
            num_nodes=x.size(self.node_dim)
        )

        # Compute (2I-L)^k * x for k = 0, ..., K
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # Bernstein basis evaluation
        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            # Apply L^{i+1}
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """BernNet: Bernstein polynomial graph filter (He et al., 2021).

    MLP encoder + Bernstein polynomial propagation.
    """

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
```
