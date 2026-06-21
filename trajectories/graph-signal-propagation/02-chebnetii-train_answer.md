The Bernstein filter landed with the asymmetry its design predicted. On the heterophilic WebKB graphs the non-negative basis did its job — Texas reached $0.9093$, the highest of anything on this ladder, and Cornell $0.8443$ — but on the homophilic citation graphs it was soft: Cora $0.8554$ and Citeseer $0.7795$, the lowest homophilic numbers a learnable filter produces here, with the Citeseer gap the loud one, a clear step below where a smooth low-pass should land on a clean citation graph. The diagnosis is sharp and it is *not* a constraint problem and *not* an expressivity problem: Bernstein is expressive enough and its constraint is helping. It is **approximation efficiency** — at the fixed $K=10$ the Bernstein basis resolves a smooth target too slowly ($\omega(K^{-1/2})$ error), and its $O(K^2 m d)$ cost means I cannot just raise $K$ to buy resolution. I want the same near-best approximation quality with *linear* cost and *faster* per-degree convergence — a Chebyshev construction — while keeping BernNet's lesson that response constraints should be parameter constraints.

There is a puzzle on that path I have to resolve first, or I will make Citeseer worse. The truncated Chebyshev expansion is *near-minimax*: among degree-$K$ polynomials its worst-case uniform error is within a small factor of the best possible, which is the whole reason numerical analysts abandoned the monomial powers a century ago. So a naive expectation is that dropping the Chebyshev basis in and learning $w_k$ freely should beat everything. Yet the empirical record on these citation graphs is the opposite — a free-coefficient Chebyshev filter comes in *last*, below both the monomial GPR filter and Bernstein. The theory is about the *best* polynomial in the basis; gradient descent picking coefficients freely is a different thing. Any continuous response on $[-1,1]$ has an expansion $h(x) = \sum_k w_k T_k(x)$ with $T_k(x) = \cos(k\arccos x)$ a cosine of increasing frequency, so the large-$k$ coefficients control the high-frequency wiggle. If $h$ is analytic — the smooth kind of response a sensible filter should be — its Chebyshev coefficients must *decay*, asymptotically like $1/k^q$; the high-frequency coefficients are forced to die off. Free $w_k$ fit by gradient descent chasing training accuracy have nothing enforcing that decay, so the optimizer piles weight onto the large-$k$ coefficients and fits a jagged response that memorizes labels — exactly the disease that made the original ChebNet lose to its own first-order GCN special case and worsen as $K$ rose. The basis was never the problem; the *missing coefficient constraint* is.

So I propose **ChebNetII**, which forces the decay by *tying the coefficients to a well-behaved response* — and does it without BernNet's quadratic cost. A crude fix, learning $w_k$ but propagating $w_k/k$, mechanically pushes the spectrum toward the $1/k^q$ shape and does beat the other bases, but it is a hack and it cannot express a non-negativity constraint, since with abstract coefficients there is no transparent relation between $w_k$ and the value $h(\hat\lambda)$. The principled move is the one BernNet taught me: if my trainable numbers *were* the filter's values, "non-negative at the control points" would just be "these numbers are non-negative." So instead of parameterizing the *coefficients*, parameterize the polynomial by the **values** it takes at $K+1$ fixed nodes and *derive* the coefficients — switching from expansion to **interpolation**. The free parameters become $\gamma_j = h(x_j)$, the response values, and when those values come from a smooth response the recovered coefficient vector is that smooth response's interpolation vector, so the decay comes from the response itself rather than from a heuristic penalty.

Interpolation has its own famous trap, the **Runge phenomenon** — equispaced nodes make the high-degree interpolant oscillate harder near the interval ends — and locating it tells me exactly what to do. The error at any point is

$$R_K(\hat\lambda) = \frac{h^{(K+1)}(\zeta)}{(K+1)!}\,\pi_{K+1}(\hat\lambda),\qquad \pi_{K+1}(\hat\lambda) = \prod_{k}(\hat\lambda - x_k),$$

where the *nodal polynomial* $\pi_{K+1}$ — the monic degree-$K+1$ polynomial whose roots are my chosen nodes — is *entirely mine*. The whole Runge problem reduces to: where do I place $K+1$ nodes so $\max|\pi_{K+1}|$ over $[-1,1]$ is smallest? Among all monic degree-$K+1$ polynomials, the minimizer is the scaled Chebyshev polynomial $2^{-K}T_{K+1}$, with uniform norm exactly $2^{-K}$. So choosing the nodes to be the **roots of $T_{K+1}$**,

$$x_j = \cos\!\Big(\frac{(j+\tfrac12)\pi}{K+1}\Big),\qquad j=0,\dots,K,$$

makes the nodal polynomial that minimizer — $\|\pi_{K+1}\| = 2^{-K}$, shrinking geometrically in $K$. These **Chebyshev nodes** cluster near the endpoints, denser exactly where equispaced nodes let the error run wild, which is the geometric reason they tame it; and the Lebesgue constant of Chebyshev-node interpolation grows like $\log K$ (versus $\sim 2^K$ for equispaced), so the interpolant stays within a $\sim\log K$ factor of the minimax polynomial. The basis was never the problem — interpolation at the *right nodes* is near-best, and free-coefficient fitting was wrong.

The coefficients then come in closed form, and that structure is the payoff. Writing $x_j = \cos\theta_j$ with $\theta_j = (j+\tfrac12)\pi/(K+1)$ gives $T_k(x_j) = \cos(k\theta_j)$, and the Chebyshev polynomials are *discretely orthogonal* at their own nodes — $\sum_j T_m(x_j)T_l(x_j)$ is $0$ for $m\neq l$, $(K+1)/2$ for $m=l\neq0$, and $K+1$ for $m=l=0$, the half-integer offset in $\theta_j$ making the cross sums of unit roots cancel and the $m=l=0$ case twice the others. Plugging the interpolation conditions $P_K(x_j)=\gamma_j$ into $P_K = \tfrac{c_0}{2}T_0 + \sum_{k\geq1} c_k T_k$ and inverting via that orthogonality gives

$$c_k = \frac{2}{K+1}\sum_{j=0}^{K}\gamma_j\,T_k(x_j),$$

a discrete cosine transform of the sampled values, with the constant term applied as $c_0/2$ to compensate $T_0$'s doubled discrete norm. Two implementation choices close it, and both are the opposite trade-off from BernNet. Chebyshev lives on $[-1,1]$ while the normalized-Laplacian spectrum lives on $[0,2]$; ChebNet needed $\lambda_{\max}$ to set $\hat L = 2L/\lambda_{\max}-I$, an extra eigen-computation, but I know $\lambda_{\max}\leq 2$ a priori, so $\lambda_{\max}=2$ gives the free shift $\hat L = L - I$ — build $L$ once, add self-loops with weight $-1$, no spectrum estimate. And I propagate with the standard three-term recurrence $T_k(\hat L)x = 2\hat L\,T_{k-1}(\hat L)x - T_{k-2}(\hat L)x$, carrying two running vectors through one sparse mat-vec each and accumulating $\sum c_k T_k(\hat L)x$ from $(c_0/2)T_0 + c_1 T_1$. Forming the coefficients is $O(K^2)$ scalar work on the fixed $T_k(x_j)$ table; the propagation is $K$ mat-vecs. The forward pass is $O(K^2 + Kmd)$ — **linear in $K$**, where BernNet was quadratic — and the per-degree convergence is faster, $\omega(K^{-1})\log K$ against Bernstein's $\omega(K^{-1/2})$, so at the *same* fixed $K=10$ it resolves a smooth target more sharply, which is precisely the deficit I diagnosed on Citeseer.

The initialization mirrors BernNet's logic in the new parameterization: the parameters are response values $\gamma_j = h(x_j)$, so every $\gamma_j = 1$ interpolates to the constant $h\equiv 1$, the all-pass start with no a-priori bias, the same flat start that let the Bernstein filter find heterophilic responses. The ReLU on $\gamma$ before the DCT keeps non-negativity at the interpolation nodes — the BernNet lesson preserved, though now a *sampled-value* constraint (non-negative at the $K+1$ nodes), weaker than BernNet's global certificate between nodes. Propagation dropout and the training overrides stay at scaffold defaults. What I expect against the Bernstein numbers is an inversion of its profile: ChebNetII should *win the homophilic graphs* — Citeseer climbing clearly off $0.7795$ toward $0.80$ and Cora off $0.8554$ toward $0.87$, with the per-seed variance *shrinking* because the Chebyshev-node parameterization ties the coefficients to a stable interpolation problem — while possibly *losing Texas*, since BernNet's $0.9093$ is a high bar its global certificate earned on a tiny graph, and my weaker sampled-value constraint plus the higher-resolution basis could overfit there.

```python
class CustomProp(MessagePassing):
    """ChebNetII propagation: Chebyshev interpolation filter.

    Learns filter values at Chebyshev interpolation nodes, then converts
    to Chebyshev polynomial coefficients. Uses ReLU to ensure non-negative
    interpolation values.

    Filter: h(L_tilde) = sum_{k=0}^{K} c_k * T_k(L_tilde)
    where L_tilde = L - I (shifted Laplacian), T_k is the k-th Chebyshev
    polynomial, and c_k are computed from interpolation values via DCT-like transform.
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        # Convert interpolation values to Chebyshev coefficients
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # L_tilde = L - I (shifted to [-1, 1] range)
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=x.size(self.node_dim)
        )

        # Chebyshev recurrence: T_0(x)=x, T_1(x)=x, T_{k+1}=2xT_k - T_{k-1}
        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """ChebNetII: Chebyshev interpolation graph filter (He et al., 2022).

    MLP encoder + ChebNetII propagation with Chebyshev interpolation.
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
