A spectral graph neural network classifies a node by pushing its features through the graph with a *graph filter* — a polynomial $h(L)$ of the symmetric normalized Laplacian $L = I - D^{-1/2} A D^{-1/2} = U \Lambda U^T$ whose frequency response $h(\lambda)$ decides what the network can see. Homophilic graphs want a smooth low-pass response; heterophilic graphs put the discriminative content at high frequency and need a high- or band-pass shape. A single learnable filter should reach any of these from labels alone, and the existing learnable filters already can: GPR-GNN learns monomial coefficients, BernNet learns Bernstein coefficients, ChebNetII interpolates at Chebyshev nodes. They differ only in *which polynomial basis* parameterizes the response, and since each basis is complete over the degree-$K$ polynomial space, they all have the same expressive power. Yet they reach very different accuracies. So the sharper question is not whether the filter can take any shape, but why these equally expressive bases behave so differently — and whether some basis is optimal.

The clue is in GPR-GNN's own numbers. Its flat, unconstrained monomial filter wins the ladder (Texas 0.9065, Cornell 0.8705, Cora 0.8890, Citeseer 0.8020), but with the highest seed-to-seed variance — the Texas standard deviation across seeds ranges from 0.0262 to 0.0471. That spread is the fingerprint of a hard *optimization* problem, not an expressiveness one: the monomial powers are complete but become collinear as the order grows (the Vandermonde collinearity), so the optimizer lands in different places depending on the seed. This reframes the goal. I stop asking which basis can *represent* the target filter and start asking which basis makes the coefficient fit *well-conditioned*. There is also a license to simplify the harness: a linear spectral GNN $Z = g(\hat L) X W$ can already produce any one-dimensional prediction when the Laplacian has no repeated eigenvalues and the features carry every frequency component — conditions that hold in practice on the standard benchmarks (fewer than 1% of eigenvalues are multiple and no frequency component is missing). ReLU's only role is to mix frequencies across the spectrum, which the no-missing-frequency condition makes unnecessary, so the filter can be studied purely linearly.

I propose JacobiConv: replace the monomial powers with the Jacobi orthogonal polynomial family, whose weight function can be tuned to match the graph's spectral density. The reason this is the right family follows from a direct look at the optimization landscape. For the linear filter $Z = \sum_k \gamma_k\, g_k(\hat L) X$ trained under squared loss $R = \tfrac12 \|Z - Y\|^2$, the Hessian in the coefficients is the Gram matrix of the basis under the signal-density inner product,
$$H_{k_1 k_2} = \int_0^2 g_{k_1}(\lambda)\, g_{k_2}(\lambda)\, f(\lambda)\, d\lambda,$$
where $f(\lambda)$ is the spectral density of the graph signal. Gradient descent's convergence rate is governed by the condition number $\kappa(H)$, and $\kappa(H) = 1$ — the best possible — exactly when $H = I$, i.e. when the normalized basis is *orthonormal under the inner product weighted by $f$*. This makes the optimal basis explicit and data-dependent: it is the orthogonal family whose weight function matches the signal density. The monomial basis is provably non-orthogonal under any valid weight, so it can never diagonalize this Gram matrix — that is the source of its conditioning trouble. Chebyshev *is* orthogonal, but only under the single fixed weight $(1-x^2)^{-1/2}$, which cannot be right for both homophilic graphs (low-frequency energy) and heterophilic graphs (high-frequency energy) at once. What I want is an orthogonal family whose weight can *move* with the graph.

The Jacobi polynomials $P_k^{(a,b)}$ are exactly that: orthogonal under $(1-x)^a (1+x)^b$ on $[-1,1]$, with two free exponents $a, b$ that reshape the endpoint emphasis. Chebyshev is the special case $a=b=-\tfrac12$ and Legendre is $a=b=0$. On $\hat A = I - L$, whose spectrum lies in $[-1,1]$, low graph frequency maps to $x=1$ and high graph frequency to $x=-1$; lowering $a$ relative to $b$ shifts mass toward $x=1$, lowering $b$ relative to $a$ shifts it toward $x=-1$. I never need to compute $f(\lambda)$ exactly — that would require the $O(n^3)$ eigendecomposition I am trying to avoid — I only need a tunable orthogonal family in place of the monomial powers, keeping the unconstrained signed coefficients that made GPR-GNN expressive.

The filter is evaluated through the three-term recurrence, which is the fragile part and must be carried exactly. The base cases are $P_0^{(a,b)}(x) = 1$ and $P_1^{(a,b)}(x) = \tfrac{a-b}{2} + \tfrac{a+b+2}{2}\,x$. For $k \ge 2$,
$$P_k^{(a,b)}(x) = (\theta'_k + \theta_k x)\, P_{k-1}^{(a,b)}(x) - \theta''_k\, P_{k-2}^{(a,b)}(x),$$
with
$$\theta_k = \frac{(2k+a+b)(2k+a+b-1)}{2k(k+a+b)}, \quad \theta'_k = \frac{(2k+a+b-1)(a^2-b^2)}{2k(k+a+b)(2k+a+b-2)}, \quad \theta''_k = \frac{(k+a-1)(k+b-1)(2k+a+b)}{k(k+a+b)(2k+a+b-2)}.$$
In code these become $A_k = \theta'_k$, $B_k = \theta_k$, and $C_k = \theta''_k$ over the shared denominator $2k(k+a+b)(2k+a+b-2)$; the extra factor of two cancels in $B_k$ and $C_k$, leaving the canonical recurrence. Each step requires only one sparse mat-vec: $x\, P_{k-1}^{(a,b)}$ is realized by propagating the previous polynomial signal once through the shifted operator, so the whole filter is $O(K)$ sparse mat-vecs with $K+1$ learnable scalars $\gamma_k$ combining the propagated signals, $h(\hat L) X = \sum_{k=0}^K \gamma_k P_k^{(a,b)}(\hat L) X$.

Two implementation conventions need care. The full JacobiConv is a linear GNN — it drops the ReLU encoder, uses a separate coefficient vector per output channel so multi-class predictions get individual filters, and adds polynomial coefficient decomposition $\alpha_{kl} = \beta_{kl} \prod_{i \le k} \gamma_i$ to control coefficient scales. The harness here keeps the two-layer ReLU encoder and exposes only one shared coefficient vector `temp` of length $K+1$, with $a=b=1$ fixed and all coefficients initialized uniformly to $1/(K+1)$, and no slot for the decomposition; so the artifact landed here is the shared-coefficient, fixed-$(a,b)$ Jacobi recurrence core, not the full per-channel PCD model. The operator convention is also safe: the recurrence is evaluated on $\tilde L = L - I = -\hat A$, built by adding self-loops of weight $-1$ to the normalized Laplacian. Because $a=b=1$ is symmetric, the reflection identity $P_k^{(a,b)}(-x) = (-1)^k P_k^{(a,b)}(x)$ holds, and the learned scalar `temp[k]` absorbs the alternating sign, so the representable filter family is identical whether one evaluates on $\hat A$ or on $\tilde L$. The bar this core must clear is the GPR-GNN ladder (Cora 0.8890, Citeseer 0.8020, Texas 0.9065, Cornell 0.8705), with the most important signal being lower seed-to-seed variance than that Texas std spread of 0.0262–0.0471.

```python
class CustomProp(MessagePassing):
    """Jacobi polynomial propagation layer.

    Filter: h(L_tilde) = sum_{k=0}^{K} gamma_k * P_k^{(a,b)}(L_tilde)
    where P_k^{(a,b)} are Jacobi polynomials with parameters a, b,
    L_tilde = L - I (shifted Laplacian), and gamma_k are learnable.

    Jacobi polynomials generalize Chebyshev (a=b=0), Legendre (a=b=0.5),
    and other orthogonal polynomial families.
    """

    def __init__(self, K, alpha=0.1, a=1.0, b=1.0, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.a = a
        self.b = b
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize uniformly
        self.temp.data.fill_(1.0 / (self.K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # L_tilde = L - I (shifted to [-1, 1])
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=x.size(self.node_dim)
        )

        a, b = self.a, self.b

        # Jacobi three-term recurrence
        # P_0^{(a,b)}(x) = 1
        Px_0 = x
        out = self.temp[0] * Px_0

        if self.K >= 1:
            # P_1^{(a,b)}(x) = (a+1) + (a+b+2)/2 * (x-1)
            #                 = ((a-b)/2) + ((a+b+2)/2) * x
            # Using matrix form: P_1 = c1 * L_tilde @ x + c0 * x
            c0 = (a - b) / 2.0
            c1 = (a + b + 2.0) / 2.0
            Px_1_prop = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
            Px_1 = c1 * Px_1_prop + c0 * x
            out = out + self.temp[1] * Px_1

        for k in range(2, self.K + 1):
            # Three-term recurrence coefficients for Jacobi polynomials
            k_f = float(k)
            denom1 = 2.0 * k_f * (k_f + a + b) * (2.0 * k_f + a + b - 2.0)
            A_k = ((2.0 * k_f + a + b - 1.0) * (a * a - b * b)) / denom1
            B_k = ((2.0 * k_f + a + b - 1.0) * (2.0 * k_f + a + b - 2.0) * (2.0 * k_f + a + b)) / denom1
            C_k = (2.0 * (k_f - 1.0 + a) * (k_f - 1.0 + b) * (2.0 * k_f + a + b)) / denom1

            # P_k(x) = (A_k + B_k * x) * P_{k-1}(x) - C_k * P_{k-2}(x)
            Px_1_prop = self.propagate(edge_index_tilde, x=Px_1, norm=norm_tilde, size=None)
            Px_2 = (A_k * Px_1 + B_k * Px_1_prop) - C_k * Px_0
            out = out + self.temp[k] * Px_2
            Px_0, Px_1 = Px_1, Px_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """JacobiConv: Jacobi polynomial graph filter (Wang & Zhang, 2022).

    MLP encoder + Jacobi polynomial propagation with learnable coefficients.
    """

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha, a=1.0, b=1.0)
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
