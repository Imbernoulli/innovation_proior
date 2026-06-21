The monomial filter won the ladder, but it left a crack I want to pry open. Dropping the constraint and starting flat paid off across the board: Texas $0.9065$ (from ChebNetII's $0.8770$), Cornell $0.8705$ (from $0.8470$), and the homophilic graphs did not regress — Cora $0.8890$ and Citeseer $0.8020$, both above ChebNetII. But the seed-to-seed variance is the largest of any rung: Texas std spans $0.0262$ to $0.0471$ across seeds (mean $0.0361$), Cornell $0.0275$ to $0.0434$, and one seed dropped a Cora run entirely. That spread is the fingerprint of a hard optimization landscape — the filter reaches a good response *on average*, but where it lands depends on the seed, which means the coefficient loss surface is poorly conditioned. And I chose the monomial basis knowing it is ill-conditioned, the powers $P^k$ going collinear as $k$ grows; the variance is the bill for that hand-wave arriving. So the question is not "what other constraint" — constraints cost more than they pay here — but: can I keep the unconstrained, sign-free, flat-start freedom that made the monomial filter win, while fixing the conditioning that made it noisy?

Answering that forces me to look harder at something I treated as obvious across all three rungs. Every rung differed in exactly one thing, the polynomial *basis* — Bernstein, then Chebyshev nodes, then monomial — and every one of those bases is *complete*: they all span the same degree-$K$ function space, so they have the *same expressive power*. They are linear changes of variable for one function space. Yet they reached visibly different accuracies. If they can all express the same filters, the difference cannot be *what* they represent; it has to be *how the optimizer reaches* the good filter inside each. The basis was never an expressiveness choice — it was an **optimization** choice, and I had been picking bases by intuition instead of by which one conditions the descent. Let me make that precise. Take the decoupled filter $Z = \sum_k \gamma_k\, g_k(\hat L)\, X$ and look at the optimization over $\gamma$ alone, near the optimum, under squared loss — convex in $\gamma$, so gradient descent's rate is set by the condition number $\kappa(H)$ of the Hessian. Its $(k_1,k_2)$ entry is $X^T g_{k_2}(\hat L) g_{k_1}(\hat L) X = \sum_i g_{k_1}(\lambda_i) g_{k_2}(\lambda_i) \hat X^2_{\lambda_i}$, and in the continuum

$$H_{k_1 k_2} = \int_0^2 g_{k_1}(\lambda)\, g_{k_2}(\lambda)\, f(\lambda)\, d\lambda,$$

where $f(\lambda)$ is the **spectral density of the graph signal** — how its energy is distributed over frequency. The Hessian is exactly the Gram matrix of the basis functions under the inner product weighted by $f$, and the whole optimization difficulty is $\kappa(H)$. It is smallest, equal to $1$, exactly when $H = I$ — when the basis is **orthonormal under the density-weighted inner product** $\langle g,h\rangle = \int g h f\, d\lambda$. That explains every result on the ladder at once: the monomial basis is *provably non-orthogonal under any weight function*, the worst case, which is why it optimizes well on average but with the highest variance; Bernstein is also non-orthogonal; and the Chebyshev basis ChebNetII used *is* orthogonal — but only under the *one fixed* weight $(1-x^2)^{-1/2}$, and since $f$ differs across these four graphs (Cora and Citeseer have their energy at low frequency, Texas and Cornell at high), one fixed weight cannot match all four, which is the stiffness I diagnosed two rungs ago. No fixed-weight basis can be optimal across graphs of differing density; I need a basis whose weight I can *tune to match $f$*.

That requirement names it. I propose **JacobiConv**, a filter in the **Jacobi basis** $P_k^{a,b}$, the classical orthogonal family under the *tunable* weight $(1-x)^a(1+x)^b$ on $[-1,1]$, with two free parameters $a,b$ that reshape the weight continuously. Chebyshev is the single special case $a=b=-\tfrac12$ and Legendre is $a=b=0$, so Jacobi *contains* the basis ChebNetII used and adds two knobs to slide the weight toward whichever end of the spectrum the graph's energy occupies. On the normalized adjacency $\hat A = I - L$, whose spectrum is $[-1,1]$, low graph frequency $\lambda\approx0$ maps to $x\approx1$ and high $\lambda\approx2$ to $x\approx-1$, so for a homophilic graph I push the weight toward $x=1$ (smaller exponent on $(1-x)$) and for a heterophilic graph toward $x=-1$ (smaller exponent on $(1+x)$). Tuning $a,b$ *is* matching the weight to $f$ — the $\kappa(H)\to1$ knob made an explicit hyperparameter. This keeps everything the monomial filter got right — unconstrained, sign-free, learned coefficients and the flat uniform $1/(K+1)$ start — while replacing the non-orthogonal monomial basis whose bad conditioning produced the seed variance with the *adaptable* orthogonal basis that drives the Hessian toward identity. I do not have to *compute* the optimal $a,b$ (that needs the eigendecomposition I cannot afford); I pick the flexible family and a sensible default and the weight sits close enough to $f$ to condition the descent. The filter applied is

$$h(\hat A) = \sum_{k=0}^{K} \texttt{temp}_k\, P_k^{a,b}(\hat A).$$

I have to be honest about what this harness's edit surface lets me build, because the full principled method has refinements the scaffold cannot express, and I land the version it supports. The full method is a *linear* GNN — it drops the ReLU, justified because ReLU's only spectral effect is to *mix* frequency components, the repair for repeated Laplacian eigenvalues or missing frequency components, and on these irregular benchmark graphs fewer than 1% of eigenvalues are multiple and no component is missing, so the activation fixes a disease the data does not have. It also gives each output channel its own coefficient vector $\alpha_{kl}$ (multi-class universality, since the one-dimensional universality argument means a single shared filter cannot produce an arbitrary multi-class prediction), and it rescales the per-$k$ magnitudes by a polynomial-coefficient decomposition $\alpha_{kl} = \beta_{kl}\prod_i \gamma_i$ with shared bounded ratios, which is what realizes the orthogonal basis's good conditioning in practice. But this task's `CustomProp` owns a *single shared* `temp` vector of length $K+1$ applied identically to every channel — no per-channel filter slot and no PCD — and the harness keeps the ReLU between its two linear layers. So what I land is the **shared-coefficient, fixed-$(a,b)$ Jacobi core**: the same unconstrained learned coefficients, the same uniform start, the same decoupled encoder as the monomial filter, but the monomial powers $P^k$ replaced by $P_k^{a,b}(\hat A)$, with $a=b=1$ as the symmetric default — symmetric because it makes the recurrence's $(a^2-b^2)$ term vanish, a basis neutral between the two spectral ends.

The propagation stays linear in $K$. I build $L$ with the symmetric-normalized Laplacian, shift to $\tilde L = L - I = -\hat A$ by adding self-loops of weight $-1$, and run the Jacobi three-term recurrence: $P_0 = x$, then $P_1 = \tfrac{a+b+2}{2}(\tilde L x) + \tfrac{a-b}{2}\,x$, and for $k\geq2$,

$$P_k = (A_k + B_k\tilde L)\,P_{k-1} - C_k\,P_{k-2},$$

with $A_k, B_k, C_k$ the standard Jacobi recurrence coefficients computed from $a,b,k$. Each step is one `propagate`, one sparse mat-vec, so the whole filter is $K$ propagations — $O(Kmd)$, the same order as the monomial and Chebyshev rungs, strictly cheaper than BernNet, with no DCT. One subtlety I get right deliberately: running the recurrence on $\tilde L = -\hat A$ rather than $\hat A$ evaluates $P_k(-x) = (-1)^k P_k^{a,b}(x)$ — the Jacobi reflection identity, symmetric when $a=b$ — an alternating sign that the *learned* `temp` absorb completely, so the filter family is identical to the $\hat A$ convention and I lose nothing by using the operator the harness builds for free. Against the monomial filter's numbers, the motivation was its *conditioning*, not its means, so I expect a quieter result than a dramatic mean jump: the Jacobi basis should match or modestly exceed each accuracy — Cora at or above $0.8890$, Citeseer at or above $0.8020$, Texas at or above $0.9065$, Cornell at or above $0.8705$ — while *shrinking the seed-to-seed variance* that betrayed the monomial basis's ill-conditioned Hessian. The honest risk: with $a=b$ fixed (no per-dataset tuning of the weight toward each density) and no PCD, the advantage on this harness may be small, since orthogonality is only *exactly* matched to $f$ when $a,b$ track each graph's density. But even the untuned symmetric Jacobi basis is orthogonal under *a* weight where the monomial basis is orthogonal under *none*, so I expect the conditioning to improve and the variance to fall — and that, more than a fractional accuracy gain, is the result that justifies the orthogonal basis as the ceiling for this ladder.

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
