**Problem.** GPR-GNN's unconstrained monomial filter won the ladder but with the *highest* seed-to-seed
variance (Texas std spanning 0.0262–0.0471; a Cora run dropped entirely) — the fingerprint of an
ill-conditioned optimization, because the monomial powers `P^k` go collinear and the monomial basis is
*provably non-orthogonal under any weight*. Keep GPR-GNN's unconstrained, sign-free, flat-start freedom,
but fix the conditioning by changing the *basis*.

**Key idea (Jacobi-basis filter).** Among complete polynomial bases — equal expressive power — the one
that optimizes fastest is the one whose Gram matrix under the graph signal's spectral density `f(λ)` is
closest to identity (the convex coefficient-fit Hessian is `H_{k₁k₂}=∫ g_{k₁}g_{k₂} f dλ`; `κ(H)=1` iff
the basis is orthonormal under `f`). The monomial basis is non-orthogonal (worst case); Chebyshev is
orthogonal under one *fixed* weight, mismatched to graphs of differing density. The **Jacobi** family
`P_k^{a,b}` is orthogonal under the *tunable* weight `(1-x)^a(1+x)^b` (Chebyshev = `a=b=-1/2`), so it
can be slid toward each graph's `f` — driving the Hessian toward identity. Apply `h(Â)=sum_k temp_k
P_k^{a,b}(Â)`, `Â = I − L`, via the Jacobi three-term recurrence.

**Why.** Same unconstrained, sign-free coefficients and uniform `1/(K+1)` flat start as GPR-GNN — only
the basis changes, from non-orthogonal monomial to orthogonal Jacobi — so the conditioning improves
(variance should fall) without re-imposing any of the constraints that capped BernNet/ChebNetII.
Linear cost `O(K m d)`: `K` sparse mat-vecs through the recurrence, no DCT, no quadratic blow-up.

**Same-named-vs-paper / what this harness lands.** The full method is a *linear* GNN (drops ReLU; the
universality conditions — no repeated eigenvalues, no missing frequency components — hold on real
graphs) with *per-channel* coefficients `α_{kl}` (multi-class universality) and *PCD*
`α_{kl}=β_{kl}Π_iγ_i` (per-`k` rescaling that realizes the conditioning). This task's `CustomProp` owns
a *single shared* `temp` vector applied to all channels and keeps the ReLU encoder, so it lands the
**shared-coefficient, fixed-`(a,b)` Jacobi core** — no per-channel filters, no PCD, `a=b=1` fixed (which
zeros the recurrence's `θ'_k`, a basis neutral between the spectral ends). Running the recurrence on
`L̃ = L − I = −Â` flips `P_k` by `(−1)^k` (Jacobi reflection, symmetric for `a=b`), absorbed by the
learned `temp`.

**Hyperparameters.** `K=10`, `a=b=1.0`, `temp` init **all `1/(K+1)`** (uniform, basis-neutral),
`hidden=64`, `dropout=0.5`, `dprate=0.5` (the scaffold default for this fill; the model passes `dprate`
through). Training overrides left at the scaffold defaults.

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
