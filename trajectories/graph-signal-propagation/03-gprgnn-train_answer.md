ChebNetII landed where its design predicted, and the *way* it landed reframes the ladder. On Cora it climbed to $0.8725$ and on Citeseer to $0.8004$, both clearly above BernNet's $0.8554$ and $0.7795$ — the linear-cost near-minimax interpolation bought back the smooth-low-pass resolution Bernstein under-resolved — and the seed-to-seed variance collapsed to almost nothing (Cora $0.8724/0.8726/0.8724$, Citeseer within $0.0006$). But the heterophilic side regressed: Texas *fell* to $0.8770$, below BernNet's $0.9093$, and Cornell sat flat at $0.8470$. That near-perfect seed-invariance is not only a feature, it is a *symptom* — a filter whose three seeds land within $0.0006$ of each other is barely moving off its initialization. The ReLU-on-values plus DCT plus near-minimax node placement together pin the filter so tightly it cannot reach the sharper, possibly sign-changing response Texas needs, which is exactly why Texas regressed. Looking across both failures, the pattern is specific: BernNet constrained the response to be non-negative *everywhere*, ChebNetII to be a near-minimax interpolant of non-negative *sampled values*, and both lose not on the large citation graphs where overfitting capacity would be dangerous but on the 183-node WebKB graphs, where the heterophilic response is sharp and sign-changing — and a true high-pass response can want to *alternate sign* across frequency, exactly what a non-negative-everywhere or non-negative-at-nodes constraint cannot express. The *constraint*, the thing both prior rungs were proud of, is the thing capping heterophilic performance. So the move is to drop it.

That sounds like the free-coefficient ChebNet disease all over again unless I change two things at once. I propose **GPRGNN**, a learnable **monomial / Generalized-PageRank filter**: learn the hop-weights of a monomial polynomial in the GCN-normalized adjacency,

$$h(P) = \sum_{k=0}^{K} \gamma_k\, P^k,\qquad P = D^{-1/2} A D^{-1/2},$$

with *unconstrained, sign-free* coefficients. The first change is the *basis*. ChebNet failed because in the Chebyshev basis the high-$k$ terms are high-frequency cosines, so unconstrained learning piles capacity exactly where it overfits. In the monomial basis $P^k$ is the $k$-hop propagation operator and $\gamma_k$ is the weight on information from $k$ hops away — a directly interpretable quantity, not an abstract amplitude. A node's label, homophilic or heterophilic, depends on its neighborhood at a few hops; learning *how much* to trust each hop, with the sign free, is exactly the degree of freedom Texas needs — it can learn to *subtract* the 1-hop average to build a contrast — and exactly what the non-negative bases forbade. This is the basis the scaffold default already uses, frozen at PPR; the move is to unfreeze it and let every hop-weight be learned, sign and all. The monomial basis is ill-conditioned — the powers $P^k$ go collinear as $k$ grows — and that is a real objection, but conditioning is an approximation-theory concern about fitting an arbitrary target to high precision; here $K=10$ is modest, the targets are not pathological, and the empirical record on these exact graphs is blunt: a learned monomial filter outperforms a free Chebyshev filter on the citation graphs despite the conditioning. I accept the conditioning cost to buy unconstrained, sign-free, hop-interpretable coefficients.

The second change is what keeps the unconstrained filter from overfitting the way ChebNet did: the regularization moves from a hard architectural constraint to *soft* ones — weight decay, a learning-rate split, and the right initialization — and these are the deliberate departures from the textbook monomial filter. The textbook initializes the hop-weights to the PPR pattern $\gamma_k = \alpha(1-\alpha)^k$, a decaying low-pass that *bakes in the homophily prior*: it starts the filter as a low-pass and asks the optimizer to climb out of it toward high-pass on heterophilic graphs, and on a 183-node graph with little data that starting bias is a headwind the optimizer may not travel far enough to overcome. So I initialize **uniformly**, every $\gamma_k = 1/(K+1)$ — an equal-weight average over all hops $0$ through $K$, dataset-agnostic, committing to neither low- nor high-pass (the same neutrality logic that made all-ones right for the previous two rungs, here the uniform hop-average), so the optimizer can move freely toward whichever response the labels demand without first undoing a baked-in prior. This is the single most important departure of the rung: *uniform, not PPR, initialization*, precisely so the heterophilic graphs are reachable. The training split is the second: I give the filter coefficients the **same** fast learning rate as the MLP — $0.05$ for both, where the scaffold default would throttle the propagation parameters to $0.01$. ChebNetII's stiffness was partly that its parameterization plus a slow propagation LR meant the filter barely moved; here I *want* the unconstrained hop-weights to move fast enough to find the heterophilic response. To control the overfitting that fast, unconstrained, ill-conditioned coefficients invite, I spend the regularization budget where it belongs: a small $5\times10^{-4}$ weight decay on the MLP encoder (the high-capacity part, where overfitting on a tiny graph is the real danger), and **zero** weight decay on the $K+1$ filter coefficients — decaying the filter toward zero would bias it back toward an over-smooth response, the wrong prior. Propagation dropout stays off (`dprate=0.0`), consistent with the task's finding that it hurts spectral filters on heterophilic data.

The propagation itself is the cheapest of all three rungs, which I state plainly since BernNet's quadratic cost was a named weakness: $\texttt{hidden} = \gamma_0 x$; for each $k$, $x \leftarrow Px$, $\texttt{hidden} \mathrel{+}= \gamma_{k+1} x$ — $K$ sparse mat-vecs of the GCN-normalized adjacency accumulating the learned hop-weights, $O(Kmd)$, linear in $K$, the same order as ChebNetII and strictly cheaper than BernNet, with no DCT and no Laplacian shift, just the normalized adjacency powers. So this rung is simultaneously the *least constrained* and the *cheapest*: the bet is that on these graphs the right design is not a clever constrained basis but the plainest learnable hop-mixer, freed of every prior, regularized softly, and started flat. Against ChebNetII's numbers I expect Texas to climb back off $0.8770$ toward $0.90$ and Cornell off $0.8470$ toward $0.87$ — the unconstrained sign-free coefficients with a uniform start building the contrast response both non-negative bases forbade — while Cora and Citeseer do *not* regress and may edge slightly higher, since a decaying $\gamma_k$ is well within reach of a uniform-init monomial filter and the encoder weight decay controls overfitting without stiffening the filter. The one cost I expect to pay is *more* seed-to-seed variance than ChebNetII's near-zero, the natural price of an unconstrained, fast-learning filter that will not land in the identical place every run.

```python
class CustomProp(MessagePassing):
    """GPR propagation: learnable polynomial in the monomial basis.

    Filter: h(A) = sum_{k=0}^{K} gamma_k * A^k
    where A is the GCN-normalized adjacency and gamma_k are learnable.

    Initialized with uniform coefficients (1/(K+1)) so the filter starts
    as an equal-weight average of all hops. This is dataset-agnostic and
    lets the optimizer freely learn both low-pass (homophilic) and
    high-pass (heterophilic) filters.
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Uniform initialization for dataset-agnostic starting point.
        nn.init.constant_(self.temp, 1.0 / (self.K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """GPRGNN: Generalized PageRank GNN (Chien et al., 2021).

    MLP encoder + learnable monomial polynomial filter.
    """

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha)
        self.dropout = dropout
        self.dprate = 0.0  # GPRGNN paper: no propagation dropout
        # Override training hyperparams (read by template's training loop)
        self.custom_lr = 0.05
        self.custom_wd = 0.0005
        self.custom_prop_lr = 0.05  # same lr for filter coefficients
        self.custom_prop_wd = 0.0

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
