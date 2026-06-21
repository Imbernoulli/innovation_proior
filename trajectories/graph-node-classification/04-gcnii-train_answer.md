The renormalized convolution won the way I bet it would: PubMed at 0.7863 past every prior baseline, CiteSeer up to 0.7177 with its seed spread finally collapsed under a point, Cora at 0.8207 essentially tied with attention. It is the strongest and most stable of the three — but notice what *all three* share, the winner included: every one is a two-layer model, capped at a two-hop receptive field. That is not coincidence, it is a wall. A label that needs evidence from a four- or five-hop-distant node simply cannot reach it, and on CiteSeer — the sparsest graph, where many nodes have few same-class neighbors within two hops — that horizon is plausibly *why* it stays the worst dataset even after I stabilized it. The obvious move is to stack more layers, and the obvious move fails: accuracy drops as I deepen. So before bolting on a vision trick I derive *why* depth poisons graph convolution. Strip the weights and nonlinearities and $K$ layers is $\tilde{\mathbf P}^{K}\mathbf x$. The renormalization trick that made depth stable pinned $\tilde{\mathbf P}$'s top eigenvalue to exactly 1 with all others strictly smaller, and that is precisely what makes its powers collapse — every eigencomponent but the top decays geometrically. Conjugating the lazy operator into a random walk, $\tilde{\mathbf P}^{K}\mathbf x\to\boldsymbol\pi=\frac{\langle\tilde{\mathbf D}^{1/2}\mathbf 1,\mathbf x\rangle}{2m+n}\tilde{\mathbf D}^{1/2}\mathbf 1$: every node's value becomes $\sqrt{\tilde d_j}$ times one global scalar, so two nodes of equal degree become identical and the per-node signal needed to classify is gone. That is over-smoothing, derived rather than asserted. And the rate is uneven — the relative deviation shrinks like $1/\tilde d_j$, so high-degree nodes over-smooth first.

The reflex cure is a residual connection, but the literal ResNet move fails for a derivable reason. Adding the previous layer back makes the propagation $(\mathbf I+\tilde{\mathbf P})\mathbf H^{(\ell)}$ — exactly a lazy random walk, which has the *same* stationary distribution $\boldsymbol\pi$. Laziness only slows mixing; it never prevents the collapse, which is why deep residual GCNs go a few layers further and then collapse while a 2-layer GCN still beats them. The skip is to the wrong target: the previous layer is itself already partly smoothed, so adding it back feeds smoothed signal into more smoothing. That reframes the question — not *whether* to skip but *what to skip back to* — and the one representation guaranteed un-smoothed is layer zero, the freshly transformed input before any propagation touched it.

I propose GCNII: deep graph convolution made to work by two changes, initial residual and identity mapping. The update is
$$\mathbf H^{(\ell+1)}=\sigma\!\Big(\big((1-\alpha)\tilde{\mathbf P}\mathbf H^{(\ell)}+\alpha\mathbf H^{(0)}\big)\big((1-\beta_\ell)\mathbf I+\beta_\ell\mathbf W^{(\ell)}\big)\Big),$$
with $\mathbf H^{(0)}$ a learned map of the features. The first change is the **initial residual**: at every layer mix the smoothed representation with a fixed fraction $\alpha$ of the *original* input rather than the previous layer. Unrolling, $\mathbf H^{(K)}$ becomes a geometric mixture $\sum_\ell\alpha(1-\alpha)^\ell\tilde{\mathbf P}^\ell\mathbf H^{(0)}$ that converges as $K\to\infty$ to $\alpha(\mathbf I-(1-\alpha)\tilde{\mathbf P})^{-1}\mathbf H^{(0)}$ — the personalized-PageRank diffusion of the input, a fixed point that *carries the features* instead of erasing them. A surfer that with probability $\alpha$ teleports back to where it started keeps locality forever; with $\alpha$ small (0.1) propagation dominates so the receptive field genuinely widens — fetching the far-hop CiteSeer evidence the two-hop wall could not — while the anchor never vanishes. This fixes the over-smoothing of the *propagation*.

But initial-residual-only is linear in $\mathbf H^{(\ell)}$ — deep in propagation, shallow in representation, with no learned nonlinear transform along the way, which is the very expressive power that made depth worth wanting. Put the weight and ReLU back and a *second*, different depth pathology appears: a $K$-layer stack drives features into a low-dimensional subspace at rate $s^{K}$ where $s$ is the largest singular value of the per-layer weights, and at ~20 labels per class against $K$ free $H\times H$ matrices the over-parameterization overfits, the damage being too much cross-dimension feature mixing. The second change, **identity mapping**, takes the ResNet lesson at the level of the matrix: write the effective transform as $(1-\beta_\ell)\mathbf I+\beta_\ell\mathbf W^{(\ell)}$. At $\beta_\ell\to0$ it is the identity and the layer falls back to the pure PageRank diffusion that does not over-smooth; at moderate $\beta_\ell$ it adds a $\beta$-sized learned transform. This is safe to regularize hard for a real reason: for a linear residual network the optimal weights have small norm and the only critical point is the global minimum, so strong $L_2$ on $\mathbf W$ is not fighting the model — the optimum already lives at small norm — and it removes the spurious minima scarce-label training would fall into. Small $\|\mathbf W\|$ keeps $(1-\beta)\mathbf I+\beta\mathbf W$ near identity, $s\approx1$, $s^{K}=O(1)$, collapse defused. And $\beta_\ell$ should *decay* with depth — the deeper a layer, the more I want pure propagation rather than transform, since the cumulative product of transforms is what blows the singular values around. The natural single-knob schedule is $\beta_\ell=\log(\frac{\lambda}{\ell}+1)\approx\lambda/\ell$ with $\lambda=0.5$: layer 1 gets the most transformation budget, deep layers fade toward identity. The two cures interlock — initial residual stops the propagation smoothing, identity mapping stops the weights collapsing rank and overfitting.

That depth is finally a resource has a formal reason: vanilla GCN's $K$ layers realize a *fixed*-coefficient polynomial $\tilde{\mathbf P}^{K}$, whereas a $K$-layer model of this form expresses an arbitrary order-$K$ filter $\sum_k\theta_k\tilde{\mathbf L}^{k}$ — matching coefficients telescopes to $\gamma_{K-1}=\sum_k\theta_k$ and $\gamma_{K-\ell-1}=c_\ell/c_{\ell-1}$, solvable for any target — so the free $\beta_\ell/\mathbf W^{(\ell)}$ can land on coefficients whose deep limit keeps the input rather than being forced into the degree-only collapse. The network can *learn* not to over-smooth.

Mapping this onto the harness adds one constraint the headline configuration does not face. GCNII's signature is very deep — 64 layers on Cora — but the scaffold pins the hidden width at $H=64$ and enforces a $1.05\times$-largest-baseline parameter budget; every conv is an $H\times H=4096$-parameter weight, so the count grows linearly in depth and 64 layers blows Cora's ~194k budget. The deepest budget-compliant uniform choice is $L=16$ layers, which fits all three datasets, so I keep the method exactly — initial residual to $\mathbf H^{(0)}$, identity-mapped weights with $\beta_\ell=\log(\lambda/\ell+1)$, $\alpha=0.1$, $\lambda=0.5$ — at $L=16$, still eight times deeper than every baseline and far past the over-smoothing wall, with the budget rather than a grid search setting the depth. The scaffold also fixes dropout at 0.5 and a single weight-decay scalar, so I cannot use a split $L_2$ for conv-versus-dense weights or the 0.6/0.7 dropout the headline config used; I accept those and rely on the identity mapping itself to do the weight regularization the split $L_2$ would have. Structurally one GCNII conv goes into `CustomMessagePassingLayer` — propagate through $\tilde{\mathbf P}$, form the teleported support $(1-\alpha)\tilde{\mathbf P}\mathbf H^{(\ell)}+\alpha\mathbf H^{(0)}$, apply the identity-mapped weight via the clean blend $(1-\theta)\mathbf s+\theta\,\mathbf s\mathbf W$ — and `CustomGNN` becomes input-FC → 16 convs (all $H\to H$, carrying $\mathbf H^{(0)}$ into every layer) → output-FC, ignoring the passed `num_layers=2` since the whole point is to set the depth myself. The bar is gcn's Cora 0.8207, CiteSeer 0.7177, PubMed 0.7863; the biggest gain should be on CiteSeer, where the two-hop wall most plausibly capped the baselines and a 16-hop field that does not over-smooth should reach the far same-class labels, with Cora up past ~0.82 and PubMed past 0.7863 by the smallest margin since its large clean neighborhoods already gave the two-hop model most of what depth offers.

```python
# EDITABLE region of custom_nodecls.py — finale: deep GCN via initial residual + identity mapping
import math


class CustomMessagePassingLayer(MessagePassing):
    """One deep conv: ((1-alpha) P̃ H + alpha H0) ((1-theta) I + theta W), theta = log(lambda/l + 1).
    P̃ = D̃^{-1/2} Ã D̃^{-1/2} (renormalization trick)."""

    def __init__(self, channels: int):
        super().__init__(aggr="add")
        self.weight = nn.Parameter(torch.empty(channels, channels))   # H x H, depth-independent size
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor, edge_index: Adj, h0: Tensor,
                alpha: float, lam: float, layer: int) -> Tensor:
        theta = math.log(lam / layer + 1.0)                 # identity-mapping strength, decays with depth
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        hi = self.propagate(edge_index, x=x, norm=norm)     # P̃ H^{(l)}
        support = (1 - alpha) * hi + alpha * h0             # initial residual: anchor to H^{(0)}
        # (1-theta) support + theta (support W) == support ((1-theta) I + theta W)
        return (1 - theta) * support + theta * torch.mm(support, self.weight)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    """Deep GCN: input FC -> L identity-mapped, initial-residual convs -> output FC.
    Sets its own depth (L) to go past the over-smoothing wall within the parameter budget;
    the passed num_layers is ignored."""

    NUM_PROP_LAYERS = 16          # 8x deeper than the baselines; fits the 1.05x-largest-baseline budget
    ALPHA = 0.1                   # initial-residual fraction
    LAMBDA = 0.5                  # identity-mapping schedule scale

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            CustomMessagePassingLayer(hidden_channels) for _ in range(self.NUM_PROP_LAYERS))
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        h0 = F.relu(self.fc_in(x))                          # H^{(0)} = transformed features
        h = h0
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(conv(h, edge_index, h0, self.ALPHA, self.LAMBDA, i + 1))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc_out(h)                               # class logits (loop applies cross-entropy)
```
