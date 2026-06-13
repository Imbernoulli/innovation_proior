**Problem.** Link prediction with a principled version of the scaffold default: a GCN encoder feeding
an inner-product decoder. The question for the first rung is whether framing this as a generative model
of the adjacency — with the regularizer that framing demands — buys anything before adding structural
features or a learned decoder.

**Key idea (VGAE).** Treat each edge as a Bernoulli draw $p(A_{ij}=1\mid z_i,z_j)=\sigma(z_i^\top z_j)$
with per-node latents $z_i$. A shared GCN plus two GCN heads produce $\mu$ and $\log\sigma$; the
embedding is the reparameterized sample $z=\mu+\epsilon\odot\sigma$ at train time and $\mu$ at eval.
Train against the ELBO: reconstruction (the loop's BCE on the inner-product scores) minus
$\mathrm{KL}[q(Z\mid X,A)\,\|\,\mathcal N(0,I)]$.

**Why / harness adaptation.** The fixed loop computes BCE on returned scores and offers no hook for an
auxiliary loss, so the KL cannot be a separate term. It is injected into the scores: every score gets a
uniform additive shift $w\cdot\mathrm{KL}_{\text{per-node}}$ with
$\mathrm{KL}_{\text{per-node}}=-\tfrac12\,\overline{\sum_f(1+\log\sigma^2-\mu^2-\sigma^2)}$, so the KL
gradient flows into the encoder during backprop while barely perturbing the ranking. $w=0.005$ keeps
the reconstruction gradient dominant. No BatchNorm on the $\mu$/$\log\sigma$ heads, to preserve
embedding magnitude for the dot product. This is a deliberate floor: bare inner-product decoder, no
pairwise/structural features, plus sampling noise — expected to separate (AUC) but be soft on the
ranking metrics, weakest on large-pool `ogbl-collab` Hits@50.

**Hyperparameters.** `hidden_channels=256`, `num_layers=2`, `dropout=0.0`, `kl_weight=0.005`, Adam
`lr=0.01`, BCE loss, 200 epochs, early-stop patience 20.

```python
class LinkPredictor(nn.Module):
    """Variational Graph Auto-Encoder (VGAE).

    GCN encoder produces mean + logstd, samples via reparameterization.
    Dot-product decoder. KL regularization is injected into the computation
    graph so that it participates in the backward pass even though the
    external training loop only sees BCE on the returned scores.

    The KL term is added to each score as  w * KL / num_nodes  (the
    standard VGAE normalisation), NOT divided by num_scores.  This
    ensures the KL gradient is strong enough to regularise the latent
    space while remaining small enough not to overwhelm the
    reconstruction gradient on any single score.

    No BatchNorm on the final (mu/logstd) layers to preserve embedding
    magnitude for dot-product scoring.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        # Standard VGAE uses 1/N weighting for KL; we keep a small coefficient
        # because the KL is already normalized per node below.
        self.kl_weight = 0.005

        # Shared GCN layers (all but last)
        self.shared_convs = nn.ModuleList()
        self.shared_bns = nn.ModuleList()
        if num_layers > 1:
            self.shared_convs.append(GCNConv(in_channels, hidden_channels))
            self.shared_bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.shared_convs.append(GCNConv(hidden_channels, hidden_channels))
                self.shared_bns.append(nn.BatchNorm1d(hidden_channels))
            last_in = hidden_channels
        else:
            last_in = in_channels

        # Separate heads for mean and log-variance (no BN on these)
        self.conv_mu = GCNConv(last_in, hidden_channels)
        self.conv_logstd = GCNConv(last_in, hidden_channels)

        self.__mu = None
        self.__logstd = None
        self.__num_nodes = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.__num_nodes = x.size(0)
        # Shared layers with BN + ReLU
        for conv, bn in zip(self.shared_convs, self.shared_bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        self.__mu = self.conv_mu(x, edge_index)
        self.__logstd = self.conv_logstd(x, edge_index)

        if self.training:
            std = torch.exp(0.5 * self.__logstd)
            eps = torch.randn_like(std)
            return self.__mu + eps * std
        return self.__mu

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        return (z_src * z_dst).sum(dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        scores = self.decode(edge_label_index, z,
                             edge_index=edge_index, num_nodes=x.size(0))
        # Inject KL divergence into the computation graph so its gradient
        # flows through the encoder during backprop.  We add a uniform
        # per-score shift:  scores + w * KL_per_node.
        # KL_per_node = (1/N) * sum_i KL(q(z_i|X,A) || p(z_i)).
        # The coefficient w controls the strength of the regularisation.
        if self.training and self.__mu is not None:
            kl_per_node = -0.5 * torch.mean(
                torch.sum(1 + self.__logstd - self.__mu.pow(2)
                          - self.__logstd.exp(), dim=-1)
            )
            scores = scores + self.kl_weight * kl_per_node
        return scores
```
