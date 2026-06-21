The scaffold I am handed is already a GCN encoder feeding a dot product, trained by binary cross-entropy on sampled positive and negative pairs. Rather than immediately bolting structural features or a learned decoder onto that, I want to start from the *principled* version of exactly that object, because the dot-product-of-GCN-embeddings shape is not an arbitrary baseline — it is the likelihood of a real generative model of the graph. Starting there tells me whether the probabilistic framing buys anything at all before I add machinery, and gives every later rung a clean floor to be measured against.

I propose the variational graph auto-encoder (VGAE): the same GCN-encoder / inner-product-decoder skeleton, but derived as a latent-variable model of the adjacency and trained with the regularizer that framing demands. The whole method falls out of taking the generative story seriously. I have an undirected graph on $N$ nodes with adjacency $A$ and features $X$, and I posit a latent embedding $z_i\in\mathbb{R}^F$ per node, with a generative model in which each edge is a Bernoulli draw whose probability is a function of the two latents,
$$p(A_{ij}=1\mid z_i,z_j)=\sigma(z_i^\top z_j).$$
This *inner-product decoder* is not a modeling afterthought — it is the natural likelihood for a symmetric binary relation: $z_i^\top z_j$ is large when two embeddings point the same way, the sigmoid turns it into an edge probability, and reconstructing the whole adjacency is $\hat A=\sigma(ZZ^\top)$. The prior on the latents is the usual isotropic Gaussian $p(Z)=\prod_i\mathcal N(z_i\mid 0,I)$.

The inference side — the encoder — is where the graph structure and features enter. Exact posterior inference over $Z$ is intractable, so I take the variational route: an approximate posterior $q(Z\mid X,A)=\prod_i\mathcal N(z_i\mid \mu_i,\operatorname{diag}(\sigma_i^2))$ whose per-node mean and log-standard-deviation are produced by a GCN read off the features and connectivity. A shared GCN stack computes a hidden representation, and then two separate GCN heads on top of it produce $\mu=\mathrm{GCN}_\mu(X,A)$ and $\log\sigma=\mathrm{GCN}_\sigma(X,A)$. The reason both heads are GCNs and not plain linear layers is that I want the *uncertainty* of a node's embedding to depend on its neighborhood too — a node with few, ambiguous connections should be allowed a wider posterior than a hub with a clear structural role. This is the GCN-as-encoder idea (Kipf & Welling 2016), now producing the parameters of a distribution rather than a point.

The objective is the variational lower bound,
$$\mathcal L=\mathbb E_{q(Z\mid X,A)}[\log p(A\mid Z)]-\mathrm{KL}\big[q(Z\mid X,A)\,\big\|\,p(Z)\big].$$
The first term is the expected reconstruction log-likelihood; under the Bernoulli–inner-product decoder it is exactly a cross-entropy between the adjacency entries and $\sigma(z_i^\top z_j)$ — the very edge-classification loss the harness already runs. The second term is the KL from the approximate posterior to the prior, which for two Gaussians has the closed form
$$\mathrm{KL}=-\tfrac12\sum_i\sum_f\big(1+\log\sigma_{if}^2-\mu_{if}^2-\sigma_{if}^2\big).$$
To make the expectation differentiable I use the reparameterization trick: instead of sampling $z_i$ directly, I sample $\epsilon\sim\mathcal N(0,I)$ and set $z_i=\mu_i+\epsilon\odot\sigma_i$ during training, moving the stochasticity off the parameters so gradients flow through $\mu$ and $\sigma$. At evaluation I drop the noise and use $z_i=\mu_i$, the posterior mean, as the deterministic embedding.

The part that has to be gotten right, and the whole reason to think carefully rather than copy a generic VGAE, is how the ELBO survives this harness. The fixed training loop does not know about an ELBO. It calls my `forward`, gets back a vector of edge scores, and computes binary cross-entropy on those scores against $\{1$ for positives, $0$ for negatives$\}$. That BCE *is* the reconstruction term, handed to me for free. But the loop will never add a KL term, because it only ever sees my scalar scores — there is no hook for an auxiliary loss. If I do nothing, the KL gradient never flows, the encoder is unregularized, $\sigma$ collapses, $\mu$ drifts to whatever reconstruction wants, and I am left with a plain (non-variational) auto-encoder plus sampling noise — strictly worse than fitting point embeddings. So the KL must be smuggled into the computation graph *of the scores themselves*.

The mechanism is to add the KL, as a single scalar, onto every returned score:
$$\text{score}\leftarrow\text{score}+w\cdot\mathrm{KL}_{\text{per-node}},\qquad \mathrm{KL}_{\text{per-node}}=-\tfrac12\,\overline{\textstyle\sum_f(1+\log\sigma^2-\mu^2-\sigma^2)},$$
the mean-over-nodes KL (the standard $1/N$ VGAE normalization, realized here as a mean). This is a uniform additive shift on all scores, so it barely perturbs the *ranking* the BCE cares about; but because the KL scalar is built from $\mu$ and $\log\sigma$ — differentiable functions of the GCN parameters — adding it to a quantity the loop then differentiates makes the KL gradient flow into the encoder during backprop. The coefficient $w$ controls how hard the regularizer pulls, and I keep it small, $w=0.005$: the KL is already normalized per node, and the dominant gradient on any single score must remain the reconstruction signal — a large $w$ would drag every logit toward a common value and degrade the BCE. This injection trick is specific to this score-only interface; a standalone VGAE simply sums ELBO terms in its loss, and the small coefficient is a direct consequence of the KL entering through the scores rather than as a separate term. One more harness-driven detail: BatchNorm goes only on the shared intermediate layers and *not* on the $\mu$/$\log\sigma$ heads, because the decoder is a raw inner product and normalizing the final embeddings would crush the magnitude the dot product needs to spread positive and negative pairs apart.

This is a deliberate floor, and I can already name where it will be weak. The Gaussian prior pulls every embedding toward the origin while the inner-product likelihood wants to push them outward and apart; those two forces fight, the prior is a poor match for an inner-product likelihood, and the latent space the encoder may use is squeezed. The train-time sampling noise is pure variance from the ranking metrics' point of view — it aids calibration but blurs the exact top-of-list ordering that MRR and Hits@K reward. And the decoder is the weakest possible one: a single inner product, no learned interaction between the two embeddings, no structural feature about the *pair* such as how many neighbors $i$ and $j$ share. The model only ever knows a pair through the geometry of two independently-encoded points. I expect this to *separate* — respectable AUC — while being soft on the ranking-sensitive metrics, and weakest on the large-pool `ogbl-collab` Hits@50, where a single dot product is a blunt instrument against tens of thousands of non-edges and there is no explicit common-neighbor signal. That is the verdict that should tell the next rung to drop the variational noise for a deterministic embedding and replace the bare inner product with a *learned* decoder that mixes the two embeddings.

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
