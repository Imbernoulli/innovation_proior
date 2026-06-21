Barlow cleared the floor convincingly: 89.12 on ResNet-18, 90.28 on ResNet-34, 90.62 on ResNet-50 — the ~75-point hole essentially recovered, a clean monotone with backbone scale, no straggler stuck near 10%. So the information-maximization family is the right family on this contract; I am not looking for a different *kind* of fix but a cleaner member of it. Two things in those numbers point the way. The spread across backbones is about 1.5 points and ResNet-18 sits lowest, and Barlow's whole objective lives in *one* $D\times D$ matrix — the cross-correlation between view 1 and view 2 — so each entry mixes a dimension of branch A with a dimension of branch B and *couples* the two branches; and to make the entries genuine correlations in $[-1,1]$ it has to *standardize* the embeddings, with the diagonal-to-1 condition then doing double duty as invariance *and* the only thing pinning per-feature scale. Collapse-prevention is entangled: it rides on a cross-branch matrix and an embedding standardization, and I cannot point at one line and say "this, by itself, forbids the constant."

I propose **VICReg** — variance, invariance, covariance regularization — which separates the two failure modes and forbids each with an explicit term applied to *each branch on its own*, with no cross-branch matrix and no embedding standardization. Trivial collapse is exactly "each embedding dimension carries no variation across the batch," so the most direct forbiddance is the **variance** term: look at one branch's embeddings, compute per dimension its batch std, and demand it not be zero. I do not want to *maximize* variance — that would push without bound and fight invariance — I want a *floor*: a hinge $v(Z)=\mathrm{mean}_j\,\max(0,\;\gamma - S(z^j))$ where $S(z^j)$ is the batch std of column $j$ and $\gamma$ the target floor; above $\gamma$ a dimension contributes nothing, below it the hinge pushes back up. Applied to each branch separately, $v(Z_1)$ and $v(Z_2)$, with no coupling.

One subtlety decides whether the term even works: the hinge must act on the standard deviation, not the variance. They carry the same information, but the gradients differ enormously. With the *variance* in the hinge, $\partial\mathrm{Var}/\partial z_{b,j} \propto (z_{b,j}-\bar z_j)$, and near collapse every $z_{b,j}$ sits at the mean, so that gradient $\approx 0$ — the cure goes silent exactly where the disease is worst. With the *standard deviation*, $S=\sqrt{\mathrm{Var}}$, we have $dS/d\mathrm{Var}=1/(2\sqrt{\mathrm{Var}})\to\infty$ as $\mathrm{Var}\to0$; the $(z-\bar z)$ factor still shrinks near collapse, but the deviations are themselves of order $\sqrt{\mathrm{Var}}$, so dividing by $\sqrt{\mathrm{Var}}$ keeps the ratio order 1, and the restoring force *survives* at collapse. So it has to be the std, with a small $\epsilon$ inside the root for stability: $S=\sqrt{\mathrm{Var}+\epsilon}$. With $\gamma=1$ — the absolute scale is arbitrary since the network can rescale; what matters is a fixed positive floor — the term is $\mathrm{mean}_j\,\max(0,\,1-\sqrt{\mathrm{Var}(z^j)+\epsilon})$, and a constant gives every column zero variance so $\sqrt{0+\epsilon}\approx0<1$ and the hinge is fully active on every dimension. Collapse is now the *most* penalized configuration, from a single per-branch, per-dimension statistic.

The variance floor alone is not enough, for the same cheap escape I had to block in Barlow: pin every dimension's std at 1, but make all $D$ dimensions copies of one informative direction. Each dimension has variance 1, so the variance term is happy, yet the representation has $D$ coordinates but one effective degree of freedom — informational collapse, to which a per-dimension statistic is blind. So the second term forbids redundancy *between* dimensions, the decorrelation idea from Barlow but on a single branch. Center one branch's embeddings and form its **covariance** $C(Z)=(\tilde z^\top \tilde z)/(B-1)$; its diagonal is the per-dimension variances (owned by the variance term), and its off-diagonal $C_{ij}$ ($i\neq j$) is exactly how much dimensions $i$ and $j$ co-vary. Drive every off-diagonal to zero, $c(Z)=\sum_{i\neq j}[C(Z)]_{ij}^2$, on each branch separately. Note what I did *not* do: I never normalized the embeddings into correlations. Barlow had to standardize because its cross-correlation entries needed to be in $[-1,1]$ for the diagonal-to-1 target to mean anything; here I have no diagonal target on the covariance — the variance term already pins each dimension's scale to $\gamma$ — so the covariance is left unnormalized, the variance term owning the scale and the covariance term doing the decorrelating. That is the standardization Barlow needed and I do not.

The division of labor is the crux. Variance forbids trivial collapse (each dimension keeps std $\geq\gamma$, nothing shrinks to a point); covariance forbids informational collapse (decorrelated dimensions, so the guaranteed variance is spread across all $D$ dimensions rather than duplicated). Neither alone suffices: variance alone permits the copy-one-direction redundancy, and covariance alone collapses outright, since the cheapest way to zero all off-diagonals is to send everything to a constant where the whole covariance matrix is zero. They are complementary — variance gives the covariance term something to spread, covariance makes the variance meaningful. The third leg is plain-MSE **invariance**, `F.mse_loss(z1, z2)` with no normalization (the variance term owns the scale, so l2-normalizing would fight it), tying the two views together so the variance-and-decorrelation budget is spent on augmentation-stable features rather than independent noise that would satisfy variance and covariance perfectly while telling me nothing about the image.

The coefficients here are not the generic ones, and that is load-bearing. The invariance term has coefficient 1; the variance term uses `std_margin = 1` and `std_coeff = 1.0` (the hinge summed over the two branches); the covariance term is the off-diagonal-squared *mean* of each branch's $(x^\top x)/(B-1)$, summed over branches, weighted at `cov_coeff = 100.0`. That cov weight of 100 is far heavier than the generic recipe's relative weighting, and it pairs with a deliberate reshape: `CONFIG_OVERRIDES = {"proj_output_dim": 1024}`, narrowing the projector output from 2048 to 1024. The two interlock — a narrower embedding has fewer off-diagonal pairs to decorrelate, so a large cov coefficient can fully decorrelate the 1024-wide space within budget without the $\sim D^2$ off-diagonal gradient destabilizing training, and 1024 is the projector width the upstream "impact of the projector" comparison ranks best for this method on CIFAR-10 ResNet-18. So unlike Barlow, which kept the default 2048 projector, VICReg here narrows to 1024 and leans hard on covariance (100) with a light variance floor (1) and unit invariance — the variance term only holds the floor while the heavy covariance term does the spreading, in a deliberately narrower, more fully decorrelated space.

The delta from Barlow is precise: where Barlow rode one coupled, standardized cross-correlation matrix and pinned scale through its diagonal, VICReg splits collapse-prevention into a per-branch variance floor on the standard deviation (kills trivial collapse, gradient survives at collapse) and a per-branch unnormalized covariance penalty (kills informational collapse), with plain-MSE invariance, no embedding standardization and no cross-branch coupling, in a 1024-wide projector with a heavy cov weight. I expect it at or slightly above Barlow on the larger backbones — high-89s to low-91s — and the interesting bet is ResNet-18, where Barlow sat lowest: the decoupled per-branch geometry in a 1024-wide space should transfer at least as well, so I expect VICReg's ResNet-18 to match or edge past. A *lower* ResNet-18 would be the tell that the cov weight of 100 over-decorrelates before the representation is informative. And a near-tie with Barlow on aggregate would say that on this harness the *kind* of anti-collapse term matters more than its exact formulation — which then points the final rung at a regularizer that does not just decorrelate but pins the *whole* embedding distribution to a target, not only its second moments.

```python
class CustomRegularizer(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization."""

    def __init__(self, std_coeff=1.0, cov_coeff=100.0, std_margin=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_margin = std_margin

    def _std_loss(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        return torch.mean(F.relu(self.std_margin - std))

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _cov_loss(self, x):
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)
        return self._off_diagonal(cov).pow(2).mean()

    def forward(self, z1, z2):
        sim_loss = F.mse_loss(z1, z2)
        var_loss = self._std_loss(z1) + self._std_loss(z2)
        cov_loss = self._cov_loss(z1) + self._cov_loss(z2)
        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# Paper README "Impact of the projector" table ranks VICReg's best
# projector as 2048->1024 (90.12% on CIFAR-10 ResNet-18, 300 epochs).
CONFIG_OVERRIDES = {"proj_output_dim": 1024}
```
