The naive floor came back exactly as broken as the construction predicted: 14.56 on ResNet-18, 13.77 on ResNet-34, 17.28 on ResNet-50 — a tight band of two-digit accuracies hugging the ten-class chance floor, with the larger backbone only marginally higher because its wider features give the probe a slightly bigger chance-level cushion. That signature is collapse, not under-training: a merely weak invariance term would let one backbone climb and the others lag, but all three sitting in the same broken band says the failure is in the *objective*, identical across architectures. So the hole is roughly 75 points of `val_acc`, and the fix is forced in kind — I have to add a term that makes the collapsed configuration a *high*-loss state. The only question is which statistic to penalize, and the contract here — `forward(z1, z2)` returns a scalar, no negatives slot, no second network, no stop-gradient — rules out two of the three classic answers. Contrastive repulsion is a non-parametric spread estimate from pairwise sample distances, brutally sample-hungry, and this harness fixes the batch at 256 with no queue; asymmetric methods need a predictor head and a stop-gradient between branches, which the two-tensors-to-one-number contract gives me no room to insert. That leaves the third family — the one that states what a good representation *is* and gets non-collapse as a *consequence* of the objective — and it fits the contract perfectly, because the whole statistic lives inside the two tensors I am handed.

I propose **Barlow Twins**, built on Barlow's redundancy-reduction principle: a good sensory code recodes redundant input into statistically non-overlapping components — if two units always carry the same information, one is wasted. That gives two desiderata that pull against each other in exactly the way I need: invariance (two views of one image should produce the same feature values) and non-redundancy (distinct feature components should not duplicate each other, i.e. be decorrelated). Bare invariance alone collapses everything to a point; a constant has no variance and duplicated features are maximally *correlated*, not decorrelated — so the place where the two desiderata balance cannot be the constant. Non-collapse becomes a consequence of asking for non-redundancy.

To make both desiderata one differentiable scalar, I work with the cross-correlation between the two views across the batch. With $b$ indexing the batch and $i,j$ the features, I first standardize each feature across the batch — subtract its batch mean, divide by its batch std — and then form, for every pair $(i,j)$,
$$C_{ij} = \frac{1}{B}\sum_b \hat z^{1}_{b,i}\,\hat z^{2}_{b,j},$$
a $D\times D$ matrix with each entry in $[-1,1]$. The two desiderata read straight off it. The diagonal $C_{ii}$ is the correlation of feature $i$ in view 1 with feature $i$ in view 2; if feature $i$ is invariant to the augmentation the two views move together and $C_{ii}=1$, so invariance is "drive every diagonal to 1." The off-diagonal $C_{ij}$ ($i\neq j$) is how much feature $i$ and a *different* feature $j$ co-vary, so non-redundancy is "drive every off-diagonal to 0." Together, push $C$ to the identity, with a knob trading the two halves:
$$L = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i\neq j} C_{ij}^2.$$

This excludes collapse by construction, and the standardization is the load-bearing piece. If the encoder emits a constant $z_{b,i}=c_i$, then standardizing feature $i$ maps every value to $(c_i-c_i)/\mathrm{std}=0$ — a zero-variance feature whose self-correlation $C_{ii}$ can never reach 1, because a correlation needs non-zero variance in both arguments even to be defined as 1. So the diagonal target alone already forbids the constant: "be perfectly correlated with yourself across the batch" is impossible for a feature that does not vary. That is exactly the counter-pressure the naive floor lacked. The subtler cheap escape — find one invariant direction and copy it into all $D$ features — is killed by the off-diagonal term, since the copies are the same signal, so $|C_{ij}|=1$ off-diagonal and the redundancy term slams them. The only zero-loss target is $C=I$: $D$ features each invariant *and* mutually decorrelated. Delete the off-diagonal term and the copy-one-direction escape returns; delete the on-diagonal term and nothing ties the views together — both halves are necessary, the decorrelation doing the job that negatives or stop-gradients did elsewhere, but inside the objective.

What this harness actually runs is the practical CIFAR port, and three implementation choices matter for the numbers. First, rather than dividing by per-feature std by hand I run each view through a non-affine `BatchNorm1d` — which *is* the center-and-divide-by-batch-std — so the cross-correlation is just `bn(z1).T @ bn(z2) / B`; because the loss module is constructed with no arguments and only sees $D$ at the first forward, I make it `nn.LazyBatchNorm1d(affine=False)` so it registers in `__init__` (riding along with `.to(device)`/dtype) but materializes $D$ on the first call. Second, the loss uses the on- and off-diagonal terms as *sums*, with $\lambda = 0.0051$ — small because there are $\sim D^2$ off-diagonal entries against $D$ diagonal ones, and an unweighted sum would let the off-diagonal block drown the diagonal alignment. Third, the one easy to miss: the raw summed loss with a 2048-wide projector is on the order of $10^3$–$10^4$, and LARS rescales each layer's step by $\lVert p\rVert/(\lVert g\rVert+\dots)$, so an enormous gradient norm starves that adaptive rescaling and the diagonal never climbs to 1 (a backbone can stay stuck near 10%). The CIFAR recipe's fix is a fixed multiplier `scale_loss = 0.1` on the whole loss, taming the gradient norm so LARS can move the diagonal. This is the solo-learn CIFAR recipe — default $2048\to2048$ projector, batch 256, the three CIFAR backbones — not the ImageNet recipe (8192 projector, smaller `scale_loss`, batch 2048, 1000 epochs), which at this budget would leave the diagonal stuck, so I keep `CONFIG_OVERRIDES = {}`.

The delta from the naive floor is concrete: where naive returned `F.mse_loss(z1, z2)` and let the encoder relax to a point, I now standardize each view across the batch, form the $D\times D$ cross-correlation, and push it toward the identity — diagonal to 1 for invariance, off-diagonal to 0 for decorrelation, scaled by 0.1 so LARS can drive it. The two-digit band should vanish into the high-80s on every backbone, and the cross-backbone order should *invert* relative to naive: under collapse ResNet-50 led only on a chance cushion, but with a working objective the larger backbones should genuinely separate classes better. A straggler stuck near 10% would be the LARS-starvation tell. And if Barlow lands solidly but a touch below where an explicit *per-branch* variance floor would — its diagonal-to-1 condition couples the two branches and standardizes the embeddings, which may shape a geometry that transfers a hair worse on the smallest backbone — that gap is the thread the next rung pulls.

```python
class CustomRegularizer(nn.Module):
    """Barlow Twins (Zbontar et al. ICML 2021).

    NB on scale_loss: the ImageNet 8192-projector recipe includes a
    `--scale-loss 0.024` multiplier. Without it the raw loss is on the
    order of 1e3-1e4, and LARS' adaptive rescaling
    (lars_lr = p_norm / (g_norm + ...)) starves the optimizer so the
    diagonal of the cross-correlation matrix never approaches 1. Here I
    use the CIFAR recipe (scale_loss=0.1) with the 2048 projector.
    """

    def __init__(self, lambd=0.0051, scale_loss=0.1):
        super().__init__()
        self.lambd = lambd
        self.scale_loss = scale_loss
        # Use LazyBatchNorm1d so the module is registered in __init__
        # (with proper to(device)/dtype propagation) but the feature dim
        # is materialized on the first forward call.
        self.bn = nn.LazyBatchNorm1d(affine=False)

    @staticmethod
    def _off_diagonal(x):
        # Return a flattened view of the off-diagonal elements.
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        B = z1.shape[0]

        # Cross-correlation matrix.
        c = self.bn(z1).T @ self.bn(z2)
        c = c / B

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = self._off_diagonal(c).pow(2).sum()
        total_loss = self.scale_loss * (on_diag + self.lambd * off_diag)

        return {
            "loss": total_loss,
            "on_diag": on_diag,
            "off_diag": off_diag,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# Use the solo-learn CIFAR-10 Barlow Twins recipe (proj=2048,
# scale_loss=0.1) instead of the ImageNet recipe
# (proj=8192, scale_loss=0.024, batch=2048, epochs=1000). Our setup
# matches solo-learn's: CIFAR-10, batch=256, ResNet-{18,34,50}, LARS
# with eta=0.02 and clip_lr=True. The 8192 recipe needs
# epochs=1000 + batch=2048 to converge — at our 100-epoch budget it
# leaves the diagonal stuck (see logs from v3: rn34 only reaches 10%).
# https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/barlow.yaml
CONFIG_OVERRIDES = {}
```
