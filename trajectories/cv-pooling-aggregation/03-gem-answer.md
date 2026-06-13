**Problem (from step 2).** Average + Max recovered the floor on the real-map backbones (ResNet-56
69.96 → 71.06, MobileNetV2 94.27 → 94.52) by reporting both statistics, confirming the midpoint of the
mean↔max axis is the right neighborhood. But it hard-codes a *fixed* 50/50 blend for every channel and
every backbone, when different channels and architectures want different points on the axis.

**Key idea (GeM — generalized-mean pooling).** The mean↔max axis has an exact one-parameter form: the
power-mean `f_c(p) = ( mean_x x^p )^{1/p}`, which is the arithmetic mean at `p=1` (GAP) and the max as
`p→∞` (GMP). Make `p` a single learnable scalar (init `3.0`) and let SGD slide it along the axis per
backbone — replacing Avg+Max's fixed endpoint-blend with the continuous interpolation it approximated.
Reference: Radenović, Tolias & Chum, arXiv:1711.02512 (generalized-mean pooling).

**Why it works.** `p` is differentiable, so the classification loss itself decides whether more
peakiness (raise `p`) or more averaging (lower `p`) helps; it strictly generalizes every earlier rung
(it *contains* GAP, GMP, and the Avg+Max region) and adds adaptivity the fixed 50/50 blend lacked. Two
clamps keep it safe as a classification pool: `x.clamp(min=eps)` (positive bases for `x^p`) and
`p.clamp(min=1.0)` (pins the operating range to the mean↔max segment, never falling off toward the
geometric/harmonic side or the `1/p` blow-up).

**Divergence from the retrieval descriptor (what the harness drops).** No `L2`-normalization after the
pool — the output feeds the frozen linear head, which wants the channel magnitudes that `L2` would
erase. A single *shared* `p`, not a per-channel `p_k` vector. Trained purely by cross-entropy, not a
contrastive loss. One parameter total.

**Hyperparameters.** `p = nn.Parameter(ones(1) * 3.0)`, `eps = 1e-6`. Power-mean implemented as
`adaptive_avg_pool2d(x.pow(p), 1).pow(1/p)`. On VGG-16-BN's `1×1` map GeM degenerates to GAP and `p`
gets no gradient (uninformative, as for every rung).

**What to watch.** Expect GeM to *exceed* Avg+Max on the two real-map backbones (ResNet-56 into the
low-72s, MobileNetV2 high-94s) by learning a per-backbone `p`; VGG near the low-74s as upstream noise.

```python
# EDITABLE region of pytorch-vision/custom_pool.py (lines 31-48) — step 3: GeM (generalized-mean) pooling
class CustomPool(nn.Module):
    """Generalized Mean (GeM) Pooling.

    Learnable generalized mean with parameter p (init=3.0).
    Interpolates between average pooling (p=1) and max pooling (p->inf).

    """

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * 3.0)
        self.eps = 1e-6

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps)
        return F.adaptive_avg_pool2d(x.pow(p), 1).pow(1.0 / p).view(x.size(0), -1)
```
