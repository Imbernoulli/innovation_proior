The activation-clustering rung came back almost exactly as I had feared, and the recall numbers say why. On resnet20-cifar10-badnets poison recall was $0.0016$; on vgg16bn-cifar100-blend and mobilenetv2-fmnist-badnets it was a flat $0.0000$. The filter caught essentially none of the real poison, the trigger shortcut survived, and `asr` stayed pinned at $0.97/1.00/1.00$ with `defense_score` mean $0.4237$ — the retrained model simply relearned the backdoor from the poison the filter failed to remove. This is the high-dimensional clustering collapse made concrete: 2-means run directly on the penultimate features did not find a clean-versus-poison split, it bisected one blob arbitrarily, and the "smaller cluster" was half of the clean data. And it is not a tuning problem. Distance-based clustering needs distance contrast, and in hundreds of feature dimensions the pairwise distances concentrate, so the contrast the method depends on is gone. The lesson is sharp: I should not be asking a distance metric to *partition* the cloud into compact blobs at all. I should find the single *direction* the poison shift inflates and measure along it — a spectral question, not a clustering one, because a direction can be read off the covariance even when no compact cluster is visible to a distance metric.

I propose **per-class spectral signatures**, and the direction it measures along is not gestured at — it is derived from the mixture covariance. Within one class the training points are a mixture $F = (1-\varepsilon)D + \varepsilon W$ of clean points $D$ (mean $\mu_D$, fraction $1-\varepsilon$) and poison $W$ (mean $\mu_W$, fraction $\varepsilon$), with $\Delta = \mu_D - \mu_W$ the gap between the two sub-population means. Compute the mixture covariance about the mixture mean $\mu_F = (1-\varepsilon)\mu_D + \varepsilon\mu_W$. For the clean part $\mu_D - \mu_F = \varepsilon\Delta$, so $\mathbb{E}_{X\sim D}[(X-\mu_F)(X-\mu_F)^\top] = \Sigma_D + \varepsilon^2\Delta\Delta^\top$; for the poison part $\mu_W - \mu_F = -(1-\varepsilon)\Delta$, so $\mathbb{E}_{X\sim W}[\cdot] = \Sigma_W + (1-\varepsilon)^2\Delta\Delta^\top$. Mixing with weights $(1-\varepsilon)$ and $\varepsilon$, the $\Delta\Delta^\top$ coefficient is $(1-\varepsilon)\varepsilon^2 + \varepsilon(1-\varepsilon)^2 = \varepsilon(1-\varepsilon)$, giving
$$\Sigma_F = (1-\varepsilon)\,\Sigma_D + \varepsilon\,\Sigma_W + \varepsilon(1-\varepsilon)\,\Delta\Delta^\top.$$
There it is. The contamination contributes a **rank-one variance bump** $\varepsilon(1-\varepsilon)\Delta\Delta^\top$ on top of the within-population covariances: the mean shift between clean and poison announces itself as extra variance pointing exactly along $\Delta$. If that bump is large relative to the within-population spread, the top eigenvector of $\Sigma_F$ lines up with $\Delta$, and the squared projection onto it separates poison from clean. This is the lever clustering missed — I do not need to *group* the points, I need the one direction of anomalous variance, and the covariance hands it to me directly. Crucially, reading a covariance eigenvector lives in the second moment, not in pairwise distances, so it does not suffer the distance-concentration that killed 2-means.

Why this succeeds where pixels fail comes down to the size of the bump $\varepsilon(1-\varepsilon)\lVert\Delta\rVert^2$ having to beat the within-population variance to dominate the spectrum. At the pixel level $\lVert\Delta\rVert$ is tiny (a trigger barely moves the mean) while the natural-image variance is enormous, so the bump is buried; in the *learned representation* the network amplifies the trigger — it is rewarded for keying on a near-perfect predictor of the target — so $\lVert\Delta\rVert$ in penultimate-feature space is pushed past the within-class variance. The same amplification I leaned on for clustering is what makes the spectral bump visible, but now the test reads the bump off the covariance rather than asking a distance metric to find a cluster, so it survives the high dimension that defeated 2-means.

I owe the condition under which separation is guaranteed, because "should separate" is what I said about clustering too. Call $D, W$ **$\varepsilon$-spectrally separable** if for the top eigenvector $v$ of $\Sigma_F$ there is a threshold $t>0$ with $\Pr_D[\lvert\langle X-\mu_F,v\rangle\rvert > t] < \varepsilon$ (almost no clean point projects beyond $t$) and $\Pr_W[\lvert\langle X-\mu_F,v\rangle\rvert < t] < \varepsilon$ (almost no poison point within it); then removing the largest projections removes nearly all poison while sacrificing little clean data. The claim is that if $\Sigma_D, \Sigma_W \preceq \sigma^2 I$ and $\lVert\Delta\rVert^2 \geq 6\sigma^2/\varepsilon$ (with $\varepsilon < 1/2$), then $D, W$ are $\varepsilon$-spectrally separable. Three moves. (1) Chebyshev: along any unit $u$, each sub-population concentrates within $\sim\sigma$ of its own mean, $\Pr_D[\lvert\langle X-\mu_D,u\rangle\rvert > t] \leq \sigma^2/t^2$. (2) Correlation implies separation: with $c = \lvert\langle u,\Delta\rangle\rvert$, the clean and poison projected centers about $\mu_F$ sit at $\varepsilon c$ and $-(1-\varepsilon)c$, and a threshold between them with Chebyshev room on both sides drives both error tails below $\varepsilon$ once $c$ is large relative to $\sigma/\sqrt{\varepsilon}$. (3) The top eigenvector is correlated with $\Delta$: from $\Sigma_F \succeq \varepsilon(1-\varepsilon)\Delta\Delta^\top$ the top eigenvalue is at least $\varepsilon(1-\varepsilon)\lVert\Delta\rVert^2$, and expanding $v^\top\Sigma_F v \leq \sigma^2 + \varepsilon(1-\varepsilon)\langle v,\Delta\rangle^2$ gives $\langle v,\Delta\rangle^2 \geq \lVert\Delta\rVert^2 - \sigma^2/(\varepsilon(1-\varepsilon))$, which under the constant-$6$ hypothesis and $\varepsilon<1/2$ exceeds $(2/3)\lVert\Delta\rVert^2 > 4\sigma^2/\varepsilon$, supplying the correlation move (2) needs. The $6$ is a proof-sketch constant, not an optimized threshold; the finite-sample version costs a little slack ($\varepsilon < 1/4$, $\lVert\Delta\rVert^2 \geq 10\sigma^2/\varepsilon$, $n = \Omega(d\log n/\varepsilon)$ samples) via matrix concentration and gives separation with probability at least $9/10$.

The same derivation names the regime where the method breaks, and it is exactly what the task design warned about: the whole argument needs $\varepsilon < 1/2$ within the class. If the within-class poison fraction climbs past a half, the "outlier" sub-population becomes the majority, centering subtracts a mean dominated by poison, and the *clean* images become the extreme points of $v$ — I would remove clean data. That is why vgg16bn-cifar100-blend uses 1% global poison (target class ~33% poison) rather than 5% (which would make the target class 83% poison): at 83% the spectral test is pointed at the wrong sub-population by construction. So on cifar100 I do not expect miracles even from a faithful test — but I expect it to stop being blind, which is the bar clustering failed.

Grounding the computation in the scaffold: for each training label I gather that class's features, compute the class mean $\mu_c$ (this is $\mu_F$ per class, the only mean I can compute since I do not know which points are poison), center, and take the top *right* singular vector $v_c$ of the centered matrix — the top eigenvector of the class covariance, the direction the rank-one bump inflates. I use the right singular vector, not the left, because the centered matrix is $n\times d$ with one row per sample, so its right singular vectors are the $d$-dimensional feature-space directions I project onto. Each point scores by the squared centered projection $\tau_i = ((r_i - \mu_c)\cdot v_c)^2$ — its per-point variance contribution along the contaminated direction. Squaring rather than absolute value only monotonically reshapes the ranking, so "remove the largest" is unchanged, and squaring is the natural form because the bump I derived is a variance. I do this per class because the poison all sits inside one trained label; decomposing the whole dataset at once would let ordinary between-class mean differences dominate the spectrum and I would just rediscover the labels — the pooling mistake I avoided in clustering.

One subtlety about *which* label to group by matters more than it looks. The poison's signature lives in the target class as it was *trained* — the bag of points the network saw as "this class" and whose representation carries the amplified trigger direction. If I grouped by the model's *predicted* class at scoring time, a hard poisoned example whose prediction happens to disagree with its assigned label would be pulled into the wrong group and projected onto a direction fit for a different class, corrupting its score. So `fit` remembers the training labels and `score_samples(features, logits)` applies each point's class-specific direction using those cached labels; the logits the harness passes are not the grouping signal, only present because the interface offers them. A degenerate class with fewer than two points gets a zero direction rather than a crash, and the harness's fixed $1.5\,\varepsilon$ removal is again the right asymmetry — only an upper bound on $\varepsilon$ is known, surviving poison preserves the trigger shortcut, and discarding a few extra clean points out of thousands costs little.

```python
# EDITABLE region of custom_backdoor_defense.py — step 2: spectral signatures (Tran et al., 2018)
import numpy as np
import torch


class BackdoorDefense:
    """Per-class spectral signatures (Tran et al., NeurIPS 2018).

    For each class c: compute the mean mu_c of its penultimate-layer
    features, center them, take the top right singular vector v_c of
    the centered matrix, and score each sample by
        tau_i = ((r_i - mu_c) . v_c)^2.
    Training labels from fit() route each sample to its class direction.
    """

    def __init__(self):
        self.class_centers = {}       # cls -> mean vector (D,)
        self.class_directions = {}    # cls -> top right singular vector (D,)
        self.cached_labels = None     # training labels from fit()

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self.cached_labels = labels.copy()
        for cls in np.unique(labels):
            mask = labels == cls
            cls_feat = features[mask]
            if cls_feat.shape[0] < 2:
                # Degenerate class; zero direction
                self.class_centers[int(cls)] = cls_feat.mean(axis=0) if len(cls_feat) else np.zeros(features.shape[1])
                self.class_directions[int(cls)] = np.zeros(features.shape[1])
                continue
            mu = cls_feat.mean(axis=0)
            centered = cls_feat - mu
            # Top right singular vector of the centered class matrix.
            # np.linalg.svd returns vh with rows = right singular vectors.
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            self.class_centers[int(cls)] = mu
            self.class_directions[int(cls)] = vh[0]

    def score_samples(self, features, logits):
        features = np.asarray(features, dtype=np.float64)
        # Use TRAINING LABELS (cached in fit) to pick the per-class
        # direction.  Using argmax(logits) would mismatch any sample whose
        # prediction disagrees with its training label.
        if self.cached_labels is not None and len(self.cached_labels) == len(features):
            cls_for_sample = self.cached_labels
        else:
            cls_for_sample = np.asarray(logits).argmax(axis=1)

        scores = np.zeros(len(features), dtype=np.float64)
        for cls, direction in self.class_directions.items():
            mask = (cls_for_sample == cls)
            if not mask.any():
                continue
            centered = features[mask] - self.class_centers[cls]
            proj = centered @ direction
            scores[mask] = proj * proj  # tau_i = (centered . v)^2
        return scores
```
