The spectral rung did exactly what the derivation promised and then stopped exactly where I worried it would. On resnet20-cifar10-badnets poison recall climbed from the clustering rung's $0.0016$ to $0.2688$, and on mobilenetv2-fmnist-badnets from $0.0000$ to $0.1192$ — the spectral test surfaces real poison where 2-means found none. But `asr` was $0.9708/0.9999/1.0000$, essentially unmoved, and on vgg16bn-cifar100-blend recall stayed flat at $0.0000$. Recall went up; the backdoor did not break; `defense_score` mean $0.4225$ was, if anything, a hair below the clustering rung. A $27\%$ recall means the top eigenvector caught roughly one poisoned point in four and missed three. The spectral test bets everything on the *single* top-variance direction of the *combined* class covariance, and that bet pays off only when the contaminated direction $\Delta$ is also the largest-variance direction of the data. Two things break it. The clean penultimate features of a class are not isotropic — a few directions of large legitimate spread (pose, sub-type, lighting) and many of small spread — so if the poison shift falls along a clean direction that is *not* the top one, the top eigenvector is owned by clean variance and the poison projection looks clean. And the blended attack on cifar100 spreads its trigger more diffusely than a single bright BadNets pixel, so its signature is not one sharp direction but several low-variance ones, of which the top eigenvector inspects exactly one — which is why recall there is pinned at zero. The missing $0.73$ on cifar10 is sitting in directions the top eigenvector never inspected. I do not need a *better* top eigenvector; I need the poison's excess variance to stand out *wherever* it sits in the spectrum.

The geometric operation that makes excess variance stand out regardless of direction is **whitening**, and the method I propose is **SPECTRE**: robust-whitening followed by a Quantum-Entropy (QUE) score, per class. Suppose I knew the *clean* covariance $\Sigma_{\text{clean}}$ of the class. Whiten every point by $\Sigma_{\text{clean}}^{-1/2}$ and the clean cloud becomes isotropic, unit variance in every direction; there is then no longer a "big clean direction" for the poison to hide behind. The poison's mean shift $\Delta$ becomes $\Sigma_{\text{clean}}^{-1/2}\Delta$ and its excess variance is now the *only* above-unit-variance structure left, exposed no matter which clean direction it originally overlapped. Whitening converts the question I cannot control — "is the signature a top-variance direction?" — into one whose answer is yes whenever poison is present — "is there *any* residual high-variance direction?"

This trades the problem for a harder one: I do not have $\Sigma_{\text{clean}}$. All I have is the *contaminated* class covariance, and if I whiten by that I whiten away the poison along with everything else — the contaminated covariance already contains the rank-one bump $\varepsilon(1-\varepsilon)\Delta\Delta^\top$, so $\Sigma_F^{-1/2}$ flattens precisely the direction I want to keep tall. The whole approach hinges on estimating the *clean* covariance from contaminated samples with no trusted clean set, which is exactly the problem high-dimensional robust statistics solves: estimate the mean and covariance of $D$ when an $\varepsilon$-fraction of samples are adversarial outliers, recovering them up to error $O(\varepsilon\sqrt{\log(1/\varepsilon)})$ from enough samples — close enough that whitening by the robust covariance makes the clean cloud nearly isotropic while leaving the poison's excess variance intact. The workhorse underneath the polynomial-time robust estimators is iterative filtering: start with all points, compute a mean and covariance, measure each point's Mahalanobis distance under the current estimate, keep the points closest to the running core and drop the most extreme, recompute, repeat. The fixed point is a clean core whose mean and covariance are not dominated by the outliers, because the high-Mahalanobis-distance points are exactly the ones trimmed. The trim fraction should track the contamination level (on the order of $\varepsilon$, with margin, since over-trimming clean points costs almost nothing while under-trimming leaves poison in the covariance and re-pollutes the whitening — the same asymmetry the harness's $1.5\,\varepsilon$ removal already encodes), and because it is below a half the clean majority is never trimmed away.

There is a dimensionality obstacle I must clear before whitening, and it is the difference between this working and amplifying noise. Robust covariance estimation in $d$ dimensions needs on the order of $d^2$ samples, and $d$ here is the penultimate-feature dimension — hundreds to thousands — while a class has only a few thousand examples. I cannot robustly estimate a thousand-dimensional covariance from that. So I reduce dimension first: project the centered class features onto their top-$k$ singular subspace and do all robust estimation and whitening inside that $k$-dimensional space. This is safe for the poison as long as $k$ is large enough to contain the contaminated direction — the poison bump, even if not the single top direction, is *some* above-average-variance direction, so it lives within the top-$k$ subspace for a reasonable $k$ — and it makes the estimation feasible ($k^2$ samples, not $d^2$) while restoring distance contrast.

But $k$ is a real knob with failure on both sides, and the spectral rung's cifar100 zero is a warning that a fixed choice can be exactly wrong. Too small a $k$ and the subspace may not contain the poison direction, projecting the signature away before I start; too large a $k$ and I drag in clean directions that are heavy-tailed or poorly estimated from limited samples, the robust estimator mis-whitens them, and whitening *inflates* a clean direction so the score flags clean points. The sweet spot is attack-dependent — a sharp BadNets trigger has low effective dimension and wants a small $k$; a diffuse blend wants a larger $k$. Rather than sit on a knife's edge I *select* $k$ by the signal it produces: run the pipeline for a grid of candidate $k$, and for each measure how strong the residual signature is *after* whitening — the top eigenvalue of the whitened empirical covariance. A $k$ that contains the poison and estimates the clean directions well leaves a tall residual eigenvalue (the un-whitened poison); a $k$ too small (no poison) or too large (mis-whitened clean dilution) leaves a flatter whitened spectrum. I pick the $k$ that maximizes the post-whitening top eigenvalue, so the data reveals the effective dimensionality of the signature instead of my guessing it; a geometric grid from $1$ up to a cap of $64$ suffices and the choice is not sensitive to the exact grid.

After robust whitening, how do I score? The two endpoints I already know are both wrong for half the cases. Scoring by the squared whitened *norm* — the total excess across all directions — is good when the signature is *spread* (a diffuse blend) but bad when it is one sharp direction (diluted by $k-1$ unit-variance clean directions in the sum). Scoring by the squared projection onto the whitened *top eigenvector* is perfect for one sharp direction (this is the spectral rung, post-whitening) but throws the signal away when the signature is spread — precisely the cifar100 case that scored zero. I have spent this whole rung making the method handle the spread case, so I cannot commit to the top-projection, but I also do not want to lose the sharp cifar10 case to the norm score. I want a score that *interpolates* and adapts to the effective dimensionality automatically, and that is exactly the **quantum-entropy** score. Take the empirical covariance $\tilde\Sigma$ of the whitened reps, form
$$Q = \exp\!\Big(\alpha\,\frac{\tilde\Sigma - I}{\lVert\tilde\Sigma\rVert_2 - 1}\Big),$$
and score each whitened point $h_i$ by
$$\tau_i = \frac{h_i^\top Q\, h_i}{\operatorname{Tr}(Q)}.$$
Read what $Q$ does. After whitening the clean directions have variance near $1$, so $\tilde\Sigma - I$ is near zero on them and $Q$ weights them lightly; the contaminated directions have variance above $1$, so $\tilde\Sigma - I$ is positive there and the exponential weights them *heavily*. The normalization $\lVert\tilde\Sigma\rVert_2 - 1$ scales the temperature by the strength of the top excess direction so it is comparable across classes and settings. The single parameter $\alpha$ controls how aggressively $Q$ concentrates: as $\alpha\to 0$, $Q\to I$ and the score is the squared whitened norm (the spread-friendly endpoint); as $\alpha\to\infty$, $Q$ collapses onto the top whitened eigenvector and the score is the top-projection (the sharp endpoint); intermediate $\alpha$ weights each excess direction by $\exp(\alpha\cdot\text{its excess})$, a soft-max over directions that automatically follows whatever effective dimensionality the whitening exposed. A sharp BadNets trigger leaves one tall direction and $Q$ concentrates there; a diffuse blend leaves several moderately-tall directions and $Q$ spreads weight across them. I set $\alpha = 4$, large enough to emphasize excess over clean noise, small enough not to bet everything on the single top direction the spectral rung already over-bet on. The QUE score is not an arbitrary nonlinearity; it is the precise object that makes the method agnostic to how concentrated or spread the signature is, which is the entire reason I did robust whitening.

Grounding it in the scaffold: same `BackdoorDefense` contract, same per-training-label routing as the spectral rung — the target label is not exposed, so the pipeline runs on every class, grouped by cached training labels rather than predicted class, for the same reason as before (a hard poisoned point's prediction can disagree with its label). Per class I center, then for each candidate $k$ project onto the top-$k$ subspace, robustly estimate the clean mean and covariance by iterative Mahalanobis trimming, whiten by the robust $\Sigma^{-1/2}$, record the post-whitening top eigenvalue, keep the best $k$, and compute the QUE score for it. The numerical guards fall out of the construction: clip the matrix-exponential exponent at $50$ so it cannot overflow, add a $10^{-6}$ ridge so the robust covariance is invertible, skip classes with fewer than 4 points, and fall back to the squared whitened norm if the trace degenerates. `fit` does the per-class work and caches the scores; `score_samples` reads them out; the harness's fixed $1.5\,\varepsilon$ removal does the rest.

```python
# EDITABLE region of custom_backdoor_defense.py — finale: SPECTRE (Hayase et al., ICML 2021)
import numpy as np
import torch


class BackdoorDefense:
    """Per-class SPECTRE: robust-whitening + Quantum-Entropy (QUE) scoring.

    For each training label c, center its penultimate features, project onto a
    top-k SVD subspace, robustly estimate the clean mean/covariance (iterative
    Mahalanobis trimming), whiten by the robust Sigma^{-1/2}, and score each
    sample by the QUE quadratic form
        tau_i = (h_i^T Q h_i)/Tr(Q),
        Q = exp(alpha (Sigma_tilde - I)/(||Sigma_tilde||_2 - 1)),
    with Sigma_tilde the empirical covariance of the whitened reps and k chosen
    to maximize the post-whitening top eigenvalue.
    """

    def __init__(self, k=64, alpha=4.0):
        self.k = k
        self.alpha = alpha
        self.class_scores = {}        # c -> per-sample QUE scores (class order)
        self.class_indices = {}       # c -> global indices for this class
        self.cached_labels = None

    def _robust_cov(self, Y, eps, n_iter=8):
        """Iterative robust mean/cov: keep the points closest (Mahalanobis) to
        the running clean core, re-estimate, repeat."""
        Y = np.atleast_2d(Y)
        n = Y.shape[0]
        trim = min(0.5, 1.5 * eps)
        n_keep = max(2, n - int(np.floor(trim * n)))
        keep = np.ones(n, dtype=bool)
        for _ in range(n_iter):
            Yc = Y[keep]
            mu = Yc.mean(axis=0)
            sigma = np.atleast_2d(np.cov(Yc, rowvar=False, bias=False))
            sigma = sigma + 1e-6 * np.eye(sigma.shape[0])
            centered = Y - mu
            sol = np.linalg.solve(sigma, centered.T).T
            md = np.einsum("ij,ij->i", centered, sol)
            new_keep = np.zeros(n, dtype=bool)
            new_keep[np.argsort(md)[:n_keep]] = True
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        Yc = Y[keep]
        mu = Yc.mean(axis=0)
        sigma = np.atleast_2d(np.cov(Yc, rowvar=False, bias=False))
        sigma = sigma + 1e-6 * np.eye(sigma.shape[0])
        return mu, sigma

    def _que_for_k(self, Y, eps, k):
        """SPECTRE for one reduced dimension k; returns (scores, signal)."""
        n = Y.shape[0]
        _, _, vh = np.linalg.svd(Y, full_matrices=False)
        U = vh[:k].T                        # (D, k)
        T1 = Y @ U                          # (n, k)
        mu_r, sigma_r = self._robust_cov(T1, eps)
        evals, evecs = np.linalg.eigh(sigma_r)
        evals = np.clip(evals, 1e-8, None)
        inv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
        H = (T1 - mu_r) @ inv_sqrt          # (n, k) whitened
        sigma_t = (H.T @ H) / n
        top = float(np.linalg.eigvalsh(sigma_t)[-1])
        denom = max(top - 1.0, 1e-6)
        A = self.alpha * (sigma_t - np.eye(k)) / denom
        ev, evec = np.linalg.eigh(A)
        ev = np.clip(ev, None, 50.0)        # guard matrix-exp overflow
        Q = (evec * np.exp(ev)) @ evec.T
        trace = np.trace(Q)
        if trace <= 0 or not np.isfinite(trace):
            return np.einsum("ij,ij->i", H, H), top
        scores = np.einsum("ij,jk,ik->i", H, Q, H) / trace
        return scores, top

    def _que_scores(self, Y, eps):
        """Auto-select k by the post-whitening signal strength."""
        n, D = Y.shape
        if n < 4:
            return np.zeros(n)
        cap = int(min(self.k, D, n - 1))
        grid = sorted({int(round(v)) for v in np.linspace(1, np.sqrt(cap), 8) ** 2})
        grid = [k for k in grid if 1 <= k <= cap]
        best_scores, best_signal = None, -np.inf
        for k in grid:
            scores, signal = self._que_for_k(Y, eps, k)
            if signal > best_signal and np.isfinite(scores).all():
                best_signal, best_scores = signal, scores
        return best_scores if best_scores is not None else np.zeros(n)

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self.cached_labels = labels.copy()
        eps = float(poison_fraction)
        for c in np.unique(labels):
            mask = labels == c
            feat_c = features[mask]
            self.class_indices[int(c)] = np.where(mask)[0]
            if feat_c.shape[0] < 4:
                self.class_scores[int(c)] = np.zeros(len(feat_c))
                continue
            Y = feat_c - feat_c.mean(axis=0)
            self.class_scores[int(c)] = self._que_scores(Y, eps)

    def score_samples(self, features, logits):
        features = np.asarray(features, dtype=np.float64)
        scores = np.zeros(len(features), dtype=np.float64)
        for c, idx in self.class_indices.items():
            scores[idx] = self.class_scores[c]
        return scores
```
