**Problem.** Spectral signatures raised recall off the floor (0.2688 on cifar10) but `asr` stayed pinned
near 1.0 and cifar100 recall was 0.0000 — the single top eigenvector catches only poison aligned with
that one direction, and an ill-conditioned clean covariance or a diffuse/m-way trigger hides the
signature in directions it never inspects. I need the poison's excess variance to stand out *wherever* it
sits in the spectrum.

**Key idea.** Whiten each class by an estimate of its **clean** covariance: the clean cloud becomes
isotropic (unit variance everywhere), so the poison's excess variance is the only above-unit structure
left, exposed regardless of which clean direction it overlaps. Since there is no clean reference,
estimate the clean covariance **robustly** (iterative Mahalanobis trimming to a clean core), inside a
top-`k` SVD subspace so the estimation is not sample-starved. Then score with the **Quantum-Entropy
(QUE)** quadratic form `tau_i = (h_i^T Q h_i)/Tr(Q)`, `Q = exp(alpha (Sigma_tilde - I)/(||Sigma_tilde||_2
- 1))`, which weights each whitened direction by `exp(alpha * its excess variance)`.

**Why.** Whitening turns "is the signature the top direction?" (uncontrollable) into "is there any
residual high-variance direction?" (yes whenever poison is present). QUE's `alpha` interpolates between
the squared whitened norm (`alpha->0`, spread signatures) and the squared top-projection (`alpha->inf`,
sharp signatures), so a single rule handles both a sharp BadNets pixel and a diffuse blend. `k` is
auto-selected by the post-whitening top eigenvalue, letting the data reveal the attack's effective
dimensionality rather than committing to a fixed number that failed on cifar100.

**What the harness exposes.** Same `BackdoorDefense` contract, per-training-label routing (target label
not exposed; group by cached training labels, not predicted class). The robust estimator is iterative
trimming (a self-contained stand-in for the paper's ROBUSTEST); model access and a clean oracle are not
available, so everything is computed from the penultimate features the harness extracts.

**Hyperparameters.** `alpha = 4` (QUE temperature; robust across {2,4,8}); `k` auto-selected over a
geometric grid up to 64; robust trim `~1.5*eps`, few iterations; ridge `1e-6`; exponent clip `50`;
classes `< 4` points scored zero. Removal budget `1.5*eps` fixed by the harness.

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
