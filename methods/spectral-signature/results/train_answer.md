I will describe the backdoor-poisoning defense I am building and explain why its central operation is a class-conditional spectral decomposition of learned representations. The canonical name for this method is Spectral Signatures, introduced by Tran, Li, and Madry at NeurIPS 2018. It is designed for the setting where a small fraction of the training data has been altered so that a chosen trigger, when present, causes a classifier to output an attacker-chosen target label, while ordinary clean inputs continue to be classified correctly. Because the attack preserves clean validation accuracy, the defender cannot rely on a simple accuracy drop to detect the problem. Instead, Spectral Signatures uses the internal geometry that the network learns during its first training run.

The starting observation is that a trigger is a very reliable statistical cue. When a network is trained on poisoned data, gradient descent has an incentive to build features that respond strongly to that trigger, because the trigger is almost perfectly predictive of the target label inside the corrupted subset. The trigger itself may be visually subtle in pixel space, but its learned representation can separate the poisoned examples from the clean examples of the target class much more cleanly than the raw images do. The defense therefore trains the model once on the suspect data, extracts a penultimate-layer feature vector for every training example, and then looks for the poison inside those feature vectors.

The key structural fact is that, within a single training label, the data are now a mixture of two populations. The majority population is the clean examples that genuinely belong to that label. The minority population is the poisoned examples that were relabeled to that label because they carry the trigger. Let me call the clean population D and the poisoned population W, and let their feature means be mu_D and mu_W, with Delta equal to mu_D minus mu_W. The mixture mean mu_F is a convex combination of the two class means, so the clean mean lies a small distance epsilon times Delta from the mixture mean, while the poisoned mean lies a larger distance (one minus epsilon) times Delta on the opposite side, where epsilon is the poison fraction within that label.

When I compute the empirical covariance of the mixed population around the mixture mean, the algebra reveals the opening. The clean points contribute their own covariance plus a small rank-one term proportional to Delta Delta^T, while the poisoned points contribute their covariance plus a larger rank-one term in the same direction. Putting the two groups together gives a covariance matrix that equals the weighted average of the within-population covariances plus epsilon times (one minus epsilon) times Delta Delta^T. That last term is a rank-one bump along the direction connecting the clean and poisoned means. If the learned representation has amplified the trigger enough, this bump can dominate the spectrum of the class covariance, even though the poison is a small minority. The top eigenvector of the covariance therefore tilts toward Delta, and the squared projection of a centered feature vector onto that eigenvector gives a natural suspicion score.

This derivation also explains why raw-pixel outlier removal often fails. In pixel space, the displacement Delta is dominated by ordinary image variation such as pose, background, lighting, and object subtype, so the rank-one bump is buried. In the learned representation, the same trigger can produce a much larger directional signal relative to the within-class variance, making the bump visible in the covariance spectrum. The defense does not need to know what the trigger looks like; it only needs the network to have learned a feature that distinguishes the triggered inputs from the clean target-class inputs.

The algorithm follows directly from this picture. For each training label, I collect the representations of the examples carrying that label and subtract the class mean. I then compute the top right singular vector of the centered class matrix, which is equivalent to the top eigenvector of the empirical class covariance. Each example is scored by the squared centered projection onto that direction. Higher scores indicate that an example is contributing disproportionately to the suspicious variance direction, which is the spectral signature of the poisoned minority. I then remove the top-scoring examples from each class, using a budget that is about one and a half times the estimated poison fraction as a safety margin, and retrain the model from scratch on the remaining data. Retraining from scratch is important because it removes the shortcut that the network previously learned from the poisoned examples.

Several implementation choices matter. The grouping must be by the cached training label, not by the model's predicted label at scoring time. If I regroup by prediction, a hard or suspicious example can be routed to the wrong class covariance and the per-class mixture structure is destroyed. The scoring is class-conditional because the poison lives inside one trained label; pooling all classes together would simply rediscover the ordinary differences between class means rather than the subtle trigger direction inside the target class. For degenerate classes with fewer than two examples, the direction is set to zero because there is no covariance to decompose. Using float64 for the SVD helps avoid numerical issues when the covariance bump is small compared with the bulk variance.

The theory gives useful guardrails but should not be overstated. Under bounded within-population covariance assumptions, the top eigenvector of the mixture covariance is guaranteed to have nontrivial correlation with Delta provided the mean gap is large enough relative to the noise level and the poison fraction. For the squared-projection score to cleanly separate clean and poisoned examples, the projected mean gap must beat a two-sided margin that grows as the poison fraction approaches one half. This is why the method is most reliable when the poison fraction within a label stays well below fifty percent, which is the usual regime in backdoor attacks. The paper states a simple population condition of order six sigma squared over epsilon, which captures the low-poison-rate intuition; a fully rigorous Chebyshev argument for the absolute-threshold version requires a slightly stronger epsilon-dependent constant, but the practical behavior remains the same.

There is a subtle distinction between the published Algorithm 1 and the reference implementation sometimes circulated with the paper. Algorithm 1 centers the features before computing the singular vector and then scores by the squared centered projection. Some released code instead centers before SVD but scores by the absolute uncentered projection of the raw features onto the learned direction, because it already knows the target label. The generic class-conditional scorer I will give below follows Algorithm 1: it centers for both SVD and scoring, and it runs on every training label, because the target label is not exposed in the general defense harness.

The main limitation of Spectral Signatures is that it looks for a single dominant direction per class. If the attack spreads its signature across several orthogonal directions, or if the poisoned minority is so small that finite-sample noise swamps the covariance bump, the top singular vector may not align well with the poison direction. It also assumes that the learned representation amplifies the trigger; against attacks whose triggers are not strongly represented in the penultimate layer, the defense may degrade. Nevertheless, the method remains a clean and theoretically grounded baseline: it replaces distance-based clustering with a spectral test, it requires no knowledge of the trigger, and it fits naturally into the standard workflow of training once, filtering, and retraining.

```python
import numpy as np


def spectral_signature_scores(features, labels):
    """Return a per-sample suspicion score using class-conditional Spectral Signatures."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    scores = np.zeros(len(features), dtype=np.float64)

    for cls in np.unique(labels):
        mask = labels == cls
        Xc = features[mask]
        mu = Xc.mean(axis=0)
        centered = Xc - mu
        if centered.shape[0] < 2:
            direction = np.zeros(features.shape[1], dtype=np.float64)
        else:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        proj = centered @ direction
        scores[mask] = proj * proj
    return scores


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_clean, n_poison = 400, 40
    dim = 10

    clean = rng.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=n_clean)
    poison = rng.multivariate_normal(mean=np.zeros(dim), cov=0.5 * np.eye(dim), size=n_poison)
    poison[:, 0] += 6.0

    X = np.vstack([clean, poison])
    labels = np.zeros(len(X), dtype=int)

    scores = spectral_signature_scores(X, labels)
    budget = int(np.ceil(1.5 * n_poison))
    threshold = np.sort(scores)[-budget]
    flagged = scores >= threshold
    true_poison = np.arange(len(X)) >= n_clean

    print("poison recall:", flagged[true_poison].mean())
    print("clean removal rate:", flagged[~true_poison].mean())
```
