The problem is to certify a fundamental limit for statistical estimation. For a model family, a target parameter, and a loss, we care about the minimax risk, which takes an infimum over all estimators and a supremum over all parameters. An upper bound only says that one estimator is good enough; it cannot rule out a better estimator that has not yet been imagined. To prove that no procedure can beat a certain error scale, the argument must be independent of any particular algorithm and instead show that the data themselves do not contain enough information to resolve the parameter at the desired resolution.

The central difficulty is that the risk quantifies over every measurable estimator. We cannot attack one rule at a time. We need a single reduction that forces every estimator to fail whenever the data channel is too noisy. The classical tool is the Fano minimax method, which converts an estimation problem into a testing problem on a carefully constructed finite set of alternatives. This method is canonical because it cleanly connects geometric separation in the parameter space with the information that the sample carries about the hidden truth.

The Fano minimax method proceeds in three conceptual steps. First, we choose a finite hard set of parameters that are well separated under the loss metric. Specifically, we pick parameters θ_1, ..., θ_M such that any two distinct members are at least 2ε apart. Second, we draw an index V uniformly from this set and generate the data from the distribution indexed by θ_V. Third, we observe that any estimator θ̂ naturally induces a test for V by rounding θ̂ to the nearest hard-set parameter. If θ̂ lands within ε of the true θ_V, the rounding recovers V exactly, because every other hard-set parameter is at least 2ε away. Therefore, failure to identify V implies estimation error of at least ε.

Once this reduction is in place, Fano's inequality gives a lower bound on the probability that any test fails to recover V. For a uniform index over M alternatives, the probability of error is at least 1 minus the ratio of the mutual information I(V; X) plus a small binary-entropy constant to log M. Combining the reduction and Fano's inequality yields the basic lower bound: the minimax risk is at least Φ(ε) times the Fano error probability, where Φ is a nondecreasing loss transform. The remaining analytical work in any application is to construct a large packing with small mutual information. A common way to bound the mutual information is by averaging KL divergences, either against a reference distribution or pairwise over the hard set.

A refined form, sometimes called the Duchi-Wainwright corollary, avoids exact index recovery. Instead of asking whether the estimator identifies the exact index, it asks whether the estimator lands inside a neighborhood of radius t in an index metric. The effective number of distinguishable alternatives then becomes the total number of alternatives divided by the largest neighborhood size, and the separation scale is the parameter distance guaranteed for indices farther than t apart. This generalization recovers the classical packing proof as the special case t = 0 and can simplify proofs where the parameter space has nontrivial local geometry.

The method has limitations. It often gives constant-probability or weak-converse lower bounds, and the quality of the bound depends on a careful choice of packing and on a sharp information bound. In adaptive or sequential settings, controlling the mutual information can become more involved. Nevertheless, when the parameter space contains many separated alternatives whose induced data laws remain statistically close, the Fano minimax method is the standard way to prove that the obstacle is not a missing algorithm but the information content of the sample.

```python
import numpy as np
from scipy.spatial.distance import cdist


def fano_minimax_bound(packings, samples, epsilon, phi=lambda x: x, radius=0.0, index_metric=None):
    """
    Compute the Fano minimax lower bound from a finite hard set.

    Parameters
    ----------
    packings : array-like, shape (M, d)
        Hard-set parameters theta_1, ..., theta_M. Must be a 2*epsilon-packing
        under the estimation metric when radius == 0.
    samples : list of array-like, length M
        samples[i] contains observations drawn from P_{theta_i}.
    epsilon : float
        Separation scale used in the bound.
    phi : callable
        Nondecreasing loss transform.
    radius : float
        Approximate recovery radius for the Duchi-Wainwright form.
    index_metric : callable or None
        Metric on indices for the Duchi-Wainwright form. If None, uses
        indicator of inequality and assumes radius == 0.

    Returns
    -------
    float
        Lower bound on the minimax risk.
    """
    packings = np.asarray(packings, dtype=float)
    M = packings.shape[0]

    # Estimate each P_v by its empirical distribution over the provided samples.
    # Mutual information I(V; X) is bounded by the average KL to a reference Q.
    # Here we use the uniform mixture as Q and approximate KL with a histogram.
    def empirical_histogram(x, bins):
        probs, _ = np.histogram(x, bins=bins, density=False)
        probs = (probs + 1e-12) / (probs.sum() + 1e-12 * len(probs))
        return probs

    all_data = np.concatenate(samples)
    bins = np.histogram_bin_edges(all_data, bins="auto")
    histograms = [empirical_histogram(np.asarray(s), bins) for s in samples]
    Q = np.mean(histograms, axis=0)

    # KL divergence D(P_v || Q) averaged over v.
    def kl(p, q):
        return np.sum(p * np.log(p / q))

    avg_kl = np.mean([kl(h, Q) for h in histograms])
    mutual_info_bound = avg_kl

    if radius == 0.0:
        # Classical Fano packing bound.
        log_card = np.log(M)
        fano_prob = max(0.0, 1.0 - (mutual_info_bound + np.log(2)) / log_card)
        return phi(epsilon) * fano_prob

    # Duchi-Wainwright approximate recovery form.
    if index_metric is None:
        # Default: indices are distinct iff different, so distance is 0/1.
        index_metric = lambda i, j: 0.0 if i == j else 1.0

    # Largest number of index points within radius t of any index.
    dist_mat = np.array([[index_metric(i, j) for j in range(M)] for i in range(M)])
    N_t_max = max(np.sum(row <= radius) for row in dist_mat)

    if M <= N_t_max:
        return 0.0

    delta_t = epsilon  # caller should supply the separation for indices > radius apart.
    log_card = np.log(M / N_t_max)
    fano_prob = max(0.0, 1.0 - (mutual_info_bound + np.log(2)) / log_card)
    return phi(delta_t / 2) * fano_prob


# Example: Gaussian location model.
if __name__ == "__main__":
    np.random.seed(0)
    n = 100
    sigma = 1.0
    # Construct a 2*epsilon-packing in R^1 by placing points 2*epsilon apart.
    eps = 0.5
    M = 4
    centers = np.arange(M) * 2 * eps
    samples = [np.random.normal(loc=c, scale=sigma, size=n) for c in centers]
    bound = fano_minimax_bound(centers, samples, epsilon=eps)
    print("Fano minimax lower bound:", bound)
```
