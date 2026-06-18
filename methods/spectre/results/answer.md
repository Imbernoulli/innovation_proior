# SPECTRE, Distilled

SPECTRE is a target-label backdoor filter for hidden representations. It keeps the spectral-signature idea that poisoned examples create excess covariance, but first estimates a clean covariance core, whitens by that estimate, and then scores the whitened points with Quantum Entropy (QUE) scoring.

## Final Algorithm

For target-label representations `S = {h_i in R^d}_{i=1}^n`, poison fraction `epsilon`, and QUE temperature `alpha = 4`:

1. Center `S` and take a top-`k` SVD/PCA subspace.
2. Project into that subspace.
3. Robustly estimate clean mean/covariance in the reduced space.
4. Whiten all projected points by the robust covariance estimate.
5. Compute
   `Sigma_tilde = (1/n) sum_i h_tilde_i h_tilde_i^T`,
   `Q_alpha = exp(alpha * (Sigma_tilde - I) / (||Sigma_tilde||_2 - 1))`, and
   `tau_i = h_tilde_i^T Q_alpha h_tilde_i / Tr(Q_alpha)`.
6. Remove the `1.5 * epsilon * n` largest `tau_i` scores and retrain from scratch.

The published `k` identifier tries candidate dimensions and chooses the one with the largest mean QUE score after whitening by the covariance of the examples not removed by that candidate. The canonical repository uses a squared grid up to `100`: `[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]`.

## What Must Be Correct

- The mixture covariance bump is `epsilon * (1 - epsilon) * (mu_D - mu_W)(mu_D - mu_W)^T`; the sign of the mean difference is irrelevant because it is an outer product.
- Whitening is by a robust estimate of the clean covariance, not by the contaminated mixture covariance.
- The QUE exponential uses `(Sigma_tilde - I)` and divides by `||Sigma_tilde||_2 - 1`, because identity is the clean whitened baseline.
- `Q_alpha` is trace-normalized in the score. Equivalently compute `M = exp(...)`, then score `h^T M h / Tr(M)`.
- `alpha = 0` gives `(1/k)||h||^2`; `alpha -> infinity` gives squared projection onto the leading whitened eigenspace.
- In the repository code, `eps` is a poison count, not a fraction. The robust covariance filter receives `eps / n`; the final removal count is `round(Int, 1.5 * eps)`.
- The repository code filters the target-label representation file. The full algorithm description also gives a target-label identification wrapper, but `run_filters.jl` reads the target label from the experiment name.

## Canonical Code Shape

The faithful implementation is the Julia filter in `SewoongLab/spectre-defense`, especially `quantum_filters.jl`, `dkk17.jl`, `util.jl`, and `run_filters.jl`. The core shape is:

```julia
function rcov_quantum_filter(reps, eps, k, alpha=4, tau=0.1; limit1=2, limit2=1.5)
    d, n = size(reps)
    reps_pca, _ = pca(reps, k)

    if k == 1
        reps_white = reps_pca
        sigma_white = ones(1, 1)
    else
        selected = cov_estimation_iterate(
            reps_pca,
            eps / n,
            tau,
            nothing;
            limit=round(Int, limit1 * eps),
        )
        sigma_clean = cov(reps_pca[:, selected]', corrected=false)
        reps_white = sigma_clean^(-1 / 2) * reps_pca
        sigma_white = cov(reps_white')
    end

    weights = k > 1 ? exp(alpha * (sigma_white - I) / (opnorm(sigma_white) - 1)) : ones(1, 1)
    weights /= tr(weights)

    scores = [x' * weights * x for x in eachcol(reps_white)]
    remove = k_lowest_ind(-scores, round(Int, limit2 * eps))
    return .!remove
end

function rcov_auto_quantum_filter(reps, eps, alpha=4, tau=0.1; limit1=2, limit2=1.5)
    reps_pca_100, _ = pca(reps, 100)
    best_score = -Inf
    best_selected = nothing

    for k in round.(Int, range(1, sqrt(100), length=10) .^ 2)
        selected = rcov_quantum_filter(reps, eps, k, alpha, tau; limit1=limit1, limit2=limit2)
        sigma_clean = cov(reps_pca_100[:, selected]')
        sigma_white = cov((sigma_clean^(-1 / 2) * reps_pca_100)')
        weights = exp(alpha * (sigma_white - I) / (opnorm(sigma_white) - 1))
        mean_que = tr(sigma_white * weights) / tr(weights)

        if mean_que > best_score
            best_score = mean_que
            best_selected = selected
        end
    end

    return best_selected
end
```

`pca` centers the representation matrix before taking the top singular vectors. `cov_estimation_iterate` is the DKK-style covariance filter: it repeatedly removes high-norm or high-quadratic-tail points until the remaining set's covariance passes the robust Gaussian covariance test or the removal cap is reached.

## Practical Interface

The repository pipeline is:

```text
train.py EXPERIMENT
rep_saver.py EXPERIMENT
julia --project=. run_filters.jl EXPERIMENT
train.py EXPERIMENT mask-rcov-target
```

`run_filters.jl` parses the target label and poison count from the experiment name, reads `output/<name>/label_<target>_reps.npy`, runs the PCA, clustering, and SPECTRE masks, and writes `mask-rcov-target.npy` for retraining.
