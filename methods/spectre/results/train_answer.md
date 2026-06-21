A user has trained an image classifier on data they cannot trust. A small fraction of the training examples may have been stamped with an attacker-chosen trigger and relabeled to a target class, so the resulting network keeps ordinary clean accuracy yet maps any triggered input to the target label. The defender holds the corrupted training set, the trained model, and an upper bound on the corruption fraction, but no clean validation data, no knowledge of the trigger, and no visible label mismatch. The only useful trace of the attack lives in the hidden representations of the already-trained network, where the poisoned examples behave like a shifted subpopulation inside the target class. The reason a spectral approach can find them at all comes straight from the mixture algebra: if clean target-class representations have mean $\mu_D$ and the poisoned ones have mean $\mu_W$ at fraction $\epsilon$, then the class covariance picks up the rank-one between-population term $\epsilon(1-\epsilon)(\mu_D-\mu_W)(\mu_D-\mu_W)^\top$ on top of the within-clean and within-poison covariances. The sign of $\mu_D-\mu_W$ is irrelevant because the term is an outer product; the attack simply injects extra variance along the clean-poison separation direction. Spectral Signatures turns this into a cheap filter — center the class representations, take the top singular direction, score each point by its squared projection — and it needs no clean data. But the same formula exposes its weakness: the top singular direction is the direction of largest *total* variance in the combined cloud, not the direction that separates poison from clean. Real representations are not isotropic. If the trigger-induced shift sits in a low-clean-variance direction, the separation can be real while never being the loudest direction, and the filter projects onto an unrelated clean axis where poison and clean scores blend together. A multi-way trigger makes this failure systematic: splitting the poison across two or three trigger variants smears the mean shift into several weak directions, so a squared top projection catches the strongest one and misses the rest, while a pure squared-norm score dilutes a one-direction signature among many ordinary directions. Activation Clustering and confidence/loss filtering fare no better — the former is unstable when the poison signature is diffuse or ordinary class variation dominates, and the latter is blind once the poisoned model fits both clean and triggered examples confidently.

What we need is two changes: make the clean directions comparable before hunting for excess variance, and then score that excess without committing to either the squared-norm endpoint or the single-direction endpoint. I propose SPECTRE, a target-label, representation-space, two-stage filter that keeps the spectral insight but removes the brittle assumption that the poison is already the loudest direction. The first change is whitening, but whitening done correctly. If I had the clean target-label covariance $\Sigma_{\text{clean}}$, I would map each representation $h$ to $\Sigma_{\text{clean}}^{-1/2}(h-\mu_{\text{clean}})$; in those coordinates clean data has roughly identity covariance, so there is no high-variance clean direction left for the poison to hide behind, and the contaminated directions are exactly those where the whitened covariance of *all* target-label points rises above the identity. The crucial detail is that I must not whiten by the contaminated mixture covariance itself, because that would flatten the poison bump along with everything else; the covariance has to come from a robust estimate of the clean core. High-dimensional robust statistics permits this in principle — under $\epsilon$-corruption a robust covariance estimator recovers the clean Gaussian's covariance up to small error — but only with enough samples, on the order of $d^2/\epsilon^2$ up to logarithmic factors for dimension $d$. A CIFAR target class has roughly five thousand examples while the representation can have thousands of coordinates, so full-dimensional robust estimation is not merely expensive, it is statistically underdetermined. Hence the first operational step is dimension reduction: center the target-label representations, take a top-$k$ SVD/PCA subspace, and run robust covariance estimation only in that $k$-dimensional system.

That introduces a genuine choice of $k$. Too small and I project away the poison separation before robust estimation even begins; too large and I include low-sample, heavy-tailed clean directions where the robust filter is unreliable and whitening amplifies clean artifacts. A fixed moderate dimension is unsafe because different attacks have different effective dimensions, and — this is the binding constraint — the choice must use only observable quantities, never the poison labels. The $k$-identifier resolves this with a self-consistency test. For each candidate $k$ I run the full detector and remove its $1.5\,\epsilon n$ highest-scoring points; then I take the survivors, estimate their covariance in a fixed top-$k_{\max}$ PCA system (the canonical grid uses $k_{\max}=100$), whiten *all* target-label points by that survivor covariance, compute QUE scores, and average them. If $k$ truly found the poison, the removed set leaves a survivor covariance close to clean, and whitening by it turns the remaining poisoned points into loud outliers, so the average score is high; if $k$ missed the poison, the average is weak. So I sweep the squared grid $[1,4,9,16,25,36,49,64,81,100]$ and select the $k$ with the largest mean QUE objective — not the largest raw top eigenvalue, which would just reproduce the spectral-signature blind spot. The robust covariance step inside each candidate is the Diakonikolas-style covariance filter: after PCA centering it computes the current empirical covariance, whitens by its inverse square root, first removes points whose whitened norm is excessively large, then searches over even quadratic polynomials by an implicit Kronecker operator; if the worst quadratic variance is within the robust threshold it accepts the current Gaussian, otherwise it finds a tail threshold for that quadratic score and removes the points beyond it. This robust-estimation removal is capped at $\mathrm{round}(2\,\text{eps})$ and uses the corruption fraction $\text{eps}/n$ internally, which is a separate budget from the final defense removal of $\mathrm{round}(1.5\,\text{eps})$.

With a robust survivor set in the PCA subspace, the score has to handle both concentrated and spread signatures, and this is where Quantum Entropy scoring enters. Let the whitened vectors be $\tilde h_i \in \mathbb{R}^k$ and let $\tilde\Sigma = \frac{1}{n}\sum_i \tilde h_i \tilde h_i^\top$. Define the score matrix and per-point score
$$Q_\alpha = \exp\!\left(\alpha\,\frac{\tilde\Sigma - I}{\|\tilde\Sigma\|_2 - 1}\right), \qquad \tau_i = \frac{\tilde h_i^\top Q_\alpha \tilde h_i}{\operatorname{Tr} Q_\alpha}.$$
Every piece here is deliberate. The exponential uses $\tilde\Sigma - I$, not $\tilde\Sigma$, because after whitening the identity is the clean baseline, so only the excess covariance over the identity should drive the emphasis. The denominator is $\|\tilde\Sigma\|_2 - 1$, again subtracting the baseline, so that the scale is set by how far the worst direction rises *above* one rather than by its absolute magnitude. The matrix exponential gives a soft, temperature-controlled emphasis on the high-excess-variance eigenspaces: at $\alpha = 0$, $Q_\alpha = I$ and the trace-normalized score collapses to $\frac{1}{k}\|\tilde h_i\|^2$, the squared-norm endpoint; as $\alpha \to \infty$ it concentrates on the leading whitened eigenspace and approaches squared projection onto that single direction, the spectral-signature endpoint. The default $\alpha = 4$ sits between them as a soft maximum over excess-covariance directions, which is exactly what catches a multi-direction poison signature that either endpoint would miss. The score is trace-normalized so that $\tau_i$ is a calibrated nonnegative quadratic form; higher $\tau_i$ means more suspicious. In the implementation this ranking is done by calling `k_lowest_ind` on the negated scores, so the routine returns a keep-mask and the removal mask is its complement. The complete detector is then: center the target-label representations, project to the candidate PCA dimension, robustly select a clean covariance core, whiten by it, compute QUE scores, remove the $1.5\,\epsilon n$ budget; wrap that in the $k$-identifier that reruns over the squared grid and keeps the candidate whose survivor-covariance whitening yields the largest average QUE score; then retrain from scratch on the retained set.

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
