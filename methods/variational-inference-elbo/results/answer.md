# Variational inference and the evidence lower bound (ELBO)

## The problem

Given a model with observed data `x` and latent variables `z` and a joint `p(x, z)` that is
cheap to evaluate, we want the posterior `p(z | x) = p(x, z) / p(x)` and the evidence
`p(x) = ∫ p(x, z) dz`. The evidence is intractable — an exponential sum or high-dimensional
integral — so the posterior cannot be normalized and the model cannot be scored directly.

## The key idea

Stop computing the posterior; **approximate it by optimization**. Posit a tractable family of
distributions `q(z)` and find the member closest to the true posterior in KL divergence:

    q* = argmin_q KL(q(z) ‖ p(z | x)).

The direction `KL(q ‖ p)` is chosen because it is an expectation under the tractable `q`, not
under the intractable posterior. This objective still contains the intractable `log p(x)`, but
that term is constant in `q`, and the following identity removes it.

## The ELBO decomposition

Substituting `p(z | x) = p(x, z) / p(x)` into the KL and using that `log p(x)` is constant
under `E_q`:

    log p(x) = E_q[log p(x, z)] − E_q[log q(z)]  +  KL(q ‖ p(z | x))
             =  ELBO(q)  +  KL(q ‖ p(z | x)),

where

    ELBO(q) = E_q[log p(x, z)] − E_q[log q(z)] = E_q[log p(x, z)] + H(q).

Because `KL ≥ 0`, the ELBO is a lower bound on `log p(x)` (hence the name). Because `log p(x)`
is constant, **maximizing the ELBO is exactly minimizing `KL(q ‖ p(z | x))`**, and the gap
between the bound and the log-evidence is precisely that KL — so a tighter bound is a better
posterior approximation, and the maximized ELBO is the best lower-bound surrogate for
`log p(x)` within the chosen family. The ELBO uses only the tractable joint `p(x, z)` and the
entropy of the chosen `q`; it never references `p(x)` or `p(z | x)`. Equivalently, Jensen's
inequality applied to `log p(x) = log E_q[p(x,z)/q(z)]` gives the same lower bound with slack
`KL(q ‖ p(z | x))`. In the convex-duality view, `log Σ_z exp(u_z)` has conjugate
`Σ_z q_z log q_z` on the probability simplex and `+∞` outside it, so the log-sum-exp evidence
has the dual form `max_q {qᵀu − Σ_z q_z log q_z}`.

## Mean-field family and the CAVI update

Take the fully factorized ("mean-field") family

    q(z) = ∏_{j=1}^m q_j(z_j),

which makes the ELBO separable. Optimizing one factor at a time with the others fixed
(coordinate ascent) reduces the per-factor subproblem to a single KL minimization, whose
closed-form optimum is

    q*_j(z_j) ∝ exp{ E_{−j}[ log p(x, z) ] }
              ∝ exp{ E_{−j}[ log p(z_j | z_{−j}, x) ] },

the exponentiated expectation (over the other factors) of the log of `z_j`'s **complete
conditional** — the same complete conditional the Gibbs sampler draws from. Cycling these
updates raises the ELBO monotonically to a local optimum. This is **coordinate ascent
variational inference (CAVI)**:

```
Initialize variational factors q_j(z_j)
repeat
    for j = 1..m:
        q_j(z_j) ∝ exp{ E_{−j}[ log p(z_j | z_{−j}, x) ] }
    compute ELBO(q) = E_q[log p(x,z)] − E_q[log q(z)]
until ELBO converges
```

## Conjugate-exponential closed form

If each complete conditional is in the exponential family,
`p(z_j | z_{−j}, x) = h(z_j) exp{η_j(z_{−j}, x)ᵀ z_j − a(η_j)}`, then the optimal factor is in
the **same** family with natural parameter set to the expected natural parameter:

    q*_j(z_j) = h(z_j) exp{ ν_jᵀ z_j − a(ν_j) },     ν_j = E_{−j}[ η_j(z_{−j}, x) ].

For a conjugate Bayesian model with global `β` (conjugate prior) and locals `z_i`,
`p(β, z, x) = p(β) ∏_i p(z_i, x_i | β)`, the updates are closed-form:

    local:   ϕ_i = E_{q(β)}[ η(β, x_i) ]
    global:  λ   = α + Σ_i E_{q(z_i)}[ t(z_i, x_i) ].

## Link to EM

With model parameters `θ`, `L(q, θ) = E_q[log p(x, z | θ)] + H(q)` lower-bounds
`log p(x | θ)` with gap `KL(q ‖ p(z | x, θ))`. Coordinate ascent on `L`:

- **E step** (max over `q`, current `θ` fixed): optimal `q = p(z | x, θ)`; the bound becomes tight.
- **M step** (max over `θ`, `q` fixed): maximizes `E_q[log p(x, z | θ)]`, the expected
  complete-data log-likelihood.

This is exactly the **EM algorithm**, recovered as the special case where the variational
family is rich enough to contain the true posterior so the E step is exact. When the posterior
is intractable, the mean-field variational E step (CAVI) replaces the exact E step — still
coordinate ascent on the same bound, over a restricted family.

## Worked instance: Bayesian mixture of Gaussians

`K` components with means `μ_k ~ N(0, σ²)`; each point `x_i` has assignment `c_i` and
`x_i ~ N(c_iᵀ μ, 1)`. Mean-field family `q(μ, c) = ∏_k N(μ_k; m_k, s_k²) ∏_i Cat(c_i; ϕ_i)`.
The CAVI updates derived from `q*_j ∝ exp{E_{−j}[log p(x,z)]}` are:

    assignments:  ϕ_ik ∝ exp{ E[μ_k] x_i − E[μ_k²]/2 },   E[μ_k]=m_k, E[μ_k²]=m_k²+s_k²
    means:        m_k = (Σ_i ϕ_ik x_i) / (1/σ² + Σ_i ϕ_ik),   s_k² = 1 / (1/σ² + Σ_i ϕ_ik)

— soft responsibilities and a responsibility-weighted conjugate Gaussian posterior.

```python
import numpy as np

# CAVI for a Bayesian mixture of K unit-variance Gaussians with N(0, sigma^2) prior on each
# component mean. Latents: global means mu_k (Gaussian factor), local assignments c_i
# (categorical factor). Each step is the closed-form mean-field update
# q_j ∝ exp{ E_{-j}[ log p(x,z) ] }, i.e. set each factor's natural parameter to the expected
# natural parameter of the corresponding complete conditional. We climb the ELBO.

def elbo(x, m, s2, phi, sigma2):
    # ELBO = E_q[log p(x, z)] + H(q): the bound we maximize and use to check convergence.
    x = np.asarray(x, dtype=float)
    n, K = phi.shape
    Emu, Emu2 = m, m**2 + s2
    e_log_lik = (np.sum(phi * (np.outer(x, Emu) - 0.5 * Emu2[None, :]))
                 - 0.5 * np.sum(x**2) - 0.5 * n * np.log(2*np.pi))
    e_log_prior_c  = -n * np.log(K)
    e_log_prior_mu = np.sum(-0.5*np.log(2*np.pi*sigma2) - 0.5*Emu2/sigma2)
    e_log_joint = e_log_lik + e_log_prior_c + e_log_prior_mu
    H_mu = np.sum(0.5 * np.log(2*np.pi*np.e*s2))
    safe_phi = np.where(phi > 0, phi, 1.0)
    H_c  = -np.sum(np.where(phi > 0, phi * np.log(safe_phi), 0.0))
    return e_log_joint + H_mu + H_c

def cavi_gmm(x, K, sigma2=10.0, max_iters=200, tol=1e-6, seed=0, return_history=False):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    m   = rng.normal(np.mean(x), np.std(x) + 1e-3, size=K)  # component-mean means
    s2  = np.ones(K)                                        # component-mean variances
    phi = rng.dirichlet(np.ones(K), size=n)                # soft assignments
    history = [elbo(x, m, s2, phi, sigma2)]
    for _ in range(max_iters):
        # local update (E-step-like): phi_ik ∝ exp{ E[mu_k] x_i - E[mu_k^2]/2 }
        Emu, Emu2 = m, m**2 + s2
        log_phi = np.outer(x, Emu) - 0.5 * Emu2[None, :]
        log_phi -= log_phi.max(axis=1, keepdims=True)      # log-sum-exp stabilization
        phi = np.exp(log_phi); phi /= phi.sum(axis=1, keepdims=True)
        # global update (M-step-like): responsibility-weighted conjugate Gaussian posterior
        Nk = phi.sum(axis=0)
        s2 = 1.0 / (1.0/sigma2 + Nk)
        m  = (phi.T @ x) * s2
        L = elbo(x, m, s2, phi, sigma2)                    # monitor the bound
        improvement = L - history[-1]
        if improvement < -1e-8:
            raise FloatingPointError("ELBO decreased; check the coordinate updates.")
        history.append(L)
        if improvement < tol:
            break
    if return_history:
        return m, s2, phi, np.array(history)
    return m, s2, phi

def fit_cavi_gmm(x, K, sigma2=10.0, max_iters=200, tol=1e-6, n_init=10, seed=0,
                 return_history=False):
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        run_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        m, s2, phi, history = cavi_gmm(
            x, K, sigma2=sigma2, max_iters=max_iters, tol=tol,
            seed=run_seed, return_history=True
        )
        score = history[-1]
        if best is None or score > best[0]:
            best = (score, m.copy(), s2.copy(), phi.copy(), history.copy())
    if return_history:
        return best[1], best[2], best[3], best[4]
    return best[1], best[2], best[3]
```

CAVI is non-convex in the variational parameters, so it is run from several initializations and
the highest-ELBO solution is kept (a higher ELBO means a smaller KL to the true posterior). The
fitted `q` is then used as a proxy for the posterior — most-likely assignments
`ĉ_i = argmax_k ϕ_ik`, component means `m_k`, and an approximate predictive density.
