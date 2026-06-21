We observe an unknown matrix $M \in \mathbb{R}^{n_1 \times n_2}$ only on a random subset $\Omega$ of $m \ll n_1 n_2$ of its entries and want the whole matrix back. With fewer samples than unknowns there are infinitely many completions, so the problem is hopeless unless something structural pins down the answer — and the thing that does is the assumption, true of recommender systems, structure-from-motion, and system identification alike, that $M$ is (approximately) low rank. Under an incoherence condition on its singular vectors, most rank-$r$ matrices are the *unique* minimum-nuclear-norm matrix consistent with the observed entries once $m \gtrsim n^{6/5} r \log n$, so the convex program

$$\text{minimize } \|X\|_* \quad \text{subject to } P_\Omega(X) = P_\Omega(M)$$

recovers $M$ exactly, where $\|X\|_* = \sum_i \sigma_i(X)$ is the sum of singular values — the $\ell_1$ norm of the spectrum — and $P_\Omega$ keeps the observed entries and zeros the rest. The nuclear norm is the right surrogate because it is the convex envelope of rank on the spectral-norm ball, the tightest convex relaxation of an NP-hard problem. So the objective is not in question; *solving it at scale* is. The way this gets done today is to recast it as a semidefinite program and feed it to an interior-point solver, but each Newton step forms and factors a dense linear system whose size grows with the problem, so these solvers choke around $n = 100$, their Newton system becomes ill-conditioned exactly when closing in on the optimum, and they throw away the one fact most worth exploiting — that the answer is described by $r(2n - r)$ numbers, not $n^2$. We need a first-order method that only ever touches the matrix through cheap operations and never assembles anything of size $n^2 \times n^2$. The penalized iterative-shrinkage route the imaging community would suggest — minimizing $\lambda\|X\|_* + \tfrac12\|P_\Omega(X) - P_\Omega(M)\|_F^2$ — does not escape the bind either: to fit the data one must take $\lambda$ small, but the shrinkage threshold is $\lambda\delta$, so small $\lambda$ means a tiny threshold, the iterates are not low rank, and the working matrix is not sparse. The very things that make each iteration cheap are forbidden by accuracy in that formulation.

I propose Singular Value Thresholding (SVT). The core primitive borrows the compressed-sensing intuition that a low-rank matrix is one whose vector of singular values is sparse, so the rank-producing operation is to *soft-threshold the singular values*: for $Y = U\Sigma V^*$, define the singular value shrinkage operator

$$D_\tau(Y) = U\,\mathrm{diag}\big((\sigma_i - \tau)_+\big)\,V^*, \qquad t_+ = \max(t, 0),$$

which shrinks each singular value toward zero and drops those that fall to zero, adaptively discovering the basis in which the matrix is sparse rather than thresholding in a fixed transform. What makes this principled rather than ad hoc is that $D_\tau$ is exactly the proximity operator of the nuclear norm, $D_\tau(Y) = \arg\min_X \{\tfrac12\|X-Y\|_F^2 + \tau\|X\|_*\}$ — the matrix analogue of scalar soft-thresholding being the prox of $\ell_1$. The proof is the load-bearing step: the objective is strictly convex with unique minimizer characterized by $0 \in \hat X - Y + \tau\,\partial\|\hat X\|_*$, and using the nuclear-norm subdifferential $\partial\|X\|_* = \{UV^* + W : U^*W = 0,\ WV = 0,\ \|W\|_2 \le 1\}$, one splits $Y$ at the threshold into the part $U_0\Sigma_0 V_0^*$ above $\tau$ and $U_1\Sigma_1 V_1^*$ at or below, so that $Y - D_\tau(Y) = \tau(U_0 V_0^* + W)$ with $W = \tau^{-1}U_1\Sigma_1 V_1^*$; here $U_0 V_0^*$ is precisely the $UV^*$ term of $D_\tau(Y)$, $W$ is orthogonal to both subspaces, and $\|W\|_2 = \tau^{-1}\max(\Sigma_1) \le 1$ since every singular value in $\Sigma_1$ is at most $\tau$ — so the inclusion holds and $D_\tau$ is the exact prox.

To respect the hard constraint without the penalized formulation's trap, the key design choice is to let the data residual accumulate on its own track and decouple the threshold from the step size. We keep a matrix $Y$ that grows only by the residual and read $X$ off it by shrinking:

$$X^k = D_\tau(Y^{k-1}), \qquad Y^k = Y^{k-1} + \delta_k\,P_\Omega(M - X^k), \qquad Y^0 = 0.$$

Now $\tau$ is a fixed large constant of its own and $Y$ is a running sum of residuals, not $X^k$ plus a residual. This is what reading the iteration as Uzawa's dual ascent reveals: for $\min_X f(X)$ s.t. $P_\Omega(X) = P_\Omega(M)$ with Lagrangian $L(X,Y) = f(X) + \langle Y, P_\Omega(M-X)\rangle$, the dual step is exactly $Y^k = Y^{k-1} + \delta_k P_\Omega(M - X^k)$, and reverse-engineering which $f$ makes the inner minimization a single shrink (since every $Y^k$ stays supported on $\Omega$, so $P_\Omega Y = Y$) yields

$$f(X) = \tau\|X\|_* + \tfrac12\|X\|_F^2.$$

So SVT solves $\min\ \tau\|X\|_* + \tfrac12\|X\|_F^2$ s.t. $P_\Omega(X) = P_\Omega(M)$. The extra $\tfrac12\|X\|_F^2$ term is the price of a closed-form inner step, and as a bonus it makes $f$ strongly convex, so the solution is unique. It also means we are not minimizing the bare nuclear norm — but that gap vanishes as $\tau \to \infty$: comparing optimality of $X_\tau$ (the proximal minimizer) and $X_\infty$ (the minimum-Frobenius solution among nuclear-norm minimizers) gives $\|X_\tau\|_F^2 \le \|X_\infty\|_F^2$ uniformly, and passing to a convergent subsequence forces $X_\tau \to X_\infty$. So large $\tau$ recovers the genuine nuclear-norm answer — and large $\tau$ is *also* exactly what makes the iterates low rank and $Y^k$ sparse. The two goals that fought each other in the penalized version now agree: crank $\tau$ up.

Convergence rests on the strong convexity of $f$. For subgradients $Z = \tau Z_0 + X \in \partial f(X)$ and $Z'$, the Frobenius part contributes $\|X-X'\|_F^2$ and the nuclear part is nonnegative by duality ($\langle Z_0, X\rangle = \|X\|_*$, $\|Z_0\|_2 \le 1$), giving the strong-monotonicity lemma $\langle Z - Z', X - X'\rangle \ge \|X - X'\|_F^2$. Tracking the dual distance $r_k = \|P_\Omega(Y^k - Y^*)\|_F$, expanding its square, using this lemma on the cross term and $\|P_\Omega(\cdot)\|_F \le \|\cdot\|_F$ on the quadratic term yields

$$r_k^2 \le r_{k-1}^2 - (2\delta_k - \delta_k^2)\,\|X^k - X^*\|_F^2,$$

so for $0 < \inf \delta_k \le \sup \delta_k < 2$ the $r_k$ are nonincreasing and telescoping forces $\|X^k - X^*\|_F \to 0$. The bound $\delta < 2$ is safe but slow; it came from $\|P_\Omega(X^* - X^k)\|_F^2 \le \|X^* - X^k\|_F^2$, but the near-isometry $\|P_\Omega(A)\|_F^2 \approx p\|A\|_F^2$ with $p = m/(n_1 n_2)$ suggests that projection term is really about $p\|X^* - X^k\|_F^2$, loosening the sufficient condition to roughly $\delta < 2/p$. With a safety factor we take the aggressive $\delta = 1.2/p = 1.2\,n_1 n_2/m$ — not a theorem, since $X^* - X^k$ depends on $\Omega$ and the isometry need not apply, but empirically convergent and far faster. For the threshold we calibrate against the standard synthetic generator $M = M_L M_R^*$ with Gaussian factors, where $\|M\|_F \approx n\sqrt r$ and $\|M\|_* \approx nr$, so the ratio of the two terms is $\tau\|M\|_*/(\tfrac12\|M\|_F^2) \approx 2\tau/n$; setting it to about $10$ gives $\tau = 5n$ (and $\tau = 5\sqrt{n_1 n_2}$ in the rectangular case). Two final refinements: a warm start that leaps over the predetermined opening steps, since while $k\delta\|P_\Omega(M)\|_2 \le \tau$ the shrink kills everything and $Y^k = k\delta P_\Omega(M)$, so we define $k_0$ by $\tau/(\delta\|P_\Omega(M)\|_2) \in (k_0 - 1, k_0]$ and start at $Y^0 = k_0\delta P_\Omega(M)$; and a stopping rule from the KKT conditions ($X = D_\tau(Y)$ holds by construction, leaving only $P_\Omega(X-M)=0$), namely the relative residual $\|P_\Omega(X^k - M)\|_F/\|P_\Omega(M)\|_F \le \varepsilon$, which by the near-isometry tracks the true reconstruction error on the unseen entries. What makes the whole thing scale is that $Y^k$ stays supported on $\Omega$ (sparse, $m$ nonzeros, $O(m)$ residual update) and large $\tau$ keeps $X^k$ low rank, so only the singular triplets above $\tau$ are needed — a partial Lanczos SVD at large scale, where a dense SVD suffices for moderate $n$.

```python
import math
import torch


class NuclearNormSVT:
    """Singular Value Thresholding for matrix completion."""

    def __init__(self, tau_factor=5.0, delta_factor=1.2, tol=1e-4):
        self.tau_factor = float(tau_factor)      # tau = tau_factor * n   (=> 5n)
        self.delta_factor = float(delta_factor)  # delta = delta_factor / p  (=> 1.2/p)
        self.tol = float(tol)                    # relative-residual stop on Omega

    @torch.no_grad()
    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        mask = observed_mask.to(device).to(torch.float32)
        M_obs = observed_values.to(device).to(torch.float32)      # P_Omega(M)
        n_observed = max(int(mask.sum().item()), 1)
        p = n_observed / float(n * n)                             # observation ratio

        tau = self.tau_factor * float(n)                          # tau = 5n
        delta = self.delta_factor / max(p, 1e-6)                  # delta = 1.2 / p

        # Warm start: jump over the trivial steps where the shrink kills everything.
        norm_proj_m = float(torch.linalg.matrix_norm(M_obs, ord=2).item())
        k0 = max(1, math.ceil(tau / (delta * max(norm_proj_m, 1e-6))))
        Y = (k0 * delta) * M_obs                                  # Y^0 = k_0 * delta * P_Omega(M)

        X = torch.zeros_like(M_obs)
        norm_obs = max(float(M_obs.norm().item()), 1e-6)
        log_every = max(int(log_iters), 1)
        for it in range(1, max_iters + 1):
            # X^k = D_tau(Y^{k-1}): soft-threshold the singular values.
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            S_thresh = torch.clamp(S - tau, min=0.0)             # (sigma_i - tau)_+
            X = (U * S_thresh) @ Vh

            # Stop on the same relative residual used by Algorithm 1 and SVT.m.
            residual = (M_obs - X) * mask
            rel_res = float(residual.norm().item() / norm_obs)
            train_mse = float(residual.pow(2).sum().item() / float(n_observed))
            if it == 1 or it % log_every == 0 or it == max_iters or rel_res <= self.tol:
                print(
                    f"TRAIN_METRICS iter={it} rel_res={rel_res:.6e} "
                    f"train_mse={train_mse:.6e}",
                    flush=True,
                )
            if rel_res <= self.tol:
                break

            # Dual / residual update, supported on Omega.
            Y = Y + delta * residual

        return X.detach().cpu()


def build_strategy():
    return NuclearNormSVT(tau_factor=5.0, delta_factor=1.2, tol=1e-4)
```
