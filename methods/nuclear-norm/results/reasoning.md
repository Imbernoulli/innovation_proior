Let me start from what actually blocks me. I have a matrix `M` I can only see on a random subset `Omega` of its entries, far fewer than `n^2` of them, and I want the whole thing back. With fewer samples than unknowns there are infinitely many completions, so on its own this is hopeless — but I get to assume `M` is (approximately) low rank, and that assumption is enough: Candès and Recht showed that under an incoherence condition on the singular vectors, most rank-`r` matrices are the *unique* minimum-nuclear-norm matrix that fits the observed entries, recoverable once the number of samples is on the order of `n^{6/5} r log n`. So I already know *what* to solve,

  minimize ||X||_* subject to P_Omega(X) = P_Omega(M),

where `||X||_* = Σ σ_i(X)` is the sum of singular values, the `l1` norm of the spectrum, and `P_Omega` keeps the observed entries and zeros the rest. The nuclear norm is the right surrogate because it is the convex envelope of rank on the spectral-norm ball — the tightest convex thing under rank — and the program is convex, even recastable as a semidefinite program. So the objective isn't in question. What's in question is solving it at the scale I care about: `n` in the thousands, hundreds of millions of unknowns, a fraction of a percent observed.

And that's where I'm stuck. The way this gets solved today is to feed the SDP to an interior-point solver — SDPT3, SeDuMi. They work, but each Newton step forms and factors a dense linear system whose size grows with the problem, so they choke around `n = 100`; the Newton system gets badly conditioned right when I'm closing in on the optimum; and worst of all they throw away the one fact I most want to exploit — that the answer has low rank, so it's described by `r(2n - r)` numbers, not `n^2`. I need a *first-order* method, something that only ever touches the matrix through cheap operations and never assembles anything `n^2 × n^2`. So let me stop thinking about the constrained program as a monolith and ask what cheap, repeatable operation could possibly drive me toward a low-rank, data-consistent matrix.

The shape of the answer should look like compressed sensing. There, to find a sparse vector `x` with `Ax = b`, the workhorse is soft-thresholding: `S_tau(x) = sign(x)(|x| - tau)_+`, applied over and over in iterative-shrinkage / linearized-Bregman schemes, because shrinking small coefficients to exactly zero is what *produces* sparsity. My situation is the matrix analogue: a low-rank matrix is one whose vector of singular values is sparse. So the move I want is "soft-threshold the singular values" — shrink each `σ_i` toward zero by some amount and drop the ones that fall to zero. Write that operator out: take the SVD `Y = U Σ V^*` and define

  D_tau(Y) = U diag((σ_i - tau)_+) V^*,

the singular-value shrinkage at level `tau`. If many singular values sit below `tau`, the output has far lower rank than the input — exactly the spectrum-sparsifying behavior I want, and it's adaptive in a way the vector case isn't, because it discovers the basis (the singular vectors) in which the matrix is sparse rather than thresholding in a fixed transform.

But before I build an algorithm on `D_tau`, I should make sure it's the *right* operation and not just an appealing one. In the scalar case, soft-thresholding is not an ad hoc shrink — it's the proximity operator of the `l1` norm, `S_tau(x) = argmin_u { (1/2)(u-x)^2 + tau|u| }`. If `D_tau` is going to play the analogous role, it had better be the proximity operator of the *nuclear* norm:

  D_tau(Y) =? argmin_X { (1/2)||X - Y||_F^2 + tau||X||_* }.

Let me actually prove this, because the whole construction hangs on it. The objective `h(X) = (1/2)||X-Y||_F^2 + tau||X||_*` is strictly convex (the quadratic is strictly convex, the nuclear norm convex), so it has a unique minimizer `X̂`, characterized by `0 ∈ ∂h(X̂)`, i.e.

  0 ∈ X̂ - Y + tau ∂||X̂||_*.

I need the subdifferential of the nuclear norm. For `X = U Σ V^*` in reduced SVD it's

  ∂||X||_* = { U V^* + W : U^*W = 0, W V = 0, ||W||_2 <= 1 } —

the `U V^*` is the matrix analogue of `sign(x)`, and `W` lives in the orthogonal complement with spectral norm at most one. Now guess `X̂ = D_tau(Y)` and check the inclusion. Split the SVD of `Y` by the threshold: let `U_0, Σ_0, V_0` carry the singular values strictly above `tau`, and `U_1, Σ_1, V_1` the ones at or below `tau`, so `Y = U_0 Σ_0 V_0^* + U_1 Σ_1 V_1^*`. By definition of the shrinkage, `X̂ = U_0 (Σ_0 - tau I) V_0^*`. Then

  Y - X̂ = U_0 (tau I) V_0^* + U_1 Σ_1 V_1^* = tau ( U_0 V_0^* + W ),  W := tau^{-1} U_1 Σ_1 V_1^*.

Is `tau^{-1}(Y - X̂)` a subgradient of `||·||_*` at `X̂`? The leading term `U_0 V_0^*` is the `U V^*` of `X̂` (whose SVD is built from `U_0, V_0`). For `W`: `U_0^* W = 0` and `W V_0 = 0` because `U_1 ⟂ U_0`, `V_1 ⟂ V_0`; and `||W||_2 = tau^{-1} max(Σ_1) <= tau^{-1}·tau = 1` since every singular value in `Σ_1` is at most `tau`. So `Y - X̂ ∈ tau ∂||X̂||_*`, which is precisely `0 ∈ X̂ - Y + tau ∂||X̂||_*`. So `D_tau(Y)` is the unique minimizer — `D_tau` *is* the proximity operator of the nuclear norm. Good. The shrink isn't a heuristic; it's the exact prox.

Now, how do I turn one prox into a method that respects the hard constraint `P_Omega(X) = P_Omega(M)`? The obvious thing, copying the imaging people, is to relax: penalize the data mismatch and minimize `lambda||X||_* + (1/2)||P_Omega(X) - P_Omega(M)||_F^2`, whose proximal-forward-backward fixed point is `X = D_{lambda δ}(X + δ P_Omega(M - X))`, iterated as `X^k = D_{lambda δ}(Y^{k-1})`, `Y^k = X^k + δ P_Omega(M - X^k)`. Let me think about whether this actually serves me. It converges to the minimizer of the *penalized* objective, which trades off fitting the data against small nuclear norm — so it does not exactly fit the observations, and its nuclear norm isn't the minimal one. To make it fit the data well I'd have to take `lambda` small. But here's the trap: the threshold in the shrink is `lambda δ`, so small `lambda` means a *small* threshold, which means the shrink barely kills any singular values, the iterates `X^k` are *not* low rank, and the working matrix isn't sparse. The two things that would make each iteration cheap — a large threshold producing low-rank, sparse iterates — are exactly what fitting the data forbids in this formulation. So the penalized route puts cheapness and accuracy in direct opposition. Wall. I want large threshold *and* exact data fit at the same time.

Let me look harder at where that PFBS recursion came from, because maybe a small change buys me both. The recursion feeds `X^k` back into the `Y`-update: `Y^k = X^k + δ P_Omega(M - X^k)`. What if instead the residual accumulates on its own track and I shrink *that*? Concretely, keep a matrix `Y` that I only ever grow by the data residual, and read `X` off it by shrinking:

  X^k = D_tau(Y^{k-1}),
  Y^k = Y^{k-1} + δ_k P_Omega(M - X^k),    Y^0 = 0.

This looks almost identical — but the difference is that the threshold `tau` is now *decoupled* from the step size, a fixed constant of its own, and `Y` is a running sum of residuals rather than `X^k` plus a residual. Let me check what it costs and what it converges to. Cost first: `Y^0 = 0` is supported on `Omega`, and `P_Omega(M - X^k)` is supported on `Omega`, so by induction *every* `Y^k` is supported on `Omega` — it's sparse, with at most `m` nonzeros, and the residual update touches only those `m` entries, an `O(m)` operation. The only real work is the shrink, one (partial) SVD of a sparse matrix. So if I'm free to take `tau` large, the iterates `X^k` are low rank and the `Y^k` stay sparse, and *both* hold simultaneously — the coupling that doomed the penalized version is gone.

But "looks like it works" isn't a proof, and I have no idea yet what this iteration even converges to. Let me figure out its fixed-point objective. The cleanest way to understand a residual-accumulating scheme is as a dual method, so let me try to read it as Uzawa's algorithm on *some* constrained problem and see what problem that is. Uzawa solves `min_X f(X) s.t. P_Omega(X) = P_Omega(M)` by forming the Lagrangian `L(X,Y) = f(X) + <Y, P_Omega(M - X)>`, alternating an inner minimization over `X` with a dual ascent step `Y^k = Y^{k-1} + δ_k ∂_Y g(Y^{k-1})`, where `g(Y) = min_X L(X,Y)` is the dual function and `∂_Y g(Y) = P_Omega(M - X̃)` at the inner minimizer `X̃`. So the dual step is exactly `Y^k = Y^{k-1} + δ_k P_Omega(M - X^k)` — that matches my `Y`-update for free. The question is which `f` makes the inner minimization come out to a single shrink `D_tau`.

Try `f(X) = (1/2)||X - P_Omega Y||...` no — let me just do the inner minimization for a candidate `f` and reverse-engineer it. The inner problem is `argmin_X f(X) + <Y, P_Omega(M - X)> = argmin_X f(X) - <P_Omega Y, X> + const`. I want this to equal `D_tau(Y) = argmin_X (1/2)||X - Y||_F^2 + tau||X||_*` — but only when `Y` is supported on `Omega`, which mine always is, so `P_Omega Y = Y`. Expand the prox objective: `(1/2)||X||_F^2 - <Y, X> + (1/2)||Y||_F^2 + tau||X||_*`. Dropping the `X`-independent `(1/2)||Y||_F^2`, that's `(1/2)||X||_F^2 + tau||X||_* - <Y, X>`. Compare to `f(X) - <P_Omega Y, X> = f(X) - <Y, X>`. They match exactly if

  f(X) = tau||X||_* + (1/2)||X||_F^2.

So my residual-accumulating iteration *is* Uzawa's algorithm for

  minimize tau||X||_* + (1/2)||X||_F^2  subject to P_Omega(X) = P_Omega(M).

That's a real, clean characterization, and it immediately tells me three things. First, why the extra `(1/2)||X||_F^2` term is there even though I never asked for it: it's the price of getting a single closed-form shrink as the inner step, and as a bonus it makes `f` *strongly* convex, so the solution is unique and the dual method is well-behaved — neither of which is true for the bare nuclear norm. Second, it tells me what I'm *not* solving: I'm minimizing nuclear norm plus a small Frobenius penalty, not nuclear norm alone. But — third — I can make that gap vanish. As `tau → ∞`, the `tau||X||_*` term dominates `(1/2)||X||_F^2`, so the minimizer of `f` is pushed toward the minimum-nuclear-norm solution. Let me make that precise rather than hand-wave it. Let `X_tau` minimize `f` over the feasible set and `X_∞` be the minimum-Frobenius-norm solution among all nuclear-norm minimizers. Optimality of each gives two inequalities: `||X_tau||_* + (1/2tau)||X_tau||_F^2 <= ||X_∞||_* + (1/2tau)||X_∞||_F^2` (because `X_tau` minimizes `||X||_* + (1/2tau)||X||_F^2`, the scaled objective), and `||X_∞||_* <= ||X_tau||_*` (because `X_∞` is a nuclear-norm minimizer). Add them: `||X_tau||_F^2 <= ||X_∞||_F^2`, so `||X_tau||_F` is bounded uniformly in `tau`. Then along any convergent subsequence `X_tau → X_c`, feasibility passes to the limit, the first inequality forces `||X_c||_* = ||X_∞||_*` (so `X_c` is a nuclear-norm minimizer) and the bound forces `||X_c||_F <= ||X_∞||_F`; by definition of `X_∞` as the *minimum*-Frobenius minimizer, `||X_c||_F = ||X_∞||_F`, and uniqueness gives `X_c = X_∞`. So `X_tau → X_∞` as `tau → ∞`. Picking `tau` large makes the proximal solution as close as I want to the genuine nuclear-norm answer — and large `tau` is *also* what makes the iterates low rank and sparse. The two goals that fought each other in the penalized formulation now *agree*: crank `tau` up.

So I've resolved the wall. The decoupling of `tau` from `δ` — `tau` a fixed large threshold owning the rank, `δ` a step size owning the dual ascent — is what lets large threshold and exact data fit coexist, and the Uzawa reading is what told me the iteration converges to (the proximal surrogate of) the right thing rather than to the penalized object.

Now I owe myself an actual convergence proof, with conditions on `δ`. The engine is the strong convexity of `f`. Let me first nail that down as a lemma. Take subgradients `Z ∈ ∂f(X)`, `Z' ∈ ∂f(X')`. Since `f = tau||·||_* + (1/2)||·||_F^2`, any such `Z = tau Z_0 + X` with `Z_0 ∈ ∂||X||_*`. Then

  <Z - Z', X - X'> = tau <Z_0 - Z_0', X - X'> + ||X - X'||_F^2.

The Frobenius part already gives me the `||X - X'||_F^2` I want, so I just need the nuclear part nonnegative. Expand `<Z_0 - Z_0', X - X'> = <Z_0,X> + <Z_0',X'> - <Z_0,X'> - <Z_0',X>`. Using the two facts about nuclear-norm subgradients — `<Z_0, X> = ||X||_*` and `||Z_0||_2 <= 1` — the diagonal terms are `||X||_* + ||X'||_*`, and the cross terms are bounded by duality `|<Z_0, X'>| <= ||Z_0||_2 ||X'||_* <= ||X'||_*` and likewise `|<Z_0', X>| <= ||X||_*`. So `<Z_0 - Z_0', X - X'> >= ||X||_* + ||X'||_* - ||X'||_* - ||X||_* = 0`. Hence

  <Z - Z', X - X'> >= ||X - X'||_F^2.

That's the strong-monotonicity lemma. Now run it through the iteration. Let `(X^*, Y^*)` be primal-dual optimal. The optimality of the inner minimization says `0 = Z^k - P_Omega(Y^{k-1})` for some `Z^k ∈ ∂f(X^k)`, and at the optimum `0 = Z^* - P_Omega(Y^*)`. Subtract: `(Z^k - Z^*) = P_Omega(Y^{k-1} - Y^*)`. Pair with `X^k - X^*` and apply the lemma:

  <X^k - X^*, P_Omega(Y^{k-1} - Y^*)> = <Z^k - Z^*, X^k - X^*> >= ||X^k - X^*||_F^2.

Now track the dual distance `r_k := ||P_Omega(Y^k - Y^*)||_F`. Since `P_Omega(X^*) = P_Omega(M)`, the residual at the optimum vanishes on `Omega`, so the `Y`-update gives `P_Omega(Y^k - Y^*) = P_Omega(Y^{k-1} - Y^*) + δ_k P_Omega(X^* - X^k)`. Square it:

  r_k^2 = r_{k-1}^2 + 2 δ_k <P_Omega(Y^{k-1} - Y^*), P_Omega(X^* - X^k)> + δ_k^2 ||P_Omega(X^* - X^k)||_F^2.

The cross term: `<P_Omega(Y^{k-1}-Y^*), X^* - X^k>` (the projection can move onto either factor of an inner product since `P_Omega` is an orthogonal projector and the other factor is already on `Omega`) `= -<P_Omega(Y^{k-1}-Y^*), X^k - X^*> <= -||X^k - X^*||_F^2` by the line above. And `||P_Omega(X^* - X^k)||_F^2 <= ||X^* - X^k||_F^2` because a projection only shrinks Frobenius norm. So

  r_k^2 <= r_{k-1}^2 - 2 δ_k ||X^k - X^*||_F^2 + δ_k^2 ||X^k - X^*||_F^2
        = r_{k-1}^2 - (2 δ_k - δ_k^2) ||X^k - X^*||_F^2.

For this to be a genuine decrease I need `2 δ_k - δ_k^2 >= β > 0` for all `k`, i.e. `0 < δ_k < 2` (with `inf δ_k > 0`, `sup δ_k < 2`). Under that, `r_k^2 <= r_{k-1}^2 - β ||X^k - X^*||_F^2`: the dual distances `r_k` are nonincreasing, hence convergent, and telescoping forces `Σ_k ||X^k - X^*||_F^2 < ∞`, so `||X^k - X^*||_F → 0`. The iterates converge to the unique solution of the proximal problem. So convergence is guaranteed for any constant `δ ∈ (0, 2)`.

That's the safe bound — but it's conservative, and `δ < 2` makes convergence slow, so let me push on the step size. Where did the `2` come from? From bounding `||P_Omega(X^* - X^k)||_F^2 <= ||X^* - X^k||_F^2` — the projection. But the near-isometry says that for a fixed incoherent matrix `A`, `||P_Omega(A)||_F^2 ≈ p ||A||_F^2` with `p = m/(n_1 n_2)` the observation ratio. If I could treat `A = X^* - X^k` as such a matrix, the projection term would be about `p ||X^* - X^k||_F^2`, far smaller, and re-running the sufficient condition `−2δ||X^*-X^k||_F^2 + δ^2 ||P_Omega(X^*-X^k)||_F^2 <= −β||X^*-X^k||_F^2` becomes `δ < 2/p` roughly. With a safety factor (taking `ε = 1/4` in the near-isometry gives `δ <= 1.6/p`), a practical aggressive choice is

  δ = 1.2 / p = 1.2 · n_1 n_2 / m,

i.e. `1.2` times the inverse undersampling ratio. I should be honest that this isn't a theorem: `X^* - X^k` is *not* a fixed matrix — it depends on `Omega`, because the iterates were computed using the observed entries — so the near-isometry doesn't rigorously apply to it. But empirically the iterates stay incoherent enough that `1.2/p` converges, and it takes much bigger, faster steps than the worst-case `δ < 2`. I'll take `δ = 1.2/p`.

Now `tau`. I argued large `tau` is good in the limit, but I need a concrete value, and "as large as possible" fights convergence speed (a huge `tau` means tiny initial iterates and many steps to build up). I want `tau` just large enough that the proximal solution is essentially the nuclear-norm solution — concretely, that `tau||X||_*` dominates `(1/2)||X||_F^2` for the kind of matrix I'm recovering. Calibrate against the standard synthetic generator `M = M_L M_R^*` with Gaussian factors: random-matrix theory says `||M||_F` concentrates near `n√r` and `||M||_*` near `n r`. The ratio of the two terms is then

  tau ||M||_* / ( (1/2) ||M||_F^2 ) ≈ tau · n r / ( (1/2) n^2 r ) = 2 tau / n.

To make the nuclear term about `10×` the Frobenius term I want `2 tau / n ≈ 10`, i.e. `tau ≈ 5 n`. So I'll set

  tau = 5 n,

which keeps the nuclear term dominant as long as the rank is bounded away from `n`, while not being so large as to crawl.

There's one more piece of slack to exploit, at the very start. I initialize `Y^0 = 0`. Then `X^1 = D_tau(0) = 0`, and `Y^1 = δ P_Omega(M)`; if `||Y^1||_2 = δ ||P_Omega(M)||_2 <= tau`, the shrink kills everything again, `X^2 = 0`, and `Y^2 = 2 δ P_Omega(M)`. In general, as long as `k δ ||P_Omega(M)||_2 <= tau`, every singular value of `Y^k = k δ P_Omega(M)` is below threshold, `X^k = 0`, and the iteration is just summing the same residual `P_Omega(M)`. Those steps are pure bookkeeping — I can leap over them. Define `k_0` as the integer with

  tau / ( δ ||P_Omega(M)||_2 ) ∈ (k_0 - 1, k_0],

and warm-start directly at `Y^0 := k_0 δ P_Omega(M)`, the first iterate where the shrink will actually produce something nonzero. This is the simplest case of a "kicking" device — jump over steps whose outcome is predetermined.

Now stopping. I don't want to run a fixed number of iterations; I want to stop when `X^k` is close to the solution. The KKT conditions of the proximal problem are `X = D_tau(Y)` (which my iterates satisfy by construction) together with `P_Omega(X - M) = 0` — the constraint. So the only optimality residual left to monitor is how badly the constraint is violated on the observed set, and a natural scale-free test is the relative residual on `Omega`:

  ||P_Omega(X^k - M)||_F / ||P_Omega(M)||_F <= ε,

for a small `ε` like `1e-4`. This also controls the *true* error: by the near-isometry, `||P_Omega(X^k - M)||_F^2 ≈ p ||X^k - M||_F^2` and `||P_Omega(M)||_F^2 ≈ p ||M||_F^2`, so the ratio on `Omega` tracks the relative reconstruction error `||X^k - M||_F / ||M||_F` on the full matrix. Controlling what I can measure (residual on the observed entries) controls what I actually care about (error on the unseen entries).

Two structural properties make each iteration genuinely cheap, and they're worth saying out loud because they're what make this scale. The `Y^k` are supported on `Omega` by induction from `Y^0`, so they're sparse with `m` nonzeros and the residual update is `O(m)`. And because `tau` is large, the shrink kills most singular values, so `X^k` is low rank — empirically the rank is even nondecreasing, climbing to its final value only near the end — which means I never need the full SVD of `Y^k`. I only need the singular triplets *above* `tau`. So in the large-scale realization I'd compute a *partial* SVD with an iterative Lanczos bidiagonalization (e.g. PROPACK), asking for `s_k = r_{k-1} + 1` singular values and incrementing by a few until one falls below `tau`; since `Y^k` is sparse it multiplies vectors fast, and the partial SVD is far cheaper than a dense one. Storage is just the current low-rank `X^k` in factored form plus the sparse `Y^k`. For the synthetic sizes here a dense SVD per step is fine; the partial-SVD machinery is what carries the same algorithm to a billion entries.

Let me assemble the whole thing into code I'd actually run, filling the single empty slot in the recovery harness. The pieces map one-to-one to the reasoning: `tau = 5n` for the square case, `δ = 1.2/p`, the `k_0` warm start, the shrink `D_tau` via SVD, the residual-accumulating `Y`-update on the mask, and the relative-residual stop.

```python
import math
import torch


class NuclearNormSVT:
    """Singular Value Thresholding for nuclear-norm matrix completion.

    Uzawa / dual-ascent iteration for  min tau||X||_* + (1/2)||X||_F^2
    s.t. P_Omega(X) = P_Omega(M):
        X^k = D_tau(Y^{k-1})                       # soft-threshold singular values
        Y^k = Y^{k-1} + delta * P_Omega(M - X^k)   # dual step (residual on Omega)
    Large tau -> proximal solution approaches the min-nuclear-norm solution,
    and makes X^k low rank and Y^k sparse.
    """

    def __init__(self, tau_factor=5.0, delta_factor=1.2, tol=1e-4):
        self.tau_factor = float(tau_factor)      # tau = tau_factor * n  (=> 5n)
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

        # Warm start: skip the trivial steps where the shrink kills everything.
        # While k*delta*||P_Omega(M)||_2 <= tau, X^k = 0 and
        # Y^k = k*delta*P_Omega(M); jump to k_0.
        norm_proj_m = float(torch.linalg.matrix_norm(M_obs, ord=2).item())
        k0 = max(1, math.ceil(tau / (delta * max(norm_proj_m, 1e-6))))
        Y = (k0 * delta) * M_obs                                  # Y^0 = k_0 * delta * P_Omega(M)

        X = torch.zeros_like(M_obs)
        norm_obs = max(float(M_obs.norm().item()), 1e-6)
        log_every = max(int(log_iters), 1)
        for it in range(1, max_iters + 1):
            # X^k = D_tau(Y^{k-1}): soft-threshold the singular values of Y.
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            S_thresh = torch.clamp(S - tau, min=0.0)             # (sigma_i - tau)_+
            X = (U * S_thresh) @ Vh

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

            # Dual / residual update, supported on Omega:
            # Y^k = Y^{k-1} + delta * P_Omega(M - X^k)
            Y = Y + delta * residual

        return X.detach().cpu()


def build_strategy():
    return NuclearNormSVT(tau_factor=5.0, delta_factor=1.2, tol=1e-4)
```

For a rectangular version I would replace `tau = 5n` by `tau = 5 sqrt(n_1 n_2)`, exactly as the numerical driver does. For the warm start I need the spectral norm of the sparse observed matrix; in this dense harness I compute it directly, while a large implementation would use a `normest`-style approximation.
