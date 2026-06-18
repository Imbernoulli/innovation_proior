Let me start from the thing that actually bugs me. I have a matrix `W*` and only a small random handful of its entries. If I just look for any matrix that agrees with those entries, I am sunk: the observed set is a tiny fraction of the `n^2` coordinates, so the constraints leave a huge affine set of interpolants, and most of them are wrong on the entries I care about. Fitting the data is not the problem. The problem is that fitting the data does not, by itself, prefer the right completion. When `W*` is low rank, the right completion is the low-rank one, so I need a low-rank bias that comes from the parameterization and the optimizer rather than from an explicit penalty.

The classical move is to relax rank to nuclear norm, `||W||_* = sum_r sigma_r(W)`, and minimize that subject to matching the observations. With enough observations and the usual incoherence conditions, this can recover the ground truth exactly. But the regime where implicit regularization matters is the data-poor one, where the convex surrogate can stop matching minimum rank. So nuclear norm is the reference point, but it may not be the whole story.

The empirical fact that keeps this alive is shallow factorization. If I write `W = W_2 W_1`, keep the hidden dimension full, initialize near zero, and run gradient descent on observed-entry squared loss, I often get a low-rank completion even without a rank cap. Gunasekar and coauthors conjecture that this depth-2 implicit bias is minimum nuclear norm, and they prove that statement in a clean commuting positive-semidefinite sensing case with initialization `alpha I` and `alpha -> 0`. That makes the next guess almost unavoidable: if two factors look like Schatten-1, perhaps more factors act like a Schatten-`p` quasi-norm with `p < 1`, closer to rank.

I should test that guess in the setting where the existing proof works. Take symmetric PSD sensing, `ell(W) = 1/2 ||A(W) - y||_2^2`, with commuting symmetric measurement matrices. Since the measurements commute, an orthogonal matrix `O` diagonalizes all of them. In the diagonal basis, `W_tilde = O W O^T`, the end-to-end product dynamics from balanced deep linear networks become

dot W_tilde = - sum_{j=1}^N [W_tilde W_tilde^T]^((j-1)/N)
                         A_tilde^dagger(r)
                         [W_tilde^T W_tilde]^((N-j)/N),

where `r = A(W) - y`. Starting from `W_tilde(0) = alpha^N I`, the product stays diagonal: the adjoint gradient is diagonal, powers of a diagonal matrix stay diagonal, and the right-hand side has no off-diagonal entries. So I get one scalar ODE per diagonal coordinate:

dot W_tilde_kk = -N (W_tilde_kk^2)^((N-1)/N) A_tilde^dagger_kk(r),
W_tilde_kk(0) = alpha^N.

Before integrating, I need the sign. The ODE has the form `dot s = (s^2)^beta g(t)` with `beta = (N-1)/N`. For `N = 2`, `beta = 1/2` and a nonzero solution is exponential in `int g`, so it never crosses zero. For `N >= 3`, the separated solution has exponent `1/(1-2 beta)` and can blow up in finite time, but before blow-up it cannot change sign. Thus every diagonal coordinate that starts positive remains positive. In this commuting PSD setting, the asymmetric product implicitly stays in the PSD cone.

For `N >= 3`, integrating the diagonal ODE gives

W_tilde_kk(t) =
  alpha^N (1 + (N-2) alpha^(N-2) A_tilde^dagger_kk(s(t)))^(-N/(N-2)),

where `s(t) = int_0^t r(t') dt'`. At `t = 0` this returns `alpha^N`, so the constant is right. If the limit as `t -> infinity` exists, define `nu_infty(alpha) = -(N-2) alpha^(N-2) lim_t s(t)`. Then

W_tilde_deep,infty(alpha) =
  alpha^N [I - A_tilde^dagger(nu_infty(alpha))]^(-N/(N-2)).

The negative power is positive definite, so `A_tilde^dagger(nu_infty(alpha)) < I`. Now let `alpha -> 0+` and assume the limiting zero-loss solution exists. Any diagonal entry of the limit that is nonzero can only survive the front factor `alpha^N -> 0` if the corresponding slack `1 - A_tilde^dagger_kk(nu_infty(alpha))` goes to zero. Therefore the complementary slackness quantity `<I - A_tilde^dagger(nu_infty(alpha)), W_tilde*>` goes to zero. Undo the diagonalization and I have a primal feasible PSD matrix, a dual-feasible sequence for the SDP `min <I,W>` subject to `A(W)=y, W >= 0`, and vanishing primal-dual gap. Since trace equals nuclear norm on PSD matrices, the deep product limit is a minimum-nuclear-norm solution.

That is not what I hoped for. The same proof line that supports the depth-2 nuclear-norm story also extends to every `N >= 3` in this restricted setting. So the simple hypothesis "depth means Schatten-`p`" is already in trouble.

I can make the contradiction sharper. Choose diagonal constraints `W_11 = W_22` and `W_11 = W_kk + 1` for `k >= 3`. The minimum-trace PSD solution is `diag(1,1,0,...,0)`. Now add `epsilon` to the `(1,2)` and `(2,1)` entries. The constraints are unchanged, the matrix remains PSD for `0 < epsilon < 1`, and the nonzero eigenvalues become `1+epsilon` and `1-epsilon`. For any `0 < p < 1`,

(1+epsilon)^p + (1-epsilon)^p < 2

by strict concavity of `x^p`. Thus arbitrarily close feasible PSD perturbations have smaller Schatten-`p` quasi-norm. The nuclear-norm limit point is not even a local minimizer of any Schatten-`p` objective. So I should stop trying to name one fixed norm and look at the trajectory itself.

Now I set up the product dynamics directly. Let `ell` be analytic and let the factorized objective be `phi(W_1,...,W_N) = ell(W_N ... W_1)`. Balanced initialization gives the end-to-end ODE for `W(t)`. Because the factors are analytic under gradient flow, the product is analytic in time, and I can use an analytic SVD

W(t) = U(t) S(t) V(t)^T

with signed diagonal entries `sigma_r(t)`. Differentiating and sandwiching by `U^T` and `V` gives

U^T dot W V = U^T dot U S + dot S + S dot V^T V.

On the diagonal, `<u_r, dot u_r> = 0` and `<dot v_r, v_r> = 0`, because the singular vectors keep unit norm. So the value velocity is simply

dot sigma_r = u_r^T dot W v_r.

Plug in the end-to-end ODE. Since `W W^T = U S^2 U^T` and `W^T W = V S^2 V^T`, the `j`th term contributes

-(sigma_r^2)^((j-1)/N)
  <grad ell(W), u_r v_r^T>
  (sigma_r^2)^((N-j)/N).

The exponents add to `(N-1)/N` for every `j`, so all `N` terms are identical:

dot sigma_r =
  -N (sigma_r^2)^(1-1/N) <grad ell(W), u_r v_r^T>.

This is the central mechanism. For `N = 1`, the multiplier is `1`, so direct gradient flow treats singular modes without a size-dependent throttle. For `N >= 2`, the multiplier is `N sigma_r^(2-2/N)`: large modes move faster and small modes move more slowly. The exponent increases with depth, so the separation between large and small modes becomes sharper. The sign lemma also applies to `sigma_r`, so for non-degenerate factorization I can take singular values nonnegative after absorbing fixed signs into the vectors. Near-zero initialization puts every mode on the slow part of the curve; only modes with sustained gradient alignment switch on.

But I still need to know what happens to the singular vectors, because the scalar gradient projection depends on `u_r v_r^T`. The off-diagonal part of the differentiated SVD gives

Ibar o (U^T dot W V) = U^T dot U S + S dot V^T V.

Combining it with its transpose and using skew-symmetry of `U^T dot U` and `dot V^T V`, I solve for the in-subspace rotation:

U^T dot U =
  H o [U^T dot W V S + S V^T dot W^T U],
H_rr' = (sigma_r'^2 - sigma_r^2)^(-1)   for r != r'.

The orthogonal-complement term is

(I - U U^T) dot U = (I - U U^T) dot W V S^(-1).

Substituting the product ODE into these two pieces yields

dot U =
  -U (F o [U^T grad ell V S + S V^T grad ell^T U])
  -(I-UU^T) grad ell V (S^2)^(1/2 - 1/N),

with

F_rr' = ((sigma_r'^2)^(1/N) - (sigma_r^2)^(1/N))^(-1)   for r != r',
F_rr = 0.

The denominator is important: it comes from `F = H o G`, where

G_rr' = sum_{j=1}^N (sigma_r^2)^((j-1)/N) (sigma_r'^2)^((N-j)/N),

and the finite geometric-sum identity gives `G_rr'/(sigma_r'^2 - sigma_r^2) = 1 / ((sigma_r'^2)^(1/N) - (sigma_r^2)^(1/N))`. The complement exponent is also fixed: only the `j=1` term survives the left projection onto `U`'s orthogonal complement, leaving `(S^2)^((N-1)/N) S^(-1) = (S^2)^(1/2 - 1/N)`.

Now I get the alignment statement I wanted. Combining the `U` and `V` equations gives

U^T dot U S - S V^T dot V =
  -Ibar o G o [U^T grad ell(W) V].

The off-diagonal entries of `G` are nonzero away from isolated singular-value collisions, and the equation extends through those times by continuity. If the singular vectors are stationary, the left side vanishes, so the off-diagonal part of `U^T grad ell(W) V` must vanish. Stationary singular vectors therefore align with the singular vectors of the gradient. Earlier analyses could show the forward direction; this gives the converse I need for the trajectory picture.

A one-measurement toy makes the depth effect concrete. Let `ell(W) = 1/2 (<A,W> - y)^2`, so `grad ell(W) = delta(t) A`. Once the singular vectors are stationary and aligned, write

u_r^T grad ell(W) v_r = delta(t) e_r rho_r,

where `rho_r` is a singular value of `A` and `e_r` is a sign. Then

dot sigma_r = -N (sigma_r^2)^(1-1/N) delta(t) e_r rho_r.

For two modes, the common factor `delta(t)` cancels. With
`alpha_12 = e_1 rho_1 / (e_2 rho_2)`, integration gives

sigma_1 = alpha_12 sigma_2 + const                         for N = 1,
sigma_1 = const * sigma_2^alpha_12                         for N = 2,
sigma_1 = (alpha_12 sigma_2^(-(N-2)/N) + const)^(-N/(N-2))  for N >= 3.

If `0 < alpha_12 < 1`, the first mode is weaker. At depth one it grows linearly with the stronger mode. At depth two it grows only polynomially with it. At depth at least three it approaches a finite asymptote as the stronger mode grows. This is the low-rank bias in motion: a few modes cross the slow-growth region and accelerate, while weaker modes stay small; increasing depth sharpens that separation.

The design choices now follow from the derivation. I want depth at least three because that is where the toy dynamics produce saturation rather than just a polynomial gap, and the experiments show depth four is not worth extra cost over depth three. I want near-zero initialization because all singular values should begin in the throttled regime. I want balanced or approximately balanced factors because the end-to-end ODE is the math I am relying on; small Gaussian factors are the practical approximation, while identity initialization is the exact balanced case. I keep the hidden dimension full because otherwise low rank is imposed explicitly and the experiment no longer isolates implicit bias. I train to very small observed-entry loss because the choice among interpolating solutions is exactly where the parameterization matters.

When I translate this into code, I should mirror the canonical implementation rather than a convenient harness variant. The public reference builds a sequence of bias-free `Linear` layers, computes the end-to-end matrix by starting with the first weight transposed and applying each later layer to that running matrix, initializes Gaussian factors with per-factor standard deviation `init_scale^(1/depth) n^(-1/2)`, and trains full-batch observed-entry squared loss. Its default executable uses `GroupRMSprop` with global gradient-norm scaling, `eps=1e-4`, learning rate `1e-3` in the matrix-completion config, and `train_thres=1e-6`; SGD and Adam exist as options, but Adam is not the canonical default. So the final artifact should present depth-three, near-zero Gaussian, full-dimensional, bias-free linear factors trained with the canonical optimizer path, and only describe Adam as an optional harness substitution if a benchmark requires it.
