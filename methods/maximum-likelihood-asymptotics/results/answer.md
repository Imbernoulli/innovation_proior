# Maximum Likelihood Asymptotics

## Regular Local Theorem

In a regular correctly specified dominated parametric model with true interior parameter `theta_0`, maximum likelihood becomes a local quadratic problem only after the estimator has been localized near `theta_0`. With

`ell_n(theta) = sum_{i=1}^n log p_theta(X_i)`,

let `hat h_n = sqrt(n)(hat theta_n - theta_0)`. Assume `hat h_n = O_p(1)`, `I(theta_0)` is nonsingular, and for every compact set `K`,

`sup_{h in K} |ell_n(theta_0 + h / sqrt(n)) - ell_n(theta_0) - h^T Delta_n + (1/2) h^T I(theta_0) h| ->p 0`,

where `Delta_n = n^{-1/2} dot ell_n(theta_0)` and `Delta_n -> N(0, I(theta_0))`.

Equivalently,

`ell_n(theta_0 + h / sqrt(n)) - ell_n(theta_0) = h^T Delta_n - (1/2) h^T I(theta_0) h + o_p(1)`.

Any local approximate maximizer then satisfies

`hat h_n = I(theta_0)^{-1} Delta_n + o_p(1)`,

so

`sqrt(n)(hat theta_n - theta_0) -> N(0, I(theta_0)^{-1})`.

## Sign And Constant Check

The signs and constants come from the second-order Taylor expansion:

`ell_n(theta_0 + h / sqrt(n)) - ell_n(theta_0) = h^T n^{-1/2} dot ell_n(theta_0) + (1/2) h^T [n^{-1} ddot ell_n(theta_0)] h + o_p(1)`.

Under information equality and Hessian convergence, `n^{-1} ddot ell_n(theta_0) -> -I(theta_0)`, so the quadratic term is negative: `-(1/2) h^T I(theta_0) h`. The score equation gives the same sign:

`0 = Delta_n - I(theta_0) sqrt(n)(hat theta_n - theta_0) + o_p(1)`.

Thus the displacement is `I(theta_0)^{-1} Delta_n`, with covariance `I(theta_0)^{-1} I(theta_0) I(theta_0)^{-1} = I(theta_0)^{-1}`. There is no missing factor of `n` or `2`.

## Artifact Faithfulness

There is no computational reference implementation for this non-computational method. The canonical artifact is the local-quadratic theorem in `refs/final_artifact/regular_mle_local_quadratic_theorem.md`, and the deliverable follows it: localization precedes local approximation; the approximation is uniform on bounded local neighborhoods; the score supplies the Gaussian linear term; the negative Hessian supplies Fisher-information curvature; and the conclusion is for regular localized maximizers, not arbitrary global argmaxes.

## Efficiency

The inverse-information covariance is not just a variance calculation. In the local asymptotic normality framework, the localized statistical experiment converges to a Gaussian shift experiment. Regular estimators inherit the Gaussian experiment's lower bounds, and the localized maximum likelihood estimator attains that first-order bound under the regularity assumptions. Pointwise superefficiency or other nonregular behavior is outside this claim.

## Failure Cases

The theorem is conditional. It can fail before the local quadratic limit is reached:

- The likelihood may have no attained, measurable, or version-stable maximizer.
- Single observations can create infinite or dominating peaks in mixture, shifted-support, lognormal, gamma, or singular-density examples.
- Boundary models such as uniform endpoint estimation can have nonnormal rates and different constants.
- Incidental parameters, as in the Neyman-Scott construction, can make the likelihood estimator converge to the wrong value.
- False peaks can drive estimates away even under smooth parametrizations, and bounded likelihood ratios do not by themselves prevent failure.
- Singular or flat information prevents score noise from being converted into a stable displacement.
- Ordinary Wald quadratics can be parametrization-dependent or nonuniform near boundaries; the safer local object is likelihood-ratio or Hellinger/Gaussian-shift geometry.

The distinctive insight is therefore not "argmax implies normality"; it is that a localized likelihood with nonsingular information curvature becomes a Gaussian quadratic, and that geometry controls the estimator's uncertainty.
