I start with the discomfort that the usual update is pretending not to have. I have a stochastic policy written with parameters theta, and I keep saying that the gradient of reward tells me which way to move theta. But theta is only a coordinate chart for the policy. If I rename the same family of policies with a smooth change of variables, the raw derivative changes by the inverse transpose Jacobian, while the displacement I actually apply should transform like a tangent vector. So the step I take is not an intrinsic move of the policy. It is a move chosen by my coordinates.

That means the first question is not how to estimate the gradient more accurately. The first question is what I mean by a small move. The ordinary gradient is steepest only after I secretly decide that the squared length of a parameter displacement is `sum_i d theta_i^2`. That is the identity metric in the current coordinates. If I choose a different metric `G(theta)`, the small-step problem becomes: maximize the linearized reward change `grad eta dot d theta` subject to `d theta^T G d theta` being fixed. The Lagrange equation gives `grad eta = lambda G d theta`, so the direction is `G^{-1} grad eta`. The raw gradient is just the special case where I made the arbitrary choice `G = I`.

So the real problem is choosing the metric. The policy is not valuable because its coordinate vector has moved a short Euclidean distance; it is valuable because the action distributions it induces have changed in a controlled way. A one-unit shift in the mean of a narrow Gaussian policy can almost replace the distribution, while the same one-unit shift in a broad Gaussian barely changes it. The Euclidean parameter distance says those moves are equal. The distributions say they are not. So I want a metric whose length measures how much the distribution moved, not how far the label moved.

For a parameter that indexes a single probability distribution, that object already exists: the Fisher information. It is the second-order local form of the KL divergence between nearby members of the family, so a small distributional change has size `d theta^T F d theta` to leading order. If I cap a Fisher length, I cap local policy change rather than coordinate motion. So `F^{-1} grad` would be the steepest direction measured on the distribution manifold itself. The reason to reach for it here is not that it preconditions nicely; it is that its length is exactly the quantity I just argued I care about.

But I cannot simply paste that supervised-learning metric into a Markov decision process, because a policy is not one probability distribution. At every state `s` the parameters define a conditional distribution over actions, `pi(.|s,theta)`, and the objective averages consequences under the stationary distribution of the current policy. So for one state I can define the Fisher metric of that conditional distribution,

`F_s(theta) = E_{a~pi(.|s,theta)}[grad log pi(a|s,theta) grad log pi(a|s,theta)^T]`.

Then I need a single metric for the policy as it is used by the Markov chain. The objective weights states by `rho^pi(s)`, so the natural thing is to measure local policy distance with the same weighting the objective uses:

`F(theta) = E_{s~rho^pi}[F_s(theta)]`.

This is already a conceptual leap away from parameter space. I am not measuring how far the parameter vector goes. I am measuring the average infinitesimal change in the action distributions at the states the current policy visits. The dynamics enter only through `rho^pi`; the per-state Fisher itself depends only on how the policy distribution reacts to theta.

I need to check that defining the metric this way has not destroyed the main advantage of policy gradients. The exact average-reward gradient is

`grad eta(theta) = sum_{s,a} rho^pi(s) grad pi(a|s,theta) Q^pi(s,a)`.

There is no explicit derivative of `rho^pi`; the difficult state-distribution derivative cancels, which is what makes the gradient estimable from rollouts without differentiating the environment. The metric I just wrote is also an expectation under the on-policy state-action distribution `rho^pi(s) pi(a|s,theta)`, only with score outer products instead of score times value. So both objects live under the same sampling distribution, and I have changed the geometry of the direction without reintroducing a state-distribution derivative.

Now the practical worry. Computing `F` and then solving `F v = grad eta` looks like extra machinery I have to bolt on. Before I accept that cost I want to know whether the solve is genuinely separate from things an actor-critic already does, because if it is hiding inside a critic fit I get it for free. Sutton's compatibility condition is the lead: if I approximate `Q^pi` with a linear critic whose features are the policy score,

`psi(s,a) = grad log pi(a|s,theta)`,

then substituting the critic into the policy-gradient formula does not bias the gradient, provided the approximation error is orthogonal to those score features. So suppose I fit

`f(s,a;w) = w^T psi(s,a)`

by weighted least squares under `rho^pi(s) pi(a|s,theta)`. The normal equations are

`[sum_{s,a} rho^pi(s) pi(a|s,theta) psi(s,a) psi(s,a)^T] w = sum_{s,a} rho^pi(s) pi(a|s,theta) psi(s,a) Q^pi(s,a)`.

The left bracket is the metric `F(theta)`. The right side looks like the gradient if `pi psi = grad pi`. That identity is just `grad log pi = grad pi / pi`, but I have been burned by "obvious" identities before, so I want to actually see all three pieces line up on a concrete policy rather than wave at them.

Take one state, two actions, a Gibbs policy with features `phi_0 = (1,0)`, `phi_1 = (0,1)` and `theta = (0.3, -0.5)`. Then `pi = (0.6900, 0.3100)`, and the mean feature is `phibar = (0.6900, 0.3100)`. For a Gibbs policy the score is the centered feature `psi_a = phi_a - phibar`, giving `psi_0 = (0.3100, -0.3100)` and `psi_1 = (-0.6900, 0.6900)`. First sanity check: `sum_a pi_a psi_a` should vanish, and it comes out `(-3e-17, 0)`, i.e. zero up to rounding, as a score average must. Next the identity I was nervous about. Finite-differencing the probabilities, `grad pi_0 = (0.2139, -0.2139)`, and `pi_0 psi_0 = (0.2139, -0.2139)` — they match, and likewise for action 1. So `pi psi = grad pi` holds, and the right-hand side of the normal equations really is `sum rho pi psi Q = grad eta`.

Now assign `Q = (2.0, -1.0)` (single state, so `rho = 1`). The gradient is `g = sum_a pi_a psi_a Q_a = (0.6417, -0.6417)`. The Fisher matrix is `F = sum_a pi_a psi_a psi_a^T`, which evaluates to `[[0.2139, -0.2139], [-0.2139, 0.2139]]`. This `F` is rank one — with two actions the scores span only one direction, since they sum to zero under pi — so the solve needs a pseudoinverse, exactly the rank-deficiency I should expect and handle with a ridge in general. Solving with the pseudoinverse, `v = F^+ g = (1.5, -1.5)`. Then I fit the compatible critic by the same pseudoinverse on the normal equations and get `w = (1.5, -1.5)` as well, and `F w = (0.6417, -0.6417) = g` exactly. So on this example `w = v = F^+ g`: the least-squares critic weights are not merely related to the natural-gradient direction, they are it. The inverse-metric operation I was treating as extra machinery is the critic regression. That is worth more than the algebra alone told me, because I might have feared the pseudoinverse broke the equivalence on a singular `F`; it does not.

This also clarifies what kind of object the metric is, and what it is not. A different tempting route is to do second-order optimization on reward directly — use the Hessian of `eta(theta)`. But the Hessian of the average reward is

`grad^2 eta = sum_{s,a} rho^pi(s)(grad^2 pi Q^pi + grad pi grad Q^pi^T + grad Q^pi grad pi^T)`,

and it is full of `grad Q^pi` couplings and of `Q^pi` weighting the policy curvature. The Fisher metric has none of that — it is built from first derivatives of the policy distribution alone, with no value terms. So `F` is not an approximation to the reward Hessian, and I should not expect it to inherit Newton's convergence or Amari's parameter-estimation efficiency story; it is chosen because its length measures policy displacement and is invariant to reparameterization. That difference is a practical strength far from an optimum: a reward Hessian for a maximization problem can be indefinite and point into unstable curvature before I am near a maximum, whereas `F` is positive on the identifiable score subspace (or made so with a ridge), so `F^{-1} grad eta` is an ascent direction whenever the solve is positive and the gradient is nonzero.

I still want to understand why this direction aims at good actions rather than merely making a tidy geometric statement. Back on the worked example: I can ask which action the critic prefers and whether stepping along `v` actually moves probability there. The critic scores are `w^T psi = (0.93, -2.07)`, and the bare-feature scores are `w^T phi = (1.5, -1.5)`. Those two vectors differ — but they have the same argmax, action 0 — and they differ by a state-constant, which is exactly what the centering `psi = phi - phibar` predicts. So `argmax_a w^T psi(s,a) = argmax_a w^T phi_sa`: the compatible critic and the raw logit direction agree on which action is best. For a Gibbs policy, stepping `theta' = theta + alpha v` adds `alpha v^T phi_sa` to each logit, so as `alpha` grows the policy concentrates on that argmax — the greedy action under the compatible critic. A raw gradient step lacks this; pushed to the limit it can favor an action merely above the current critic mean rather than the critic's maximizer.

For a general smooth policy I cannot take the infinite-step geometry literally, so I expand locally. With `Delta theta = alpha v` and `grad pi = pi psi`,

`pi(a|s,theta + alpha v) = pi(a|s,theta)(1 + alpha psi(s,a)^T v) + O(alpha^2)`,

so the local step scales each action probability by `psi^T v`, the compatible critic's advantage-like value. I want to confirm this is the real behavior and not just a formal series. On the example, take a small step toward a KL budget `epsilon = 1e-3`. The first-order prediction gives `pi_0 -> 0.7107`; the exact reweighted policy gives `pi_0 = 0.7103`. They agree to three digits, with the small residual being the `O(alpha^2)` term — so the linear law genuinely governs how this step redistributes probability, and it does so in proportion to the critic's judgment.

That same budget gives me the step size honestly. If I cap the local KL by `epsilon`, then to second order `0.5 alpha^2 grad eta^T F^{-1} grad eta = epsilon`, and since `grad eta^T v = grad eta^T F^{-1} grad eta`, I get `alpha = sqrt(2 epsilon / (grad eta^T v))`. On the example with `epsilon = 1e-3` this yields `alpha = 0.0322`, and the actual `KL(pi_old || pi_new)` after stepping is `9.88e-4` against the target `1e-3`. So the quadratic budget formula is accurate to within the `O(alpha^3)` error I should expect, and it is the right way to set the step rather than a fixed learning rate.

Putting the pieces together gives the procedure. Roll out the current policy. Estimate `grad eta` from score times return or differential value. Estimate `F` from score outer products, or equivalently fit the compatible critic by least squares — these are the same computation. Solve `F v = grad eta`, using a pseudoinverse or small ridge when the scores are rank deficient, as they were even in the two-action example above. Step `theta <- theta + alpha v`, with `alpha` set by the KL-budget formula or a line search.

What the reasoning lands on is a direction that is neither the steepest direction in parameter coordinates nor the Newton direction for reward, but the steepest direction in the space of policy distributions under Fisher geometry — assembled in the MDP from per-state action distributions weighted by where the policy goes. The worked checks tie the geometry to operation: the score covariance that defines the metric is the same score covariance whose least-squares solve produces the compatible critic, so the critic weights equal the inverse-Fisher direction, and that direction moves the policy toward the action the critic calls greedy.
