I start with the discomfort that the usual update is pretending not to have. I have a stochastic policy written with parameters theta, and I keep saying that the gradient of reward tells me which way to move theta. But theta is only a coordinate chart for the policy. If I rename the same family of policies with a smooth change of variables, the raw derivative changes by the inverse transpose Jacobian, while the displacement I actually apply should transform like a tangent vector. So the step I take is not an intrinsic move of the policy. It is a move chosen by my coordinates.

That means the first question is not how to estimate the gradient more accurately. The first question is what I mean by a small move. The ordinary gradient is steepest only after I secretly decide that the squared length of a parameter displacement is `sum_i d theta_i^2`. That is the identity metric in the current coordinates. If I choose a different metric `G(theta)`, the small-step problem becomes: maximize the linearized reward change `grad eta dot d theta` subject to `d theta^T G d theta` being fixed. The Lagrange equation gives `grad eta = lambda G d theta`, so the direction is `G^{-1} grad eta`. The raw gradient is just the special case where I made the arbitrary choice `G = I`.

So the real problem is choosing the metric. The policy is not valuable because its coordinate vector has moved a short Euclidean distance; it is valuable because the action distributions it induces have changed in a controlled way. A one-unit shift in the mean of a narrow Gaussian policy can almost replace the distribution, while the same one-unit shift in a broad Gaussian barely changes it. The Euclidean parameter distance says those moves are equal. The distributions say they are not.

Amari's information-geometric answer now becomes the right primitive. When a parameter indexes a probability distribution, the Fisher information is the invariant local metric on that distribution family. It is also the second-order local form of KL divergence: to leading order, a small distributional change has size `d theta^T F d theta`. So if I cap a Fisher length, I am capping local policy change, not coordinate motion. The direction `F^{-1} grad` is not a trick for preconditioning; it is the steepest direction measured on the distribution manifold.

But I cannot simply paste the supervised-learning formula into a Markov decision process. A policy is not one probability distribution. At every state `s`, the parameters define a conditional distribution over actions, `pi(.|s,theta)`. Each state has its own statistical manifold, while the reward objective averages consequences under the stationary distribution of the current policy. So for one state I can define the Fisher metric of that conditional distribution,

`F_s(theta) = E_{a~pi(.|s,theta)}[grad log pi(a|s,theta) grad log pi(a|s,theta)^T]`.

Then I need a single metric for the policy as it is used by the Markov chain. The objective weights states by `rho^pi(s)`, so I use the same weighting for the local policy-distance measurement:

`F(theta) = E_{s~rho^pi}[F_s(theta)]`.

This is already a conceptual leap. I am not measuring how far the parameter vector goes. I am measuring the average infinitesimal change in the action distributions at the states the current policy visits. The dynamics enter through `rho^pi`; the per-state Fisher itself depends only on how the policy distribution reacts to theta.

I need to check that this has not destroyed the main advantage of policy gradients. The exact average-reward gradient is

`grad eta(theta) = sum_{s,a} rho^pi(s) grad pi(a|s,theta) Q^pi(s,a)`.

There is no explicit derivative of `rho^pi`. That is the policy-gradient theorem's gift: the difficult state-distribution derivative cancels. The metric I just wrote is also an expectation under the same on-policy state-action distribution, but with score outer products rather than score times value. So both the gradient and the metric are estimable from rollouts. I have changed the geometry of the direction without asking to differentiate the environment.

Now I wonder whether this inverse Fisher solve is an extra burden or whether it is already hiding in actor-critic machinery. Sutton's compatibility condition says that if I approximate `Q^pi` with a linear critic whose features are the policy score,

`psi(s,a) = grad log pi(a|s,theta)`,

then substituting the critic into the policy-gradient formula does not bias the gradient once the approximation error is orthogonal to those score features. So I fit

`f(s,a;w) = w^T psi(s,a)`

by weighted least squares under `rho^pi(s) pi(a|s,theta)`. The normal equations are

`[sum_{s,a} rho^pi(s) pi(a|s,theta) psi(s,a) psi(s,a)^T] w = sum_{s,a} rho^pi(s) pi(a|s,theta) psi(s,a) Q^pi(s,a)`.

The left bracket is exactly the policy Fisher metric I just constructed. The right side is exactly the policy gradient, because `pi psi = grad pi`. So the critic fit gives

`F(theta) w = grad eta(theta)`.

When the metric solve is well defined, `w = F(theta)^{-1} grad eta(theta)`. The compatible critic weights are the geometric steepest direction. That is the point where two lines of thought meet: the score features that make the critic unbiased are the same score features whose covariance is the Fisher metric. Least squares with those features performs the inverse-metric operation.

This is why the idea is more than bolting second-order optimization onto policy gradients. If I were using the Hessian of reward, I would be modeling curvature of `eta(theta)`. That is not what happens. The Hessian of the average reward contains terms involving `grad Q^pi` and terms where `Q^pi` weights policy curvature:

`grad^2 eta = sum rho^pi(grad^2 pi Q^pi + grad pi grad Q^pi^T + grad Q^pi grad pi^T)`.

The Fisher metric has none of the `grad Q^pi` coupling and no value weighting. It is built from the first derivatives of the policy distribution alone. Therefore it is not Newton's method for the control objective, and it should not inherit Amari's parameter-estimation efficiency story automatically. The matrix is chosen because it measures policy-distribution displacement, not because it approximates the reward Hessian.

That distinction is a strength far from an optimum. A Hessian for a maximization problem can be indefinite and can point into unstable curvature before I am near a local maximum. A Fisher metric is positive on the identifiable score subspace, or can be made usable with a ridge, so `F^{-1} grad eta` is an ascent direction whenever the solve is positive and the gradient is nonzero. It gives a coordinate-free direction through policy space, not a promise of Newton convergence.

I still need to understand why this direction aims at good actions rather than merely making a tidy geometric statement. In a Gibbs policy, `pi(a|s,theta) proportional to exp(theta^T phi_sa)`, the score feature is the centered action feature, `phi_sa - E_pi[phi_s.]`. The compatible critic value `w^T psi(s,a)` differs from `w^T phi_sa` only by a state-dependent constant, so it has the same maximizing action. If I step with `theta' = theta + alpha w`, the new logit contains

`theta^T phi_sa + alpha w^T phi_sa`.

As `alpha` grows, the second term dominates and the policy concentrates on actions maximizing the compatible critic. A raw gradient step does not have this property; pushed to the limit, it can select an action that is merely above the current mean under the critic rather than the critic's maximizer. The inverse Fisher direction points at the greedy improvement encoded by the compatible value approximation.

For a general smooth policy I cannot take that infinite-step geometry literally, so I expand locally. With `Delta theta = alpha w`,

`pi(a|s,theta + Delta theta) = pi(a|s,theta) + grad pi(a|s,theta)^T Delta theta + O(alpha^2)`.

Since `grad pi = pi psi`, this becomes

`pi(a|s,theta + alpha w) = pi(a|s,theta)(1 + alpha psi(s,a)^T w) + O(alpha^2)`.

And `psi^T w` is the compatible critic. So the local step increases or decreases action probability in proportion to the critic's advantage-like judgment. The same object is doing three jobs: it is the unbiased critic feature, the Fisher metric's score coordinate, and the direction toward greedy policy improvement.

The final method is now forced. Roll out the current policy. Estimate `grad eta` from score times return or differential value. Estimate `F` from score outer products, or equivalently fit the compatible critic by least squares. Solve `F v = grad eta`, using a pseudoinverse or small ridge if the scores are rank deficient. Step `theta <- theta + alpha v`, preferably with a line search or local KL budget. If I use a local KL budget `epsilon`, then `0.5 alpha^2 grad eta^T F^{-1} grad eta = epsilon`, so `alpha = sqrt(2 epsilon / (grad eta^T v))`.

So the discovery is: the right direction is not the steepest direction in parameter coordinates, and it is not the Newton direction for reward. It is the steepest direction in the space of policy distributions under Fisher geometry. In reinforcement learning that geometry is assembled from per-state action distributions weighted by where the policy goes. The compatible critic then reveals why this geometry matters operationally: the same score covariance that defines the metric makes the fitted critic weights equal the inverse-Fisher direction, and that direction moves the policy toward the action the critic calls greedy.
