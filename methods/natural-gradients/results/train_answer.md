Direct policy search is attractive for large or partially observed control problems because I can adjust the parameters of a stochastic policy rather than building a value function and acting greedily from it. A small parameter change makes a small change in behavior, stochastic policies remain available, and convergence arguments can be phrased directly about the policy parameters. The weak point of this approach is the update direction. The conventional rule follows the raw derivative of long-run reward with respect to the chosen coordinates, and in doing so it silently treats the parameter vector as a flat Euclidean object. If the same policy class is reparameterized, the derivative transforms as a covector while the applied parameter displacement transforms as a tangent vector, so two coordinate descriptions of the same policy can produce different physical moves in policy space. The step I take therefore depends on my coordinates, not on the policy I am changing. A usable replacement must make the step a property of the policy itself, while still allowing the direction to be estimated from rollouts.

I propose the Natural Policy Gradient as that replacement. The first question is not how to estimate the gradient more accurately, but what I mean by a small move. The ordinary gradient is steepest only after I secretly decide that the squared length of a parameter displacement is the sum of squared coordinate changes, which is the identity metric in the current chart. If instead I maximize the linearized reward change subject to a fixed length measured by a metric G, the Lagrange condition gives the direction G inverse times the ordinary gradient. The raw gradient is just the special case where I made the arbitrary choice G equals the identity. So the real problem is choosing the metric, and a policy is valuable not because its coordinate vector moved a short Euclidean distance, but because the action distributions it induces changed in a controlled way. A one-unit shift in the mean of a narrow Gaussian policy can almost replace the distribution, while the same one-unit shift in a broad Gaussian barely changes it. The Euclidean parameter distance says those moves are equal, but the distributions say they are not.

The right primitive is Amari's information geometry. When a parameter indexes a probability distribution, the Fisher information is the invariant local metric on that distribution family, and it is also the second-order local form of KL divergence. Capping a Fisher length caps local policy change rather than coordinate motion, and the direction F inverse times the gradient is genuinely the steepest direction on the distribution manifold rather than a preconditioning trick. The complication is that a policy is not one distribution. At every state the parameters define a conditional action distribution, so each state carries its own statistical manifold, while the objective averages consequences under the stationary distribution of the current policy. I therefore build the metric in two stages. For a single state I use the Fisher metric of that conditional, with score defined as the gradient of the log probability, and then I weight the per-state metrics by exactly the distribution the objective uses, namely the current stationary distribution. The dynamics enter only through the stationary distribution; the per-state Fisher depends only on how the action distribution reacts to the parameters.

The natural policy-gradient direction is then the metric steepest ascent direction, obtained by solving the linear system F times v equals the policy gradient g, using a pseudoinverse or a small ridge when the score covariance is rank deficient. This preserves the central advantage of policy gradients. The average-reward gradient from the policy-gradient theorem contains no explicit derivative of the stationary distribution; the hard state-distribution derivative cancels. The metric F is an expectation under the same on-policy state-action distribution, just with score outer products instead of score times value. Both objects are estimable from rollouts, so I have changed the geometry of the direction without ever differentiating the environment.

What makes this more than information geometry pasted onto policy search is that the inverse-Fisher solve is already hiding inside actor-critic machinery. By Sutton's compatibility condition, if I approximate the action-value function with a linear critic whose features are precisely the score, then substituting it into the gradient formula introduces no bias once the approximation error is orthogonal to those features. Fitting the critic by weighted least squares gives normal equations whose left bracket is exactly the Fisher metric and whose right side is exactly the policy gradient, since probability times score equals the gradient of the probability. Hence the compatible critic weights are exactly the natural policy-gradient direction. The same score features that make the critic unbiased are the ones whose covariance is the Fisher metric, and least squares with them performs the inverse-metric operation.

This also explains why the direction aims at good actions rather than merely making a tidy geometric statement, and why it is emphatically not Newton's method. For an exponential-family policy the score is the centered feature, so the action that maximizes the compatible critic is the action that maximizes the raw feature inner product. After stepping along the natural direction, the new logit contains the original term plus the critic's weight vector times the feature, and as the step size grows the policy concentrates exactly on the actions maximizing the compatible critic, which is the greedy move of approximate policy iteration. For a general smooth policy the same fact holds locally: the update scales each action's probability by the critic's advantage-like value. By contrast the reward Hessian contains value-coupled terms and policy curvature weighted by values; the Fisher metric contains only score outer products and is chosen because it measures local policy-distribution displacement and is invariant to reparameterization, not because it approximates objective curvature. It therefore stays positive on the identifiable score subspace and yields an ascent direction even far from an optimum where a Hessian could turn indefinite.

The method is then forced. I roll out the current policy, estimate the policy gradient from score times return or differential value, estimate the Fisher matrix from score outer products or equivalently fit the compatible critic by least squares, solve F v equals g with a pseudoinverse or ridge if the scores are rank deficient, and step the parameters along v. To keep each step a controlled change in the policy distributions I size the step length by a local KL budget, which gives a step size proportional to the square root of the budget divided by the directional derivative along v.

```python
import numpy as np

np.random.seed(0)

# A tiny average-reward MDP: two states, two actions.
states = [0, 1]
actions = [0, 1]
R = np.array([[1.0, 0.0],    # rewards for (state, action)
              [0.0, 1.0]])
P = np.array([[[0.7, 0.3],   # P(next_state | state, action)
               [0.3, 0.7]],
              [[0.3, 0.7],
               [0.7, 0.3]]])

def stationary(rho, pi, P, eps=1e-10):
    """Return stationary distribution over states."""
    P_pi = np.einsum('sap,sa->sp', P, pi)
    while True:
        rho_next = rho @ P_pi
        if np.linalg.norm(rho_next - rho, ord=1) < eps:
            return rho_next
        rho = rho_next

def softmax_policy(theta):
    """theta[s,a] -> stochastic policy pi[s,a]."""
    e = np.exp(theta - theta.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def score_vector(theta, pi, s, a):
    """Vectorized policy score: grad_theta log pi(a|s) for softmax logits."""
    psi = np.zeros_like(theta)
    psi[s, a] = 1.0
    psi[s, :] -= pi[s, :]
    return psi.ravel()

def q_value(pi, R, P, gamma=0.99, max_iter=1000):
    """Tabular action-value function under policy pi."""
    Q = np.zeros((2, 2))
    for _ in range(max_iter):
        V = (pi * Q).sum(axis=1)
        Q_next = R + gamma * np.einsum('sap,p->sa', P, V)
        if np.linalg.norm(Q_next - Q) < 1e-10:
            return Q_next
        Q = Q_next
    return Q

def policy_gradient_and_fisher(theta):
    pi = softmax_policy(theta)
    rho = stationary(np.array([0.5, 0.5]), pi, P)
    Q = q_value(pi, R, P)

    grad = np.zeros(theta.size)
    fisher = np.zeros((theta.size, theta.size))

    for s in states:
        for a in actions:
            psi = score_vector(theta, pi, s, a)
            grad += rho[s] * pi[s, a] * psi * Q[s, a]
            fisher += rho[s] * pi[s, a] * np.outer(psi, psi)

    return grad, fisher, pi, rho, Q

# Initialize a nearly uniform softmax policy.
theta = np.array([[0.1, -0.1],
                  [-0.1, 0.1]], dtype=float)

grad, fisher, pi0, rho0, Q0 = policy_gradient_and_fisher(theta)
v = np.linalg.solve(fisher + 1e-4 * np.eye(fisher.shape[0]), grad)

# Step size from a small KL budget.
epsilon = 0.01
alpha = np.sqrt(2 * epsilon / max(grad @ v, 1e-12))
theta_new = theta + alpha * v.reshape(theta.shape)
pi1, rho1 = softmax_policy(theta_new), stationary(np.array([0.5, 0.5]), softmax_policy(theta_new), P)

print("Initial policy:")
print(pi0)
print("Stationary rho0:", rho0)
print("Average reward eta0:", (rho0[:, None] * pi0 * R).sum())
print("Natural gradient direction (reshaped):")
print(v.reshape(theta.shape))
print("Step size alpha:", alpha)
print("Updated policy:")
print(pi1)
print("Average reward eta1:", (rho1[:, None] * pi1 * R).sum())
```
