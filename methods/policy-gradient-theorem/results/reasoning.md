I start with the part that blocks the whole program. I want a smooth stochastic policy `pi_theta(a|s)` and I want to improve `rho(theta)`, the long-run reward. If I write the average-reward objective in the most direct way, I get `rho(theta) = sum_s d_theta(s) sum_a pi_theta(a|s) R(s,a)`. The temptation is to differentiate this directly. When I do, the product rule gives me two pieces: a piece where the derivative lands on `pi`, and a piece where it lands on `d_theta(s)`. The second piece is `sum_s [d d_theta(s)/d theta] sum_a pi(a|s) R(s,a)`. That is the derivative of the stationary distribution of the Markov chain induced by the policy. It depends on the environment's transition probabilities and on how the whole chain shifts when I perturb the controller. I can sample states by acting, but I cannot sample a derivative of the environment's induced state distribution.

Before I assume that term is a real problem, I should check that it is actually large rather than something that quietly vanishes. I take a concrete tiny MDP: two states, two actions, transition rows `P(.|0,0)=(0.8,0.2)`, `P(.|0,1)=(0.3,0.7)`, `P(.|1,0)=(0.4,0.6)`, `P(.|1,1)=(0.9,0.1)`, rewards `R = [[1,2],[0.5,-1]]`, and a soft-max policy at preferences `theta=[[0.3,-0.2],[0.1,0.4]]`. I compute the true gradient of `rho` by finite differences, and separately the "hold `d` fixed" piece where I only differentiate the one-step reward through `pi`. They are not the same: the finite-difference gradient is `[[-0.0288, 0.0288],[0.0611,-0.0611]]`, the policy-only piece is `[[-0.150, 0.150],[0.132,-0.132]]`, and the gap — which is exactly the distribution-derivative term I want to avoid — has magnitude about `0.12`. So this term is genuinely part of the gradient. Ignoring it is not an option, and computing it would require modeling the environment. If it really has to stay, direct policy search is not going to be model-free in the way I need.

So differentiating the one-step reward expression is the wrong move; it exposes the distribution derivative immediately. I should instead differentiate the value recursion, because the value recursion is where the future-state dependence has a chance to telescope across steps rather than sit inside a stationary-distribution derivative. In the average-reward setting the differential action value obeys `Q(s,a) = R(s,a) - rho + sum_s' P(s'|s,a) V(s')`, and `V(s) = sum_a pi(a|s) Q(s,a)`. Differentiating the value gives

`dV(s)/dtheta = sum_a [dpi(a|s)/dtheta Q(s,a) + pi(a|s) dQ(s,a)/dtheta]`.

The reward and transition probabilities do not depend on theta, so differentiating the Bellman equation gives

`dQ(s,a)/dtheta = -d rho/dtheta + sum_s' P(s'|s,a) dV(s')/dtheta`.

Substituting this back into the derivative of `V` gives

`dV(s)/dtheta = sum_a dpi(a|s)/dtheta Q(s,a) - d rho/dtheta + sum_a pi(a|s) sum_s' P(s'|s,a) dV(s')/dtheta`.

Now I isolate the thing I actually want:

`d rho/dtheta = sum_a dpi(a|s)/dtheta Q(s,a) + sum_a pi(a|s) sum_s' P(s'|s,a) dV(s')/dtheta - dV(s)/dtheta`.

This identity holds for every current state `s`. The `dV/dtheta` terms have not disappeared, and right now they look just as model-dependent as the term I was running from. But their form is suggestive. One term is the derivative of value after one policy/environment transition, the other is the derivative of value before that transition, and both carry the same unknown function `dV/dtheta`. If I weight states by something that one policy/environment step leaves invariant, the two copies could line up and subtract.

The natural invariant weighting is the stationary distribution. I multiply by `d^pi(s)` and sum over `s`. The left side stays `d rho/dtheta` because the stationary weights sum to one. The `-dV(s)/dtheta` term becomes `- sum_s d^pi(s) dV(s)/dtheta`. The transported term `sum_s d^pi(s) sum_a pi(a|s) sum_s' P(s'|s,a) dV(s')/dtheta` collects, in front of each `dV(s')/dtheta`, the coefficient `sum_s d^pi(s) sum_a pi(a|s) P(s'|s,a)`. That coefficient is `(d^pi P_pi)(s')`, where `P_pi` is the policy-induced state-to-state matrix. For these two terms to cancel I need `d^pi P_pi = d^pi`, which is exactly the defining property of the stationary distribution.

I should not just assert that this holds; the whole derivation rests on it, so I check it on the same two-state MDP. I form `P_pi[s,s'] = sum_a pi(a|s) P(s'|s,a)`, solve for the stationary left eigenvector, and get `d^pi = (0.6387, 0.3613)`. Then `d^pi P_pi = (0.6387, 0.3613)` as well — equal to `d^pi` to the last printed digit (the difference is exactly zero in the computation). So the coefficient in front of each `dV(s')/dtheta` from the transported term really is `d^pi(s')`, and the transported term is `sum_s' d^pi(s') dV(s')/dtheta`, which is identical to the `sum_s d^pi(s) dV(s)/dtheta` term. They cancel for any value of the unknown `dV/dtheta`, because the cancellation is in the weights, not in the values. What remains is

`d rho/dtheta = sum_s d^pi(s) sum_a dpi(a|s)/dtheta Q^pi(s,a)`.

The state distribution still appears, but only as a weighting. I do not differentiate it. Acting under the current policy gives me that weighting in the average-reward case once the chain is in stationarity. The derivative lands only on the policy probability, which is under my control.

I want to confirm this is really the gradient and not an expression that merely looks like one, so I compute both sides on the same MDP. I solve the differential Bellman equations for `V` (normalizing `d^pi . V = 0`) and `Q`, form `sum_s d^pi(s) sum_a [dpi/dtheta] Q`, and compare against the finite-difference gradient of `rho`. The two agree to about `4e-11` per entry — both equal `[[-0.0288,0.0288],[0.0611,-0.0611]]`. That is the same finite-difference gradient I computed at the start, the one the naive policy-only piece missed by `0.12`. So the value-recursion route recovers the full gradient, distribution-derivative term included, without ever differentiating the distribution. The obstacle was real, and it is the stationarity identity that absorbs it.

I need to check that this is not just an average-reward accident. In the start-state discounted case, `rho(theta) = V(s0)`, and `Q(s,a) = R(s,a) + gamma sum_s' P(s'|s,a) V(s')`. Differentiating gives

`dV(s)/dtheta = sum_a dpi(a|s)/dtheta Q(s,a) + sum_a pi(a|s) sum_s' gamma P(s'|s,a) dV(s')/dtheta`.

Here there is no `-d rho/dtheta` term, so there is nothing to cancel against and the stationarity trick does not apply directly. Instead I unroll the recursion. The first expansion gives the local score-weighted term at `s`, then a discounted copy of the same derivative at the next state, then another discounted copy, and so on. After `k` steps, the contribution from state `x` is weighted by `gamma^k Pr(s -> x in k steps under pi)`. Setting `s = s0`, I get

`d rho/dtheta = sum_x [sum_{k>=0} gamma^k Pr(s0 -> x in k steps under pi)] sum_a dpi(a|x)/dtheta Q^pi(x,a)`.

The bracket is the discounted occupancy of state `x` from the start state. It is not a normalized stationary distribution — and I want to be careful about that, because if I quietly normalized it I would get a wrong scale. On the MDP, with `gamma=0.9` and `s0=0`, the occupancy `e0^T (I - gamma P_pi)^{-1}` is `(6.725, 3.275)`, which sums to `10`, i.e. `1/(1-gamma)`, not to one. So it is genuinely an unnormalized weighting generated by walking forward from `s0` with discount weights. With that weighting the same form holds:

`d rho/dtheta = sum_s d^pi(s) sum_a dpi(a|s)/dtheta Q^pi(s,a)`,

where `d^pi` means stationary occupancy in the continuing average-reward setting and unnormalized discounted occupancy in the start-state setting. I check this numerically too: solving the discounted Bellman equations for `Q`, weighting the score-times-`Q` term by that unnormalized occupancy, and comparing to the finite-difference gradient of `V(s0)`. They agree to about `3e-9` per entry. The discounted case lands on the same expression, by unrolling instead of by stationarity, with the derivative of the occupancy again absent.

Now I have a finite action sum, but a sampled trajectory gives me one action, not every action. I rewrite the action sum under the policy distribution:

`sum_a dpi(a|s)/dtheta Q(s,a) = sum_a pi(a|s) [dpi(a|s)/dtheta / pi(a|s)] Q(s,a)`.

The ratio is `grad_theta log pi_theta(a|s)`. Therefore

`d rho/dtheta = sum_s d^pi(s) E_{a~pi(.|s)}[grad_theta log pi_theta(a|s) Q^pi(s,a)]`.

This is the sampleable form. The state comes from the on-policy occupancy; the action comes from the policy; the derivative is the log-probability score of the sampled action. In the discounted start-state case I carry the `gamma^t` occupancy weight along the trajectory. In the average-reward case I use the stationary on-policy stream and a differential return.

The obvious estimator is to replace `Q^pi(s_t,a_t)` by a sampled return. That gives the REINFORCE-style update, but it has high variance. I need to know what I can subtract. If I subtract a state-only baseline `b(s)`, the extra term in the exact gradient is

`sum_a dpi(a|s)/dtheta b(s) = b(s) d/dtheta sum_a pi(a|s) = b(s) d/dtheta 1 = 0`.

The baseline costs nothing in expectation because action probabilities sum to one. In sampled form this is the expected-score identity `E[grad log pi(a|s)] = 0`. I confirm it does not perturb the gradient on the MDP: subtracting `V^pi(s)` from `Q^pi(s,a)` and recomputing the weighted score term changes the gradient by about `3e-17`, i.e. nothing beyond floating-point noise. So I can replace `Q` by `Q - b(s)` and choose `b` to reduce variance, with `V(s)` as the natural target. The signal becomes an advantage.

The next question is the critic. If I simply plug in an arbitrary learned `f_w(s,a)` for `Q^pi(s,a)`, I have a biased direction, because the approximation error couples into the gradient. I want the approximation error to be invisible specifically to the policy-gradient direction. Suppose I train `f_w` by on-policy least squares toward an unbiased estimate of `Q^pi`. At a local optimum, the residual is orthogonal to the critic's parameter-gradient features:

`sum_s d^pi(s) sum_a pi(a|s) [Q^pi(s,a) - f_w(s,a)] df_w(s,a)/dw = 0`.

But the error I need to vanish is the one carried by the policy derivative:

`sum_s d^pi(s) sum_a dpi(a|s)/dtheta [Q^pi(s,a) - f_w(s,a)]`.

These two are different sums in general. They coincide if the critic feature `df_w/dw` equals `dpi(a|s)/dtheta`, and since the orthogonality already carries a `pi(a|s)` weight, what I actually need is

`df_w(s,a)/dw = dpi(a|s)/dtheta / pi(a|s) = grad_theta log pi_theta(a|s)`.

With this compatibility condition, the least-squares orthogonality becomes exactly

`sum_s d^pi(s) sum_a dpi(a|s)/dtheta [Q^pi(s,a) - f_w(s,a)] = 0`.

Now I can subtract this zero from the exact gradient:

`sum_s d^pi sum_a dpi/dtheta Q - sum_s d^pi sum_a dpi/dtheta (Q - f_w) = sum_s d^pi sum_a dpi/dtheta f_w`.

So if the critic is at the fixed point of the least-squares objective and its features are the score, plugging it in does not merely approximate the gradient; it gives the same value as the true action value. I want to see this actually happen rather than trust the algebra alone, because "the residual is orthogonal at the fixed point" is the kind of statement that is easy to state and easy to get slightly wrong. On the MDP I take one-hot `(s,a)` features, center them per state, fit `w` by `d^pi`-and-`pi`-weighted least squares to the true differential `Q`, and then compute the gradient both with the true `Q` and with the fitted `f_w`. With true `Q` the gradient is `[[-0.0288,0.0288],[0.0611,-0.0611]]`; with the fitted compatible critic it is the same to about `1e-17`. And the fitted `f_w` has policy-weighted mean exactly `0` in each state. The fit does not reproduce `Q` itself — it cannot, since it is mean-zero per state — but it reproduces `Q` along the only direction the gradient reads.

For a soft-max policy with preferences `theta^T phi(s,a)`, the score is

`grad_theta log pi(a|s) = phi(s,a) - sum_b pi(b|s) phi(s,b)`.

Therefore the compatible critic must be linear in the centered policy features:

`f_w(s,a) = w^T [phi(s,a) - sum_b pi(b|s) phi(s,b)]`.

This is consistent with the mean-zero behavior I just saw in the fit: `sum_a pi(a|s) f_w(s,a) = w^T [sum_a pi(a|s) phi(s,a) - sum_b pi(b|s) phi(s,b)] = 0` in every state. So `f_w` cannot be an absolute `Q` estimate unless `V(s)` happens to be zero; it is really an advantage estimator, which is exactly what the baseline argument said the gradient only ever uses.

The whole route now fits together, and each joint has been checked rather than asserted. I stop trying to differentiate the state distribution — which the first finite-difference comparison showed is a real, order-`0.1` part of the gradient on a concrete MDP — and instead differentiate the value recursion. Stationarity (verified as `d^pi P_pi = d^pi`) cancels the transported `dV/dtheta` terms in the average-reward case; discounted unrolling creates the unnormalized occupancy weights (verified to sum to `1/(1-gamma)`) in the start-state case; both forms reproduce the finite-difference gradient. The remaining derivative is only a policy derivative. Dividing by the sampled action probability turns it into a log-policy score. The expected score is zero, so state baselines do not bias the update (verified to machine precision). Finally, a critic whose features equal the log-policy score has an error orthogonal to the only direction the controller update uses, so the actor-critic update reproduces the exact gradient when the critic is fitted to its fixed point (verified to machine precision on the MDP).
