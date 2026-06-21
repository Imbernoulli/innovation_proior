# The Policy Gradient Theorem

For a differentiable stochastic policy `pi_theta(a|s)`, the performance gradient can be written without differentiating the environment-induced state weighting:

```text
d rho(theta) / d theta
  = sum_s d^pi(s) sum_a [d pi_theta(a|s) / d theta] Q^pi(s,a).
```

In the average-reward setting, `d^pi` is the stationary distribution. In the start-state discounted setting, `d^pi(s) = sum_{t>=0} gamma^t Pr(S_t=s | S_0=s0, pi)` is the unnormalized discounted occupancy. The point is that `d^pi` is sampled by acting under the policy; its derivative does not appear.

Using `d pi / d theta = pi grad_theta log pi`, the sampleable form is:

```text
d rho(theta) / d theta
  = sum_s d^pi(s) E_{a~pi_theta(.|s)}
      [grad_theta log pi_theta(a|s) Q^pi(s,a)].
```

Any state-only baseline is valid:

```text
sum_a [d pi_theta(a|s)/d theta] b(s)
  = b(s) d/dtheta sum_a pi_theta(a|s)
  = 0.
```

So the estimator may use `Q^pi(s,a) - b(s)`, usually an advantage estimate with `b(s) ~= V^pi(s)`, to reduce variance without changing the expected gradient.

## Compatible Critic

If a learned critic `f_w(s,a)` is fitted on-policy to a least-squares fixed point,

```text
sum_s d^pi(s) sum_a pi_theta(a|s)
  [Q^pi(s,a) - f_w(s,a)] d f_w(s,a)/d w = 0,
```

and satisfies compatibility,

```text
d f_w(s,a) / d w = grad_theta log pi_theta(a|s),
```

then it can replace the true action value exactly:

```text
d rho(theta) / d theta
  = sum_s d^pi(s) sum_a [d pi_theta(a|s)/d theta] f_w(s,a).
```

For soft-max preferences `theta^T phi(s,a)`:

```text
grad_theta log pi_theta(a|s)
  = phi(s,a) - sum_b pi_theta(b|s) phi(s,b),

f_w(s,a)
  = w^T [phi(s,a) - sum_b pi_theta(b|s) phi(s,b)].
```

This critic is mean-zero over actions under the policy in each state, so it estimates an advantage-like signal.
