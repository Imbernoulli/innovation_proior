CRPO is a primal safe-RL policy-optimization rule for

```text
max_pi J_0(pi)    subject to    J_i(pi) <= d_i,  i = 1, ..., p.
```

It removes the Lagrange multipliers and chooses the next unconstrained policy-optimization target from
the current constraint estimates.

## Algorithm

At iteration `t`:

1. Estimate reward and cost action-values `\bar Q_t^i` under `pi_{w_t}` for `i = 0, ..., p`.
2. Estimate each constraint return
   `\bar J_{i,B_t} = sum_{j in B_t} rho_{j,t} \bar Q_t^i(s_j, a_j)`.
3. If `\bar J_{i,B_t} <= d_i + eta` for every `i >= 1`, add `w_t` to `N_0` and take one
   unconstrained policy step maximizing `J_0`.
4. Otherwise choose a violated constraint `i_t` with `\bar J_{i_t,B_t} > d_{i_t} + eta` and take one
   unconstrained policy step minimizing `J_{i_t}`.
5. Return `w_out` sampled uniformly from `N_0` in the theoretical algorithm.

For tabular softmax NPG, the update signs are:

```text
reward step:          w_{t+1} = w_t + alpha * \bar Delta_t
violated-cost step:   w_{t+1} = w_t - alpha * \bar Delta_t
\bar Delta_t(s,a) = (1 - gamma)^(-1) \bar Q_t^i(s,a)
```

Equivalently, a policy-gradient implementation uses the reward advantage when approximately feasible and
`-A_i` for a violated cost.

## Guarantee

With tabular softmax policies, sufficiently accurate TD policy evaluation, and
`K_in = Theta(T^(1/sigma) (1 - gamma)^(-2/sigma) log^(2/sigma)(T^(1+2/sigma) / delta))`,
the theorem uses

```text
eta   = Theta(sqrt(|S||A|) / ((1 - gamma)^1.5 sqrt(T)))
alpha = (1 - gamma)^1.5 / sqrt(|S||A| T)
```

and, with probability at least `1 - delta`,

```text
J_0(pi*) - E[J_0(w_out)]
  <= Theta(sqrt(|S||A|) / ((1 - gamma)^1.5 sqrt(T)))

E[J_i(w_out)] - d_i
  <= Theta(sqrt(|S||A|) / ((1 - gamma)^1.5 sqrt(T)))   for all i = 1, ..., p.
```

With neural softmax policies, the update is applied to the temperature-scaled parameter:

```text
tau_{t+1} = tau_t + alpha
reward step:          tau_{t+1} W_{t+1} = tau_t W_t + alpha * \bar Delta_t
violated-cost step:   tau_{t+1} W_{t+1} = tau_t W_t - alpha * \bar Delta_t
```

Under the stated regularity assumptions, `K_in = Theta((1 - gamma)^2 sqrt(m))`,
`alpha = Theta(1 / sqrt(T))`, and
`eta = Theta(m / ((1 - gamma) sqrt(T)) + 1 / ((1 - gamma)^2.5 m^(1/8)))`. The objective and every
constraint violation are bounded by a `1 / ((1 - gamma) sqrt(T))` term plus a finite-width term of order
`(1 - gamma)^(-2.5) m^(-1/8) log^(1/4)(((1 - gamma)^2 T sqrt(m)) / delta)`.

## Code Reference

OmniSafe's upstream `OnCRPO` at commit `15603dd7a654a991d0a4648216b69d60b81a6366` is a
single-constraint, on-policy TRPO-backed implementation of the same switch:

```python
Jc = self._logger.get_stats('Metrics/EpCost')[0]
if Jc <= self._cfgs.algo_cfgs.cost_limit + self._cfgs.algo_cfgs.distance:
    return adv_r
return -adv_c
```

The default config names the tolerance `distance` and sets it to `2.0`; the cost estimate comes from
logged episodic cost. The local reference files match the upstream
`omnisafe/algorithms/on_policy/primal/crpo.py` and `omnisafe/configs/on-policy/OnCRPO.yaml`
byte-for-byte. This is faithful to the theoretical sign and threshold logic, while the theoretical
algorithm is more general: it allows multiple constraints, chooses one violated constraint, and samples
the final output from the approximately feasible set `N_0`.
