# The Distributional Bellman Operator

Let an MDP have state space `X`, action space `A`, random reward `R`, transition kernel `P`, policy `pi`, and discount `gamma in [0,1)`. Define the random return

```text
Z^pi(x,a) = sum_{t >= 0} gamma^t R(x_t,a_t),
x_0 = x, a_0 = a, x_t ~ P(.|x_{t-1},a_{t-1}), a_t ~ pi(.|x_t).
```

The ordinary value is the mean `Q^pi(x,a) = E Z^pi(x,a)`. The distributional object is the law of `Z^pi(x,a)` for every `(x,a)`.

## Operator

Define

```text
P^pi Z(x,a) =_D Z(X',A'),       X' ~ P(.|x,a), A' ~ pi(.|X')
T^pi Z(x,a) =_D R(x,a) + gamma P^pi Z(x,a).
```

The equality is equality in distribution. The backup combines three independent sources of randomness: immediate reward, successor state-action, and successor return.

## Evaluation Theorem

For cdfs `F,G`, let

```text
d_p(F,G) = inf_{U,V} ||U - V||_p
         = ||F^{-1}(U_0) - G^{-1}(U_0)||_p,  U_0 ~ Uniform[0,1].
```

Lift this to value distributions by

```text
bar d_p(Z_1,Z_2) = sup_{x,a} d_p(Z_1(x,a), Z_2(x,a)).
```

Then, for every fixed policy `pi`,

```text
bar d_p(T^pi Z_1, T^pi Z_2) <= gamma bar d_p(Z_1,Z_2).
```

The proof uses three Wasserstein facts: common additive noise does not increase the distance, multiplying both variables by `gamma` scales the distance by `gamma`, and the same successor draw can be coupled on both sides. Therefore `T^pi` has a unique fixed point, and that fixed point is the true return distribution `Z^pi`. Iterating the distributional backup converges geometrically in `bar d_p`.

This metric choice is essential. Discounting moves probability mass along the return axis; Wasserstein measures that transport. Total variation, KL, and Kolmogorov-style comparisons do not see the discount as a contraction in the same way.

## Control Caveat

A distributional optimality operator chooses a greedy policy with respect to `E Z`:

```text
T Z = T^pi Z for some pi in G_Z.
```

The mean still contracts because `E T_D Z = T_E E Z`. The full distribution does not. Greedy selection can be discontinuous: an arbitrarily small perturbation in an action's mean can change the argmax and swap in a completely different successor distribution. Consequently the control operator is not a contraction in distribution space, may cycle under tie-breaking, and in general converges only toward nonstationary optimal value distributions. A fixed stationary limit is recovered only under additional consistent tie-breaking over optimal policies.

## Categorical Approximation

Use fixed atoms

```text
z_i = V_min + i Delta z,  i = 0,...,N-1,
Delta z = (V_max - V_min)/(N-1),
p_i(x,a) = softmax(theta_i(x,a)).
```

For a sampled transition `(x,a,r,x')`, choose the next action greedily by expected return, transform each atom,

```text
T_hat z_j = [r + gamma z_j]_{V_min}^{V_max},
```

and project it back onto the fixed grid:

```text
(Phi T_hat Z_theta(x,a))_i =
  sum_j [1 - |[T_hat z_j]_{V_min}^{V_max} - z_i| / Delta z]_0^1
        p_j(x', pi(x')).
```

Train with cross-entropy, equivalently the KL term

```text
D_KL(Phi T_hat Z_target(x,a) || Z_theta(x,a)).
```

The projection is needed because the raw transformed support `{r + gamma z_i}` usually differs from the model support, and the Wasserstein loss cannot in general be optimized by unbiased single-transition stochastic gradients. Later categorical analysis shows the projected operator is contractive in Cramer geometry, explaining why the fixed-grid projection is more than a convenience.

```python
import numpy as np

def project_distribution(rewards, terminals, next_probs, gamma, v_min, v_max, n_atoms):
    z = np.linspace(v_min, v_max, n_atoms)
    dz = (v_max - v_min) / (n_atoms - 1)
    out = np.zeros((len(rewards), n_atoms))

    for b, (r, terminal) in enumerate(zip(rewards, terminals)):
        g = gamma * (1.0 - float(terminal))
        for j, prob in enumerate(next_probs[b]):
            tz = min(v_max, max(v_min, r + g * z[j]))
            bj = (tz - v_min) / dz
            lo, hi = int(np.floor(bj)), int(np.ceil(bj))
            if lo == hi:
                out[b, lo] += prob
            else:
                out[b, lo] += prob * (hi - bj)
                out[b, hi] += prob * (bj - lo)

    return out
```
