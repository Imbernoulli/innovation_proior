The problem is to do reinforcement learning with the full random return rather than only its mean. In standard value-based methods the return is collapsed to an expectation as soon as Bellman's equation is written, so a state-action pair that is usually safe but occasionally catastrophic can look identical to one that is merely mediocre. Earlier work tried to repair this by tracking variances, parametric densities, or risk-sensitive summaries, but those are side computations around a scalar value, not a replacement for the value object itself. What is missing is a fixed-point theory that treats the entire probability law of the return as the Bellman quantity.

The obstacle is that the usual proof tools do not survive the move from scalars to distributions. Discounting slides probability mass along the return axis, so a distance that only measures overlap or likelihood will not contract under the backup. Greedy control is worse: a tiny change in an action's mean can flip the argmax and replace one return law with another, so the optimality operator is discontinuous in distribution space. A usable theory must therefore separate policy evaluation, where contraction is possible, from greedy control, where it is not.

The method is the Distributional Bellman operator. For a fixed policy pi, define the random return Z^pi(x,a) as the discounted sum of rewards along trajectories starting from (x,a). The distributional Bellman evaluation operator is T^pi Z(x,a) =_D R(x,a) + gamma Z(X',A'), where the right-hand side mixes immediate reward, random successor, and successor return. To prove contraction we measure laws with the maximal Wasserstein distance over state-action pairs. Scaling a random variable by gamma scales its Wasserstein distance by gamma, adding a common reward does not increase distance, and the same successor draw can be coupled on both sides. This gives bar d_p(T^pi Z_1, T^pi Z_2) <= gamma bar d_p(Z_1, Z_2), so T^pi has a unique fixed point and iteration converges geometrically to the true return law. The ordinary Q-value is only the first moment of this law.

For control, the distributional optimality operator selects actions greedily by expected return and applies the corresponding evaluation backup. The mean sequence still contracts because expectation commutes with the backup, but the full distribution does not: a small perturbation can change the greedy action and swap in a very different successor law. So the distributional theory gives a strong fixed-point result for evaluation and simultaneously explains why greedy control over distributions is intrinsically discontinuous.

To make this learnable, the return law at each (x,a) is represented as a categorical distribution over a fixed grid of atoms z_i from V_min to V_max. Given a sampled transition, the next action is chosen by expected return, each atom z_j is transformed to r + gamma z_j, and the resulting probability mass is projected linearly back onto the grid. The network is trained to predict this projected target by minimizing the cross-entropy between the projected target distribution and the predicted distribution. This categorical projection, known as the C51 algorithm, turns the continuous backup into a stable supervised update and can be shown to contract in Cramer distance.

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
