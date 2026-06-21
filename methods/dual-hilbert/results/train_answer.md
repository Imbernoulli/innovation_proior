The problem is how to represent goals for an offline goal-conditioned reinforcement-learning agent. The usual approach passes the raw observation of the goal straight into the value function and the policy, so the agent receives something like V(s, g) and pi(a | s, g). Raw observations are a poor coordinate system for reaching because visual or coordinate similarity is not reachability similarity. Two states can look almost identical yet be separated by a wall, while two visually different states can be one controllable step apart. They also carry exogenous nuisance variation such as textures, lighting, and proprioceptive noise. When the goal is given in raw observation space, the downstream networks must simultaneously learn to ignore the noise, recover the environment's long-horizon reachability geometry, and learn control, all from a fixed offline dataset. Existing representation ideas also fall short for this slot: reconstruction-based encoders preserve appearance details that are irrelevant to reaching, temporal contrastive methods capture local trajectory smoothness without tying the embedding to the Bellman objective, and frozen feature extractors hand the downstream agent geometry it still has to relearn. What is needed is a goal code that is trained by the same goal-reaching value objective the agent already uses, so that distances in the code space directly encode temporal reachability.

The method is Hilbert Goal Representation. It learns a single shared encoder phi that maps both states and goals into a Euclidean, inner-product space, so that the latent l2 distance between phi(s) and phi(g) approximates the optimal temporal distance d*(s, g). The key fact that makes this possible is the classical identity V*(s, g) = -d*(s, g), which holds for the goal-reaching reward that costs minus one for every step until the goal. Because temporal distance and goal-conditioned value are the same object, we can learn the representation by parameterizing the value function as a distance and fitting it with standard offline value-learning machinery. The parameterization is V(s, g) = -||phi(s) - phi(g)||. Training this V to satisfy the goal-conditioned Bellman equations pushes ||phi(s) - phi(g)|| toward d*(s, g), so phi becomes an approximate isometry of the MDP's temporal structure. The goal code fed to the downstream actor and value is simply phi(g).

Several choices in the design flow directly from this parameterization. First, using a single shared encoder for states and goals is natural because the value expression is symmetric in its two arguments. Second, V(s, s) = 0 and V <= 0 are guaranteed by construction, facts that an unconstrained value network would otherwise have to discover. Third, the embedding space is deliberately a Hilbert space rather than an arbitrary metric space because the l2 norm is induced by an inner product. That inner product gives the space an algebra of directions, so phi(g) - phi(s) is itself an actionable control signal and the optimality argument can run through Cauchy-Schwarz. The representation is learned offline with action-free expectile regression in the style of IQL and HIQL. Offline, the Bellman max over next states is unsafe because it would query out-of-distribution transitions; instead an upper expectile over the dataset's own next states approximates the in-sample max. A twin-head target network and a shared target advantage are used for stability, and the square root is floored to avoid gradient explosion when phi(s) is close to phi(g). Because optimal temporal distance is a quasimetric, because not every finite metric embeds isometrically into l2, and because a discount is used for numerical stability, the learned phi is best described as the best discounted symmetric Hilbert approximation of the reachability structure, which is sufficient for the downstream agent.

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence

from jaxrl_m.networks import MLP, ensemblize, default_init
from jaxrl_m.typing import PRNGKey, Shape, Dtype, Array


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable = nn.gelu
    activate_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    skill_dim: int = 32
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self):
        repr_class = LayerNormMLP if self.use_layer_norm else MLP
        if self.ensemble:
            repr_class = ensemblize(repr_class, 2)
        phi = repr_class((*self.hidden_dims, self.skill_dim), activate_final=False)
        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi

    def get_phi(self, observations):
        return self.phi(observations)[0]

    def __call__(self, observations, goals=None, info=False):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        return -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))


def expectile_loss(adv, diff, expectile):
    weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * (diff ** 2)


def compute_value_loss(agent, batch, network_params):
    batch['masks'] = 1.0 - batch['rewards']
    batch['rewards'] = batch['rewards'] - 1.0

    next_v1, next_v2 = agent.network(
        batch['next_observations'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    v1_t, v2_t = agent.network(
        batch['observations'], batch['goals'], method='target_value')
    v_t = (v1_t + v2_t) / 2.0
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    v1, v2 = agent.network(
        batch['observations'], batch['goals'], method='value', params=network_params)
    v = (v1 + v2) / 2.0

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'accept prob': (adv >= 0).mean(),
    }
```
