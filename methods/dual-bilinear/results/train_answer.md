Offline goal-conditioned reinforcement learning asks a single policy to reach many goals from many starts using only a fixed, reward-free dataset. The first description the policy sees is usually the raw goal observation, but that observation is a sensing format rather than a task format. It bundles together the controllable aspects that matter for reaching with exogenous noise such as background pixels, lighting, distractor objects, or nuisance sensor coordinates. Two observations can describe the same underlying task while looking different, and two observations that look close can require very different control. Existing representations do not cleanly strip away that noise while preserving what is needed for optimal control. The identity mapping keeps everything, including the exogenous variation, so the downstream learner must infer invariances entirely from finite data. A variational information bottleneck compresses generically, but its KL penalty has no way to know that a robot pose matters while a texture patch does not. VIP and HILP tie geometry to temporal distance, but they impose a metric structure that is symmetric, whereas reachability is generally directed: falling down can be easy while climbing back is hard. Temporal representation alignment and BYOL-gamma learn from how the data moves, not from how an optimal agent would move, so they can encode behavioral occupancy instead of optimal reaching.

The right coordinate system is the optimal temporal relation itself. In a deterministic environment the optimal time-to-goal d*(s, g) is the shortest path length; with a sparse absorbing reward r(s, g) = I(s = g), the optimal goal-conditioned value satisfies V*(s, g) = gamma^{d*(s, g)}. A goal should therefore be represented by the whole function of its incoming temporal distances, phi^vee(g) = (s |-> d*(s, g)). This dual representation is sufficient because the optimal greedy action only needs successor values gamma^{d*(s', g)}, and a policy that selects actions by maximizing the expected successor value under the representation recovers the optimal Q-function. It is also noise-invariant in a block controlled Markov process: two goal observations that map to the same latent induce identical rewards on every trajectory, hence identical value functions, so observation noise that does not change the latent task is discarded.

The method is called dual goal representations, and the practical instantiation described here is dual-bilinear goal representations. It learns a square-root-scaled bilinear value V_rep(s, g) = psi(s)^T phi(g) / sqrt(d), where psi(s) and phi(g) are separate MLP embeddings. The inner product is chosen instead of a metric aggregator because a bilinear form with learned feature maps can approximate arbitrary continuous two-argument functions and naturally represents directed reaching costs, while a shared Euclidean distance is symmetric and not universal. Dividing by sqrt(d) keeps the dot-product scale order-one across embedding dimensions, matching the scaling convention used in attention mechanisms. The goal-side embedding phi(g) is exported as the compact goal code consumed by the downstream policy and value networks.

The bilinear value is trained with goal-conditioned implicit Q-learning. A twin target critic provides an in-support Q estimate, and an expectile regression loss pulls V_rep toward an upper expectile of that target. For residual u = bar Q(s, a, g) - V_rep(s, g), the loss is ell_kappa^2(u) = |kappa - I(u < 0)| u^2 with kappa > 0.5, so targets above the current value receive the larger weight. The twin online critic is trained by ordinary TD regression to the bilinear value, and the target critic is EMA-updated. This approximates an in-support maximum without evaluating actions outside the offline dataset. With OGBench's shifted reward r(s, g) = I(s = g) - 1, the converged value is still a monotone transform of d*(s, g), so the goal embedding carries the optimal temporal relation either way.

Because the structured bilinear head is useful for learning the goal code but can be too constrained for policy extraction, the learned phi(g) is passed to a separate monolithic offline GCRL learner such as GCIVL, CRL, or GCFBC. That learner trains pi(a | s, phi(g)) and its own value functions on the same relabeled batches. The representation objective and the control objective are therefore decoupled: one shapes a compact, task-relevant goal code, the other extracts actions from it. This separation matters in practice because the goal representation can focus on where to reach while the downstream controller still reasons about joint angles, velocities, and contact forces.

```python
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.networks import GCBilinearValue, GCValue


class BilinearGoalValue(nn.Module):
    """Bilinear value whose goal branch is the exported goal code."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        self.network = GCBilinearValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            layer_norm=self.layer_norm,
            ensemble=self.ensemble,
        )

    def __call__(self, observations, goals=None):
        if goals is not None:
            return self.network(observations, goals, actions=None, info=False)

        dummy_observations = jnp.zeros_like(observations)
        _, _, goal_branch = self.network(
            dummy_observations, observations, actions=None, info=True)
        return goal_branch.mean(axis=0) if self.ensemble else goal_branch


class GoalRepresentation(nn.Module):
    """Goal-side embedding from a bilinear temporal-value learner."""

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True
    rep_expectile: float = 0.7
    discount: float = 0.99

    def setup(self):
        self.rep_value = BilinearGoalValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.rep_dim,
            layer_norm=self.layer_norm,
            ensemble=True,
        )
        self.rep_critic = GCValue(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            ensemble=True,
        )
        self.target_rep_critic = GCValue(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            ensemble=True,
        )

    def encode_goal(self, goals):
        return self.rep_value(goals)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * (diff ** 2)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        q1_t, q2_t = self.target_rep_critic(observations, goals, actions)
        q_t = jax.lax.stop_gradient(jnp.minimum(q1_t, q2_t))

        v = self.rep_value(observations, goals).mean(axis=0)
        adv = q_t - v
        value_loss = self.expectile_loss(
            adv, adv, self.rep_expectile).mean()

        next_v = self.rep_value(next_observations, goals).mean(axis=0)
        td_target = jax.lax.stop_gradient(
            rewards + self.discount * masks * next_v)
        q1, q2 = self.rep_critic(observations, goals, actions)
        critic_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()

        info = {
            'rep/value_loss': value_loss,
            'rep/critic_loss': critic_loss,
            'rep/v_mean': v.mean(),
        }
        return value_loss + critic_loss, info

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(observations, goals, next_observations,
                                         rewards, masks, actions)
        return self.encode_goal(goals)
```
