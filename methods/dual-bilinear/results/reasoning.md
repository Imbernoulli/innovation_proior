Let me start from the failure mode I keep running into. I want one offline policy that reaches any goal from any state, and the first description of the task it sees is the raw goal observation. That observation is a sensing format, not a task format. In images it contains walls, shadows, backgrounds, and distractors; in state vectors it can contain nuisance coordinates and noise. Two observations can mean the same task while differing in irrelevant details, and a policy trained on raw `g` has to learn that invariance from finite offline data. If I compress the goal generically, I might throw away the controllable information; if I do not compress it, I keep the noise. The goal code I want has to keep exactly the information an optimal reaching policy needs and drop the rest.

The obvious baselines do not give me that. The identity representation `phi(g) = g` keeps everything, including exogenous variation. A variational bottleneck can force a smaller stochastic code, but the KL penalty does not know that the robot pose matters and a texture patch does not. VIP and HILP point in a better direction by making representation geometry temporal, but their metric flavor is already suspicious because reachability is directed. TRA uses an inner product, so it has a more flexible aggregator, but its contrastive target is behavioral: it describes how the dataset moves, not how an optimal goal-reaching policy would move. BYOL-gamma predicts temporal structure, but again without an optimal-reaching target. I need the target itself to tell me what information is task-relevant.

What quantity ignores the observation coordinate system entirely? The optimal time-to-goal. Write it `d*(s, g)`. In a deterministic environment it is the shortest path length; in the goal-conditioned value notation it is the same object in discounted coordinates. With the sparse absorbing reward `r(s, g) = I(s = g)`, first hitting the goal after `k` steps produces return `gamma^k`, so I expect

  `V*(s, g) = gamma^{d*(s, g)}`.

Let me make sure I have the bookkeeping right rather than just writing it down, because the rest of the argument leans on this identity. Take a four-state chain `0 -> 1 -> 2 -> 3` with `g = 3` absorbing, `gamma = 0.9`. From `s` the reaching time is `d* = 3 - s`. Reward `1` is collected once, on the step the goal is hit, so the return is exactly `gamma^{d*}`: at `s = 0`, `0.9^3 = 0.729`; at `s = 1`, `0.9^2 = 0.81`; at `s = 2`, `0.9^1 = 0.9`. Taking `log_{0.9}` of each recovers `3, 2, 1`, i.e. `d*` back out. OGBench uses the shifted reward `r = I(s = g) - 1` instead, which charges `-1` per pre-goal step; summing `-(1 + gamma + ... + gamma^{d*-1}) = -(1 - gamma^{d*})/(1 - gamma)`, and for `d* = 3` that is `-2.71`, matching the closed form. Both are strictly monotone transforms of `d*`, so either reward gives me the same temporal-distance coordinate. Good — the identity holds, and it is already a much better coordinate system than raw observations: temporal distance depends on dynamics, not on the renderer.

But one scalar distance from one anchor state cannot identify a goal's role for every possible start. A goal is not just "far from here"; operationally it is the thing that every state has some reaching relation to. So I should represent `g` by the whole function of its incoming relations:

  `phi^vee(g) = (s |-> d*(s, g))`.

If the state space is finite, this is the vector `[d*(s_1, g), ..., d*(s_K, g)]`; in general it is a functional on the state space. The move I am making is the one where a point is identified by all its relations to other points rather than by its own coordinates. Whether that is the *right* object here is not settled by the analogy — it is settled by whether the functional actually supports control and discards noise, so I need to check those two properties before I trust it.

Sufficiency comes first. Suppose I am handed only a functional `f = phi^vee(g)`, so for any successor `s'` I can query `f(s') = d*(s', g)`. The one-step greedy action for the optimal goal-conditioned value only needs successor values. Define

  `pi^vee(s, f) := argmax_a E_{s' ~ p(. | s, a)}[gamma^{f(s')}]`.

Now feed it the goal's functional:

  `tilde pi(s, g) = pi^vee(s, phi^vee(g))`
  `= argmax_a E_{s'}[gamma^{phi^vee(g)(s')}]`
  `= argmax_a E_{s'}[gamma^{d*(s', g)}]`
  `= argmax_a E_{s'}[V*(s', g)]`
  `= argmax_a Q*(s, a, g)`.

The last equality is the one-step optimality relation used by the goal-conditioned value definition: the action that maximizes the expected successor `V*` is `argmax_a Q*(s, a, g)`. Greedy action selection with respect to `Q*` attains `V*`, so `V^{tilde pi}(s, g) = V*(s, g)` for every `s, g`.

The algebra is clean, but I want to see it actually pick the right action from only the functional, because the whole point is that the raw goal is gone. Concrete instance: at state `s` I have two actions, `a1` leads deterministically to a successor at distance `d*(.,g) = 2` from the goal and `a2` leads to one at distance `5`; `gamma = 0.9`. All I am handed is `f = phi^vee(g)` queried at the two successors, `f = (2, 5)`. The greedy rule scores `gamma^{f(s')}`: `0.9^2 = 0.81` for `a1`, `0.9^5 = 0.5905` for `a2`, so it takes `a1`. Now compute `Q*(s, a, g)` directly: acting through `a` then optimally costs `1 + d*(s', g)` steps, so `Q* = (gamma^3, gamma^6) = (0.729, 0.5314)`, and `argmax_a Q*` is also `a1`. They agree — and they must, since the extra `gamma^1` from the taken step is a common positive factor that cannot flip an argmax. So the functional-greedy policy picks the `Q*`-greedy action without ever seeing `g`. It never needs the raw goal; it only needs the successor-state table of optimal goal values, and `phi^vee(g)` supplies exactly that table in log coordinates.

Now check noise invariance. I need a formal model where "different observations, same task" has a precise meaning. In a block CMP, each observation `s` maps to a latent `p^ell(s)`, emissions from different latents have disjoint supports, and the goal reward is latent:

  `r^ell(s, g) = I(p^ell(s) = p^ell(g))`.

Take two goal observations `g_1, g_2` with `p^ell(g_1) = p^ell(g_2)`. For any state `s`,

  `phi^vee(g_1)(s) = log_gamma V*(s, g_1)`
  `= log_gamma max_pi E_{tau, s_0=s}[sum_t gamma^t r^ell(s_t, g_1)]`
  `= log_gamma max_pi E_{tau, s_0=s}[sum_t gamma^t I(p^ell(s_t) = p^ell(g_1))]`.

But `p^ell(g_1) = p^ell(g_2)`, so every indicator inside every trajectory return is identical:

  `I(p^ell(s_t) = p^ell(g_1)) = I(p^ell(s_t) = p^ell(g_2)) = r^ell(s_t, g_2)`.

The maximization is over the same policy class and the same trajectory distribution, only the reward expression has been rewritten, so the value equals `log_gamma V*(s, g_2) = phi^vee(g_2)(s)`. This holds for every `s`, hence `phi^vee(g_1) = phi^vee(g_2)`. Exogenous observation noise disappears because the reward and temporal distances only see the latent.

So the functional clears both bars I set for it: it is sufficient for optimal greedy control and it collapses observation noise to the latent. As an idealized object it is what I want. But I cannot store an arbitrary function per goal and I do not know `d*`. The first practical question is how a finite vector can stand in for `phi^vee(g)`. The proof tells me how the functional is consumed: it is paired with a state and evaluated as `d*(s, g)`. So I can model the two-argument value/distance surface through two embeddings,

  `V(s, g) ~= f(psi(s), phi(g))`,

and use the goal-side vector `phi(g)` as the finite representation. For any state, `psi(s)` plus the aggregator should recover the relevant relation to the goal.

A metric aggregator is tempting because I keep saying "distance": `||phi(s) - phi(g)||_2`. But a Euclidean metric is symmetric in its arguments — `||x - y|| = ||y - x||` identically — so any `V(s, g) = -||phi(s) - phi(g)||` would force `V(s, g) = V(g, s)`. Is temporal distance actually symmetric? Quick counterexample: a directed cycle `0 -> 1 -> 2 -> 0` with only forward unit edges. Then `d*(0, 1) = 1` but going back `1 -> 0` has to traverse `1 -> 2 -> 0`, so `d*(1, 0) = 2`. The shortest-path matrix is `[[0,1,2],[2,0,1],[1,2,0]]`, plainly not symmetric. Falling down can be easy while climbing back is hard, and a shared metric cannot express that gap at all. So the metric aggregator is ruled out, not merely inelegant.

Could I keep a symmetric form anyway and hope the asymmetry is small? No — reaching is the whole task, and on a maze with one-way drops the cost gap is exactly the part a controller must respect. I need an aggregator that can be directed. A bilinear inner product is the simplest one: `psi(s)^T phi(g)` need not equal `psi(g)^T phi(s)`, because the state and goal branches are different maps. To see that it genuinely decouples the two directions, take embeddings `psi(0) = (1, 0), psi(1) = (0, 1)` and `phi(0) = (0, 2), phi(1) = (3, 0)`: then `psi(0)^T phi(1) = 3` while `psi(1)^T phi(0) = 2`. Same pair of points, two different scores — exactly what a symmetric metric forbids. So:

  `f(psi(s), phi(g)) = psi(s)^T phi(g)`.

With enough dimensions and learned feature maps, sums of separable products can approximate arbitrary continuous two-variable functions on compact domains. It is also directed because the state and goal embeddings are different maps. In the practical code I should scale the dot product by the latent width:

  `V(s, g) = psi(s)^T phi(g) / sqrt(d)`.

Without that scale, the dot product is a sum of `d` terms, so changing the representation width changes the initial value and gradient scale. Dividing by `sqrt(d)` keeps the bilinear score order-one across embedding dimensions, the same numerical reason attention uses a square-root scale.

The second practical question is how to learn the optimal temporal relation offline. I should not train on a behavioral contrastive target, because then the representation describes the dataset's occupancy rather than optimal reaching. I want the optimal goal-conditioned value, but a naive Bellman max over actions is unsafe offline. Implicit Q-learning is the right tool: fit the value to an upper expectile of dataset-action Q values, which approximates an in-support maximum without ever evaluating unseen actions.

Use the residual `u = bar Q(s, a, g) - V(s, g)` and

  `ell_kappa^2(u) = |kappa - I(u < 0)| u^2`.

For `kappa > 1/2`, if the target Q is above the current value (`u > 0`), the weight is `kappa`; if the value is already above that target, the weight is `1 - kappa`. So the value is pulled upward toward the high in-dataset targets. With the bilinear value `V_rep(s, g) = psi(s)^T phi(g) / sqrt(d)`, the representation loss is

  `L_rep_value = E[ell_kappa^2(bar Q(s, a, g) - V_rep(s, g))]`.

The critic is an ordinary TD fit to the same value:

  `L_rep_Q = E[(Q(s, a, g) - r(s, g) - gamma V_rep(s', g))^2]`.

In the implementation this is a twin critic: the expectile target uses `min(q1_t, q2_t)` from the target critic, both current critic heads regress to the TD target, and the target critic is EMA-updated. With the `0/1` reward the converged value is `gamma^{d*}`; with the OGBench `-1/0` reward it is `-(1 - gamma^{d*})/(1 - gamma)`. Either way, the learned bilinear value is a monotone transform of optimal temporal distance, and the goal-side embedding is trained to carry the information needed to reconstruct that relation.

I should also keep the representation value separate from the downstream control value. The bilinear structure is exactly what forces the goal code to be relational and compact, but it is a constrained head for extracting a policy. In `antmaze`, the goal representation can mostly care about the x-y target, while the controller still needs joint angles and velocities to choose actions. So I use the bilinear value only to learn the goal code, then pass `phi(g)` to a separate monolithic downstream GCRL value/policy such as GCIVL, CRL, or GCFBC. Two value functions are not redundant; one shapes the representation, the other extracts control.

Now write the module in the harness shape. The local `GCBilinearValue` implementation names the observation branch `phi` and the goal branch `psi`, computing `phi(observation)^T psi(goal) / sqrt(d)`. In the derivation notation, the returned goal-branch vector is the practical `phi(g)` goal representation. I will wrap that naming mismatch so the interface exposes `encode_goal` cleanly.

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

The causal chain is now tight. Raw observations are a poor goal coordinate system because they mix controllable task information with exogenous detail. Optimal temporal distance is intrinsic to the dynamics, and with sparse absorbing rewards it is equivalent to the optimal goal-conditioned value by `V*(s, g) = gamma^{d*(s, g)}`. A goal represented by all incoming temporal distances is sufficient because the optimal greedy action only needs `gamma^{d*(s', g)}` at possible successor states, and it is noise-invariant in a block CMP because two observations of the same latent induce identical latent rewards on every trajectory. The practical finite version learns a bilinear, square-root-scaled value `psi(s)^T phi(g) / sqrt(d)` with goal-conditioned IQL, exports the goal branch as `phi(g)`, and hands that code to a separate downstream offline GCRL learner.
