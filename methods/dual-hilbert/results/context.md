# Context: representing goals for offline goal-conditioned RL

## Research question

In offline goal-conditioned reinforcement learning, the agent receives a fixed dataset of unlabeled trajectories and must later reach arbitrary goal states without collecting new interaction data. The standard interface gives the actor and value function the raw current observation and the raw goal observation, for example `V(s, g)` and `pi(a | s, g)`. That interface is convenient, but it makes the goal coordinate system do almost none of the work.

Raw observations carry nuisance variation such as textures, lighting, object appearance, proprioceptive noise, and camera details. More importantly, raw similarity is often not reachability similarity. Two image frames can be nearly identical while lying on opposite sides of a wall, and two visually different states can be one controllable step apart. A downstream value function conditioned on raw goals must therefore learn the environment's long-horizon reachability structure at the same time that it learns control.

The open problem is to learn a reusable goal code from offline trajectories, put that code in front of an existing goal-conditioned agent, and make the code useful before the downstream actor has to rediscover the whole geometry. Such a representation should filter nuisance information, be trainable with only hindsight-labeled transitions, fit the bootstrapped offline value-learning machinery already used in GCRL, and expose enough structure that the same learned code can support both value prediction and goal-directed policy conditioning.

## Background

Goal-conditioned value functions already contain temporal reachability information. With the goal-reaching reward `r(s, g) = -1(s != g)` and termination at `s = g`, the undiscounted return is the negative number of steps needed to reach the goal. The optimal value is therefore

```text
V*(s, g) = -d*(s, g),
```

where `d*(s, g)` is the minimum number of time steps needed by an optimal policy. This identity appears in early GCRL work and in quasimetric-RL analyses. The quasimetric viewpoint matters because temporal distance has `d*(s, s) = 0` and a triangle inequality, but it need not be symmetric: climbing up, going down, and moving through one-way dynamics can have different costs in the two directions.

Offline optimal value learning is difficult because the Bellman optimality backup contains a maximization over actions or next states. A function approximator queried on actions outside the dataset can hallucinate high values, producing the overestimation failure familiar in offline RL. Implicit Q-Learning avoids that explicit out-of-distribution maximization by using expectile regression. The asymmetric squared loss

```text
L_2^tau(u) = |tau - 1(u < 0)| * u^2
```

fits an upper expectile when `tau > 1/2`; as `tau` increases, the fitted scalar is pulled toward the best targets in the dataset support rather than toward the mean. This gives an in-support analogue of the max backup.

HIQL adapts this idea to goal-conditioned learning without an action-value function. It samples `(s, s', g)` from an offline replay buffer with hindsight relabeling, computes the TD residual

```text
-1(s != g) + gamma * V_bar(s', g) - V(s, g),
```

and applies expectile regression over the next states seen in the data. The recipe also supplies practical stabilizers: twin value heads, target-network smoothing, layer-normalized MLPs, and a relabeling distribution that draws goals from future states in the same trajectory or from random dataset states.

Several representation-learning baselines also try to make state embeddings reflect time or controllability. Temporal contrastive methods, robotic visual pretraining, value-implicit pretraining, Laplacian features, successor features, and metric-aware skill abstractions all provide useful ingredients. Their limitations are different: some embeddings are frozen and then consumed as generic features, some preserve appearance rather than reachability, and some require online rollouts to learn the abstraction.

## Baselines

**Raw-goal concatenation.** Feed `[s, g]` directly to `V(s, g)` and `pi(a | s, g)`. This is simple and assumption-free. Its cost is that nuisance information and reachability structure are both left for the downstream networks to infer from scratch.

**Goal-conditioned IQL/HIQL with raw goals.** Learn a scalar goal-conditioned value by action-free expectile regression, then train the policy with advantage-weighted regression. This already uses hindsight goals and an in-support optimality surrogate. Its limitation is that any reachability structure remains implicit inside a scalar predictor, not exposed as a reusable goal code for later control.

**Reconstruction or dynamics-based goal encoders.** Train an encoder through autoencoding, inverse dynamics, or forward prediction. These objectives often preserve information useful for reconstructing observations or predicting local dynamics, but they can spend capacity on appearance details that are irrelevant to reaching a goal.

**Temporal visual feature pretraining.** Train an embedding so nearby points in a trajectory are closer than far-apart points, then freeze it for control. This can capture temporal regularities, but the controller still has to learn how to use the geometry after pretraining; the representation is treated as input features rather than as the object being optimized by the goal-reaching Bellman backup.

**Online skill-abstraction methods.** Learn latent directions or controllable features by collecting rollouts under the current policy. These can produce structured skills, but the online interaction assumption is incompatible with a fixed offline dataset.

## Evaluation settings

Natural evaluation settings are long-horizon offline GCRL benchmarks where raw observation similarity is a poor proxy for reachability. Maze navigation tests whether the representation respects walls and long paths rather than coordinate closeness. Manipulation tasks test whether the code keeps controllable object configuration while suppressing irrelevant visual variation.

The typical protocol is to train only from an offline dataset of unlabeled trajectories, relabel goals during training, and at evaluation condition the learned agent on goal observations supplied at test time. The main metric is success rate: the fraction of episodes that reach the specified goal. Optimizers, target-network smoothing, minibatch training, and hindsight relabeling all belong to the existing offline-GCRL stack.

## Code framework

The method has to fit into an existing JAX/Flax offline value-learning agent. The surrounding system already provides an offline replay batch with hindsight goals, an optimizer, target-network parameters, twin-head bootstrapping, and an actor loss that can consume a learned goal code. The empty slot is the goal-conditioned value module and its associated value loss.

```python
import jax
import jax.numpy as jnp
import flax.linen as nn


class GoalConditionedValueSlot(nn.Module):
    hidden_dims: tuple = (512, 512, 512)
    code_dim: int = 32
    use_layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        # TODO: define the value module we will design.
        pass

    def get_goal_code(self, observations):
        # TODO: expose the reusable code consumed by the downstream agent.
        pass

    def __call__(self, observations, goals=None, info=False):
        # TODO: evaluate a goal-conditioned value for each state-goal pair.
        pass


def expectile_loss(adv, diff, expectile):
    weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * (diff ** 2)


def compute_value_loss(agent, batch, network_params):
    # TODO: build the bootstrapped goal-reaching target and twin-head loss.
    pass


def train_step(agent, batch):
    value_loss, value_info = compute_value_loss(agent, batch, agent.network.params)
    goal_code = agent.network(batch["goals"], method="goal_code")
    actor_loss = agent.actor_loss(batch, goal_code)
    return agent.apply_grads(value_loss + actor_loss), value_info
```

The final implementation fills only this value-module slot: how observations are encoded, how `V(s, g)` is computed from them, and how the bootstrapped value loss is assembled.
