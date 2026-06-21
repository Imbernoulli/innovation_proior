# Context

## Research question

The setting is a single reinforcement-learning algorithm applied, with **one fixed set
of hyperparameters**, across a wide range of domains — visual and proprioceptive
control, discrete-action arcade games, procedurally generated 3D worlds, and a
sparse-reward open-world game — without per-task reconfiguration. Formally the
agent interacts with a partially observable Markov decision process: at each step it
receives an observation $x_t$ (a $64\times64\times3$ image, or a vector of sensor
readings, or both) and a scalar reward $r_t$, emits an action $a_t$ (continuous or
discrete), and seeks to maximize the expected discounted return
$\mathbb{E}\!\left[\sum_{\tau\ge 0}\gamma^\tau r_{t+\tau}\right]$.

Across these domains the quantities an agent must fit vary by **orders of magnitude and
in distribution shape**. Rewards might be dense and in the hundreds (an arcade score) or
sparse and binary (reach a milestone). Returns can be unimodal or multimodal.
Observations can be pixels where fine detail matters or 3D scenes full of irrelevant
texture. The question is how to build a world-model agent — a family that promises strong
sample efficiency — whose fixed configuration of loss scales, KL weights, entropy
coefficients, and value parameterizations applies uniformly across this heterogeneous
battery.

## Background

**Learning behavior inside a world model.** The classical model-based decomposition
(Sutton 1991, *Dyna*) interleaves three processes: learn a dynamics model from
collected experience, improve behavior using the model, and act to collect more
experience. Once a model exists, the expensive real environment can be replaced for
behavior improvement by cheap imagined rollouts. Imagination is as good as the model, so
the model must be accurate enough that a policy trained entirely in it transfers back to
reality.

**Latent recurrent state-space models for pixels.** Rather than predict forward in pixel
space, one predicts forward in a compact *latent* state. The recurrent state-space
model (RSSM; Hafner et al. 2018, *PlaNet*) splits the latent into a deterministic
recurrent component $h_t$ that carries memory and a stochastic component $z_t$
that captures uncertainty. A sequence model advances $h_t=f(h_{t-1},z_{t-1},a_{t-1})$;
a *prior* (dynamics predictor) proposes $z_t\sim p(z_t\mid h_t)$ without seeing the
observation, while a *posterior* (encoder) folds in the current frame,
$z_t\sim q(z_t\mid h_t,x_t)$. The model state $s_t=\{h_t,z_t\}$ is Markov by
construction, so anything trained on it inherits a clean MDP. The model is fit as a
latent-variable sequence model: reconstruct observations and rewards from the model
state while a KL term keeps the posterior and prior close, so that rollouts of the
prior alone — the imagination used to train behavior — land where the posterior would.
PlaNet derives behavior by online cross-entropy-method planning over a fixed horizon.

**Actor and critic in imagination.** Rather than re-plan at every step, one can learn
an explicit actor $\pi(a\mid s)$ and a critic $v(s)$ that both operate on latent model
states and are trained purely on imagined rollouts (Dreamer line of work). Acting is
then a single forward pass. To see past a finite imagination horizon $H$, the critic
bootstraps the tail: rewards beyond $H$ are summed up by the value, and intermediate
horizons are blended with the $\lambda$-return (Sutton 1988; Schulman et al. 2016), an
exponentially weighted average of $k$-step bootstrapped returns with a single
bias/variance dial $\lambda$. Continuous-latent versions backpropagate the *analytic*
(pathwise) gradient of imagined returns through the differentiable dynamics; with
discrete (categorical) latents, the sampling step is handled with straight-through
gradients (Bengio et al. 2013).

**Categorical value/return distributions.** Instead of regressing a scalar value, a
critic can predict a *distribution* over returns (Bellemare et al. 2017, *C51*): a
softmax over a fixed set of value bins, trained by cross-entropy. For continuous
targets the *twohot* encoding (Schrittwieser et al. 2019, *MuZero*) generalizes the
one-hot target to a two-bin linear interpolation, so a scalar maps to a soft label and
the readout (a probability-weighted average of bin positions) can land between bins.
The gradient depends on bin *probabilities*, not on the magnitude of the target.

**Scale handling in existing systems.** Several techniques address targets and returns
that differ in scale across tasks. A squared loss on large targets can diverge;
absolute/Huber losses (Mnih et al. 2015) cap the gradient; normalizing targets by
running statistics (Schulman et al. 2017) rescales by online moments; adjusting network
weights when new extreme values appear (Hessel et al. 2019, *Pop-Art*) preserves outputs
under rescaling. On the actor side, the entropy regularizer (Williams 1991) carries a
coefficient set against the reward scale and frequency: advantage normalization
(Schulman et al. 2017) standardizes the emphasis on returns; normalization by standard
deviation rescales by spread; constrained optimization to a fixed average entropy
(Haarnoja et al. 2018; Abdolmaleki et al. 2018) targets a chosen entropy level. Deep
variational models track KL behavior (Child 2020) when categorical posteriors approach
near-deterministic, and latent models track posterior informativeness relative to the
dynamics.

**The score-function gradient and its baseline.** The Reinforce / likelihood-ratio
estimator (Williams 1992) $\mathbb{E}[A\,\nabla_\theta\log\pi_\theta(a\mid s)]$ weights
the score of each action by an advantage $A$; subtracting a state-value baseline from
the return reduces variance without biasing the gradient, and an additive offset to the
multiplier leaves the gradient unchanged.

## Baselines

**Dyna (Sutton 1991).** The learn-model / improve-in-model / act loop and the idea of
training behavior on simulated experience. It is the scaffold for model-based RL.

**PlaNet / RSSM (Hafner et al. 2018).** Learns the latent recurrent state-space model
end to end and controls by online CEM planning over a fixed horizon. Provides the world
model reused here. The representation-loss scale is set per domain — complex 3D scenes
use a strong regularizer to discard irrelevant detail, static-background games use a
weak one to keep fine pixels.

**Earlier Dreamer generations (Hafner et al. 2019, 2020).** Learn actor and critic
inside the RSSM's imagination with $\lambda$-returns; the first generation handled
continuous control with analytic pathwise gradients, the second introduced discrete
categorical latents with straight-through gradients and KL balancing and reached
human-level Atari. Settings such as the KL / representation-loss scale, the reward/return
scale, and the entropy coefficient are chosen per domain.

**PPO (Schulman et al. 2017).** The general, on-policy yardstick: clipped surrogate
policy-gradient with a value baseline and normalized advantages. Applies broadly across
domains.

**SAC (Haarnoja et al. 2018).** Off-policy maximum-entropy actor-critic with a learned
action-value and an entropy term whose temperature is tuned (or auto-tuned to a target
entropy). Data-efficient on continuous control from states.

**MuZero (Schrittwieser et al. 2019).** Plans with a learned value-prediction model and
reaches strong scores on board games and Atari, contributing the twohot value
parameterization reused here. It comprises several interacting components.

**C51 (Bellemare et al. 2017).** Distributional value learning: predict a categorical
distribution over a fixed value support and train by cross-entropy. Supplies the
distributional critic idea, with a fixed, bounded value support.

**Rainbow / IMPALA (Hessel et al. 2018; Espeholt et al. 2018).** Value-based and
distributed actor-critic agents for discrete domains; tuned expert points of comparison.

## Evaluation settings

The yardsticks are a deliberately heterogeneous battery, all scored as episode
return (or human-normalized score) as a function of real environment steps, with no
per-domain hyperparameter changes:

- **Atari** — 57 games with sticky actions; human- and random-normalized scores.
- **Atari 100k** — 26 games at a tight budget of 400K environment steps (100K agent
  steps after action repeat 4), a sample-efficiency regime.
- **DeepMind Control Suite** — about 20 continuous-control tasks, observed either as
  $64\times64\times3$ images (Visual Control, action repeat 2) or as proprioceptive
  state vectors (Proprioceptive Control); includes sparse-reward tasks.
- **DMLab** — 30 first-person 3D navigation/reasoning levels with a fixed discrete
  action space.
- **ProcGen** — procedurally generated 2D games, hard difficulty, unlimited level set,
  no action repeat.
- **BSuite** — a diagnostic suite probing credit assignment, memory, and exploration.
- **Minecraft Diamond** — an open-world 3D game with a $64\times64\times3$ first-person
  image plus inventory/vital vectors and a flat categorical action space; sparse reward
  of $+1$ at each of 12 milestones leading to a diamond, episodes up to 36000 steps.

Comparison points are PPO (a high-quality implementation with an IMPALA network, tuned
within recommendations and matched in discount), plus tuned expert algorithms per
benchmark (IMPALA, Rainbow, and others). Protocol: a small number of random seeds
(typically 5), mean with one standard deviation, a fixed action-repeat per benchmark,
and a "replay ratio" (gradient steps per environment step) chosen to fit each budget.

## Code framework

The pieces below already exist as standard primitives: the RSSM world model
(deterministic recurrent part plus stochastic part, with an observation-free prior and
an observation-conditioned posterior), convolutional and MLP encoders/decoders, dense
prediction heads, reparameterized / straight-through sampling, a routine that turns
$k$-step returns into a $\lambda$-return, and a replay buffer with an environment loop.
The slot to fill is whatever lets one fixed configuration of this agent learn across all
the domains above.

```python
# --- existing primitives ---------------------------------------------------
class RSSM:
    """Recurrent state-space model: deterministic recurrent h_t plus stochastic
    z_t; observation-free prior p(z_t|h_t) and observation-conditioned posterior
    q(z_t|h_t,x_t). (exists, reused as-is.)"""
    def img_step(self, prev_state, prev_action):   # prior / transition
        ...
    def obs_step(self, prev_state, prev_action, embed):  # posterior
        ...
    def get_feat(self, state):                      # concat(z_t, h_t)
        ...

class ConvEncoder:  ...    # image -> embedding (exists)
class ConvDecoder:  ...    # model state -> image distribution (exists)
class MLP:          ...    # model state -> hidden -> head (exists)

def lambda_return(reward, value, cont, bootstrap, lambda_):
    # exponentially-weighted average of k-step returns (exists)
    ...

optimizer = SomeOptimizer(lr)   # a standard adaptive optimizer (exists)

# --- the slot the method will fill -----------------------------------------
# TODO: whatever it takes so that ONE fixed configuration of the agent below
#       learns across all the domains above. The world model, its losses, the
#       prediction/value heads, the imagined-rollout actor-critic update, and
#       the gradient step are all in play; today each carries numbers that are
#       set per domain.

def world_model_loss(model, batch):
    # reconstruct inputs/reward/continue; KL between posterior and prior
    raise NotImplementedError

def actor_critic_loss(traj, actor, critic):
    # imagined rollout -> lambda-returns -> actor and critic losses
    raise NotImplementedError

def apply_gradients(loss, params):
    raise NotImplementedError
```
