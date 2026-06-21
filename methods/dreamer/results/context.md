# Context

## Research question

The goal is to control an agent from raw images so that it maximizes the
expected sum of rewards $\mathbb{E}\!\left[\sum_{t=1}^{T} r_t\right]$, using as
few real environment interactions as possible. Concretely the
setting is a partially observable Markov decision process: at each discrete step
the agent receives a high-dimensional observation $o_t$ (a $64\times64\times3$
image) and a scalar reward $r_t$, and emits a continuous action
$a_t$; the dynamics $p(o_t,r_t\mid o_{<t},a_{<t})$ are unknown. The objects of
interest are visual continuous-control tasks — locomotion and manipulation — where
the state is hidden behind pixels and rewards can be sparse and long-delayed.

The measure of interest is **sample efficiency**: how much real interaction is
needed to reach a given level of performance. On image-based control the standard
model-free agents reach good performance on the order of $10^8$ environment steps.
Real steps are slow and expensive, so performance at one or two orders of magnitude
fewer steps is what matters.

The broad question is whether a *learned model of the world* — something that can
predict forward from the current state and action — can be used to improve sample
efficiency, and if so how a behavior should be derived from such a model.

## Background

**Learning in imagination.** The classical decomposition of a model-based agent
(Sutton 1991, *Dyna*) has three interleaved processes: learn a dynamics model from
collected experience, improve the behavior using the model, and act in the
environment to collect more experience. Once a model exists, the
real environment can be replaced for the behavior-improvement step by
predicted ("imagined") rollouts.

**Latent dynamics for pixels.** When observations are images, one can predict
forward in a compact latent space rather than in pixel space. A *latent
state-space model* encodes images into a compact state $s_t$ and predicts
forward in that space (Watter et al. 2015, *E2C*; Karl et al. 2016; Buesing et al.
2018). Because latent states are small, many trajectories can be imagined in
parallel and far ahead. Such a model is a non-linear analogue of a Kalman filter / a
hidden Markov model with real-valued states, conditioned on actions and additionally
predicting reward — so it can score hypothetical action sequences without executing
them. The standard way to train these models is as a latent-variable model with a
variational objective (Krishnan et al. 2015, *Deep Kalman Filters*; Doerr et al.
2018): an inference network produces an approximate posterior over latent states, and
the model is trained to reconstruct observations (and rewards) while a KL term keeps
the posterior close to a learned transition prior.

**The reparameterization trick.** A latent-variable model trained by the evidence
lower bound (Kingma & Welling 2014; Rezende et al. 2014) backpropagates through a
sampling step by writing a stochastic variable as a deterministic function of the
parameters and an independent noise source, $s=\mu_\theta+\sigma_\theta\,\epsilon$,
$\epsilon\sim\mathcal{N}(0,I)$. This makes a sampled latent — and, with a
tanh-transformed Gaussian, a sampled continuous action (Haarnoja et al. 2018) —
*differentiable* with respect to the network that produced its statistics. It is the
mechanism that turns "sample a trajectory" into "compute a trajectory whose gradient I
can take."

**Pathwise versus score-function gradients.** For an objective
$\mathbb{E}_{x\sim p_\theta}[f(x)]$, there are two ways to differentiate through the
randomness. The *score-function* (likelihood-ratio, REINFORCE; Williams 1992)
estimator $\mathbb{E}[f(x)\,\nabla_\theta\log p_\theta(x)]$ treats $f$ as a black box
and weights the score by the scalar value; its variance scales with $\mathrm{Var}(f)$
and grows with horizon, which is why on-policy actor-critics such as A3C (Mnih et al.
2016) and PPO (Schulman et al. 2017) lean heavily on value baselines to subtract a
state-dependent constant and reduce that variance. The *pathwise* (reparameterized)
estimator $\mathbb{E}_\epsilon[\nabla_\theta f(g_\theta(\epsilon))]$ requires $f$ to be
differentiable but uses its *gradient*, $\partial f/\partial x$, which carries much
more information per sample and typically has far lower variance. Pathwise gradients
through a *known* differentiable simulator were the basis of analytic policy search
(Schmidhuber 1990; Deisenroth & Rasmussen 2011, *PILCO*, which differentiates returns
through a learned Gaussian-process model). Some model-based control instead uses
derivative-free optimization (Parmas et al. 2019).

**The bias/variance knob for returns.** Estimating the value of a state from a
trajectory admits a family of estimators between a one-step bootstrap
$r_t+\gamma v(s_{t+1})$ (low variance, biased by the value error) and the full Monte
Carlo return (unbiased, high variance). The $\lambda$-return (Sutton 1988) and the
generalized advantage estimator (Schulman et al. 2016) interpolate them as an
exponentially weighted average of $k$-step returns with weight parameter $\lambda$,
giving a single dial between bias and variance.

**Information bottleneck.** The variational objective for a latent dynamics model can
be read through the information-bottleneck lens (Tishby et al. 2000; Alemi et al.
2016): maximize the information the latent states carry about future observations and
rewards while penalizing the information extracted from each observation. The penalty
encourages the model to rely on its temporal memory and to extract from a new frame
only what the past could not already predict, which in turn encourages long-term
dependencies in the learned state.

## Baselines

**Dyna (Sutton 1991).** Establishes the learn-model / plan-in-model / act loop and the
idea of training the policy or value on simulated experience.

**PILCO (Deisenroth & Rasmussen 2011).** Learns a probabilistic (GP) dynamics model and
computes the *analytic* gradient of the expected long-horizon cost through the model to
update a policy, achieving high data efficiency on low-dimensional tasks via
moment-matching propagation of uncertainty.

**World Models (Ha & Schmidhuber 2018).** Train a VAE to compress frames and an
MDN-RNN to predict the latent dynamics, then evolve a tiny linear controller *inside*
the model with an evolution strategy. Behavior is learned purely in a dream; the model
and controller are trained in separate stages.

**PlaNet (Hafner et al. 2018) and the RSSM.** Learns a latent dynamics model jointly
end to end and solves visual control by *online planning*: at every environment step it
searches over action sequences with the cross-entropy method, scoring each by predicted
reward over a fixed horizon. Its model is the recurrent state-space model (RSSM), whose
transition keeps both a deterministic recurrent component and a stochastic latent
component.

**DDPG / SAC (Lillicrap et al. 2015; Haarnoja et al. 2018) and DPG (Silver et al.
2014).** Off-policy actor-critics that train a policy by ascending the analytic gradient
of a learned action-value $Q(s,a)$: $\nabla_\phi Q(s,\pi_\phi(s))$, using
reparameterized or deterministic actions. This is a pathwise gradient through a learned
one-step action-value $Q$; the action-value $Q(s,a)$ carries the dependence of the
policy gradient on the action.

**SVG (Heess et al. 2015).** Stochastic value gradients: backpropagates value gradients
through *one-step* predictions of a learned model using reparameterization.

**MVE / STEVE (Feinberg et al. 2018; Buckman et al. 2018).** Use a learned model to roll
forward a few steps and form multi-step *targets* for a model-free $Q$-learner; the
policy is updated by model-free machinery (Q-learning / REINFORCE).

**REINFORCE / A3C / PPO (Williams 1992; Mnih et al. 2016; Schulman et al. 2017).** The
model-free, on-policy yardstick: they optimize the policy with score-function gradients
and value baselines, and are robust and general.

## Evaluation settings

The natural testbed is the DeepMind Control Suite: continuous-control tasks
(cartpole, cheetah, walker, hopper, quadruped, finger, reacher, cup, pendulum,
acrobot) observed as $64\times64\times3$ images, with continuous vector actions and
episodes of fixed length, including tasks with sparse rewards. Performance is the
episode return as a function of the number of real environment steps, so the
yardsticks are both *final return* and *data efficiency* (return at a fixed, small
step budget such as $5\times10^6$ steps), plus wall-clock compute per update. The
prevailing comparison points are a model-free off-policy actor-critic operating from
state features and a model-free agent operating from pixels, and a model-based planner
operating from pixels; a fixed action-repeat and a small number of random seed
episodes to initialize the dataset are part of the standard protocol. Beyond
continuous control, the discrete-action testbed of Atari (with sticky actions) and
DeepMind Lab levels, with $3$–$18$ discrete actions and early episode termination,
serves as a stress test for visual complexity and termination.

## Code framework

The pieces below already exist as standard primitives: an environment loop that
collects episodes into a replay dataset, a convolutional encoder/decoder, a recurrent
latent state-space transition model that maintains a deterministic recurrent part and a
stochastic part (with a posterior conditioned on observations and a prior that predicts
forward without them), dense network heads, reparameterized sampling, the Adam
optimizer with gradient clipping, and a routine that turns a sequence of $k$-step
returns into a $\lambda$-weighted return. The behavior — how to derive an action from
this model — is the slot to fill.

```python
# --- existing primitives ---------------------------------------------------
class ConvEncoder:  # image -> embedding (exists)
    def __call__(self, image): ...

class ConvDecoder:  # latent feature -> image distribution (exists)
    def __call__(self, feat): ...

class DenseDecoder:  # latent feature -> scalar distribution (reward / value) (exists)
    def __init__(self, shape, layers, units, dist='normal'): ...
    def __call__(self, feat): ...

class LatentDynamics:
    """Recurrent latent state-space model: a deterministic recurrent part and a
    stochastic part, with an observation-conditioned posterior and an
    observation-free prior. (exists, reused as-is.)"""
    def obs_step(self, prev_state, prev_action, embed):
        # prior = img_step(...); posterior from concat(prior_recurrent, embed)
        ...
    def img_step(self, prev_state, prev_action):
        # advance recurrent state, then sample stochastic state from prior
        ...
    def observe(self, embed, action): ...   # filter a real sequence -> posteriors, priors
    def get_feat(self, state): ...          # concat(stochastic, deterministic)
    def get_dist(self, state): ...

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # exponentially-weighted average of k-step returns (exists)
    ...

# --- the slot the method will fill -----------------------------------------
class Agent:
    def _behavior(self, post):
        # TODO: derive a behavior from the learned latent model
        raise NotImplementedError

    def _train(self, data):
        # 1. fit the latent dynamics on a real sequence batch (variational objective)
        #    embed = encoder(data); post, prior = dynamics.observe(embed, actions)
        #    model_loss = -reconstruct(o) - reconstruct(r) + beta * KL(post || prior)
        # 2. TODO: derive behavior from the model, starting at `post`
        # 3. step the optimizers
        ...
```
