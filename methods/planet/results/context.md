# Context

## Research Question

The goal is to control an agent from raw visual observations while spending as
few real environment interactions as possible. At each discrete time step there
is a hidden physical state $s_t$, an image observation $o_t$, a continuous action
$a_t$, and a scalar reward $r_t$:
$$
s_t\sim \mathrm p(s_t\mid s_{t-1},a_{t-1}),\qquad
o_t\sim \mathrm p(o_t\mid s_t),\qquad
r_t\sim \mathrm p(r_t\mid s_t),\qquad
a_t\sim \mathrm p(a_t\mid o_{\le t},a_{<t}).
$$
A single image is generally not Markovian: velocity, contact state, and hidden
objects may have to be inferred from history. The policy must therefore act in a
POMDP from image histories, not from the simulator's compact state vector.

Planning would be the natural tool if the dynamics were known. A controller
could search over future action sequences, predict rewards, execute the first
action, and re-plan after the next observation. The hard question is whether a
learned model can be accurate and cheap enough for that inner loop. The model
must survive compounding rollout errors, represent uncertainty over possible
futures, avoid brittle extrapolation away from the data, and still score many
candidate action sequences at every environment step.

## Technical Background

Latent state-space models provide the basic language. Instead of predicting
directly in pixel space, a model can infer a compact latent state and learn
$$
p_\theta(s_t\mid s_{t-1},a_{t-1}),\qquad
p_\theta(o_t\mid s_t),\qquad
p_\theta(r_t\mid s_t).
$$
The observation model gives a dense training signal, while the latent transition
is the object a planner can roll forward cheaply. This is a nonlinear analogue
of a Kalman filter or hidden Markov model with neural transition and emission
functions.

The states are latent, so learning uses variational inference. An encoder
$q_\phi(s_t\mid o_{\le t},a_{<t})$ supplies a filtering posterior, and the
standard VAE machinery gives an evidence lower bound with a reconstruction term
and a KL term from posterior to transition prior. The filtering restriction is
important before the target method is known: a training-time smoother could look
at future images, but an acting agent only has past observations.

Two modeling pressures are in tension. A deterministic recurrent state can carry
memory over long horizons, but it represents one future. A stochastic latent
state can represent uncertainty, but repeatedly sampling through it makes memory
fragile. A successful visual-control model has to decide how to use both kinds
of structure without giving the observation encoder a shortcut that bypasses the
latent state.

## Baselines And Gaps

PILCO shows the value of learned probabilistic dynamics for sample-efficient
control, but it operates on low-dimensional Markov states and scales poorly to
image observations. E2C and RCE embed images into a latent space with
locally-linear dynamics, then plan with LQR-style methods; they demonstrate
visual latent control on simple domains but keep strong Markov and local-linearity
assumptions.

PETS combines probabilistic neural dynamics ensembles with model-predictive
control and the cross-entropy method, scaling model-based control to harder
continuous-control tasks. Its inputs are low-dimensional simulator states rather
than pixels, so it does not solve the visual filtering problem. World Models
compress frames with a VAE, train an MDN-RNN in latent space, and optimize a
small controller separately; that establishes the usefulness of latent world
models but does not give an online planner trained end to end with the dynamics.

VRNNs, Deep Kalman Filters, deep variational Bayes filters, probabilistic
recurrent state-space models, and related deep state-space models supply the
latent-sequence modeling toolbox. Their remaining gap for this problem is not
just prediction quality: the model must be shaped for fast open-loop planning
from images under partial observability.

## Evaluation Frame

The natural benchmark is pixel-only continuous control from the DeepMind Control
Suite: cart-pole, reacher, cheetah, finger, ball-in-cup, and walker-style tasks
with $64\times64\times3$ camera observations, continuous actions, contact
dynamics, hidden velocities, and sparse rewards in some domains. The relevant
curves are episode return versus real environment interaction and, secondarily,
the compute cost of model learning and action selection.

Fair comparisons include strong model-free agents trained from the same visual
observations, learned-model planners that use low-dimensional states, and
architectural ablations that isolate the role of stochastic state, deterministic
memory, iterative action search, and online data collection. The pre-method
requirement is clear: any candidate must expose the same pixel-only interface at
test time and must not use simulator state during action selection.

## Code Scaffold

The implementation can assume standard components: an episode replay dataset, a
convolutional image encoder, a deconvolutional image decoder, diagonal Gaussian
distributions with reparameterized sampling, closed-form diagonal-Gaussian KL,
a recurrent cell, Adam with gradient clipping, and a population optimizer over
continuous action sequences.

```python
class Encoder:
    def __call__(self, image): ...

class ImageHead:
    def __call__(self, latent_features): ...

class RewardHead:
    def __call__(self, latent_features): ...

def diagonal_kl(q_mean, q_std, p_mean, p_std): ...

def reparameterized_sample(mean, std, eps): ...

class LatentDynamics:
    def prior(self, previous_state, previous_action):
        # Advance without seeing the current observation.
        raise NotImplementedError

    def posterior(self, previous_state, previous_action, image_embedding):
        # Fold the current image into the current belief.
        raise NotImplementedError

    def features(self, state):
        # Features consumed by image and reward heads.
        raise NotImplementedError

def model_objective(batch, encoder, dynamics, image_head, reward_head):
    raise NotImplementedError

def choose_action(current_belief, dynamics, reward_head, action_optimizer):
    raise NotImplementedError
```
