# Context

## Research question

The goal is to control an agent from raw images so that it maximizes the
expected sum of rewards $\mathbb{E}\!\left[\sum_{t=1}^{T} r_t\right]$, while
spending as few real environment interactions as possible. The setting is a
partially observable Markov decision process (POMDP): a single image does not
reveal the full state of the world, so at each discrete step $t$ there is a
hidden state $s_t$, an image observation $o_t$ (a $64\times64\times3$ frame), a
continuous action vector $a_t$, and a scalar reward $r_t$, with unknown
dynamics
$$s_t\sim\mathrm{p}(s_t\mid s_{t-1},a_{t-1}),\quad o_t\sim\mathrm{p}(o_t\mid s_t),\quad r_t\sim\mathrm{p}(r_t\mid s_t),\quad a_t\sim\mathrm{p}(a_t\mid o_{\le t},a_{<t}).$$
The objects of interest are visual continuous-control tasks — locomotion and
manipulation — that include contact dynamics, partial observability, and sparse
rewards.

The binding constraint is **sample efficiency**. On image-based continuous
control the strong model-free agents need very large amounts of environment
interaction to reach good performance, and real steps are slow and expensive. A
method that reaches comparable performance with one or two orders of magnitude
fewer episodes would change what is practical.

Planning is a natural and powerful way to make decisions *when the dynamics are
known* — it underlies model-predictive control of simulated robots and the
search in game-playing systems. The opening question is therefore whether the
agent can *learn* a dynamics model from its own experience that is accurate
enough to plan inside. If it can, planning brings three concrete advantages over
model-free learning: it is more data-efficient because it uses the rich training
signal of predicting observations and does not have to propagate reward through
long chains of Bellman backups; performance can be bought with more search
compute at decision time; and a task-agnostic dynamics model can be reused
across tasks in the same environment. The difficulty is exactly the modeling: a
model accurate enough for planning has to survive **accumulating multi-step
prediction error**, has to **represent multiple plausible futures**, and must
not be **overconfident outside the training distribution** — and it must do all
of this fast enough that thousands of candidate action sequences can be scored at
every environment step.

## Background

**Planning with known versus learned dynamics.** Trajectory optimization and
model-predictive control (Richards 2005) solve control problems by searching for
an action sequence that maximizes predicted return under a model and re-planning
as new information arrives. With a *known* simulator this is extremely effective.
The open problem is to learn a model good enough to substitute for the simulator.
Key failure modes of learned models are model inaccuracy, the compounding of
one-step errors over a multi-step rollout, an inability to capture more than one
possible future, and overconfident extrapolation off the data manifold.

**Latent state-space models for sequences.** When observations are images,
predicting forward in pixel space is expensive and largely wasteful — most pixels
are irrelevant to control. The useful object is a *latent state-space model*
(SSM): encode each frame into a compact state and predict forward in that space,
decoding to pixels only to obtain a training signal. Such a model is a non-linear
analogue of a Kalman filter / hidden Markov model with real-valued states. The
machinery to train one comes from variational autoencoders (Kingma & Welling
2014; Rezende et al. 2014): an inference (encoder) network produces an
approximate posterior over latent states, the model is trained to reconstruct
observations under a variational lower bound, and gradients flow through the
sampling step by the **reparameterization trick**,
$s=\mu_\theta+\sigma_\theta\,\epsilon$, $\epsilon\sim\mathcal N(0,I)$. Deep Kalman
Filters (Krishnan et al. 2015) trained non-linear SSMs this way. A useful fact
for the loss: the log-likelihood under a Gaussian with unit variance equals the
mean squared error up to a constant, so reconstruction terms become squared
errors.

**Deterministic recurrence versus stochastic latents.** Two families sit at the
extremes. A recurrent neural network carries information forward through a
deterministic hidden state and can remember the distant past, but a purely
deterministic model cannot represent that the future is genuinely uncertain or
multimodal. A purely stochastic latent chain $s_t\sim p(s_t\mid s_{t-1},a_{t-1})$
can represent uncertainty, but every step squeezes all information through a
sampled bottleneck, so reliably remembering anything across many steps is hard.
Models that combine the two — the variational RNN (Chung et al. 2015), deep
variational Bayes filters (Karl et al. 2016), probabilistic-recurrent SSMs (Doerr
et al. 2018), and the deep state-space model of Buesing et al. (2018) — were the
state of the art for stochastic latent sequence prediction. The variational RNN
feeds generated observations back into the recurrence, which makes pure forward
prediction (without observations) expensive; Buesing et al.'s model is close in
form but was used to feed a hybrid agent rather than for explicit planning.

**The $\beta$ weight on the KL.** The $\beta$-VAE (Higgins et al. 2016) showed
that scaling the KL-divergence term in the variational bound by a coefficient is
a useful knob, trading reconstruction fidelity against the strength of the
prior/posterior matching. This is the precedent for per-term weighting of KL
regularizers.

**Multi-step training of sequence models.** A model trained only to make good
*one-step* predictions does not, with limited capacity, automatically make good
*multi-step* predictions. Several lines of work attacked this directly: scheduled
sampling (Bengio et al. 2015) anneals the rollout length seen during training;
hallucinated replay (Talvitie 2014) mixes the model's own predictions back into
the data; data-as-demonstrator (Venkatraman et al. 2015) frames it as imitation;
and Amos et al. (2018) train a deterministic model on all multi-step predictions
at once, in *observation* space, for robotic exploration. The open question is how
to get the multi-step benefit for a *latent* sequence model without paying to
generate images at every distance.

**Population-based action search.** The cross-entropy method (CEM; Rubinstein
1997) is a derivative-free, population-based optimizer: maintain a distribution
over candidate solutions, sample a population, keep the best fraction, and refit
the distribution to them. Applied to action sequences it is robust and was used by
PETS (Chua et al. 2018), an ensemble-of-neural-networks model that scaled
model-based control with CEM/MPC up to the cheetah running task — but from the
low-dimensional true state, not from pixels.

## Baselines

**PILCO (Deisenroth & Rasmussen 2011).** Learns a probabilistic (Gaussian
process) dynamics model and computes the *analytic* gradient of expected
long-horizon cost through it to update a policy, achieving remarkable sample
efficiency on tasks with a handful of state variables (cart-pole, mountain car).
It assumes access to the low-dimensional Markovian state and often the reward
function; its GP and moment-matching uncertainty propagation do not scale to image
observations or rich dynamics. It is the proof that an accurate, uncertainty-aware
model buys extreme data efficiency, and the open problem it leaves is scaling that
to pixels.

**E2C (Watter et al. 2015) and RCE (Banijamali et al. 2017).** Embed images into a
latent space in which the transitions are made *locally linear*, then plan with a
linear-quadratic regulator. They balance simulated cart-poles and control two-link
arms from images under *dense* rewards. Their limitations: they impose a Markov
assumption on the latent (so they cannot handle partial observability), the
locally-linear structure has been hard to scale, and they were demonstrated only on
simple tasks with short horizons and dense rewards.

**PETS (Chua et al. 2018).** Uses an *ensemble* of probabilistic neural-network
dynamics models to capture uncertainty, with CEM-based MPC for action selection,
scaling model-based control to tasks like the cheetah. It plans from the
low-dimensional true state, not from images, so it does not address the visual or
partial-observability problem.

**World Models (Ha & Schmidhuber 2018).** Train a VAE to compress frames into a
latent code and a mixture-density RNN to predict the latent dynamics, then evolve a
tiny linear controller inside the learned model. It shows that behavior can be
learned in a learned latent model, but the components are trained in separate
stages and the controller is optimized by a derivative-free evolution strategy
rather than by planning; its convolutional encoder/decoder architecture is a
reusable building block.

**Latent SSM video models (Chung et al. 2015; Karl et al. 2016; Doerr et al. 2018;
Buesing et al. 2018).** Stochastic latent sequence models trained by variational
inference. They are the closest modeling ancestors. Their gaps for control: the
variational RNN feeds generated observations back into the model, making
observation-free forward prediction expensive; several are purely stochastic and
struggle to remember; and they were built for prediction or for a hybrid agent, not
to be planned inside directly.

**Model-free agents (A3C, Mnih et al. 2016; D4PG).** The model-free yardsticks for
visual continuous control. They are robust and general but need large amounts of
environment interaction — the sample-efficiency wall this work is trying to break.

## Evaluation settings

The natural testbed is the DeepMind Control Suite: continuous-control tasks
(cart-pole swing-up, reacher, cheetah run, finger spin, cup catch, walker walk)
observed as $64\times64\times3$ images with continuous action vectors and
fixed-length episodes, spanning contact dynamics, partial observability (e.g. a
cart-pole task where the agent must infer velocity from frames), and sparse
rewards (e.g. cup catch, where reward is nonzero only on success). Performance is
the episode return as a function of the number of real environment episodes (and
of environment steps), so the yardsticks are *final return* and *data efficiency*,
together with the wall-clock compute spent on each model update and on planning at
action time. The protocol uses a small number of random seed episodes to
initialize the dataset, a fixed per-domain action-repeat, and small Gaussian
exploration noise added to the planned actions; images are reduced to 5-bit color
depth before being fed to the model. The standard comparison points are a
model-free on-policy agent and a model-free off-policy agent operating from the
same pixel observations.

## Code framework

The pieces below already exist as standard primitives: an environment loop that
collects episodes into a replay dataset; a convolutional encoder and a
deconvolutional decoder for $64\times64\times3$ images; a gated recurrent unit;
reparameterized diagonal-Gaussian sampling; a closed-form KL divergence between
diagonal Gaussians; the Adam optimizer with gradient clipping; and a
population-based search routine that maintains a Gaussian over candidates, samples
a population, keeps the top fraction, and refits. What does *not* yet exist is the
form of the latent transition — how it should carry memory and represent uncertain
futures at once — the training objective that ties the inferred latent sequence to
the transition it should have produced, and how a planner scores action sequences
entirely in latent space.

```python
# --- existing primitives ---------------------------------------------------
class ConvEncoder:   # image -> embedding vector (exists, from prior video models)
    def __call__(self, image): ...

class ConvDecoder:   # latent feature -> image distribution (exists)
    def __call__(self, feat): ...

class DenseHead:     # latent feature -> scalar Gaussian (reward) (exists)
    def __call__(self, feat): ...

def gaussian_kl(q_mean, q_std, p_mean, p_std):
    # closed-form KL between diagonal Gaussians (exists)
    ...

def reparam_sample(mean, std):
    return mean + std * normal_like(mean)        # (exists)

# --- the slot the method will fill -----------------------------------------
class LatentTransition:
    """How the latent state advances from (state, action) to the next state,
    and how the current observation is folded in to infer it. (TODO: the form
    of the state — what remembers, what is uncertain — is the contribution.)"""
    def prior(self, prev_state, prev_action):
        # TODO: advance the state without seeing the observation
        raise NotImplementedError
    def posterior(self, prev_state, prev_action, embed):
        # TODO: infer the state using the current observation
        raise NotImplementedError
    def features(self, state):
        # TODO: what gets fed to the decoders / reward head
        raise NotImplementedError

def model_loss(batch):
    # TODO: variational objective tying the inferred latent sequence to the
    #       transition that should have produced it (reconstruction vs. a KL
    #       between the inference distribution and the transition)
    raise NotImplementedError

def plan(state_belief, transition, reward_head):
    # TODO: search over action sequences purely in latent space, scoring each
    #       by predicted reward over a horizon; return the first action
    raise NotImplementedError

def train_loop(env):
    # collect S seed episodes -> dataset
    # while not converged:
    #   for C updates: sample sequence chunks; step model_loss
    #   reset env; for each step: infer belief; a = plan(...); add noise;
    #               apply action (repeated R times); store transition
    #   append episode to dataset
    ...
```
