# Context

## Research question

The goal is to let a reinforcement-learning agent benefit from a *large*,
expressive predictive model of its environment while still being trainable. An
agent acting from raw pixels needs a good representation of the present and a
good model of how the world will unfold; large recurrent networks are exactly the
kind of model that could provide this. The obstacle is training. Model-free RL is
bottlenecked by the **credit-assignment problem** — attributing a sparse, delayed
reward to the right weights — and that bottleneck makes it impractical to learn
the millions of parameters of a large network directly from reward. In practice
people fall back on small networks that iterate quickly to a passable policy,
sacrificing the capacity a big model would bring.

So the precise question is: can the agent's complexity be moved off the
reward-driven optimization and onto something that can be trained cheaply and at
scale? A predictive model of the environment can be learned *unsupervised*, purely
from observed transitions, with plain backpropagation — no credit-assignment
problem, because the training signal (predict the next frame) is dense and local.
If most of the agent's capacity lived in such a model, the part that actually has
to be optimized against reward could be made tiny, and a tiny parameter vector is
something even a derivative-free optimizer can search. The target environments are
pixel-based control tasks: a top-down car-racing game with continuous steering /
gas / brake, and a first-person survival task where the agent dodges projectiles.

## Background

**Internal models and acting on predictions.** A long line of thinking holds that
animals and agents carry a compressed internal model of the world and act on its
predictions rather than on raw sensation — a batter swings before the visual
signal of the pitch could possibly be processed, because a predictive model tells
it where the ball will be. For an artificial agent the analogue is to learn a
representation of past and present plus a predictive model of the future, and to
let fast, reflexive action be driven by that model's state.

**RNN world models with a controller (Schmidhuber, 1990 onward).** The idea of a
recurrent network that models the environment, paired with a separate controller
that acts, goes back to *Making the World Differentiable* (Schmidhuber 1990) and
the recurrent controller–model systems that followed (Schmidhuber 1990–91). In
those systems the recurrent model M predicts and plans ahead step by step and the
controller C is trained against it. *Learning to Think* (Schmidhuber 2015) gave a
unifying framework: an RNN world model M, and a controller C that can query M's
internal subroutines for arbitrary computation — including ignoring M when M is
unreliable. A recurring hazard noted across this line: when M is a *deterministic*
predictor, it is easily *exploitable* — a controller optimized against it can find
trajectories that score well under M but violate the real dynamics.

**Compressing pixels before modeling dynamics.** Learning a dynamics model
directly on high-dimensional pixels is hard, so a standard move is to first learn a
compressed representation of frames and model dynamics in that latent space. Work
on embed-to-control, deep spatial autoencoders, and from-pixels-to-torques used the
bottleneck of an autoencoder as low-dimensional features to control a pendulum from
images, and showed that modeling dynamics in a compressed space makes RL much more
data-efficient. The variational autoencoder (Kingma & Welling 2014) is the relevant
encoder: it compresses each frame into a latent $z$ via an encoder that outputs the
mean and standard deviation of a diagonal Gaussian, samples $z=\mu+\sigma\odot\epsilon$
(reparameterized), reconstructs through a decoder, and is trained on a reconstruction
loss plus a KL term that pulls the latent toward a standard-normal prior. The
Gaussian prior both limits the information capacity of each code and keeps the latent
space well-behaved.

**Mixture density networks and the MDN-RNN.** A mixture density network (Bishop 1994)
makes a neural net output the parameters of a mixture of Gaussians — mixing weights
$\pi_k$, means $\mu_k$, standard deviations $\sigma_k$ — so it can represent a
*multimodal* conditional distribution rather than a single point. Combined with a
recurrent net, the MDN-RNN (Graves 2013 for handwriting; SketchRNN, Ha 2017 for
sketches) models the distribution of the next element in a sequence given the
recurrent hidden state, and is trained by minimizing the negative log-likelihood of
the next element under the predicted mixture. SketchRNN also introduced a
**temperature** $\tau$ that scales the sampling distribution to control how
uncertain/diverse the generated sequence is.

**Derivative-free optimization for small policies.** When a policy has only a few
hundred to a few thousand parameters, evolution strategies are an attractive
optimizer: they need only the *final* cumulative reward of a rollout (no per-step
credit assignment, no differentiable environment), and they parallelize trivially
across many rollouts. The covariance-matrix adaptation evolution strategy (CMA-ES;
Hansen) adapts a full search covariance and works well up to a few thousand
parameters. Evolution strategies had recently been shown to be a viable alternative
to gradient-based deep RL on strong benchmarks (Salimans et al. 2017).

## Baselines

**Deterministic RNN controller–model systems (Schmidhuber 1990–91).** Pair a
recurrent world model with a controller and plan step by step. The gap: a
deterministic M is exploitable — a controller trained inside it can discover
trajectories that look good under M but fail in reality, because M is only an
approximation and is wrong off the training distribution.

**PILCO (Deisenroth & Rasmussen 2011).** Fits a Gaussian-process dynamics model from
collected data and samples many trajectories through it to train a controller for
tasks like pendulum swing-up. Its *Bayesian* uncertainty estimates mitigate model
exploitation, but Gaussian processes do not scale to a long history of
high-dimensional pixel observations; PILCO operates on low-dimensional known states.

**Bayesian-NN dynamics (Deep PILCO, etc.).** Replace the GP with a Bayesian neural
net to scale further, but still demonstrated on low-dimensional, well-defined states
rather than raw pixels.

**Autoencoder-feature controllers (embed-to-control, deep spatial autoencoders,
from-pixels-to-torques).** Learn a compressed latent from images and train a
controller on it, controlling simple systems from pixels. They capture the *present*
frame but lack a temporal/predictive model of the *future*, so the features have
little predictive power.

**Model-free deep RL (DQN, A3C).** The reward-driven yardstick on these tasks. It
works but is bottlenecked by credit assignment when the network is large, and on the
car-racing benchmark typically needs hand-engineered frame preprocessing
(edge-detection, frame stacking) to do well.

## Evaluation settings

Two pixel-based control environments. (1) A top-down car-racing task with randomly
generated tracks, RGB frame observations, three continuous actions (steer, gas,
brake), rewarded for covering track tiles quickly; the score is the average
cumulative reward over many random trials (the published leaderboard sits below
$\sim$850, and the task is considered solved around 900). (2) A first-person
survival task in which the agent dodges projectiles; there is no explicit reward, so
the score is the number of time steps survived (rollouts capped at $\sim$2100 steps,
the task considered solved at an average survival above 750 over 100 consecutive
rollouts). The natural protocol: collect a dataset of rollouts from a random policy,
train the components on it, then evaluate the controller's average score over many
random rollouts. Comparison points are model-free deep RL agents (DQN, A3C) and the
public leaderboard.

## Code framework

The pieces below already exist as standard primitives: a convolutional variational
autoencoder for $64\times64\times3$ frames (encoder to a diagonal-Gaussian latent,
reparameterized sampling, deconvolutional decoder, reconstruction-plus-KL loss); a
recurrent network with a mixture-density output layer that, given the recurrent
hidden state, emits mixing weights, means, and standard deviations of a mixture of
Gaussians, trained by the negative log-likelihood of the next target under that
mixture, with a temperature knob on sampling; an environment-rollout loop; and a
derivative-free evolutionary optimizer that searches a parameter vector using only
the scalar return of a rollout. What does *not* yet exist is how to wire these into
an agent: what the controller should look at, how the controller is trained, and
whether the predictive model can stand in for the real environment during that
training.

```python
# --- existing primitives ---------------------------------------------------
class ConvVAE:                      # frame -> latent z, and back (exists)
    def encode(self, frame):  ...   # -> z (diagonal-Gaussian latent)
    def decode(self, z):      ...   # -> reconstructed frame
    def loss(self, frame):    ...   # reconstruction + KL

class MDN_RNN:                      # mixture-density recurrent predictor (exists)
    def forward(self, action, z, h):
        # -> (pi, mu, sigma) of a Gaussian mixture over the next z, and next h
        ...
    def nll(self, pi, mu, sigma, target):
        # negative log-likelihood of target under the mixture (exists)
        ...
    def sample(self, pi, mu, sigma, temperature):  ...   # exists

def cma_es_optimize(fitness_fn, num_params):
    # derivative-free search using only scalar rollout return (exists)
    ...

# --- the slot the method will fill -----------------------------------------
class Controller:
    def action(self, features):
        # TODO: what does the controller map to an action, and how big is it?
        raise NotImplementedError

def rollout(controller, env, vae, rnn):
    # TODO: at each step, what features does the controller see (present? future?)
    #       and how does the recurrent model's state enter?
    raise NotImplementedError

def train_controller():
    # TODO: optimize the controller's parameters against return -- in the real
    #       environment, or inside the learned predictive model itself?
    raise NotImplementedError
```
