# Context

## Research Question

An agent acting from pixels needs two kinds of knowledge at once: a compact
description of the current scene and a prediction of how the scene will change
when it acts. A large recurrent predictor is a natural fit for the second job, but
standard reward-driven reinforcement learning makes that choice hard to train. A
sparse return at the end of an episode gives weak credit information for millions
of recurrent weights, so practical model-free agents often use smaller networks
that can be optimized quickly but have less capacity.

The question is how to let a high-capacity predictive model help control without
forcing the whole high-capacity model through the reward signal. The target
setting is visual control: a racing task with randomly generated tracks and
continuous actions, and a first-person survival task where the agent must dodge
incoming projectiles. In both cases the raw observation is an image stream, and
the reward signal alone is a poor training signal for learning the visual and
temporal structure of the environment.

## Background

Internal-model approaches separate prediction from action. A recurrent model can
learn how observations evolve under actions, while a controller can use the
model's state to choose actions. Earlier controller-model systems already raised
the basic possibility: learn a model from interaction, use it to reason about
future states, and train a controller against that model. They also expose a
central danger. If the learned model is imperfect, a controller optimized against
it can exploit model errors and obtain behavior that looks good in the model but
fails in the actual environment.

Model-based policy search gives another precedent. Methods such as PILCO learn a
probabilistic dynamics model and use sampled model rollouts to train a controller.
Their uncertainty estimates help with model error, but Gaussian-process dynamics
do not scale naturally to long histories of high-dimensional images. Pixel-based
tasks need a way to first reduce frames into a lower-dimensional state and then
model dynamics in that lower-dimensional space.

## Existing Primitives

A variational autoencoder can compress a \(64 \times 64 \times 3\) frame into a
latent vector. Its encoder produces \(\mu(x)\) and \(\log \sigma^2(x)\), samples
\(z=\mu+\exp(\log\sigma^2/2)\epsilon\), and decodes \(z\) back to pixels. The
loss combines reconstruction error with a KL penalty toward \(N(0,I)\), giving a
smooth latent space with bounded information capacity.

A mixture density network can make a neural net output parameters of a mixture
distribution rather than a single prediction. Joined to an RNN, it can predict a
distribution over the next latent value from past latents and actions. This is
important when the next observation is multimodal: a projectile may be launched
or not, and averaging those alternatives is not a faithful prediction.

A covariance-matrix evolution strategy can optimize a small parameter vector using
only scalar rollout return. It does not need gradients through the environment or
through time, and it parallelizes over independent candidate rollouts. This makes
it attractive if the reward-trained part of the agent can be kept very small.

## Failure Modes

There are two separate failure modes to avoid. First, a deterministic predictor
of the next latent state can blur or collapse genuinely different futures. In a
survival task, the average of "projectile appears" and "no projectile appears" is
not a useful state. A controller trained on such predictions may learn from the
wrong dynamics.

Second, replacing the real environment with an approximate learned simulator is
dangerous. The controller can search for trajectories that manipulate the learned
state into unrealistic regions, especially if it has access to recurrent hidden
state rather than only rendered observations. A useful scaffold must therefore
ask not only how to train a predictor, but also how stochastic its generated
future should be and how to detect when a policy is merely exploiting the model.

## Code Scaffold

The available building blocks are a frame compressor, a recurrent density model,
an environment rollout loop, and a derivative-free optimizer. The missing design
choice is how to wire these pieces together so that most capacity is trained from
dense prediction losses while the reward-optimized policy remains small.

```python
class FrameVAE:
    def encode(self, frame):
        # returns a latent sample z and posterior statistics
        raise NotImplementedError

class RecurrentDensityModel:
    def step(self, z, action, state):
        # returns next recurrent state and distribution parameters for next z
        raise NotImplementedError

class Controller:
    def action(self, features):
        # TODO: choose the features and parameterization
        raise NotImplementedError

def rollout(controller, env, vae, recurrent_model):
    # TODO: decide what the controller observes and how model state is updated
    raise NotImplementedError

def optimize_controller(fitness_fn, parameter_count):
    # derivative-free optimizer over a small vector
    raise NotImplementedError
```
