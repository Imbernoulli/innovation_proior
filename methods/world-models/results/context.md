# Context

## Research Question

An agent acting from pixels needs two kinds of knowledge at once: a compact
description of the current scene and a prediction of how the scene will change
when it acts. A large recurrent predictor is a natural fit for the second job,
and training that predictor on dense next-observation prediction losses provides
a rich learning signal separate from the scalar reward.

The question is how to let a high-capacity predictive model help control without
forcing the whole high-capacity model through the reward signal. The target
setting is visual control: a racing task with randomly generated tracks and
continuous actions, and a first-person survival task where the agent must dodge
incoming projectiles. In both cases the raw observation is an image stream.

## Background

Internal-model approaches separate prediction from action. A recurrent model can
learn how observations evolve under actions, while a controller can use the
model's state to choose actions. Earlier controller-model systems already raised
the basic possibility: learn a model from interaction, use it to reason about
future states, and train a controller against that model.

Model-based policy search gives another precedent. Methods such as PILCO learn a
probabilistic dynamics model and use sampled model rollouts to train a controller.
Their uncertainty estimates are useful, but Gaussian-process dynamics do not scale
naturally to long histories of high-dimensional images. Pixel-based tasks benefit
from first reducing frames into a lower-dimensional state and then modeling
dynamics in that lower-dimensional space.

## Existing Primitives

A variational autoencoder can compress a \(64 \times 64 \times 3\) frame into a
latent vector. Its encoder produces \(\mu(x)\) and \(\log \sigma^2(x)\), samples
\(z=\mu+\exp(\log\sigma^2/2)\epsilon\), and decodes \(z\) back to pixels. The
loss combines reconstruction error with a KL penalty toward \(N(0,I)\), giving a
smooth latent space with bounded information capacity.

A mixture density network can make a neural net output parameters of a mixture
distribution rather than a single prediction. Joined to an RNN, it can predict a
distribution over the next latent value from past latents and actions. This is
useful when the next observation is multimodal: a projectile may be launched or
not, and a mixture can represent both alternatives faithfully.

A covariance-matrix evolution strategy can optimize a small parameter vector using
only scalar rollout return. It does not need gradients through the environment or
through time, and it parallelizes over independent candidate rollouts. This makes
it attractive if the reward-trained part of the agent can be kept very small.

## Code Scaffold

The available building blocks are a frame compressor, a recurrent density model,
an environment rollout loop, and a derivative-free optimizer.

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
