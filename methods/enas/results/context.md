Research question

Neural architecture design is still mostly expert trial and error: choose the
number of layers, operations, recurrent-cell nonlinearities, skip connections,
and cell wiring, then train and revise. Reinforcement-learning neural
architecture search gives a concrete way to automate this: an RNN controller
samples a discrete architecture, the sampled child model is trained, its
validation performance becomes a reward, and the controller is updated toward
architectures that score well.

In this formulation each sampled child model is initialized from scratch,
trained to convergence, measured once, and discarded. In the NASNet-scale
setting, this means 450 GPUs for 3-4 days, or 32,400-43,200 GPU-hours. The
research question is how to run reinforcement-learning architecture search so the
controller can compare candidate architectures by validation performance at much
lower compute.

Background

- Policy-gradient / REINFORCE (Williams 1992): for a discrete action m sampled
  from pi(m; theta), the score-function identity gives
  grad_theta E[R] = E[R * grad_theta log pi(m; theta)]. A baseline b can be
  subtracted without bias because E[b * grad_theta log pi] = b * grad_theta 1 =
  0, giving the practical estimator (R - b) * grad_theta log pi(m; theta).
- Autoregressive RNN controllers: an architecture can be serialized as a
  sequence of discrete decisions. An LSTM emits each decision through a softmax
  and feeds the sampled decision's embedding into the next step, so later
  choices are conditioned on earlier choices.
- Cell-based search: instead of searching an entire network directly, prior NAS
  work searches for a small recurrent cell, or for a convolutional cell and a
  reduction cell, then stacks copies. This makes the search space smaller and
  produces designs that can be reused at different depths.
- Skip connections and cheap vision operations: residual connections let deep
  networks preserve information and gradients; depthwise-separable convolutions
  and pooling give a compact menu of operations for convolutional search spaces.
- Transfer, multitask learning, and weight inheritance: learned weights are not
  always tied to one exact model. Transfer and multitask learning reuse
  parameters across related settings, while evolutionary architecture search can
  mutate a parent architecture and initialize the child from the parent's
  weights.
- Stochastic computation graphs: a computation graph can include stochastic
  choices whose sampled values decide which downstream computation is executed.

Baselines

Reinforcement-learning NAS and NASNet use an LSTM controller to sample a child
architecture, train the child to convergence, score it on validation data, and
update the controller with REINFORCE. NASNet searches for reusable convolutional
and reduction cells.

Evolutionary NAS with weight inheritance maintains a population of architectures,
mutates selected parents, and carries parent weights into nearby children, so a
new child is initialized from a previously trained parent.

SMASH / HyperNetworks generate a child's weights from an architecture encoding
through a hypernetwork rather than training each child, using tensor-product
generation to map the encoding to child weights.

Performance prediction, progressive search, and hierarchical search reduce the
number or length of child-training runs by predicting final accuracy from partial
signals, growing architectures progressively, or searching over motifs.

Evaluation settings

Penn Treebank language modeling is the recurrent-cell benchmark. The standard
preprocessed word-level corpus is used, with validation perplexity as the search
signal and test perplexity as the final metric. The recurrent model is kept near
a fixed parameter budget, with standard regularization such as variational
dropout, l2 weight penalty, and tied input/output embeddings. Post-training
processing such as neural cache or dynamic evaluation is excluded.

CIFAR-10 image classification is the convolutional benchmark. The 50,000 training
and 10,000 test images are normalized by channel statistics, padded to 40x40,
randomly cropped back to 32x32, and randomly horizontally flipped; Cutout may be
used in the final training setting. A held-out validation split is used for the
architecture-search reward, while final accuracy is reported on the test set.

For any search method, the protocol separates the data used to train child
weights from the validation data used to reward the controller. Search cost is a
first-class metric, reported in GPU-hours or GPU-days.

Code framework

The starting scaffold is the existing RL-based NAS harness: an autoregressive
controller, a child-network builder, a training loop that gives the child enough
optimization to be scored, and a REINFORCE update from validation reward.

```python
class Controller(object):
  def sample_architecture(self):
    """Return architecture tokens, the decision log-prob terms, and entropy."""
    raise NotImplementedError


def build_child_model(architecture):
  """Instantiate the sampled child architecture with its own parameters."""
  raise NotImplementedError


def train_child_to_score(child, train_data):
  """Optimize the child on training data until it is ready to evaluate."""
  for batch in train_data:
    loss = child.cross_entropy(batch)
    child.optimizer.minimize(loss)


def validation_reward(child, valid_batch):
  """Return validation accuracy, or a decreasing transform of validation ppl."""
  raise NotImplementedError


def reinforce_update(controller, log_prob_terms, entropy, reward, baseline):
  baseline.assign_sub((1.0 - baseline.decay) * (baseline.value - reward))
  advantage = reward - baseline.value
  controller.optimizer.minimize(-sum(log_prob_terms) * advantage)
  return baseline


def search(controller, train_data, valid_data):
  while not done:
    architecture, log_prob_terms, entropy = controller.sample_architecture()
    child = build_child_model(architecture)
    train_child_to_score(child, train_data)
    reward = validation_reward(child, next(valid_data))
    reinforce_update(controller, log_prob_terms, entropy, reward, baseline)
```
