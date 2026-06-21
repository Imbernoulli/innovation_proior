# Context

## Research question

The task is to make a deep model that can *learn a new task from only a handful
of examples*. Concretely: given a new task — classify images of a novel object
class after seeing one or five labeled instances, fit a new continuous function
from five points, or control an agent toward a new goal after a few rollouts —
produce a model that performs well on held-out data from that same task, using
only $K$ training examples (often $K\!=\!1$ or $K\!=\!5$) and very little
computation. The model is a high-capacity neural network with far more parameters
than $K$ data points.

A human does this routinely — recognize a new object from one picture, pick up a
new skill in minutes — by bringing prior experience to bear. So the setting is
not "train a model on this one task," but "have already trained on *many* related
tasks, and exploit that to adapt to the next one fast." The quantity we get to
optimize ahead of time is whatever prior structure the model carries into the new
task.

## Background

The problem is usually formalized as **few-shot meta-learning**. There is a
distribution over tasks $p(\mathcal{T})$. Each task $\mathcal{T}_i$ supplies a
small labeled set and a loss $\mathcal{L}_{\mathcal{T}_i}$. The model is trained
during a *meta-training* phase on tasks drawn from $p(\mathcal{T})$, and then,
at *meta-test* time, faced with new tasks (held out during meta-training) on
which it must adapt from $K$ examples. The key conceptual move, due to the
meta-learning tradition (Thrun & Pratt 1998; Schmidhuber 1987; Naik & Mammone
1992), is that *entire tasks* are the training examples: the unit of
generalization is a task, not a data point. A task in the most general form is a
tuple $\{\mathcal{L}, q(\mathbf{x}_1), q(\mathbf{x}_{t+1}\mid\mathbf{x}_t,
\mathbf{a}_t), H\}$ — a loss, an initial-observation distribution, a transition
distribution, and an episode length $H$. For i.i.d. supervised learning
$H\!=\!1$ and the task is just "draw $(\mathbf{x},\mathbf{y})$ pairs and incur a
classification or regression loss"; for control, $H\!>\!1$ and the loss is the
negative reward of a Markov decision process.

One approach is to learn a parametric adaptation procedure — a recurrent
"meta-learner" that emits updates or ingests the whole support set. Another is to
learn an embedding plus a nonparametric comparison rule at test time. A third is
to pretrain one network on data pooled from all tasks, then fine-tune on the new
task. When tasks demand *contradictory* outputs for the same input (e.g. different
sine waves, or different goal directions), pooling drives the network to predict
the average response; fine-tuning from there with only $K$ points either barely
moves or overfits. In control, policies pretrained across tasks have sometimes
been observed to be worse starting points than random initialization when
transferring across related games (Parisotto et al. 2016).

Automatic differentiation libraries (TensorFlow; Abadi et al. 2016) supported
higher-order differentiation: a quantity that is itself defined via a gradient
can be differentiated again with an additional backward pass. ReLU networks were
also known to behave nearly linearly in many local neighborhoods (Goodfellow et
al. 2015).

## Baselines

**Pretraining and fine-tuning (Donahue et al. 2014; "DeCAF").** Train a deep
network on a large source task; its intermediate features transfer broadly, and
a new task is solved by fine-tuning (or by retraining a top layer). Core
mechanism: ordinary supervised pretraining, then gradient descent on the new
task.

**Metric-based few-shot learning: Siamese / Matching / Prototypical networks
(Koch 2015; Vinyals et al. 2016; Snell et al. 2017).** Learn an embedding
$g_\phi(\mathbf{x})$ such that classification of a query reduces to comparison
against the labeled support set in embedding space — nearest neighbor, an
attention-weighted sum of support labels (Matching Networks), or distance to
per-class prototypes (Prototypical Networks):
$\hat{y} = \sum_j a(g_\phi(\hat{\mathbf{x}}), g_\phi(\mathbf{x}_j))\, y_j$.

**Learned optimizers / "learning to learn" (Schmidhuber 1987; Bengio & Bengio
1990, 1992; Hochreiter et al. 2001; Andrychowicz et al. 2016; Li & Malik
2017).** Replace hand-designed gradient descent with a *learned* update rule,
typically a recurrent network whose hidden state plays the role of the optimizer
and which emits parameter updates. "Learning to learn by gradient descent by
gradient descent" (Andrychowicz et al. 2016) trains an LSTM to output the update
for the learner's weights.

**LSTM meta-learner that also learns the initialization (Ravi & Larochelle
2017).** Frames the optimizer as an LSTM whose cell update mirrors a gradient
step, and additionally learns the learner's initial weights, applied to few-shot
image classification (and the source of the MiniImagenet split). Core idea:
$\theta_{t} = f_t \odot \theta_{t-1} - i_t \odot \nabla_{\theta_{t-1}}
\mathcal{L}$, with $f_t, i_t$ produced by the LSTM.

**Memory-augmented / recurrent meta-learners (Santoro et al. 2016, "MANN";
Duan et al. 2016, "RL$^2$"; Wang et al. 2016).** A recurrent network ingests the
support set (or the stream of experience) and adapts through its hidden
dynamics. RL$^2$ applies the recurrent idea to reinforcement learning.

**Context-vector adaptation (Rei 2015).** Adapt only a small set of free
parameters $\mathbf{z}$ concatenated to the input, by gradient, leaving the rest
of the network fixed.

## Evaluation settings

The natural yardsticks are the few-shot benchmarks and protocols that already
exist. **Omniglot** (Lake et al. 2011): 1623 handwritten characters from 50
alphabets, 20 instances each, augmented with multiples-of-90° rotations
(Santoro et al. 2016); the common protocol holds out characters and tests $N$-way
classification with $K=1$ or $K=5$ shots ($N\in\{5,20\}$), following Vinyals et
al. 2016. **MiniImagenet** (Ravi & Larochelle 2017): 64 train / 12 val / 24 test
classes of $84\times84$ natural images, 5-way 1-shot and 5-shot. The standard
embedding architecture for these is four $3\times3$ conv blocks (conv → batch
norm → ReLU → $2\times2$ pool) with 32–64 filters (Vinyals et al. 2016),
followed by a linear softmax. For **regression**, a synthetic sinusoid family
(amplitude in $[0.1,5.0]$, phase in $[0,\pi]$, inputs in $[-5,5]$), a small
two-hidden-layer ReLU MLP, mean-squared-error loss, and $K\in\{5,10,20\}$. For
**control**, simulated continuous-control tasks in the rllab suite (Duan et al.
2016) and the MuJoCo simulator (Todorov et al. 2012): 2D navigation to random
goals, and planar half-cheetah / 3D "ant" locomotion at random goal velocities
or directions, with a two-hidden-layer (size 100) ReLU policy, vanilla policy
gradient (REINFORCE; Williams 1992) for the task gradients, and TRPO (Schulman
et al. 2015) as the outer optimizer. Metrics: $N$-way classification accuracy,
MSE, and average return. A standard comparison point is an *oracle* that
receives the true task identity (class, amplitude/phase, or goal) as input, as
an upper bound. The protocol is always: meta-train on a set of tasks, then
measure performance on held-out tasks after $K$-shot adaptation.

## Code framework

The pieces that already exist: a data pipeline that samples a task and, for that
task, a small support batch and a query batch; a parametric model whose forward
pass can be run on a supplied set of weights; standard losses; a standard
optimizer; and an autodiff engine that can differentiate through a gradient
computation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Learner(nn.Module):
    def __init__(self, dim_in, hidden, dim_out):
        super().__init__()
        self.params = nn.ParameterDict()
        self.n_layers = 0
        pass  # TODO: define the plain trainable network parameters

    def functional_forward(self, x, weights):
        pass  # TODO: run the network on an explicit weight dictionary

    def init_weights(self):
        pass  # TODO: return the current trainable weights as a dictionary


def mse_loss(pred, y):
    pass  # TODO: regression loss


def meta_train(model, sample_tasks, steps=70000, meta_lr=1e-3):
    meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
    for _ in range(steps):
        tasks = sample_tasks()
        pass  # TODO: given a batch of tasks (each with its own support and
              # query data), what should be optimized, and how?
    return model
```
