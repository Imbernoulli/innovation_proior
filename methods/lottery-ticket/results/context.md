# Context: training sparse neural networks from the start

## Research question

Trained neural networks are wildly overparameterized. *Pruning* — removing weights from an
already-trained network — routinely deletes more than 90% of the parameters with no loss in test
accuracy, yielding a small network that is cheaper to store and faster at inference. This raises an
obvious efficiency question that, at the time, has a frustrating answer.

If a network with one-tenth the parameters can represent the learned function just as well, why not
*train* that small network directly and save the training cost too? The answer from practice is: you
can't. A sparse architecture obtained by pruning, when trained from scratch with a fresh random
initialization, learns more slowly and reaches *lower* accuracy than the original dense network. So
the small network is trainable only as the *endpoint* of training the big one, never as a starting
point.

The precise problem this sets up: there demonstrably exists a small subnetwork that achieves the
full network's accuracy (we obtained it by pruning), yet stochastic gradient descent starting from a
fresh random initialization of that same sparse architecture cannot find a solution of comparable
quality. Either small trainable networks are a fiction outside the shadow of a large network, or
there is some missing ingredient that makes a sparse architecture trainable from the start. The goal
is to determine which, and if the latter, to find that ingredient and a procedure that exposes it.

## Background

**Pruning.** Removing unnecessary weights is an old idea. Optimal Brain Damage (LeCun et al. 1990)
and Optimal Brain Surgeon (Hassibi & Stork 1993) prune weights using second-derivative (Hessian)
saliency. More recently, Han et al. (2015) showed that simple *magnitude-based* pruning — delete the
weights with the smallest absolute value — reduces image-recognition networks dramatically without
hurting accuracy, and that the standard recipe is train → prune → *fine-tune* the survivors. Pruning
can be unstructured (individual weights, giving sparse weight matrices) or structured (whole filters
/ channels / units, giving smaller dense networks better suited to hardware).

**The diagnostic phenomenon — sparse-from-scratch is hard.** The motivating measurement: take a
sparse architecture (e.g. a randomly sampled mask at a target sparsity, which mimics the connectivity
left by unstructured pruning) and train it from a fresh random initialization. Sweeping the sparsity
level and recording both the iteration at which validation loss bottoms out (a proxy for learning
speed) and the test accuracy there, one observes a clean monotone trend: the sparser the network, the
slower it learns and the lower its final accuracy. This is the wall, and it is consistent with
contemporary experience reported across the pruning literature — that a pruned model trained from
scratch performs worse than one retrained from its post-training weights, and that for surviving
connections it is better to keep the weights from the initial training phase than to re-initialize
them. The standard reading of this is that small-capacity networks are simply hard to train.

**Initialization.** Networks are initialized by drawing weights from a fixed distribution D_θ —
commonly Gaussian Glorot/Xavier (Glorot & Bengio 2010), which scales the variance by fan-in/fan-out
so that signal and gradient magnitudes are preserved across layers. These schemes are designed for
*dense* networks; there is no established theory for initializing sparse ones.

**Overparameterization and generalization.** It is widely held that overparameterized networks are
*easier to train* (more capacity, smoother optimization), and separately that *compact* hypotheses
generalize better (minimum description length; Occam's razor). The relationship between compression
and generalization is an active theoretical topic — tighter generalization bounds have been proven
for networks that compress further. Holding these two beliefs together — bigger is easier to train,
smaller generalizes better — is exactly the tension a sparse-but-trainable network would sit inside.

**Related directions for small/sparse networks.** Engineering compact architectures by hand
(SqueezeNet, MobileNets); representing weight matrices as low-rank factors (Denil et al. 2013);
restricting optimization to a small random subspace while still updating all parameters (Li et al.
2018, "intrinsic dimension"); learning sparsity during training via L0 regularization or Bayesian /
variational dropout that drives some weights' keep-probabilities to zero; and replacing dead weights
with new random connections during sparse training (Deep Rewiring). All of these either keep the full
parameter set live during training or learn the sparsity as they go.

## Baselines

The prior approaches to obtaining a small or sparse network — the points of comparison for "can a
sparse network be trained from the start" — are:

- **Magnitude pruning + fine-tuning (Han et al. 2015).** Train the dense network, delete
  smallest-magnitude weights, then continue training (fine-tune) the surviving weights from their
  post-training values. Core idea: the function needs few parameters, and the survivors are already
  in a good basin. Gap: the small network is only trainable because the big network's training already
  placed its weights; it provides no way to train a sparse network *from scratch*.

- **Second-derivative pruning (Optimal Brain Damage / Surgeon).** Use the loss Hessian to score
  weight saliency and remove the least salient. Gap: same — a post-training compression step,
  not a from-scratch training method, and more expensive to compute.

- **Train-from-scratch on a pruned architecture (the implicit baseline).** Take the architecture left
  by pruning and re-initialize it randomly, then train. Core idea: maybe the architecture alone
  suffices. Gap: this is precisely the procedure that *fails* — it learns slower and less accurately
  the sparser it is, which is the phenomenon to be explained.

- **Restricting optimization (intrinsic dimension; Li et al. 2018).** Optimize within a small random
  affine subspace of the full parameter space. Core idea: few effective degrees of freedom suffice.
  Gap: all parameters still vary (the subspace is dense in parameter space); it does not produce a
  sparse network with most weights fixed at zero.

- **Learned sparsity during training (L0 / variational / Bayesian-dropout pruning; Deep Rewiring).**
  Learn gating or keep-probabilities that sparsify the network as training proceeds. Core idea:
  discover the sparse network jointly with training. Gap: the sparse structure emerges from training
  the over-parameterized network; it is not a fixed sparse subnetwork trainable in isolation from a
  known starting point.

## Evaluation settings

The natural testbeds are standard small-scale supervised image classification, where a network can
be trained many times over (the discovery procedure is repetitive):

- **MNIST** with a fully-connected network (Lenet-300-100: hidden layers 300 and 100). Adam optimizer;
  minibatches; tens of thousands of iterations.
- **CIFAR-10** with small VGG-style convolutional networks (Conv-2/4/6: two/four/six 3×3 conv layers,
  max-pooling after every two, then two fully-connected layers), and with deeper networks (Resnet-18,
  VGG-19). Adam for the small conv nets; SGD with momentum 0.9 and a stepped learning-rate schedule
  (0.1 → 0.01 → 0.001) for the deep nets.

All networks initialized with Gaussian Glorot. Metrics: test accuracy, and — as a learning-speed
proxy — the early-stopping iteration (the iteration of minimum validation loss). The sparsity of a
mask m is reported as P_m = ‖m‖₀ / |θ|, the fraction of weights remaining.

## Code framework

The primitives that exist: a network whose weights can be trained by SGD/Adam, a data loader, a
training loop with early stopping, and a magnitude-pruning routine that, given a layer's weights,
zeros out the smallest. How to combine these into a procedure that yields a sparse network trainable
from a fixed starting point is left as an empty slot.

```python
import numpy as np
import copy

def initialize(model):
    """Draw weights from D_theta (e.g. Gaussian Glorot). Returns theta_0."""
    # TODO: random init
    pass

def train(model, weights, mask, num_iters):
    """Train (masked) weights by SGD/Adam for num_iters; return final weights and metrics."""
    # forward uses (mask * weights); gradients only flow to surviving weights
    # TODO
    pass

def prune_by_percent_once(percent, mask, final_weight):
    """Magnitude prune: zero out the smallest 'percent' of the still-surviving weights."""
    sorted_weights = np.sort(np.abs(final_weight[mask == 1]))
    cutoff_index = int(np.round(percent * sorted_weights.size))
    cutoff = sorted_weights[cutoff_index]
    return np.where(np.abs(final_weight) <= cutoff, 0, mask)

def find_subnetwork(model, num_rounds, prune_percent, num_iters):
    theta_0 = initialize(model)
    mask = {k: np.ones_like(v) for k, v in theta_0.items()}
    weights = copy.deepcopy(theta_0)
    for _ in range(num_rounds):
        trained, metrics = train(model, weights, mask, num_iters)
        mask = {k: prune_by_percent_once(prune_percent, mask[k], trained[k]) for k in mask}
        # TODO: prepare weights for the next round
    return mask, weights
```
