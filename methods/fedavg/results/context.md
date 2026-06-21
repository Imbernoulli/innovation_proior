## Research question

Phones and tablets have become the primary computing devices for many people, and they hold an
enormous amount of data — what users type on their keyboards, the photos they take — that would
make excellent training data for the language and image models that power those same devices. This
data is privacy-sensitive and large, so a natural design constraint is not to ship it to a data
center at all. The question is whether one can train a shared, high-quality deep network *across a
large population of devices while every device keeps its own data local*, sending only model-sized
messages, never raw examples.

The optimization problem this implies differs from the distributed optimization studied in data
centers in four specific ways:

- **Non-IID.** Each device's local data reflects one user's usage, so it is not representative of
  the population distribution. A per-device objective can be an arbitrarily bad proxy for the
  global objective.
- **Unbalanced.** Some users generate far more data than others, so local dataset sizes vary by
  orders of magnitude.
- **Massively distributed.** The number of participating devices is much larger than the average
  number of examples on any one device.
- **Communication-limited.** Devices are often offline or on slow, metered links (upload on the
  order of 1 MB/s), volunteer only when charged and on unmetered wifi, and participate in only a
  handful of rounds per day.

The economics differ from the usual data-center tradeoff. There, communication inside the cluster
is cheap and compute dominates, which is why so much effort goes into GPUs. Here a device's local
dataset is small and modern phones have fast processors, so on-device computation is inexpensive
relative to communication. The objective to optimize is therefore not wall-clock or FLOPs but the
**number of communication rounds** needed to reach a target accuracy. A deployed version must also
cope with devices that change their data, drop out, or send corrupted updates; a controlled study
can fix a set of clients with fixed local datasets and a synchronous round structure and study the
core setting of non-IID/unbalanced data under tight communication limits.

## Background

Essentially all of the recent successes in deep learning rest on stochastic gradient descent and
on shaping models so their losses are amenable to simple gradient-based optimization. So any method
for this setting will be built out of SGD. The natural formulation is a finite-sum objective

```
min_{w ∈ R^d}  f(w),    f(w) = (1/n) Σ_{i=1}^n f_i(w),
```

with `f_i(w) = loss(x_i, y_i; w)` the loss of model `w` on example `i`. If the `n` examples are
partitioned across `K` clients, with `P_k` the index set on client `k` and `n_k = |P_k|`, this
rewrites exactly as

```
f(w) = Σ_{k=1}^K (n_k/n) F_k(w),    F_k(w) = (1/n_k) Σ_{i∈P_k} f_i(w),
```

a sample-count-weighted combination of per-client objectives. If the partition were formed by
distributing examples uniformly at random, then `E_{P_k}[F_k(w)] = f(w)` — the IID assumption that
data-center distributed optimization leans on. In the on-device setting this does not hold: `F_k`
can be an arbitrarily bad approximation to `f`.

Two pieces of empirical knowledge about deep loss surfaces are relevant. First, although training
neural networks is a non-convex problem long feared to be riddled with bad local minima, a body of
work found the opposite for sufficiently over-parameterized networks: the dangerous critical points
are saddles rather than poor minima (Dauphin et al. 2014), and the loss surfaces of large networks
are surprisingly well-behaved (Choromanska et al. 2015). Most concretely, Goodfellow, Vinyals &
Saxe (2015) introduced a simple diagnostic: evaluate the loss along the straight line
`θ = (1-α)θ_i + α θ_f` connecting a network's initial parameters to its trained solution. Across a
wide range of architectures they found this 1-D cross-section is smooth and approximately convex,
with no exotic barriers — SGD essentially descends a single well-behaved slope from initialization
to solution. Read in terms of *averaging* parameters: two networks that started from the *same*
initialization and were trained separately tend to lie in the same well-behaved basin, whereas two
networks from *different* random initializations need not, because of early symmetry-breaking.

Second, there is direct evidence about averaging the parameters of independently trained networks.
Following the linear-interpolation diagnostic, one can take two MNIST classifiers trained by SGD
on different small data subsets and plot the loss of `θ w + (1-θ) w'` as `θ` sweeps across and
beyond `[0,1]`. When `w` and `w'` started from *different* random seeds, the interpolated loss
rises into a tall barrier between them — the average of the two models is far worse than either
parent. When `w` and `w'` started from a *shared* seed, the curve dips: their average achieves a
loss on the full training set *lower* than either parent achieved on its own subset.

The broad state of distributed training at the time: data-center systems either pass gradients
every step to a central server, or train local replicas and combine them, and the established
analysis assumes some combination of convexity, IID data, equal-sized shards, and many more
examples per node than nodes.

## Baselines

These are the prior methods a solution for this setting is measured against.

**Synchronous large-batch distributed SGD (Chen et al. 2016).** In a data center, distributed
SGD can be run synchronously: each round, a set of workers each compute a gradient on a minibatch
of their data, the server sums the (sample-count-weighted) gradients to form one large-batch
gradient, and applies a single SGD step. Chen, Monga, Bengio & Jozefowicz (2016) showed that this
synchronous scheme — guarding against slow workers with a few backup workers — avoids the gradient
staleness of asynchronous training and converges faster and to better test accuracy than async in
the cluster. Adapted to many clients, one selects a `C`-fraction of clients each round and takes
the gradient over all their data, so `C` sets the global batch size and `C=1` is full-batch
gradient descent (and the per-round batch gradient still satisfies `E[g] = ∇f(w)`). Exactly one
gradient step is taken per round of communication.

**Asynchronous SGD with a parameter server (Dean et al. 2012).** DistBelief's Downpour SGD runs
many model replicas in parallel; each replica repeatedly fetches the current parameters from a
central parameter server, computes a gradient on a local minibatch, and pushes the gradient back
asynchronously. This scaled deep network training to thousands of cores. Every minibatch step
requires a round-trip to the parameter server.

**One-shot / parallelized averaging (Zinkevich et al. 2010).** SimuParallelSGD partitions the data
across `k` machines, has each machine run SGD over its entire shard with *no* communication, and
averages the resulting parameter vectors exactly once at the end: `v = (1/k) Σ_i w_i`. Its
contraction-mapping analysis shows that with a small fixed learning rate the per-machine parameter
distribution converges to an asymptotically normal regime around the optimum, and averaging across
`k` machines reduces the variance of the independent noise. The analysis assumes a convex loss,
IID data, and equal shard sizes. Worst-case results in the convex IID setting (Zhang, Wainwright &
Duchi 2012; Arjevani & Shamir 2015) characterize the single averaged model relative to a model
trained on one machine's data alone.

**Iterative parameter mixing (McDonald, Hall & Mann 2010).** For the structured perceptron, this
trains a local model on each data shard, and after *each epoch* sends the local weight vectors to a
server, averages them, and redistributes the average to every shard for the next epoch. It was
shown that one-shot mixing (averaging only at the very end) can fail, while repeatedly averaging
and redistributing converges well. It was developed for a convex, non-deep model in the
data-center / cluster setting (a handful of workers, fast network), on balanced, roughly IID
shards, averaging after a fixed full epoch.

**Periodic parameter averaging for DNNs (Povey, Zhang & Khudanpur 2015).** This trains DNN
replicas in parallel and averages their parameters periodically (every minute or two of
computation), redistributing the average. The reported method pairs the periodic averaging with an
approximate natural-gradient local optimizer, run on IID data in a data center.

**Elastic Averaging SGD (Zhang, Choromanska & LeCun 2015).** EASGD keeps a separate local variable
on each worker tied to a central "center variable" `x̃` by an elastic (quadratic-penalty) force:
the local update is `x^i ← x^i − η(g_i + ρ(x^i − x̃))` and the center moves toward a space-and-time
moving average of the workers, `x̃ ← x̃ + η Σ_i ρ(x^i − x̃)`. Loosening the coupling (small `ρ`)
lets workers explore further between communications, reducing how often they must talk to the
master. It assumes every worker can sample the *entire* dataset (IID).

**Communication-efficient distributed convex optimization (CoCoA, DANE, DiSCO, and related).**
A line of work (Ma et al. 2015; Shamir, Srebro & Zhang 2013; Zhang & Xiao 2015; Zhang, Wainwright
& Duchi 2012) designs distributed optimizers that reduce communication for convex empirical-risk
problems, some using approximate second-order information. They assume convexity, that the number
of clients is much smaller than the number of examples per client, IID data distribution, and an
identical number of points per node.

## Evaluation settings

The natural yardsticks for on-device training:

- **MNIST digit recognition** (LeCun et al. 1998) with two model families: a 2-hidden-layer MLP
  (200 units each, ReLU; ~199K parameters) and a small CNN (two 5×5 conv layers, 32 then 64
  channels with 2×2 max-pooling, a 512-unit dense layer, softmax; ~1.66M parameters). Two client
  partitions to probe data heterogeneity: an **IID** split (shuffle, then 100 clients × 600
  examples) and a **pathological non-IID** split (sort by label into 200 shards of 300, give each
  of 100 clients 2 shards, so most clients see only two digits). Both partitions balanced.
- **Character-level language modeling on the Complete Works of Shakespeare:** build one client per
  speaking role per play with at least two lines (≈1146 clients), naturally unbalanced (roles
  range from a few lines to many) and non-IID by speaker; train/test split is the first 80% / last
  20% of each role's lines, temporally separated. The model is a stacked character LSTM: embed each
  character into 8 dimensions, two LSTM layers of 256 units, softmax over characters, unrolled 80
  characters (~866K parameters). A balanced IID version of the same data serves as a control.
- **CIFAR-10 image classification** (Krizhevsky 2009): 10 classes of 32×32 RGB images, 50K train /
  10K test, partitioned into 100 clients of 500 train / 100 test in a balanced IID setting (no
  natural user partition). A standard two-conv-plus-two-dense CNN (~10⁶ parameters) from the
  TensorFlow tutorial, with the usual cropping-to-24×24, random flips, and contrast/brightness/
  whitening preprocessing.
- **A large-scale next-word prediction task:** ~10M public social posts grouped by author (>500K
  clients), a realistic non-IID proxy for mobile text entry; a 256-unit LSTM over a 10K-word
  vocabulary with 192-dim input/output embeddings (~4.95M parameters), unrolled 10 words.

The protocol: a synchronous round structure with a fixed set of clients; the per-round knobs are
the client fraction `C`, the number of local epochs `E`, and the local minibatch size `B` (with
`B = ∞` meaning the whole local dataset is one batch); the learning rate `η` is selected over a
multiplicative grid (typically 11–13 values at resolution `10^{1/3}` or `10^{1/6}`); the headline
quantity is the number of communication rounds to reach a fixed target test accuracy.

## Code framework

A federated training harness already has the round mechanics: a server holds a global state, selects
some clients each round, broadcasts the current state to them, lets each selected client compute on
its own data, collects a payload and sample count from each selected client, applies one server
update rule, and repeats. The local computation and round loop are available; the single empty slot
is the server rule that turns the returned client payloads into the next global state.

```python
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader


def _local_sgd(model, loader, loss_fn, local_epochs, local_lr, device):
    """Train `model` in place with plain SGD on a client's local data."""
    opt = torch.optim.SGD(model.parameters(), lr=local_lr)
    total_loss, total_n = 0.0, 0
    for _ in range(local_epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            if outputs.dim() == 3:                       # seq models: flatten time
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()
            total_loss += loss.item() * inputs.size(0)
            total_n += inputs.size(0)
    return total_loss / max(total_n, 1)


class Strategy:
    """Server-side round coordinator with local computation, client selection,
    and one unresolved server update slot."""

    def __init__(self, global_model, args):
        self.args = args

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        # Generic local computation from the broadcast global state.
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss = _local_sgd(model, loader, loss_fn,
                              local_epochs, local_lr, device)
        payload = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in model.state_dict().items()
        )
        return payload, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # global_state_dict : OrderedDict of current global state
        # client_updates    : list of (payload, num_samples, avg_loss)
        # round_num         : current communication round
        # Returns the next global state.
        # TODO: server update rule.
        pass

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available),
                             min(num_to_select, num_available))
```
