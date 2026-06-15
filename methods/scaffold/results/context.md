# Context: optimizing across heterogeneous federated clients (circa 2018-2019)

## Research question

A model has to be trained over data that lives on a large number of clients — phones, sensors,
hospitals — and that data cannot be moved to a central server. The objective is a finite
average of per-client losses,

```
min_x  f(x) = (1/N) sum_{i=1}^N f_i(x),   f_i(x) = E_{zeta_i}[ f_i(x; zeta_i) ],
```

where `f_i` is the loss on client `i` and `g_i(x) := grad f_i(x; zeta_i)` is an unbiased
stochastic gradient with within-client variance bounded by `sigma^2`. Three structural
constraints make this hard and define the problem:

1. **Communication is the bottleneck.** The network between server and clients is slow and
   unreliable, so the number of *communication rounds* is the resource to minimize. To amortize
   it, each participating client runs many local update steps (`K` of them) between two syncs.
2. **Partial participation.** Only a small subset `S` of the `N` clients is reachable in any
   given round, sampled (effectively) at random.
3. **Heterogeneity (non-IID data).** The client distributions differ arbitrarily, so the local
   objectives `f_i` differ arbitrarily — their individual minimizers `x_i^*` can sit far from
   each other and from the global minimizer `x^*`.

The precise goal is a *server-side aggregation rule* (together with whatever per-client
local rule it exposes) that, under arbitrary heterogeneity and with only `S` of `N`
clients per round, converges to the global optimum in as few communication rounds as possible
— ideally no slower than centralized SGD, and faster when the clients happen to be similar. The
contribution sought is in *how the local work is shaped and how the updates are combined*,
not in the local optimizer or the simulation harness.

## Background

**Local-update federated optimization and its drift.** The dominant recipe takes `K` local
steps then averages. Each sampled client copies the server model `y_i <- x`, performs `K`
stochastic gradient steps on its own data, and ships back the change `y_i - x`; the server moves
its model along the (sample-)average of those changes with a step size. For *identical* clients
this is exactly parallel / local SGD, analyzed asymptotically by Zinkevich et al. (2010) and
sharpened by Stich (2018), Stich & Karimireddy (2019), Khaled et al. (2020). The trouble is
heterogeneity. Each `y_i` is pulled toward its *own* optimum `x_i^*`, so after `K` steps the
average of the client models has moved toward `(1/N) sum_i x_i^*`, which under non-IID data is
*not* the global optimum `x^*`. The gap between `(1/N) sum_i x_i^*` and `x^*` is the
**client-drift**: a systematic bias in the aggregated step that grows with the number of local
steps and with how dissimilar the clients are. This drift was first observed empirically by Zhao
et al. (2018), and it persists even with full-batch gradients and full participation — it is not
a stochastic-noise artifact. To stay stable in its presence, a local-averaging method is forced
to shrink its step size, which directly slows convergence.

Heterogeneity can be quantified. A standard measure is **gradient dissimilarity**: there are
constants `G >= 0`, `B >= 1` with `(1/N) sum_i ||grad f_i(x)||^2 <= G^2 + B^2 ||grad f(x)||^2`
for all `x`. A second, orthogonal measure is **Hessian dissimilarity**:
`||grad^2 f_i(x) - grad^2 f(x)|| <= delta` for all `x`. These are independent — one can have
`G = 0` with `delta` large, or `delta = 0` with `G` arbitrarily large (the clients then share a
common curvature but have far-apart optima). Which notion of dissimilarity actually governs the
slowdown is an open question at this point.

**Smoothness and convexity tools.** Each `f_i` is `beta`-smooth (`beta`-Lipschitz gradient),
which gives the quadratic upper bound `f_i(y) <= f_i(x) + <grad f_i(x), y-x> + (beta/2)||y-x||^2`
and, for convex `f_i` with `x^*` an optimum of the average objective `f`,
`(1/(2 beta N)) sum_i ||grad f_i(x) - grad f_i(x^*)||^2 <= f(x) - f^*`. For `mu`-strong
convexity, a gradient step is contractive: `||x - eta grad h(x) - y + eta grad h(y)||^2 <=
(1 - mu eta) ||x - y||^2` for `eta <= 1/beta`. A perturbed strong-convexity inequality —
`<grad h(x), z - y> >= h(z) - h(y) + (mu/4)||y-z||^2 - beta ||z-x||^2` — lets one use a gradient
read at a *perturbed* point `x` to make progress toward `y` measured at `z`; this is exactly the
situation when local gradients are computed at drifted iterates `y_{i,k}` rather than at the
server point `x`. These are the standard convex-analysis primitives of the time.

**Control variates / variance reduction.** A classical Monte-Carlo idea: to estimate `E[X]`
when a correlated `Y` with known `E[Y]` is cheap, use `theta = alpha (X - Y) + E[Y]`; for
`alpha = 1` this is unbiased, `E[theta] = E[X]`, and `Var(theta) = Var(X) + Var(Y) - 2 Cov(X,Y)`
is reduced when `Cov(X,Y)` is large. In finite-sum optimization this became SVRG (Johnson &
Zhang 2013) and SAGA (Defazio et al. 2014, building on SAG, Schmidt et al. 2017): the SGD
direction `X = g_j(x)` is corrected by a stored stale gradient `Y = g_j(phi_j)` plus the table
average `E[Y] = (1/n) sum_i g_i(phi_i)`, giving the linearly convergent update
`x <- x - gamma [ g_j(x) - g_j(phi_j) + (1/n) sum_i g_i(phi_i) ]`. The correction
`-g_j(phi_j) + (1/n) sum_i g_i(phi_i)` removes the component of `g_j(x)` that is specific to the
sampled index `j` rather than common to the whole sum. This finite-sum machinery is mature, but
it is built for sampling *individual components* of a static sum with one gradient step each, not
for clients that each run many local steps on a *stochastic* objective and are only intermittently
available.

## Baselines

**FedAvg (McMahan et al., AISTATS 2017).** The reference local-averaging method. Each sampled
client `i in S` sets `y_i <- x` and runs `K` local SGD steps,
`y_i <- y_i - eta_l g_i(y_i)`; the server aggregates with a global step size `eta_g`,
`x <- x + (eta_g/|S|) sum_{i in S} (y_i - x)`. It carries no server- or client-side state beyond
the model. Its strength is communication efficiency (one round per `K` local steps); its
weakness is exactly the client-drift above. Because each client descends *its own* `f_i`, the
aggregated direction is biased toward `(1/N) sum_i x_i^*`, and the bias is what the gradient
dissimilarity `G` measures. Where it breaks: under heterogeneous data its convergence stalls,
and increasing the local-step count `K` — the very lever that buys communication efficiency —
makes the drift worse, so past a point more local work *hurts*.

**FedProx (Li et al., MLSys 2020; aka EASGD, Zhang et al. 2015).** Keeps FedAvg's server average
unchanged but adds a proximal penalty to each client's local objective:
`min_w  f_i(w) + (mu/2) ||w - x||^2`, so the local gradient becomes
`g_i(w) + mu (w - x)`. The penalty *anchors* each client to the shared model `x`, shrinking how
far `w` can wander in `K` steps. Where it falls short: the penalty only damps the *magnitude* of
the local move; it does not change the *direction* the local gradient points. The corrected
direction `g_i(w) + mu(w - x)` still descends `f_i`, so the aggregate is still pulled toward
`(1/N) sum_i x_i^*` — the drift is throttled, not removed, and it pays for the throttling with
slower progress. The pull toward the wrong point remains as long as the clients are dissimilar.

**DANE (Shamir et al., 2014).** A distributed approximate-Newton method. Each machine solves a
local subproblem that adds a *linear gradient-correction* term:
`w_i <- argmin_w [ f_i(w) - <grad f_i(x) - grad f(x), w> + (mu/2) ||w - x||^2 ]`, then the server
averages the `w_i`. The correction `-(grad f_i(x) - grad f(x))` subtracts off the part of the
local gradient that is specific to client `i` — the same finite-sum-correction spirit as SAGA.
Where it falls short: the subproblem is an *exact proximal-point* solve, which in practice means
running the local optimizer to convergence each round, and its guarantees depend on the Hessian
dissimilarity quadratically (it needs `(delta/mu)^2` rounds) — far more than one would hope —
unless augmented with an extra line search. It is also a full-participation, deterministic-style
method, not designed for sampling a handful of stochastic clients per round.

**Large-batch / centralized SGD.** As a yardstick, ignore the local-step structure and treat
each round as one big stochastic gradient step over the sampled data. This is heterogeneity-free
in its analysis (`(sigma^2)/(mu N K eps) + 1/mu` rounds in the strongly convex case) but throws
away the communication advantage of taking many local steps — it never benefits from local
computation when clients are similar. Where it falls short: it sets the bar any improved
local-averaging method must *match* under arbitrary heterogeneity and ideally *beat* when the
clients are similar.

## Evaluation settings

The natural yardsticks for a federated aggregation rule, all pre-existing:

- **Simulated heterogeneous quadratics.** `N = 2` one-dimensional or low-dimensional quadratic
  clients with full-batch gradients (`sigma = 0`) and no client sampling, with controllable
  smoothness `beta`, Hessian dissimilarity `delta`, and gradient dissimilarity `G` swept over a
  range (e.g. `G in {1, 10, 100}`). This isolates the effect of heterogeneity and of the number
  of local steps `K` on convergence, decoupled from stochastic noise.
- **EMNIST (Cohen et al. 2017).** The extended-MNIST character dataset divided over `N = 100`
  clients with a controlled non-IID split: a fraction `s%` of each client's data drawn IID and
  the remaining `(100 - s)%` allocated by sorting on label (Hsu et al. 2019), so `s` tunes the
  client similarity from sorted (highly non-IID) to shuffled (near-IID). Convex model: multinomial
  logistic regression. Non-convex model: a two-layer fully connected network. Local methods take
  a few local steps per epoch with a minibatch that is a fixed fraction of the local data, with a
  global step size fixed and the local step size tuned per rule; a fraction of clients is
  sampled each round.
- **Metric and protocol.** The number of *communication rounds* to reach a target test accuracy
  (e.g. 0.5 for logistic regression), and the final test accuracy after a fixed round budget for
  the neural net. Comparators: centralized SGD, FedAvg, FedProx/EASGD, and the candidate
  aggregation rule. The same initialization and tuning protocol is applied across methods.

(In the present harness specifically: 200 communication rounds, 10 of 100 clients per round,
5 local epochs of SGD with `lr = 0.01`; benchmarks CIFAR-10 with a Dirichlet(0.1) split, FEMNIST,
and Shakespeare next-character prediction; metric is test accuracy after 200 rounds.)

## Code framework

The aggregation strategy plugs into a fixed federated-simulation harness. The harness owns the
data partition across clients, the per-client local training loop, the client-sampling schedule,
and the round loop; it hands the strategy the current global parameters and the per-client
results, and asks it for the next global parameters. Everything about *how the local work is
shaped and how the client updates are combined* is the open slot — that aggregation rule is
exactly what is to be designed, so the substrate below is only the generic federated machinery
that already exists, with empty stubs where the contribution will live.

```python
from collections import OrderedDict
import random

import torch
from torch import optim
from torch.utils.data import DataLoader


class Strategy:
    """Generic federated aggregation strategy. The harness owns the round loop,
    the data partition, and client sampling. The strategy decides any server- or
    client-side state to keep, how each client trains locally, and how the
    returned client updates are combined into the next global model."""

    def __init__(self, global_model, args):
        self.args = args
        self.num_clients = args.num_clients
        # TODO: any state required by the aggregation rule.
        ...

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        # Train a local copy of the model on this client's data and return
        # (final_state_dict, num_samples, avg_loss).
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=local_lr)
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        total_loss, total_samples = 0.0, 0
        for _ in range(local_epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                # TODO: any local-rule transform before stepping.
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        # TODO: any per-client bookkeeping derived from this round's local work.
        final_state = OrderedDict(
            (k, v.detach().cpu()) for k, v in model.state_dict().items()
        )
        avg_loss = total_loss / max(total_samples, 1)
        return final_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # global_state_dict: OrderedDict of current global parameters.
        # client_updates: list of (state_dict, num_samples, avg_loss).
        # Returns: OrderedDict of updated global parameters.
        # TODO: combine the client updates (and update any server-side state).
        ...

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available), min(num_to_select, num_available))
```

The harness supplies the data, the local SGD loop scaffolding, and the sampling; the strategy
fills in any state it keeps, any local-step transform, and the rule that turns the returned
client updates into the next global model.
