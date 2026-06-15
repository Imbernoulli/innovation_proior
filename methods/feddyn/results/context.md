# Context: aligning local and global stationary points in federated learning (circa 2020-2021)

## Research question

A neural network must be trained over data scattered across `m` client devices whose data cannot
leave them. The global objective is the average of per-client empirical risks,

```
min_theta  ell(theta) = (1/m) sum_{k in [m]} L_k(theta),
L_k(theta) = E_{(x,y) ~ P_k}[ ell_k(theta; (x,y)) ],
```

with each `P_k` an arbitrary device-indexed distribution. The structural constraints that make
this hard, and that any method must respect, are:

1. **Communication is the bottleneck.** Each round, the server transmits `theta` to a subset of
   devices, each runs many *local* steps, and ships a message back. The resource to minimize is
   the number of communication *rounds* (and, secondarily, the bits per round).
2. **Partial participation.** Only a subset `P_t subseteq [m]` of devices is active in round `t`,
   sampled at random; stale (non-participating) devices keep their state unchanged.
3. **Heterogeneity (non-IID data).** The `P_k` differ arbitrarily, so the device losses `L_k`
   differ arbitrarily; the minimizer of any one `L_k` need not have anything to do with the
   minimizer of `ell`.

The contribution sought is in *how the local objective is shaped and how the device updates are
combined* — the aggregation rule and whatever per-device correction it exposes — not in the local
optimizer or the simulation harness.

## Background

**The inconsistency at the heart of federated learning.** A global stationary point `theta^*`
satisfies `grad ell(theta^*) = (1/m) sum_k grad L_k(theta^*) = 0`. But a *device's* stationary
point `theta_k^*` satisfies `grad L_k(theta_k^*) = 0`, and under heterogeneity the per-device
gradients `grad L_k(theta^*)` are individually non-zero at the global optimum — they merely sum to
zero. So the dual goals of (i) driving the device models to a consensus `theta_k^t -> theta^t ->
theta_*` and (ii) having each device optimize its *own* empirical loss are *inconsistent*: if every
device exactly minimizes `L_k`, the consensus of their solutions is the average of the `theta_k^*`,
which is not `theta^*`. This is the same client-drift that earlier local-averaging analyses had to
contend with, here stated as a fixed-point mismatch: the place every device wants to go is not the
place the federation needs to reach.

**Local-update averaging and its drift.** The dominant recipe (local SGD / FedAvg, McMahan et al.
2017) takes `K` local steps then averages the returned models. Each active device copies the
server model, runs local SGD on its own data, and ships back its model; the server averages. For
identical devices this is parallel SGD. Under heterogeneity each device walks toward its own
`theta_k^*`, so the average is pulled toward `(1/m) sum_k theta_k^*` rather than `theta^*`, and the
bias grows with the number of local steps — the very lever that buys communication efficiency makes
the drift worse. Variants soften this without removing the inconsistency: a *decreasing learning
rate* (Li et al. 2020b) damps the move, a *proximal penalty* (FedProx, Li et al. 2020a) anchors the
local solution to the broadcast model, and *server-side modifications* (server momentum, Hsu et al.
2019; FedOpt, Reddi et al. 2020) reshape the global step. All of these "recognize the incompatibility
of local and global stationary points," but their fix is based on inexact minimization plus tuning,
and to prove convergence under non-IID data they impose an additional *bounded-dissimilarity*
assumption.

**Penalty-based anchoring (FedProx).** Closest in spirit to a regularizer fix: each device
minimizes `L_k(w) + (mu/2)||w - theta^{t-1}||^2`, a static quadratic spring to the broadcast model.
The added term penalizes updates far from the server model. But the regularizer is the *same*
function every round — it does not depend on the device's own past gradients — so its minimizer
does not align with the global stationary point; inexact minimization is *warranted* (solving it
exactly would still leave the bias), and tuning `mu` requires knowledge of the heterogeneity.

**Gradient-augmented methods (SCAFFOLD, DANE/FedDANE).** Another line transmits extra per-device
variables alongside the models to correct the local direction. SCAFFOLD (Karimireddy et al. 2019)
keeps server and client *control variates* and corrects each local gradient by `(c - c_i)`, which
provably removes the heterogeneity term from the rate. DANE (Shamir et al. 2014) and its
partial-participation version FedDANE (Li et al. 2019) add a linear gradient-correction term to the
local subproblem. These prove convergence by adding device-dependent regularizers, but they *cost
extra communication* — SCAFFOLD transmits both a model and a control variate every round, twice the
bits — and FedDANE/FedSVRG/FedPD need gradient information from all devices each round, which fails
under genuine partial participation.

**The variational / dual-variable viewpoint.** A regularizer added to a local loss can be read as a
dual variable in an augmented-Lagrangian / proximal-point method: if the regularizer is allowed to
*change every round* in response to the device's own iterates, its fixed point can be steered. The
question is whether such a dynamic regularizer can be designed so that, *if* local models converge,
they converge to a point that is a stationary point of the global loss — exactly aligning the two
fixed points rather than merely damping their disagreement — while keeping the per-round message a
single model (no extra control variate over the wire).

## Baselines

**FedAvg (McMahan et al., AISTATS 2017).** Local SGD then sample-weighted (or plain) model average;
no server- or client-side state. Communication-efficient but suffers client-drift under non-IID
data: the aggregate is biased toward the average of device optima, and more local steps worsen it.

**FedProx (Li et al., MLSys 2020).** FedAvg's server average plus a *static* proximal penalty
`(mu/2)||w - theta^{t-1}||^2` on each local objective. Anchors the local solution to the broadcast
model, damping the magnitude of the drift, but the penalty is the same every round so its minimizer
does not align with the global stationary point; the bias is throttled, not removed.

**SCAFFOLD (Karimireddy et al., ICML 2020).** Server and client control variates correcting each
local gradient by `(c - c_i)`; removes the heterogeneity dependence from the rate. The cost: it
transmits a control variate *in addition to* the model every round (2x the bits), and runs plain
SGD with hyperparameters tuned for a target round budget.

**Centralized / large-batch SGD.** The heterogeneity-free yardstick: treat each round as one large
stochastic step. Sets the bar any local-averaging method must match under arbitrary heterogeneity
and ideally beat when devices are similar.

## Evaluation settings

The pre-existing yardsticks for a federated aggregation rule:

- **IID and Dirichlet non-IID splits of CIFAR-10/CIFAR-100, EMNIST-L, MNIST, Shakespeare, plus a
  synthetic dataset** (the splits of McMahan et al. 2017 and the Dirichlet label-ratio split of
  Yurochkin et al. 2019, which controls heterogeneity), over both *moderate* (e.g. 100) and
  *massive* (e.g. 500) device counts, with partial participation handled by sampling devices at
  random each round independent of the past.
- **Metric and protocol.** The number of communication rounds (and the bits transmitted) to reach
  a target test accuracy, and final accuracy after a fixed round budget; comparators are FedAvg,
  FedProx, SCAFFOLD, and the candidate rule, under a shared tuning protocol. Convergence is also
  characterized analytically for strongly convex, convex, and non-convex device losses.

(In the present harness specifically: 200 communication rounds, 10 of 100 clients per round, 5
local epochs of SGD with `lr = 0.01`; benchmarks CIFAR-10 with a Dirichlet(0.1) split, FEMNIST, and
Shakespeare next-character prediction; metric is test accuracy after 200 rounds.)

## Code framework

The aggregation strategy plugs into a fixed federated-simulation harness. The harness owns the data
partition across clients, the per-client local-training loop scaffolding, the client-sampling
schedule, and the round loop; it hands the strategy the current global parameters and the
per-client results, and asks for the next global parameters. Everything about *how the local
objective is shaped and how the device updates are combined* is the open slot.

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
        # Train a local copy of the model and return
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

The harness supplies the data, the local SGD loop scaffolding, and the sampling; the strategy fills
in any state it keeps, any local-step transform, and the rule that turns the returned client updates
into the next global model.
