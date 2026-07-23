# FedDyn, distilled

FedDyn (Federated Dynamic Regularization, Acar et al., ICLR 2021) trains a shared model across
heterogeneous clients by making each active client minimize its empirical loss plus a *dynamic*
regularizer that is updated every round from the client's own past gradients, so that **if local
models reach consensus, that consensus is a stationary point of the global loss** — the local and
global stationary points are aligned, not merely damped. Each round costs a single model in each
direction (no separate control variate over the wire), and a single coefficient `alpha` controls
everything.

## Problem it solves

Minimize `ell(theta) = (1/m) sum_k L_k(theta)` over `m` clients with arbitrarily non-IID data,
reaching only a subset `P_t` per round. The core obstacle is a *fixed-point inconsistency*: a
global optimum has `(1/m) sum_k grad L_k(theta^*) = 0` with the individual `grad L_k(theta^*)`
non-zero, while a client minimizing its own `L_k` rests at `theta_k^*` with `grad L_k(theta_k^*) =
0`. So the consensus of clients that each minimize their own loss is `(1/m) sum_k theta_k^* != theta^*`
— client drift, stated as a fixed-point mismatch. Damping repairs (decaying LR, static proximal
penalty, server momentum) throttle the drift but cannot remove it: a *static* regularizer
`(mu/2)||theta - theta^{t-1}||^2` has its local minimizer offset by `-(1/mu) grad L_k(theta^*)`,
a bias the objective bakes in regardless of how exactly it is solved.

## Key idea

Make the regularizer **dynamic**. Each active client `k` solves

```
theta_k^t = argmin_theta [ L_k(theta) - <grad L_k(theta_k^{t-1}), theta>
                                       + (alpha/2) ||theta - theta^{t-1}||^2 ].
```

The linear term `-<grad L_k(theta_k^{t-1}), theta>` cancels the client's own gradient at the
consensus point: evaluate the subproblem gradient at `theta = theta^{t-1}` and the quadratic
contributes nothing, leaving `grad L_k(theta^{t-1}) - grad L_k(theta_k^{t-1})`, which vanishes once
the client's iterate has reached consensus — so consensus becomes a *local* stationary point. The
first-order condition,

```
grad L_k(theta_k^t) - grad L_k(theta_k^{t-1}) + alpha(theta_k^t - theta^{t-1}) = 0,
```

refreshes the stored gradient for free: `grad L_k(theta_k^t) = grad L_k(theta_k^{t-1}) -
alpha(theta_k^t - theta^{t-1})`. Writing the stored state as `H_k = -grad L_k(theta_k^{t-1})/alpha`
(the running sum of the client's net displacements), the regularizer gradient is `alpha(theta -
theta^{t-1}) + alpha H_k`, added to the data gradient at every plain-SGD step — an L2 pull to the
broadcast model plus a fixed accumulated-correction offset. The client needs only `theta^{t-1}`
(broadcast) and its own `H_k` (persistent local state); no extra vector is transmitted.

The server seals the alignment. It keeps a state mirroring the client states and reconstructs

```
h^t   = h^{t-1} - alpha * (1/m) * sum_{k in P_t} (theta_k^t - theta^{t-1}),
theta^t = (1/|P_t|) sum_{k in P_t} theta_k^t  -  (1/alpha) h^t.
```

The divisor `m` (total clients, not `|P_t|`) keeps `h` an honest all-client average under partial
participation. Unrolling, `-(1/alpha) h^t = (1/m) sum_{all k} H_k`, so the global model is the
active-client average plus the mean accumulated displacement over all clients — the *same* `H_k`
object the clients hold, reconstructible by the server from the model deltas it already receives.
At a consensus `theta^infty`, the displacements vanish, `h^t -> (1/m) sum_k grad L_k(theta^infty)
= grad ell(theta^infty)`, and the server fixed point requires `h = 0`, i.e. `grad ell(theta^infty)
= 0`. **The only point the method can rest at is a global stationary point.**

## Defaults and why

- `alpha = 0.1` on Dirichlet non-IID image/character benchmarks (canonical sweep `{1e-3, 1e-2,
  1e-1}`; my implementation scales it per client by data fraction). `alpha` is the single knob: it
  is the quadratic-anchor strength and the gradient-to-displacement scale; larger `alpha`
  convexifies the local subproblem (Hessian `grad^2 L_k + alpha I`) and pulls harder to the
  broadcast model, smaller `alpha` lets clients move more but weakens the correction.
- **Plain SGD locally.** The free gradient-state refresh relies on the local step being an ordinary
  gradient step plus the fixed regularizer offset; momentum would break the displacement/accumulated-
  gradient identity.
- **Half the bits of a control-variate method.** SCAFFOLD ships a control variate alongside the
  model each round; FedDyn's correction lives in client-persistent `H_k` and server-reconstructed
  `h`, so only the model crosses the wire.

## Convergence

For suitably chosen `alpha > 0` and `|P_t| = P` sampled uniformly, FedDyn converges *without a
bounded-dissimilarity assumption*: linearly for `mu`-strongly-convex `L_k`, at `O((1/T) sqrt(m/P))`
for convex `L_k`, and `E||grad ell||^2 = O((1/T)(L m/P)(ell(theta^0) - ell_*) + ...)` for non-convex
`L_k`. The `m/P` factor is the price of partial participation. The heterogeneity term that plagues
FedAvg/FedProx is absent — the alignment of the fixed points, not a dissimilarity bound, drives the
rate.

## Algorithm (Acar et al. 2021, Alg 1)

```
Input: T, theta^0, alpha > 0, grad L_k(theta_k^0) = 0   (so H_k = 0, h^0 = 0)
for t = 1, 2, ... T:
    sample devices P_t subseteq [m]; transmit theta^{t-1} to each
    for each active k in P_t (in parallel):
        theta_k^t = argmin_theta  L_k(theta) - <grad L_k(theta_k^{t-1}), theta>
                                             + (alpha/2)||theta - theta^{t-1}||^2
        grad L_k(theta_k^t) = grad L_k(theta_k^{t-1}) - alpha(theta_k^t - theta^{t-1})
        transmit theta_k^t to server
    stale k not in P_t keep theta_k, grad L_k unchanged
    h^t     = h^{t-1} - alpha * (1/m) * sum_{k in P_t} (theta_k^t - theta^{t-1})
    theta^t = (1/|P_t|) sum_{k in P_t} theta_k^t - (1/alpha) h^t
```

## Working code

The implementation below realizes a per-client accumulated-displacement state `H_k`, a
server running sum `h_sum = (1/m) sum_k H_k`, the regularizer-gradient add in the local loop, and
the server reconstruction `theta^t = active-average + h_sum`. Concretely,
`loss_algo = alpha * sum(theta * (-theta^{t-1} + H_k))` plus
weight-decay `alpha` realizes the same per-step gradient `alpha(theta - theta^{t-1} + H_k)`, the
client state accumulates `H_k += (theta_k - theta^{t-1})`, and the cloud model is `avg +
mean_over_m(H_k)`.

```python
import random
from collections import OrderedDict

import torch
from torch import optim
from torch.utils.data import DataLoader


class Strategy:
    """FedDyn — Federated Dynamic Regularization (Acar et al., ICLR 2021)."""

    def __init__(self, global_model, args):
        self.args = args
        self.num_clients = args.num_clients          # m
        self.alpha = 0.1                             # alpha_coef on Dirichlet CIFAR
        # running sum over ALL m clients of accumulated displacements H_k (= -(1/alpha) h)
        self.h_sum = OrderedDict(
            (n, torch.zeros_like(p, device="cpu", dtype=torch.float32))
            for n, p in global_model.named_parameters())
        self.client_states = {}                     # client_idx -> OrderedDict (H_k, CPU)

    def _zero_like_params(self, named_params):
        return OrderedDict(
            (n, torch.zeros_like(p, device="cpu", dtype=torch.float32))
            for n, p in named_params)

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()

        # Frozen broadcast model theta^{t-1} (named-parameter slots).
        theta_prev = OrderedDict(
            (n, p.detach().clone()) for n, p in model.named_parameters())
        # Client state H_k (lazily zero), on device.
        H_k_cpu = self.client_states.get(client_idx)
        if H_k_cpu is None:
            H_k_cpu = self._zero_like_params(model.named_parameters())
        H_k = OrderedDict((n, H_k_cpu[n].to(device)) for n in theta_prev)

        optimizer = optim.SGD(model.parameters(), lr=local_lr)  # plain SGD
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)

        total_loss, total_samples = 0.0, 0
        for _ in range(local_epochs):
            for batch_data in loader:
                if len(batch_data) != 2:
                    continue
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.dim() == 3:               # seq models: flatten time
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
                loss.backward()
                # Dynamic-regularizer gradient: alpha*(theta - theta^{t-1}) + alpha*H_k.
                for n, p in model.named_parameters():
                    if p.grad is None or n not in theta_prev:
                        continue
                    p.grad.add_(self.alpha * (p.detach() - theta_prev[n] + H_k[n]))
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        # This round's displacement and the H_k accumulation.
        delta_k = OrderedDict(
            (n, (p.detach() - theta_prev[n])) for n, p in model.named_parameters())
        new_H_k = OrderedDict(
            (n, (H_k[n] + delta_k[n]).detach().cpu().float()) for n in theta_prev)
        self.client_states[client_idx] = new_H_k
        self._pending = getattr(self, "_pending", {})
        self._pending[client_idx] = OrderedDict(
            (n, delta_k[n].detach().cpu().float()) for n in theta_prev)

        final_state = OrderedDict(
            (k, v.detach().cpu()) for k, v in model.state_dict().items())
        avg_loss = total_loss / max(total_samples, 1)
        return final_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # Plain average of active clients' models: (1/|P_t|) sum_k theta_k^t.
        n_active = max(len(client_updates), 1)
        avg = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():
                avg[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
            for st, _, _ in client_updates:
                acc += st[key].detach().cpu().float()
            avg[key] = acc / n_active

        # Server state: h_sum <- h_sum + (1/m) sum_{k in P_t} delta_k  (= -(1/alpha) h^t).
        deltas = getattr(self, "_pending", {})
        for n in self.h_sum:
            acc = None
            for d in deltas.values():
                if n in d:
                    acc = d[n].clone() if acc is None else acc + d[n]
            if acc is not None:
                self.h_sum[n] = self.h_sum[n] + acc / max(self.num_clients, 1)
        self._pending = {}

        # Server model: theta^t = avg + h_sum.
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point() or key not in self.h_sum:
                new_state[key] = avg[key]
                continue
            new_state[key] = (avg[key] + self.h_sum[key]).to(ref.dtype)
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available), min(num_to_select, num_available))
```
