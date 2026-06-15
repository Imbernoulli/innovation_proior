# SCAFFOLD, distilled

SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) is a federated optimization
method that corrects client-drift with **control variates**. Each client keeps a control
variate `c_i` that tracks its own gradient, the server keeps a control variate `c` that tracks
the average gradient (with the invariant `c = (1/N) sum_i c_i`), and every local step is
corrected by `(c - c_i)` so the local update direction tracks the *global* gradient instead of
the client's own. This removes the dependence on data heterogeneity from the convergence rate.

## Problem it solves

Minimize `f(x) = (1/N) sum_i f_i(x)` across `N` clients with arbitrarily heterogeneous (non-IID)
data, communicating only every `K` local steps and reaching only `S` of `N` clients per round.
Plain local averaging (FedAvg) suffers **client-drift**: each client descends its own `f_i`
toward `x_i^*`, so the aggregate is biased toward `(1/N) sum_i x_i^*` rather than the true `x^*`.
The slowdown is governed by the gradient dissimilarity `G` and is provably impossible to remove by step-size
tuning; it grows with the number of local steps `K`. The goal is an aggregation rule that
converges in few communication rounds under arbitrary heterogeneity, no slower than SGD, and
faster when clients are similar.

## Key idea

The ideal local update would use the global gradient `(1/N) sum_j grad f_j(y_i)`, which is
unaffordable. Approximate it with control variates: `c_i ~ grad f_i(y_i)` (client) and
`c ~ (1/N) sum_j grad f_j(y_i)` (server). Then the corrected local gradient

```
g_i(y_i) - c_i + c  ~  (1/N) sum_j grad f_j(y_i)
```

points along the global direction. The residual against the ideal update is bounded by
`sum_i ||c_i - grad f_i(y_i)||^2` — the control-variate tracking error — which is **independent
of the heterogeneity `G`**; `beta`-smoothness keeps it small, so the `c_i` are kept **stateful**
across rounds. If `c_i` is frozen at 0 the method reduces exactly to FedAvg, so SCAFFOLD is a
strict generalization. It is the lift of SAGA's variance reduction from "sample one finite-sum
component" to "sample `S` clients running `K` local steps each."

## Final algorithm

State: server model `x`, server control variate `c`, per-client control variates `c_i`
(all initialized to 0, preserving `c = (1/N) sum_i c_i`). Step sizes: local `eta_l`, global
`eta_g`.

Each round, sample `S` clients; send `(x, c)`. Each sampled client `i`:

```
y_i <- x
for k = 1..K:
    y_i <- y_i - eta_l ( g_i(y_i) + c - c_i )      # corrected local SGD step (plain SGD)
# refresh the client control variate:
c_i^+ <-  Option I:  g_i(x)                          # one extra gradient pass (more stable)
          Option II: c_i - c + (1/(K eta_l))(x - y_i) = (1/K) sum_k g_i(y_{i,k-1})   # free
send (Delta y_i, Delta c_i) = (y_i - x, c_i^+ - c_i);  set c_i <- c_i^+
```

Server aggregates (over the `S` sampled clients):

```
x <- x + (eta_g / |S|) sum_{i in S} Delta y_i
c <- c + (1/N)        sum_{i in S} Delta c_i      # divide by N (not |S|) to keep c = mean_i c_i
```

The Option-II refresh is free: telescoping the local trajectory,
`sum_k (y_{i,k} - y_{i,k-1}) = -eta_l [ sum_k g_i(y_{i,k-1}) + K(c - c_i) ]`, so
`(1/(K eta_l))(x - y_{i,K}) = (1/K) sum_k g_i(y_{i,k-1}) + (c - c_i)`, hence
`c_i^+ = (1/K) sum_k g_i(y_{i,k-1})` — the average of the already-computed local gradients.
The local step uses **plain SGD** (no momentum) so this identity holds exactly.

## Why each piece

- **Two control variates (server `c` + per-client `c_i`):** `(c - c_i)` estimates the per-client
  drift — the gap between the local and the global direction — and rotates the local step toward
  `x^*`. A proximal penalty (FedProx) only shortens the local move; it leaves the direction
  pointing at `x_i^*`, so the bias survives. SCAFFOLD fixes the *direction*.
- **Stateful `c_i`:** smoothness makes a slightly stale gradient still a good approximation of
  `grad f_i(y_i)`, so retaining `c_i` across rounds keeps the tracking error small for free.
- **Server update divides by `N` (factor `|S|/N`):** keeps `c` equal to the true all-client
  average `(1/N) sum_i c_i` even though only `S` clients refresh their `c_i` each round; dividing
  by `|S|` would over-weight the sampled clients and break the invariant.
- **Two step sizes `eta_l`, `eta_g`:** within-round drift scales with `eta_l`, round progress
  with the effective `eta~ = K eta_l eta_g`; a larger `eta_g` with smaller `eta_l` suppresses
  drift at fixed progress. Often `eta_g = 1` in practice.

## Convergence (no heterogeneity assumption)

Track client-drift `E_r = (1/KN) sum_{i,k} E||y_{i,k} - x||^2`, control-lag
`C_r = (1/N) sum_i ||E[c_i] - grad f_i(x^*)||^2`, and `||x - x^*||^2`. The lag contracts:
`C_r <= (1 - S/N) C_{r-1} + (S/N)(4 beta (f - f*) + 2 beta^2 E_r)`. With the Lyapunov function
`Phi_r = ||x^r - x^*||^2 + (9 N eta~^2 / S) C_r` and step sizes
`eta_l <= min(1/(81 beta K eta_g), S/(15 mu N K eta_g))`, one round contracts as

```
E[Phi_r] <= (1 - mu eta~/2) E[Phi_{r-1}] - eta~ (f(x^{r-1}) - f*) + (12 eta~^2/(KS))(1 + S/eta_g^2) sigma^2,
```

with **no `G` term**. Let `D = ||x^0 - x^*||`, let `F = f(x^0) - f*`, and hide the usual
logarithmic and control-initialization terms. Unrolling gives:

```
R = O~( sigma^2/(mu K S eps) + beta/mu + N/S )      (strongly convex)
R = O~( sigma^2 D^2/(K S eps^2) + beta D^2 / eps + N F / S )   (general convex)
R = O ( beta sigma^2 F/(K S eps^2) + (N/S)^{2/3} beta F / eps )   (non-convex)
```

The `sigma^2/(mu K S eps)` term matches SGD with a `K x` larger batch, so SCAFFOLD is at least as
fast as SGD **for arbitrarily heterogeneous clients**; `N/S` is the additive cost of partial
participation (the lag-healing time). When `sigma = 0, K = 1, S = 1` with Option I, the update is
exactly **SAGA** (sampling one client per round), and the deterministic strongly-convex rate has
the same condition-number-plus-table form, `beta/mu + N/S`.

## Usefulness of local steps (Hessian similarity)

A first-order expansion of the correction (quadratics, constant Hessians `A_i`,
`A = (1/N) sum_i A_i`):

```
grad f_i(y) - grad f_i(x) + grad f(x)
  ~ grad f(y) + ( grad^2 f_i(x) - grad^2 f(x) )(y - x)
  ~ grad f(y) + delta (y - x),
```

so the corrected local direction equals the ideal global direction up to error `delta` — the
**Hessian** dissimilarity `||grad^2 f_i - grad^2 f|| <= delta`, **not** `G`. Drift contracts via
`||I - eta(A_i - A)||^2 <= 1 + 3 eta delta`. The `sigma = 0` communication rate is
`(beta + delta K)/(mu K) = beta/(mu K) + delta/mu`; the useful knee is around
`K = beta/delta`, where the decreasing `beta/(mu K)` term reaches the Hessian-mismatch floor
`delta/mu`. If the curvatures match (`delta = 0`) the rate
improves linearly with `K` even if the client optima `x_i^*` are arbitrarily far apart. So local
steps pay off when the *Hessians* are similar, not when the optima are close. (The exact-solve
ancestor DANE needs `(delta/mu)^2` rounds; SCAFFOLD's gradient-step correction gets `delta/mu`.)

## Working code

Filling the strategy's open slots — server/client control state, the `(c - c_i)` local-gradient
correction, the Option-II refresh, and the `|S|/N`-scaled server control update:

```python
from collections import OrderedDict
import random

import torch
from torch import optim
from torch.utils.data import DataLoader


class Strategy:
    """SCAFFOLD: stochastic controlled averaging for federated learning (Algorithm 1,
    Option II). Server control variate `global_control` (= c) and per-client
    `client_controls[i]` (= c_i), both 0-initialized so c = mean_i c_i holds."""

    def __init__(self, global_model, args):
        self.args = args
        self.num_clients = args.num_clients
        self.global_control = OrderedDict(
            (k, torch.zeros_like(v)) for k, v in global_model.state_dict().items()
        )
        self.client_controls = {}      # client_idx -> OrderedDict c_i (stateful, on CPU)
        self._pending_delta_c = {}     # client_idx -> Delta c_i to aggregate this round
        self.global_lr = getattr(args, "global_lr", 1.0)   # eta_g

    def _zero_like(self, state_dict):
        return OrderedDict((k, torch.zeros_like(v)) for k, v in state_dict.items())

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()

        c = OrderedDict((k, v.to(device)) for k, v in self.global_control.items())
        if client_idx not in self.client_controls:
            self.client_controls[client_idx] = self._zero_like(model.state_dict())
        c_i = OrderedDict((k, v.to(device))
                          for k, v in self.client_controls[client_idx].items())

        # snapshot x (= y_{i,0}) and the per-parameter correction (c - c_i), fixed for K steps
        x = OrderedDict((n, p.detach().clone()) for n, p in model.named_parameters())
        correction = {id(p): c[n] - c_i[n]
                      for n, p in model.named_parameters() if n in c}

        optimizer = optim.SGD(model.parameters(), lr=local_lr)   # plain SGD
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)

        total_loss, total_samples, local_steps = 0.0, 0, 0
        for _ in range(local_epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                # corrected local gradient: g_i + (c - c_i)
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    corr = correction.get(id(p))
                    if corr is not None:
                        p.grad.add_(corr)
                optimizer.step()                                  # y_i <- y_i - eta_l (g_i + c - c_i)
                local_steps += 1
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        # Option II: c_i^+ = c_i - c + (x - y_{i,K}) / (K * eta_l)  =  mean_k g_i(y_{i,k-1})
        if local_steps > 0 and local_lr > 0.0:
            denom = local_steps * local_lr
            delta_c, new_ci = OrderedDict(), OrderedDict()
            for n, p in model.named_parameters():
                if n not in c:
                    continue
                new = c_i[n] - c[n] + (x[n] - p.detach()) / denom
                delta_c[n] = new - c_i[n]
                new_ci[n] = new
            self._pending_delta_c[client_idx] = delta_c
            self.client_controls[client_idx] = OrderedDict(
                (k, v.cpu()) for k, v in new_ci.items())

        final_state = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())
        return final_state, len(client_dataset), total_loss / max(total_samples, 1)

    def aggregate(self, global_state_dict, client_updates, round_num):
        # model update: x <- x + (eta_g/|S|) sum_i (y_i - x).
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not torch.is_floating_point(ref):
                new_state[key] = ref.clone()
                continue
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for st, n, _ in client_updates:
                acc += st[key].float() - ref.float()
            acc = ref.float() + self.global_lr * (acc / max(len(client_updates), 1))
            new_state[key] = acc.to(ref.dtype)

        # control: c <- c + (|S|/N) * mean_i Delta c_i  (divide by N, not |S|)
        deltas = self._pending_delta_c
        if deltas:
            n_updates = len(deltas)
            weight = n_updates / max(self.num_clients, 1)             # |S| / N
            for key in self.global_control:
                acc = None
                for dc in deltas.values():
                    if key in dc:
                        contrib = dc[key].to(self.global_control[key].device)
                        acc = contrib.clone() if acc is None else acc + contrib
                if acc is not None:
                    self.global_control[key] = (
                        self.global_control[key] + (weight / n_updates) * acc)
            self._pending_delta_c = {}
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available), min(num_to_select, num_available))
```
