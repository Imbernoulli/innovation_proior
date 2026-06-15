# FedAvg

FedAvg trains a shared deep model across many clients while every client's data stays on its
device. Each round, the server broadcasts the current global model to a random fraction of
clients; each client runs several epochs of plain local SGD on its own data; the server replaces
the global model with the sample-count-weighted average of the returned client models. By doing
many local SGD steps per round it trades cheap on-device computation for expensive communication
rounds, the scarce resource in the on-device setting.

## Problem it solves

Optimize `f(w) = (1/n) Σ_i f_i(w)` when the `n` examples are partitioned across `K` clients —
`f(w) = Σ_k (n_k/n) F_k(w)`, `F_k(w) = (1/n_k) Σ_{i∈P_k} f_i(w)` — under constraints that break
ordinary distributed optimization: data is **non-IID** (each `F_k` an arbitrarily bad proxy for
`f`), **unbalanced** (`n_k` varies widely), **massively distributed** (clients ≫ examples/client),
and **communication-limited** (rare, slow links). Raw data never leaves the device; only models are
sent. The quantity to minimize is the **number of communication rounds** to reach a target test
accuracy, spending the nearly-free on-device compute to reduce it.

## Key idea

Rewrite one synchronous distributed SGD step. With client fraction `C = 1` and learning rate `η`,
having each client send its local gradient `g_k = ∇F_k(w_t)` and the server step
`w_{t+1} = w_t − η Σ_k (n_k/n) g_k = w_t − η ∇f(w_t)` is **algebraically identical** to each client
taking one local step `w^k = w_t − η g_k` and the server averaging,
`w_{t+1} = Σ_k (n_k/n) w^k` (the weights sum to one, so the two updates coincide). This baseline,
one gradient step per round, is **FedSGD**.

Once a client is *stepping and shipping a model* instead of shipping a gradient, it can take many
local steps before the average: sweep its data in minibatches of size `B` for `E` epochs
(`u_k = E·n_k/B` local updates, with `B=∞` denoting one full-client batch) and only then send
`w^k`. That is **FedAvg**. The three knobs: `C` (clients per round), `E` (local epochs), `B`
(local batch size). `E=1, B=∞` recovers FedSGD; `E→∞` with one round is one-shot averaging.

Two design choices carry the method:

- **Sample-count weighting `n_k`.** Because `f` is itself the `n_k`-weighted sum of the `F_k`, the
  `n_k/n` weights are exactly what makes the average of one-step models equal a gradient step on
  `f`. Equal weighting would bias the global model toward tiny clients. With `C<1`, normalize over
  the *selected* set: `w_{t+1} = Σ_{k∈S_t} (n_k/m_t) w^k`, `m_t = Σ_{k∈S_t} n_k`, so the weights
  form a proper convex combination of the models actually received.
- **Shared per-round initialization.** Averaging parameters of independently-trained non-convex
  models can be far worse than either (the straight line between two different-initialization
  solutions humps over a ridge from differing symmetry-breaking). The server re-broadcasts one
  model `w_t` at the *start of every round*, so the averaged models began at the same point and
  drifted only modestly — same basin, well-behaved straight-line loss, an average that can preserve
  common movement while canceling idiosyncratic drift. This is what makes averaging sane; local SGD
  stays plain SGD, since the contribution is the aggregation rule.

As `E` grows, the global model influences a client's local optimization only through the
initialization, so clients increasingly move toward their own local optima and drift apart; smaller
`E` or larger `B` reduces that drift when the local-computation budget is too aggressive.

## Final algorithm

```
Server executes:
  initialize w_0
  for each round t = 1, 2, ...:
    m   <- max(C * K, 1)
    S_t <- random set of m clients
    for each client k in S_t in parallel:
      w^k_{t+1} <- ClientUpdate(k, w_t)
    m_t     <- sum_{k in S_t} n_k
    w_{t+1} <- sum_{k in S_t} (n_k / m_t) * w^k_{t+1}     # weighted average of models

ClientUpdate(k, w):                                        # run on client k
  B <- split P_k into batches of size B
  for each local epoch i = 1..E:
    for batch b in B:
      w <- w - eta * grad loss(w; b)                       # plain local SGD
  return w to server
```

The server combine normalizes over the selected set `S_t` (divide by `m_t = Σ_{k∈S_t} n_k`), not
over all `K` clients.

## Relation to prior methods

- **FedSGD** = FedAvg with `E=1, B=∞` (one local gradient step per round = synchronous distributed
  full-batch SGD on `f`).
- **One-shot / parallelized averaging** (SimuParallelSGD) = the `E→∞`, single-round corner: each
  client trains to convergence locally, average once. Reduces independent variance by about `1/k`
  (standard deviation by `1/√k`) but not the bias of each local empirical solution; worst-case
  (convex, IID) the average is no better than one client's model.
- **Iterative parameter mixing** (structured perceptron) is the direct lineage — iterate, average
  after each epoch, redistribute — but convex, balanced, data-center; FedAvg makes the local
  computation a tunable quantity and handles unbalanced, non-IID deep networks.
- **Asynchronous parameter-server SGD** (Downpour) needs a server round-trip per minibatch —
  prohibitively many messages in this setting.
- **EASGD** couples workers to a center variable by an elastic force but assumes each worker can
  sample the whole dataset and does not address data partitioned across workers.

## Working code

A concrete PyTorch server-side `Strategy`: a stateless local-SGD trainer per client and an
`n_k`-weighted average of the returned models, every round re-anchored to the broadcast global model.

```python
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader


def _client_sgd(model, loader, loss_fn, local_epochs, local_lr, device):
    """ClientUpdate: E epochs of plain SGD on a client's local data."""
    opt = torch.optim.SGD(model.parameters(), lr=local_lr)
    total_loss, total_n = 0.0, 0
    for _ in range(local_epochs):                       # E local epochs
        for inputs, targets in loader:                  # minibatches of size B
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            if outputs.dim() == 3:                      # seq models: flatten time
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()                                  # w <- w - eta * grad(loss; b)
            total_loss += loss.item() * inputs.size(0)
            total_n += inputs.size(0)
    return total_loss / max(total_n, 1)


class Strategy:
    """FedAvg: local SGD + sample-count-weighted average of client models,
    re-anchored each round to the broadcast global model."""

    def __init__(self, global_model, args):
        self.args = args                                # no state carried across rounds

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)        # start from shared w_t
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss = _client_sgd(model, loader, loss_fn,
                               local_epochs, local_lr, device)
        local_state = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in model.state_dict().items()
        )
        return local_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # w_{t+1} = sum_{k in S_t} (n_k / m_t) * w^k,  m_t = sum_{k in S_t} n_k
        if not client_updates:
            return OrderedDict((k, v.detach().clone())
                               for k, v in global_state_dict.items())
        total_samples = sum(n for _, n, _ in client_updates)            # m_t
        if total_samples <= 0:
            raise ValueError("FedAvg aggregation requires positive sample counts")
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():             # integer buffers: copy, don't average
                new_state[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
            for st, n, _ in client_updates:             # n_k-weighted average
                acc += st[key].detach().cpu().float() * (n / total_samples)
            new_state[key] = acc.to(ref.dtype)
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        # random set S_t of m = max(C*K, 1) clients
        return random.sample(range(num_available),
                             min(num_to_select, num_available))
```
