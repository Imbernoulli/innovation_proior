## Research question

Train one shared model across 100 clients whose data never leaves them and is **non-IID** — a
Dirichlet(0.1) split of CIFAR-10 and FEMNIST, and a naturally per-speaker split of Shakespeare.
Each round the server reaches only 10 of the 100 clients; each runs 5 local epochs of plain SGD,
then ships its model back. The single thing being designed is the **federated learning recipe** —
how each client shapes its local objective and how the server combines the returned models — under
the one constraint that bites here: the clients' local optima sit far from the global optimum, so
naive averaging suffers **client drift**. Everything else (data partition, the per-client local
loop scaffolding, client sampling, the round loop, evaluation) is fixed.

## Prior art before the first rung

The lineage the first rung reacts to is the local-update averaging recipe and its drift, the
problem every step on this ladder is trying to beat.

- **Naive distributed / synchronous SGD (FedSGD).** Each round, each selected client computes one
  gradient on its local data, the server averages them and takes one step. Provably a stochastic
  gradient step on the global objective, but **one gradient step per communication round** — tens of
  thousands of rounds to converge, hopeless when communication is the scarce resource. Gap: spends
  the scarce thing (rounds) one step at a time.
- **Local-update averaging (the dominant recipe).** Rewrite the FedSGD step: each client taking one
  local step and the server averaging the stepped models is *algebraically identical* to clients
  shipping gradients and the server stepping — and once a client is stepping-and-shipping a model,
  nothing stops it from taking *many* local steps before the average. This trades free on-device
  compute for scarce rounds. Gap: under non-IID data, many local steps drive each client toward its
  *own* optimum `theta_k^*`, so the average is pulled toward `(1/N) sum_k theta_k^*`, not the global
  `theta^*` — client drift, and it grows with the local-step count.
- **The fixed-point statement of the drift.** A global optimum has `(1/m) sum_k grad L_k(theta^*) =
  0` with the individual `grad L_k(theta^*)` non-zero; a client minimizing its own `L_k` rests where
  `grad L_k = 0`. The two stationary points are *inconsistent* under heterogeneity — the place every
  client wants to go is not the place the federation must reach. Gap: this is the disease; the ladder
  is repairs of increasing structural depth.

## The fixed substrate

A federated-simulation harness is frozen: it owns the Dirichlet/per-speaker data partition across
100 clients, the per-client local-SGD loop scaffolding, the client-sampling schedule, the 200-round
loop, and global-model evaluation on the held-out test set. It instantiates **one** `Strategy`
object before the round loop (so the strategy may carry server- and client-side state across rounds)
and per round calls, in order: `select_clients(N, K, round)` to pick the 10 participants;
`client_local_train(global_state, client_dataset, model_fn, loss_fn, local_epochs, local_lr,
local_batch_size, device, client_idx)` for each, which must return `(state_dict, num_samples,
avg_loss)`; then `aggregate(global_state, client_updates, round)` with `client_updates` the list of
those tuples, returning the next global `state_dict`. Models are `CifarCNN` / `FemnistCNN` /
`CharLSTM` (chosen by dataset); the harness also provides a `_default_client_sgd` helper (plain SGD,
optional `loss_aug` term per minibatch).

## The editable interface

Exactly one region is editable — the `Strategy` class in `flower/custom_fl_aggregation.py` (the
contract above). It owns the **full** recipe, both the client-side local objective and the
server-side combine; a server-only aggregator could only approximate methods whose innovation lives
in the local loop. Every method on the ladder is a fill of this same contract: `__init__`
(any persistent state), `client_local_train` (the local objective / gradient transform),
`aggregate` (the combine + any server-state update), and `select_clients`.

The starting point is the scaffold default: plain local SGD on each client + a sample-count-weighted
average of the returned models (this is FedAvg). Each method replaces exactly these definitions and
nothing else. One harness detail is load-bearing for every combine: a model `state_dict` carries
non-floating-point buffers (integer counters, batch-norm tracked-batches) that must not be averaged;
the default copies one client's value through for those and weight-averages only the float tensors,
on CPU in float32.

```python
class Strategy:
    """Default FL recipe: plain SGD locally + sample-count-weighted server average.
    The harness instantiates this once and calls select_clients -> client_local_train
    (per selected client) -> aggregate, each round."""

    def __init__(self, global_model, args):
        self.args = args

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)        # start from the shared global model
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss, _ = _default_client_sgd(
            model, loader, loss_fn, local_epochs, local_lr, device)
        return model.cpu().state_dict(), len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # sample-count-weighted average: w_{t+1} = sum_k (n_k / sum_j n_j) w^k
        total_samples = sum(max(upd[1], 1) for upd in client_updates)
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():             # integer buffers: copy, don't average
                new_state[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
            for st, n, _ in client_updates:
                acc += st[key].detach().cpu().float() * (max(n, 1) / total_samples)
            new_state[key] = acc.to(ref.dtype)
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available), min(num_to_select, num_available))
```

## Evaluation settings

Three benchmarks, all non-IID, **one seed (42)**: **CIFAR-10** (Dirichlet(0.1), 100 clients, 10-class
images, `CifarCNN`), **FEMNIST** (EMNIST ByClass, Dirichlet split, 62-class characters, `FemnistCNN`),
and **Shakespeare** (per-speaker next-character prediction, `CharLSTM`). Fixed pipeline: 200
communication rounds, 10 of 100 clients per round, 5 local epochs of SGD at `lr = 0.01`, local batch
size 64. Metric on every benchmark: **test accuracy of the global model after 200 rounds** (higher is
better).
