## Research question

Train one shared model across 100 clients whose data never leaves them and is **non-IID** — a Dirichlet(0.1) split of CIFAR-10 and FEMNIST, and a naturally per-speaker split of Shakespeare. Each round the server reaches only 10 of the 100 clients; each runs 5 local epochs of plain SGD, then ships its model back. The object of study is the **federated learning recipe**: how each client shapes its local objective and how the server combines the returned models. The binding constraint is that clients' local optima sit far from the global optimum, so naive averaging suffers **client drift**. Everything else (data partition, local-loop scaffolding, client sampling, the 200-round loop, evaluation) is fixed.

## Prior art / Background / Baselines

- **FedSGD / naive distributed SGD.** Each selected client computes one gradient on its local data, the server averages the gradients, and the global model takes one SGD step. **Gap:** one communication round buys only one global gradient step; convergence requires tens of thousands of rounds, which is impractical when communication is scarce.
- **Local-update averaging (FedAvg).** Each selected client runs many local SGD steps before sending back its updated model, and the server averages the returned models sample-count-weighted. **Gap:** under non-IID data, many local steps push each client toward its own optimum, so the averaged model is pulled toward the average of client optima rather than the global optimum — client drift grows with the number of local steps.
- **Stationary-point mismatch.** A global optimum satisfies `(1/m) sum_k grad L_k(theta^*) = 0` while individual client gradients are generally non-zero; a client minimizing its own loss rests where `grad L_k = 0`. **Gap:** these two stationary conditions are incompatible under heterogeneity, so local training alone does not make the averaged update vanish at a global optimum.

## Fixed substrate / Code framework

A federated-simulation harness is frozen: it owns the Dirichlet/per-speaker data partition across 100 clients, the per-client local-SGD loop scaffolding, the client-sampling schedule, the 200-round loop, and global-model evaluation on the held-out test set. It instantiates **one** `Strategy` object before the round loop and per round calls, in order:

1. `select_clients(N, K, round)` — pick the 10 participants.
2. `client_local_train(global_state, client_dataset, model_fn, loss_fn, local_epochs, local_lr, local_batch_size, device, client_idx)` — for each selected client, returning `(state_dict, num_samples, avg_loss)`.
3. `aggregate(global_state, client_updates, round)` — combine the list of returned tuples into the next global `state_dict`.

Models are `CifarCNN` / `FemnistCNN` / `CharLSTM` chosen by dataset. The harness also provides `_default_client_sgd`, a plain-SGD helper with an optional per-minibatch `loss_aug` term.

## Editable interface

The only editable region is the `Strategy` class in `flower/custom_fl_aggregation.py`. It owns the full recipe: client-side local objective, server-side combine, and any persistent state. The required interface is `__init__`, `client_local_train`, `aggregate`, and `select_clients`.

The starting point is plain local SGD on each client plus a sample-count-weighted average of the returned models (FedAvg). Each method replaces exactly these definitions and nothing else. One load-bearing detail: a model `state_dict` may contain non-floating-point buffers (integer counters, batch-norm tracked-batch counts) that must not be averaged; the default copies one client's value through for those and weight-averages only the float tensors, on CPU in float32.

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

Three non-IID benchmarks, **one seed (42)**:

- **CIFAR-10**: Dirichlet(0.1), 100 clients, 10-class images, `CifarCNN`.
- **FEMNIST**: EMNIST ByClass Dirichlet split, 62-class characters, `FemnistCNN`.
- **Shakespeare**: per-speaker next-character prediction, `CharLSTM`.

Fixed pipeline: 200 communication rounds, 10 of 100 clients per round, 5 local epochs of SGD at `lr = 0.01`, local batch size 64. Metric on every benchmark: **test accuracy of the global model after 200 rounds** (higher is better).
