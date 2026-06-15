**Problem.** Establish the clean, unembellished reference the whole task is built around: plain
local SGD per client + sample-count-weighted average of the returned models. Communication rounds
are the scarce resource, so each client spends free on-device compute (5 local epochs) before the
average — but under the non-IID split each client drifts toward its own optimum `theta_k^*`, so the
average is biased toward `(1/10) sum theta_k^*`, not `theta^*`.

**Key idea.** One synchronous distributed SGD step `w_{t+1} = w_t - eta sum_k (n_k/n) g_k` is
*algebraically identical* to each client taking one local step `w^k = w_t - eta g_k` and the server
averaging `w_{t+1} = sum_k (n_k/n) w^k` (weights sum to one). Once a client steps-and-ships a model,
it can take *many* local steps before the average — trading free compute for scarce rounds. Two
choices carry it:

- **Sample-count weighting `n_k`.** `f` is itself the `n_k`-weighted sum of the `F_k`, so `n_k/n`
  weights make the average of one-step models equal a gradient step on `f`; equal weighting would
  bias toward tiny clients. With 10/100 participation, normalize over the *selected* set:
  `w_{t+1} = sum_{k in S_t} (n_k/m_t) w^k`, `m_t = sum_{k in S_t} n_k`.
- **Shared per-round initialization.** Averaging independently-trained non-convex models can be far
  worse than either (the straight line humps over a ridge). Re-broadcasting one `w_t` each round
  keeps the averaged models in a shared basin, so their average sits low. This is also why the step-1
  spring was nearly a no-op — the shared init was already doing the basin-keeping.

**Why (and what this task exposes).** This is FedProx with the spring removed (`mu = 0`). The general
recipe's tunable corners — client fraction `C`, local epochs `E`, batch `B`, the one-shot-averaging
extreme, decaying local computation — are all fixed by the harness (`C` = 10/100, `E` = 5, `B` = 64),
so the rung is just plain SGD + the weighted average. It damps neither the magnitude nor the
*direction* of the drift; the aggregate stays biased toward the average of client optima — the gap
step 3 must close.

**Hyperparameters.** No method hyperparameters (no spring, no state). Local: plain SGD, harness
defaults (5 epochs, `lr = 0.01`, batch 64), uniform random client sampling. Aggregation:
sample-count-weighted mean on CPU in float32, non-floating-point buffers copied through (not
averaged), cast back to the original dtype.

```python
class Strategy:
    """FedAvg — plain SGD + weighted average of client state dicts."""

    def __init__(self, global_model, args):
        self.args = args

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss, _ = _default_client_sgd(
            model, loader, loss_fn, local_epochs, local_lr, device)
        return model.cpu().state_dict(), len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        total_samples = sum(max(upd[1], 1) for upd in client_updates)
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():
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
