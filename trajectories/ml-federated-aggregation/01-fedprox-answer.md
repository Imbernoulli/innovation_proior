**Problem.** Under the Dirichlet(0.1) non-IID split, 5 local epochs per round drive each client's
model toward its *own* optimum `theta_k^*`, so the sample-weighted average of the 10 returned models
is pulled toward `(1/10) sum theta_k^*` rather than the global `theta^*` — client drift, amplified
by exactly the local-step count that makes the recipe communication-efficient. The floor (plain
local SGD + weighted average) has one knob, the epoch count, and it fights itself.

**Key idea.** Throttle the drift with the lightest touch: add a quadratic **proximal spring**
anchored at the broadcast model to each client's local objective,
`h_k(w; w^t) = F_k(w) + (mu/2) ||w - w^t||^2`. Its gradient `mu(w - w^t)` is a restoring force added
to every local SGD step, capping how far a client strays from `w^t` regardless of how many steps it
takes; `mu = 0` recovers the floor exactly. The same quadratic convexifies the local subproblem
(`grad^2 h_k = grad^2 F_k + mu I`), the principled reason it stabilizes convergence under
heterogeneity.

**Why (and what this task exposes).** The spring damps the *magnitude* of the drift, not its
*direction*: `grad F_k(w) + mu(w - w^t)` still descends toward `theta_k^*`, so the aggregate stays
biased toward the average of client optima — the bias is throttled, not removed. This task's harness
runs every selected client for the same fixed 5 epochs, so the framework's variable-work /
straggler-inexactness half has no surface here; the rung is the spring alone. No gradient-correction
term (that would need the full global gradient, unaffordable at 10/100 participation). The server
combine is unchanged from the floor (sample-weighted mean), so this is a client-side-only change.

**Hyperparameters.** `mu = 0.01` (small, conservative tether; suggested range 0.001-0.1). Server
aggregation: sample-count-weighted average. Local: plain SGD, the harness defaults (5 epochs,
`lr = 0.01`, batch 64). Realized through the harness's `loss_aug` hook on its plain-SGD helper, with
a frozen copy of `w^t` for the prox term — no new optimizer, no server state, no extra communication.

```python
class Strategy:
    """FedProx — plain SGD + proximal term in the local objective."""

    def __init__(self, global_model, args):
        self.args = args
        self.mu = 0.01  # Li et al. 2020 suggested range 0.001-0.1 on CIFAR/FEMNIST.

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()
        # Freeze copies of the global parameters for the prox term.
        global_params = [
            p.detach().clone() for p in model.parameters() if p.requires_grad
        ]
        mu_half = 0.5 * self.mu

        def prox_loss(m):
            prox = 0.0
            for w, w0 in zip(
                [p for p in m.parameters() if p.requires_grad],
                global_params,
            ):
                prox = prox + (w - w0).pow(2).sum()
            return mu_half * prox

        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss, _ = _default_client_sgd(
            model, loader, loss_fn, local_epochs, local_lr, device,
            loss_aug=prox_loss,
        )
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
