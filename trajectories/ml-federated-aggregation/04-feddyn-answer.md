**Problem.** SCAFFOLD broke CIFAR open (0.6309 vs FedAvg 0.5954) by fixing the *direction* of the
drift with control variates — but it pays double the bits per round (a control variate alongside the
model) and leans on a per-client gradient estimate `c_i` that goes stale at 10/100 participation
(refreshed ~once per 10 rounds). The CIFAR jump happened *despite* that staleness. The goal: the same
direction-alignment, without a per-round control variate over the wire and without a staleness-prone
estimate, so it holds under low participation.

**Key idea.** Align the *fixed point* structurally rather than estimating the global gradient. Make
each active client minimize a **dynamic regularizer**:
`R_k(theta) = L_k(theta) - <grad L_k(theta_k^{t-1}), theta> + (alpha/2)||theta - theta^{t-1}||^2`.
The linear term cancels the client's own gradient at the consensus point, so consensus becomes a
*local* stationary point — the piece FedProx lacked (its `a_k = 0` left the minimizer offset by
`-(1/alpha) grad L_k`, which is why it tied FedAvg). The first-order condition refreshes the gradient
state for free: `grad L_k(theta_k^t) = grad L_k(theta_k^{t-1}) - alpha(theta_k^t - theta^{t-1})`.
Writing `H_k` = running sum of the client's net displacements, the per-step regularizer gradient is
`alpha(theta - theta^{t-1}) + alpha H_k`. The server seals it:
`h^t = h^{t-1} - alpha (1/m) sum_{k in P_t}(theta_k^t - theta^{t-1})`,
`theta^t = (1/|P_t|) sum_{k in P_t} theta_k^t - (1/alpha) h^t = active-average + (1/m) sum_k H_k`.
At a consensus `theta^infty`, displacements vanish, `h -> grad ell(theta^infty)`, and the fixed point
requires `h = 0` — so **the only point the method can rest at is a global stationary point.**

**Why this beats SCAFFOLD here.** The correction `alpha H_k` is structurally the same kind of
additive offset as SCAFFOLD's `(c - c_i)`, but `H_k` is an *exact accumulation* of the client's own
moves (no staleness) and the server reconstructs `(1/m) sum_k H_k` from the model deltas it already
receives — **no control variate crosses the wire (half the bits)**. Removes heterogeneity from the
rate with no bounded-dissimilarity assumption (linear for strongly convex `L_k`, `O(1/T)` non-convex,
with an `m/P` partial-participation factor).

**What this task exposes.** The harness runs every client a fixed 5 epochs, so the framework's
variable-work/straggler surface is unused; the rung is the dynamic regularizer + server
reconstruction. Verified line-by-line against the official impl (alpemreacar/FedDyn):
`loss_algo = alpha*sum(theta*(-theta^{t-1}+H_k))` + weight-decay `alpha` equals the per-step gradient
`alpha(theta - theta^{t-1} + H_k)`; `H_k += (theta_k - theta^{t-1})`; cloud `= avg + mean_over_m(H_k)`.

**Hyperparameters.** `alpha = 0.1` (Dirichlet CIFAR default; sweep `{1e-3, 1e-2, 1e-1}`). Plain SGD
locally (momentum breaks the displacement/gradient-state identity); harness defaults (5 epochs,
`lr = 0.01`, batch 64). Server-state divisor `m = num_clients` (not `|P_t|`). State: per-client `H_k`
(persistent), server running sum `h_sum`. Non-floating-point buffers averaged directly (server state
keyed on named parameters only).

```python
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
