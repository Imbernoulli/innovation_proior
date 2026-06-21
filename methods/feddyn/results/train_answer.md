The federated learning problem is to minimize the average of per-client losses, `ell(theta) = (1/m) sum_k L_k(theta)`, when data cannot leave the clients and only a random subset participates each round. The difficulty is not just statistical heterogeneity but a structural fixed-point mismatch: a global optimum satisfies `(1/m) sum_k grad L_k(theta^*) = 0`, yet each individual gradient `grad L_k(theta^*)` is typically non-zero. A client that minimizes its own loss `L_k` rests at a point `theta_k^*` where `grad L_k(theta_k^*) = 0`, which has nothing to do with `theta^*`. So if every client does local work and the server averages the results, the consensus drifts toward the average of the client optima, not the global optimum. More local steps make the drift worse because they amplify the bias. The standard fixes damp this drift rather than remove it: a decreasing learning rate suppresses step sizes, FedProx adds a static proximal penalty `(mu/2)||theta - theta^{t-1}||^2` that anchors clients to the broadcast model, and server-side momentum reshapes the global step. All of these leave the underlying inconsistency intact; they throttle the bias but cannot align the local and global stationary points, and to prove convergence they typically need a bounded-dissimilarity assumption.

The method I propose is **FedDyn** (Federated Dynamic Regularization, Acar et al., ICLR 2021). It changes the local objective so that consensus becomes a local stationary point, and then uses server-side bookkeeping to make consensus a global stationary point as well. Each active client solves the local subproblem `theta_k^t = argmin_theta [ L_k(theta) - <grad L_k(theta_k^{t-1}), theta> + (alpha/2)||theta - theta^{t-1}||^2 ]`. The linear term `-<grad L_k(theta_k^{t-1}), theta>` is the crucial departure from a static regularizer. If we evaluate the gradient of this objective at the consensus point `theta = theta^{t-1}`, the quadratic term contributes nothing and we are left with `grad L_k(theta^{t-1}) - grad L_k(theta_k^{t-1})`. Once the client iterate has reached consensus, this vanishes, so the broadcast model is actually a stationary point of the client's subproblem. The first-order condition at the local solution gives `grad L_k(theta_k^t) - grad L_k(theta_k^{t-1}) + alpha(theta_k^t - theta^{t-1}) = 0`, which rewrites as `grad L_k(theta_k^t) = grad L_k(theta_k^{t-1}) - alpha(theta_k^t - theta^{t-1})`. This means the stored gradient state refreshes for free from the displacement the client took.

To implement this without extra gradient passes, define `H_k` as the running sum of the client's net displacements away from the broadcast models, `H_k = -grad L_k(theta_k^{t-1})/alpha`. The regularizer gradient is then `alpha(theta - theta^{t-1}) + alpha H_k`, added to the data gradient at every local SGD step. The quadratic part pulls the client toward the broadcast model, and the `alpha H_k` part is an accumulated correction that cancels the client's residual drift. The client only needs the broadcast model and its own persistent state `H_k`; no separate control variate is transmitted.

The server completes the alignment. It maintains a state `h^t` that tracks `(1/m) sum_k grad L_k(theta_k^t)`, updated incrementally as `h^t = h^{t-1} - alpha * (1/m) * sum_{k in P_t}(theta_k^t - theta^{t-1})`. The global model is then `theta^t = (1/|P_t|) sum_{k in P_t} theta_k^t - (1/alpha) h^t`. Because `-(1/alpha)h^t = (1/m) sum_k H_k`, the server can equivalently average the active models and add the mean accumulated displacement over all clients, reconstructing the correction entirely from the model deltas it already receives. At a consensus `theta^infty`, the displacements vanish, `h^t` converges to `(1/m) sum_k grad L_k(theta^infty) = grad ell(theta^infty)`, and the server fixed point requires `h = 0`, which means `grad ell(theta^infty) = 0`. Therefore the only point the algorithm can converge to is a stationary point of the true global loss. FedDyn thus aligns the fixed points rather than damping their gap, and it does so at half the communication cost of control-variate methods like SCAFFOLD because the correction is reconstructed server-side from the model alone.

For deep-network benchmarks, a single coefficient `alpha = 0.1` is a good default on Dirichlet non-IID image and character datasets, with a sweep over `{1e-3, 1e-2, 1e-1}`. Plain SGD is used locally because momentum would break the clean displacement-to-gradient identity. The server-state divisor is the total number of clients `m`, not the active count `|P_t|`, so the reconstruction remains an honest average under partial participation. The implementation below follows the fixed strategy interface used by the harness.

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
