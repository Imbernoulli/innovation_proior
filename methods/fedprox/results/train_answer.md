The problem is to minimize a finite-sum objective f(w) = sum_k p_k F_k(w), with p_k = n_k / n, over a network of devices that are not allowed to move their raw data off-device. Two features make this hard. First, the local data distributions D_k are non-identically distributed, so the local risks F_k differ from one another and from the global objective; the minimizer of any single F_k can be far from the minimizer of f. Second, the devices themselves are heterogeneous in compute, memory, battery, and connectivity, so they cannot all be expected to perform the same amount of work in a fixed round. Communication is the bottleneck, which pushes the design toward doing many local steps per round and involving only a small fraction of devices each round.

The standard recipe is FedAvg: broadcast the current model w^t, let each selected device run several epochs of local SGD on its own F_k, and average the returned models weighted by sample count. That averaging only makes sense when the local solutions stay in a common basin, which is true when everyone starts from the same w^t and the local data are similar. Under non-IID data, however, more local epochs drive each device toward its own local optimum, so the very knob that reduces communication rounds amplifies client drift and can make the averaged model diverge. FedAvg also handles systems heterogeneity crudely: a device that cannot finish its quota is dropped, wasting its partial work and biasing the effective sampling distribution if stragglers have distinctive data. What is needed is a way to cap drift directly and to let devices contribute partial solutions instead of being excluded.

The method is FedProx. It keeps the FedAvg server-side aggregation unchanged but changes the local objective on each device. Instead of minimizing F_k(w), device k approximately minimizes

h_k(w; w^t) = F_k(w) + (mu/2) ||w - w^t||^2.

The added term is a quadratic spring anchored at the broadcast model w^t. Its gradient is mu(w - w^t), so a local SGD step on h_k becomes w <- w - eta (grad F_k(w) + mu (w - w^t)) — ordinary local SGD plus a restoring force that pulls the parameters back toward w^t. Because the spring acts on the objective itself, it tethers drift regardless of which local solver is used or how many steps the device takes; setting mu = 0 recovers FedAvg exactly. The same quadratic also convexifies the local subproblem: if F_k has negative curvature bounded below by -L_ I, then the Hessian of h_k is bounded below by (mu - L_) I, so choosing mu > L_ makes h_k strongly convex even when F_k is non-convex. That strong convexity gives a clean displacement bound, ||argmin h_k - w^t|| <= (1/(mu - L_)) ||grad F_k(w^t)||, which is the workhorse of the convergence analysis.

FedProx also handles variable device effort without dropping stragglers. A returned model w_k is called gamma-inexact for h_k if ||grad h_k(w_k; w^t)|| <= gamma ||grad F_k(w^t)||, with gamma in [0,1]. Gamma equals zero for an exact local minimizer and is close to one for a barely-started solve. A slow device simply returns a solution with larger gamma; its partial work is aggregated into the server average rather than discarded. The number of local epochs is just a proxy for gamma, so the formalism covers per-device, per-round variation in compute. This is a lightweight client-side change: the server still computes a sample-weighted mean of the returned models, and no full global gradient is needed, unlike DANE-style gradient-correction schemes that fail under low participation.

The convergence story is as follows. Under L-smooth F_k, a negative-curvature bound L_, mu > L_, and a bounded dissimilarity E_k ||grad F_k(w)||^2 <= B^2 ||grad f(w)||^2 measuring statistical heterogeneity, one round of FedProx decreases the global objective in expectation by rho ||grad f(w^t)||^2 for an explicit rho that depends on mu, L, L_, B, gamma, and the number of sampled devices K per round. Two qualitative constraints appear: gamma B < 1, meaning very sloppy local solves are only safe when the network is relatively homogeneous, and B / sqrt(K) < 1, meaning higher heterogeneity requires more devices per round. Telescoping the per-round decrease gives convergence to an epsilon-approximate stationary point in O(Delta / (rho epsilon)) rounds, where Delta = f(w^0) - f^*. In the convex case with exact solves, choosing mu proportional to L B^2 recovers the SGD complexity up to constants, showing that FedProx matches distributed SGD asymptotically while being much more robust to heterogeneous clients.

```python
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader


class Strategy:
    """FedProx: local SGD with a proximal term anchored at the broadcast model."""

    def __init__(self, global_model, args):
        self.args = args
        self.mu = getattr(args, "mu", 0.01)  # typical range 0.001--0.1

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()

        # Freeze a copy of the broadcast parameters for the prox term.
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

        loader = DataLoader(
            client_dataset,
            batch_size=local_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=local_lr)
        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y) + prox_loss(model)
                loss.backward()
                optimizer.step()

        return model.cpu().state_dict(), len(client_dataset), loss.item()

    def aggregate(self, global_state_dict, client_updates, round_num):
        total_samples = sum(max(upd[1], 1) for upd in client_updates)
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():
                new_state[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
            for state_dict, n, _ in client_updates:
                acc += state_dict[key].detach().cpu().float() * (
                    max(n, 1) / total_samples
                )
            new_state[key] = acc.to(ref.dtype)
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(
            range(num_available), min(num_to_select, num_available)
        )
```
