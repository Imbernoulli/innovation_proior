We need to train a single model over data that lives on a large number of clients — phones, sensors, hospitals — that cannot be moved to a server, minimizing the average of per-client losses $f(x) = \frac{1}{N}\sum_{i=1}^N f_i(x)$ where on each client we only ever see a noisy unbiased gradient $g_i(x)$ of its own objective. Three constraints define the difficulty: communication is the bottleneck, so the resource to minimize is the number of *rounds*, which forces each participating client to run $K$ local steps before syncing; only a small subset $S$ of the $N$ clients is reachable per round; and the client data distributions are arbitrarily different, so the local objectives $f_i$ are arbitrarily different functions with their minimizers $x_i^*$ scattered far from each other and from the global $x^*$. The dominant recipe, FedAvg, has each sampled client copy the server model $y_i \leftarrow x$, take $K$ SGD steps $y_i \leftarrow y_i - \eta_l\, g_i(y_i)$, and the server averages the changes, $x \leftarrow x + \frac{\eta_g}{|S|}\sum_{i\in S}(y_i - x)$. For identical clients this is parallel SGD and works beautifully. But with heterogeneous data each $y_i$ walks toward its own $x_i^*$, so the average of the endpoints lands near $\frac{1}{N}\sum_i x_i^*$, which is *not* $x^*$. This is client-drift: a systematic bias whose size is exactly the dissimilarity of the clients, present even with full-batch gradients and full participation, so it is not a stochastic artifact. Worse, the bias grows with $K$ — the very lever that buys communication efficiency makes the drift worse — forcing $\eta_l$ down and slowing convergence. Quantify heterogeneity by gradient dissimilarity, $\frac{1}{N}\sum_i \|\nabla f_i(x)\|^2 \le G^2 + B^2\|\nabla f(x)\|^2$; a one-dimensional construction $f_1 = \mu x^2 + Gx$, $f_2 = -Gx$ shows FedAvg leaves an error of order $G^2/(\mu R^2)$ no matter how the step size is tuned, because the drift inflates the round recursion exactly as the descent shrinks it. So $G$ is genuinely unavoidable for plain averaging. The natural patch — FedProx's proximal penalty $\frac{\mu}{2}\|w-x\|^2$, giving local gradient $g_i(w) + \mu(w-x)$ — only shortens how far $w$ wanders; the direction $g_i(w)$ still points at $x_i^*$, so the aggregate is still pulled the wrong way. Damping the *magnitude* of the local move is the wrong knob; we must fix the *direction*.

The method is SCAFFOLD, stochastic controlled averaging. Ask what direction the client *should* take: if communication were free the ideal local update would be $y_i \leftarrow y_i - \eta_l\,\nabla f(y_i) = y_i - \eta_l\frac{1}{N}\sum_j \nabla f_j(y_i)$, gradient descent on $f$ itself, unbiased toward $x^*$ with zero drift because every client descends the *same* function. The whole problem is that $\frac{1}{N}\sum_j \nabla f_j(y_i)$ needs every client's gradient. So the real task is to approximate the global direction using only client $i$'s own gradient plus cheap state, well enough that the error does not depend on heterogeneity. This is the control-variate situation: to estimate $\mathbb{E}[X]$ with a correlated $Y$ of known mean, use $X - Y + \mathbb{E}[Y]$, unbiased with variance $\mathrm{Var}(X)+\mathrm{Var}(Y)-2\,\mathrm{Cov}(X,Y)$, small when $X,Y$ are correlated — exactly how SAGA corrects the sampled component-gradient $g_j(x)$ to $g_j(x) - g_j(\phi_j) + \frac{1}{n}\sum_i g_i(\phi_i)$ so its mean is the full-sum gradient. Lift this to clients: keep a per-client control variate $c_i$ tracking $g_i(y_i)$ and a server control variate $c$ tracking $\frac{1}{N}\sum_j g_j(y_i)$, and correct the local step to

$$y_i \leftarrow y_i - \eta_l\,\big(g_i(y_i) - c_i + c\big),$$

because if $c_i \approx g_i(y_i)$ and $c \approx \frac{1}{N}\sum_j g_j(y_i)$ then $g_i(y_i) - c_i + c \approx \frac{1}{N}\sum_j g_j(y_i)$, the ideal global direction. The term $(c - c_i)$ is the estimated drift — where the average gradient points minus where mine points — the rotation that bends my local step away from $x_i^*$ and back toward $x^*$. This is what FedProx's penalty could not do: it operates on the direction, not the leash length. Impose the invariant $c = \frac{1}{N}\sum_i c_i$ so that $c$ being a good estimate of the average gradient follows from each $c_i$ being a good estimate of its own; initialize $c = c_i = 0$, which satisfies it, and note that freezing every $c_i \equiv 0$ recovers FedAvg exactly, so SCAFFOLD is a strict generalization. That this actually removes $G$ rather than hiding it is the load-bearing step: comparing the corrected direction to the ideal and summing, $\sum_i \|(\nabla f_i(y) - c_i + c) - \nabla f(y)\|^2 \le \sum_i \|c_i - \nabla f_i(y)\|^2$, because $c$ and $\nabla f(y)$ are common to all clients and cancel, leaving only the per-client mismatch. The residual depends *only* on the tracking error $\|c_i - \nabla f_i(y)\|$ and *not at all* on $G$ — whereas FedAvg's residual is the raw dissimilarity $\|\nabla f_i(y) - \nabla f(y)\|$ bounded by $G$. And the tracking error is controllable: $\beta$-smoothness means $\nabla f_i$ moves slowly with $y$, so a recently computed gradient stays close. That is why the $c_i$ are kept *stateful* across rounds — statefulness is not an accident, it is what makes the cheap approximation valid.

The refresh of $c_i$ has two options. Option I recomputes the client's gradient at the server model, $c_i^+ = g_i(x)$, clean but costing an extra pass over the local data. Option II reuses the gradients already computed during the $K$ local steps, for free, via telescoping. With $y_{i,0}=x$ and each step $y_{i,k} = y_{i,k-1} - \eta_l(g_i(y_{i,k-1}) + c - c_i)$, summing the increments gives $y_{i,K} - x = -\eta_l[\sum_k g_i(y_{i,k-1}) + K(c-c_i)]$, hence $\frac{1}{K\eta_l}(x - y_{i,K}) = \frac{1}{K}\sum_k g_i(y_{i,k-1}) + (c - c_i)$. Defining

$$c_i^+ = c_i - c + \frac{1}{K\eta_l}\,(x - y_{i,K})$$

and substituting collapses it to $c_i^+ = \frac{1}{K}\sum_{k=1}^K g_i(y_{i,k-1})$, the average of the $K$ minibatch gradients already computed — no extra pass. This is the default. It tracks gradients at the visited iterates rather than at $x$, slightly noisier but free, and fine because the $y_{i,k}$ stay near $x$ when drift is controlled. Crucially the local step uses *plain SGD*: a momentum buffer would break the exact telescoping identity that makes the Option-II refresh valid, so no momentum. On the server, the model aggregates as usual, $x \leftarrow x + \frac{\eta_g}{|S|}\sum_{i\in S}(y_i - x)$, but the control needs care under sampling. Only the $S$ participating clients refresh their $c_i$, so the change in the all-client average is $\Delta c = \frac{1}{N}\sum_{i\in S}(c_i^+ - c_i)$, giving

$$c \leftarrow c + \frac{1}{N}\sum_{i\in S}(c_i^+ - c_i) = c + \frac{|S|}{N}\cdot\frac{1}{|S|}\sum_{i\in S}\Delta c_i.$$

The divisor is $N$, not $|S|$: dividing by $|S|$ would over-weight the sampled clients, $c$ would no longer equal $\frac{1}{N}\sum_i c_i$, the invariant the whole correction rests on would break, and the drift estimate $(c-c_i)$ would be wrong. The $|S|/N$ factor keeps $c$ an honest all-client average despite seeing only $S$ of them. The two step sizes $\eta_l, \eta_g$ are decoupled because within-round drift scales with $\eta_l$ while round progress scales with the effective $\tilde\eta = K\eta_l\eta_g$; making $\eta_g$ large and $\eta_l$ small at fixed $\tilde\eta$ suppresses drift without losing speed, though often $\eta_g = 1$ suffices in practice. Tracking client-drift $E_r$, control-lag $C_r$, and $\|x - x^*\|^2$, the lag contracts as $C_r \le (1 - S/N)C_{r-1} + \frac{S}{N}(4\beta(f-f^*) + 2\beta^2 E_r)$, and the Lyapunov function $\Phi_r = \|x^r - x^*\|^2 + \frac{9N\tilde\eta^2}{S}C_r$ contracts each round to $\mathbb{E}[\Phi_r] \le (1 - \mu\tilde\eta/2)\mathbb{E}[\Phi_{r-1}] - \tilde\eta(f - f^*) + \frac{12\tilde\eta^2}{KS}(1 + S/\eta_g^2)\sigma^2$, with no $G$ term, giving a strongly-convex round count $\tilde O\big(\frac{\sigma^2}{\mu KS\epsilon} + \frac{\beta}{\mu} + \frac{N}{S}\big)$. The first term matches SGD with a $K\times$ larger batch, so SCAFFOLD is at least as fast as SGD for *arbitrarily heterogeneous* clients; $N/S$ is the additive cost of partial participation, the lag-healing time. When $\sigma = 0, K = 1, S = 1$ with Option I the update is exactly SAGA, confirming this is the right generalization. Finally, a first-order expansion of the correction shows $\nabla f_i(y) - \nabla f_i(x) + \nabla f(x) \approx \nabla f(y) + (\nabla^2 f_i(x) - \nabla^2 f(x))(y-x)$, so the leftover error is the *Hessian* dissimilarity $\delta$, not $G$: local steps pay off — improving linearly in $K$ when $\delta = 0$ — exactly when the curvatures are close, regardless of how far apart the optima sit.

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
