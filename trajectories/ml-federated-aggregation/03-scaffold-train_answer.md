FedAvg tied FedProx, and the tie is the whole diagnosis. Plain weighted averaging came in at 0.5954 on CIFAR-10 against FedProx's 0.5948 — six ten-thousandths, statistical noise — at 0.4360 on Shakespeare against 0.4326, and at 0.8112 on FEMNIST a fraction *below* FedProx's 0.8126, the one place the gentle spring earned its keep. The two are indistinguishable, confirming that the $\mu = 0.01$ spring throttled the *magnitude* of the drift without touching its *direction*, and that on top of the shared per-round initialization already doing the basin-keeping there was nothing left for a gentle tether to contribute. The deeper signal is the CIFAR number itself: 0.595 on a 10-class problem the same `CifarCNN` clears comfortably when trained centrally. That gap is the signature of residual client drift, and *neither* a magnitude-only spring nor plain averaging touches it — both leave each client's local gradient $g_i$ pointing at its own optimum $\theta_i^*$, so the aggregate of the 10 participating models stays biased toward $\frac{1}{10}\sum_i \theta_i^*$ rather than the global $\theta^*$. The wall is the drift's *direction*, and the fix has to rotate the local step, not shorten it.

Make the disease precise so the cure falls out of the mechanism. The ideal local update on client $i$, if communication were free, would descend the *global* gradient, $y_i \leftarrow y_i - \eta_l\,\frac{1}{N}\sum_j \nabla f_j(y_i)$ — unbiased toward $\theta^*$ by construction, zero client drift no matter how many local steps I take, because every client would be descending the same function. The whole problem is that $\frac{1}{N}\sum_j \nabla f_j(y_i)$ needs every other client's gradient at $y_i$, the exact communication I cannot afford every step. So the task is sharp: approximate that global direction using only client $i$'s own gradient $g_i(y_i)$ plus some cheap state, well enough that the approximation error does *not* depend on how heterogeneous the clients are.

I propose **SCAFFOLD**: a **control-variate** correction of the local step. The classical trick to estimate $\mathbb{E}[X]$ when I have a correlated $Y$ with known mean $\mathbb{E}[Y]$ is $X - Y + \mathbb{E}[Y]$ — unbiased, and low-variance when $X$ and $Y$ are tightly correlated. The finite-sum optimizers SAGA and SVRG use exactly this to kill the variance of a sampled gradient, correcting $g_j(x)$ to $g_j(x) - g_j(\phi_j) + \frac{1}{n}\sum_i g_i(\phi_i)$. The analogy writes itself: my components are the clients, the sampled component is client $i$, and I want the corrected local gradient on client $i$ to have mean $\frac{1}{N}\sum_j \nabla f_j$ instead of $\nabla f_i$. So I posit two pieces of state — a per-client control variate $c_i \approx g_i(y_i)$ tracking that client's own gradient, and a server control variate $c \approx \frac{1}{N}\sum_j g_j$ tracking the average — and correct each local step to

$$y_i \leftarrow y_i - \eta_l\,\big(g_i(y_i) - c_i + c\big).$$

Its mean is right: if $c_i \approx g_i(y_i)$ and $c \approx \frac{1}{N}\sum_j g_j$, then $g_i - c_i + c \approx \frac{1}{N}\sum_j g_j$, the ideal global direction. The correction $(c - c_i)$ is an estimate of the drift — $c_i$ is where *my* gradient points, $c$ where the *average* points — and it is the rotation that bends my step away from $\theta_i^*$ and back toward $\theta^*$. That is the structural fix the spring could not do: it operates on the direction, not the leash length. I impose the invariant $c = \frac{1}{N}\sum_i c_i$ so that $c$ being a good average estimate follows from each $c_i$ tracking its own gradient, initialize everything to zero (which satisfies the invariant), and note that freezing every $c_i = 0$ recovers plain FedAvg — a strict generalization, the extra state used only to change the local direction.

Does it remove the heterogeneity dependence rather than hide it? Measure the residual of the corrected direction against the ideal, $\sum_i \|(\nabla f_i(y) - c_i + c) - \nabla f(y)\|^2$. Substituting $c = \frac{1}{N}\sum_j c_j$, the common terms cancel and what is left is $\sum_i \|c_i - \nabla f_i(y)\|^2$ — the residual depends *only* on how well $c_i$ tracks $\nabla f_i$, and *not at all* on how dissimilar the clients are. That is the escape: where plain averaging's residual is the raw gradient dissimilarity that pins CIFAR at 0.595, this residual is a tracking error I control. And I can control it because $f_i$ is smooth, so $\nabla f_i$ does not change fast as $y$ moves; a recently-computed gradient stays close. So I keep the $c_i$ *stateful across rounds* — a client holds onto its $c_i$ and refreshes it when it participates — and smoothness makes a slightly stale gradient a good approximation. Statefulness is not an accident; it is what makes the cheap approximation valid.

What makes it cheap is the refresh. I do not want an extra gradient pass to update $c_i$, so I telescope the local trajectory. Over the round $y_{i,0} = x$ and each step is $y_{i,k} = y_{i,k-1} - \eta_l(g_i(y_{i,k-1}) + c - c_i)$. Summing the increments over the $K$ steps gives $y_{i,K} - x = -\eta_l[\sum_k g_i(y_{i,k-1}) + K(c - c_i)]$, so $\frac{1}{K\eta_l}(x - y_{i,K}) = \frac{1}{K}\sum_k g_i(y_{i,k-1}) + (c - c_i)$, where the left side is built only from $x$ and $y_{i,K}$, which the server already has. Defining the refresh

$$c_i^+ = c_i - c + \frac{1}{K\eta_l}(x - y_{i,K})$$

yields, after substitution, $c_i^+ = \frac{1}{K}\sum_k g_i(y_{i,k-1})$ — the average of the minibatch gradients *already computed* during local training, for free, no extra pass. This is the Option-II refresh. Crucially the local step must be *plain* SGD: the whole refresh rests on the telescoping identity between net displacement and the sum of corrected gradients, and a momentum buffer would break that clean relationship and cost me the free control update.

The server side is where I land *this task's* version rather than the general theory, and they differ in a way that matters. The general account aggregates the model with its own global step size, $x \leftarrow x + \frac{\eta_g}{|S|}\sum_i (y_i - x)$, decoupling a local step $\eta_l$ from a global step $\eta_g$ so within-round drift (scaling with $\eta_l$) can be suppressed independently of round-level progress. That decoupling is real, but the harness exposes no $\eta_g$: this task combines the *model* with the plain FedAvg sample-weighted average from step 2 and puts the entire control-variate contribution into the local correction and the server *control* update — effectively $\eta_g = 1$ folded into the weighted average. So on this rung the model combine is unchanged, and what is new is purely (a) the corrected local gradient and (b) the maintenance of $c$. The control update needs care because of sampling: only the 10 participating clients refreshed their $c_i$, the other 90 kept theirs, so to maintain $c = \frac{1}{N}\sum_i c_i$ over *all* clients the change in the global average is $\Delta c = \frac{1}{N}\sum_{i \in S}(c_i^+ - c_i)$ — each participating client sends its delta $\Delta c_i$, the server sums the 10 deltas and divides by $N = 100$, not $|S| = 10$. Equivalently $c \leftarrow c + \frac{|S|}{N}\,\text{mean}_{i \in S}\,\Delta c_i$. Dividing by $|S|$ instead would over-weight the few clients I happened to sample, break the invariant, and make the drift estimate $(c - c_i)$ wrong; with only 10 of 100 participating, getting that divisor right is what keeps $c$ an honest all-client average round over round.

In the edit surface I keep server-side state $c$ (the global control, one tensor per parameter, zero-initialized) and a dictionary of client controls $c_i$ keyed by client index, persistent across rounds and resident on CPU between participations. In `client_local_train` I load the broadcast model, move $c$ and this client's $c_i$ to the device, snapshot the broadcast parameters $x$ for the telescoping refresh, precompute the per-parameter correction $(c - c_i)$ once (fixed across the $K$ steps), then run plain SGD adding that correction to each gradient before stepping. After training I compute the Option-II refresh, stash the delta $\Delta c_i$ for the server, and update the stored $c_i$. In `aggregate` I do the FedAvg sample-weighted model average and then apply $c \leftarrow c + \frac{|S|}{N}\sum_i \Delta c_i$. Two harness details carry over: non-floating-point buffers are copied through, not averaged, and the control tensors track the parameter set with non-FP keys carried unchanged. The benchmark that should move most is the one most crippled by drift — CIFAR under Dirichlet(0.1) — which should break clear of the high-0.59 band, because that gap was pure direction-of-drift and direction is exactly what I am now fixing. The one caveat to watch is staleness: at 10/100 participation a given client's $c_i$ ages about ten rounds between refreshes, so if CIFAR does not clear 0.595 the next move is to attack control staleness directly.

```python
class Strategy:
    """SCAFFOLD — Alg 1 with Option-II control-variate update."""

    def __init__(self, global_model, args):
        self.args = args
        self.num_clients = args.num_clients
        self.global_control = OrderedDict(
            (k, torch.zeros_like(v, device="cpu"))
            for k, v in global_model.state_dict().items()
        )
        self.client_controls = {}   # client_idx -> OrderedDict (CPU)

    def _zero_like_state(self, state_dict):
        return OrderedDict(
            (k, torch.zeros_like(v, device="cpu")) for k, v in state_dict.items()
        )

    def _get_client_control(self, client_idx, reference_state):
        c_i = self.client_controls.get(client_idx)
        if c_i is None:
            c_i = self._zero_like_state(reference_state)
            self.client_controls[client_idx] = c_i
        return c_i

    def _ensure_global_control_on(self, device, reference_state):
        # Lazily move global_control to the model's device once and keep it there.
        if (not hasattr(self, "_gc_dev")) or self._gc_dev_id != id(device):
            self.global_control = OrderedDict(
                (k, v.to(device) if v.is_floating_point() else v)
                for k, v in self.global_control.items()
            )
            self._gc_dev = device
            self._gc_dev_id = id(device)

    def _get_client_control_on(self, client_idx, reference_state, device):
        c_i = self.client_controls.get(client_idx)
        if c_i is None:
            c_i = OrderedDict(
                (k, torch.zeros_like(v, device=device) if v.is_floating_point()
                 else torch.zeros_like(v, device="cpu"))
                for k, v in reference_state.items()
            )
            self.client_controls[client_idx] = c_i
        elif any(v.device != device for v in c_i.values() if v.is_floating_point()):
            c_i = OrderedDict(
                (k, v.to(device) if v.is_floating_point() else v)
                for k, v in c_i.items()
            )
            self.client_controls[client_idx] = c_i
        return c_i

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)
        model.to(device)
        model.train()

        # Move global_control + c_i to device once; keep them resident.
        self._ensure_global_control_on(device, model.state_dict())
        c_i = self._get_client_control_on(client_idx, model.state_dict(), device)

        # Snapshot global params x ON DEVICE for Option-II later.
        x_dev = OrderedDict(
            (n, p.detach().clone())
            for n, p in model.named_parameters()
            if n in self.global_control
        )

        # Pre-compute (c - c_i) on device once per client.
        correction_dev = {}
        for name, p in model.named_parameters():
            if name in self.global_control:
                correction_dev[id(p)] = self.global_control[name] - c_i[name]

        optimizer = optim.SGD(model.parameters(), lr=local_lr)  # plain SGD
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)

        total_loss, total_samples, local_steps = 0.0, 0, 0
        for _ in range(local_epochs):
            for batch_data in loader:
                if len(batch_data) != 2:
                    continue
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.dim() == 3:
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
                loss.backward()
                # Corrected gradient: g + (c - c_i). Pure in-place add on device.
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    corr = correction_dev.get(id(p))
                    if corr is not None:
                        p.grad.add_(corr)
                optimizer.step()
                local_steps += 1
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        # Option-II update — stay on device.
        if local_steps > 0 and local_lr > 0.0:
            denom = local_steps * local_lr
            new_ci = OrderedDict()
            delta_c = OrderedDict()
            for name, p in model.named_parameters():
                if name not in self.global_control:
                    continue
                # c_i+ = c_i - c + (x - y) / (K * eta)
                update = c_i[name] - self.global_control[name] + (x_dev[name] - p.detach()) / denom
                delta_c[name] = (update - c_i[name]).clone()
                new_ci[name] = update
            # Carry over non-FP buffer keys unchanged from existing c_i.
            for k, v in c_i.items():
                if k not in new_ci:
                    new_ci[k] = v
                    delta_c[k] = torch.zeros_like(v)
            self._pending_delta_c = getattr(self, "_pending_delta_c", {})
            self._pending_delta_c[client_idx] = delta_c
            self.client_controls[client_idx] = new_ci

        # Single GPU→CPU transfer for the returned state_dict (server aggregates on CPU).
        final_state = OrderedDict(
            (k, v.detach().cpu()) for k, v in model.state_dict().items()
        )
        avg_loss = total_loss / max(total_samples, 1)
        return final_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # FedAvg-style weighted model average.
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

        # Server-side global c update: c <- c + (|S|/N) * mean_i Δc_i.
        # δc_i are on the same device as global_control — stay there.
        deltas = getattr(self, "_pending_delta_c", {})
        if deltas:
            weight = len(client_updates) / max(self.num_clients, 1)
            n_updates = len(deltas)
            for key in self.global_control:
                if not self.global_control[key].is_floating_point():
                    continue
                acc = None
                for dc in deltas.values():
                    if key in dc and dc[key].is_floating_point():
                        contrib = dc[key].to(self.global_control[key].device)
                        acc = contrib.clone() if acc is None else acc + contrib
                if acc is not None:
                    self.global_control[key] = (
                        self.global_control[key] + (weight / n_updates) * acc
                    )
            self._pending_delta_c = {}
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        return random.sample(range(num_available), min(num_to_select, num_available))
```
