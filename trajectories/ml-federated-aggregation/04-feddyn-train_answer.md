SCAFFOLD's numbers are the breakthrough I was watching for, and they name the next move precisely. CIFAR-10 broke clean out of the high-0.59 band that pinned both FedAvg (0.5954) and FedProx (0.5948), landing at 0.6309 — a three-and-a-half-point jump on the benchmark most crippled by drift, with the test loss falling from FedAvg's 1.2639 to 1.0659. That confirms the diagnosis: the wall was the *direction* of the drift, and rotating each local step toward the global gradient with the $(c - c_i)$ correction knocked it down. FEMNIST moved modestly as predicted (0.8187 vs 0.8112) and Shakespeare stayed flat (0.4355 vs 0.4360), so the control-variate idea is the right family. But the *cost* tells me where to go. SCAFFOLD pays for the correction in communication: its logic requires each client's control variate to be available, which in the full method means transmitting a control variate alongside the model — twice the bits per round, a steep tax where communication is the entire scarce resource. And at 10-of-100 participation a given client's $c_i$ is refreshed only about once every ten rounds, so the correction leans on a gradient estimate that is, for most clients most of the time, stale by ten rounds of global movement. The CIFAR jump happened *despite* that staleness, which means there is headroom if I can get the same direction-correction without a per-client gradient estimate that ages between rare participations and without a control variate over the wire.

So I go back to the disease itself, the fixed-point inconsistency. A global optimum $\theta^*$ satisfies $\frac{1}{m}\sum_k \nabla L_k(\theta^*) = 0$ with the individual $\nabla L_k(\theta^*)$ non-zero — they only sum to zero. A client minimizing its own $L_k$ rests where $\nabla L_k = 0$, at $\theta_k^*$. The place every client wants to go is not the place the federation must reach, so the consensus of clients that each minimize their own loss is the average of the $\theta_k^*$, not $\theta^*$. SCAFFOLD attacks this by *estimating* the missing global-gradient information and correcting the *direction* at the current iterate. But notice the shape of what I actually need: I need each client's local subproblem to have its rest point *at the consensus*, so that driving the clients to agree automatically drives them to $\theta^*$. I want to make the *fixed point* right rather than the direction at each step — then I do not need a fresh estimate of the global gradient every step, because the alignment is baked into what each client is solving.

I propose **FedDyn**, Federated Dynamic Regularization. Have client $k$ minimize $L_k(\theta) + \langle a_k, \theta\rangle + \frac{\alpha}{2}\|\theta - \theta^{t-1}\|^2$ for a per-client linear term $a_k$ I choose. Its gradient is $\nabla L_k(\theta) + a_k + \alpha(\theta - \theta^{t-1})$; evaluate it at the consensus point $\theta = \theta^{t-1}$, where the quadratic vanishes, leaving $\nabla L_k(\theta^{t-1}) + a_k$. For consensus to be a stationary point of *every* client's subproblem I need $a_k = -\nabla L_k(\theta^{t-1})$. That is the whole idea in one line: subtract off, with a linear term, the client's own gradient at the consensus, so that *at* the consensus the client feels no pull away from it. The active client therefore minimizes the **dynamic regularizer**

$$R_k(\theta) = L_k(\theta) - \langle \nabla L_k(\theta_k^{t-1}), \theta\rangle + \frac{\alpha}{2}\|\theta - \theta^{t-1}\|^2.$$

This is the missing piece relative to FedProx, which had the quadratic spring but no linear term ($a_k = 0$): its minimizer was offset by exactly $-\frac{1}{\alpha}\nabla L_k$, the residual gradient — precisely why FedProx tied FedAvg, the spring shortened the leash but left the rest point in the wrong place. The linear term cancels the residual gradient that *causes* the drift, and because it depends on the client's own latest gradient it is "dynamic," changing round to round. The spring damps; the dynamic linear term *aligns*.

The elegant part is what makes this cheaper than SCAFFOLD. The first-order condition of the dynamic subproblem is $\nabla L_k(\theta_k^t) - \nabla L_k(\theta_k^{t-1}) + \alpha(\theta_k^t - \theta^{t-1}) = 0$, which I read as a recursion for the gradient state, $\nabla L_k(\theta_k^t) = \nabla L_k(\theta_k^{t-1}) - \alpha(\theta_k^t - \theta^{t-1})$. So I never recompute the gradient to maintain the linear term — solving the subproblem *gives* me the updated gradient as a function of the displacement the client just took. Store the state as $H_k$, the running sum of the client's net displacements away from the broadcast models, so the linear term is $+\alpha H_k$ and the per-step regularizer gradient is

$$\alpha(\theta - \theta^{t-1}) + \alpha H_k,$$

an L2 pull to the broadcast model plus a fixed accumulated-correction offset, added to the data gradient at every plain-SGD step. The client needs only the broadcast model and its own $H_k$; nothing extra crosses the wire. Compared to SCAFFOLD's $(c - c_i)$ — structurally the same kind of additive offset — where $c_i$ is a gradient *estimate* that ages between participations, $H_k$ is an exact accumulation of this client's own moves: it does not go stale, it just records history.

The server seals the alignment, and the sealing removes the communication cost. Maintain a server state $h$ that mirrors the client states. Using the gradient recursion and updating only the active clients $P_t$,

$$h^t = h^{t-1} - \alpha\,\frac{1}{m}\sum_{k \in P_t}(\theta_k^t - \theta^{t-1}), \qquad \theta^t = \frac{1}{|P_t|}\sum_{k \in P_t}\theta_k^t - \frac{1}{\alpha}h^t.$$

The divisor in the server update is $m$, the total number of clients, not $|P_t|$, because $h$ is the average over *all* clients' gradient states and only the active ones changed — the same $|S|/N$ honesty SCAFFOLD needed, here applied to the server state. Check the fixed point, because that is the whole justification. At a consensus $\theta^\infty$ the displacements $\theta_k^t - \theta^{t-1}$ go to zero, so $h$ stops moving and $\theta^t \to \theta^\infty$; the stored gradients converge to $\nabla L_k(\theta^\infty)$; $h$ unrolls to $\frac{1}{m}\sum_k \nabla L_k(\theta^\infty) = \nabla \ell(\theta^\infty)$; and the server fixed point requires the correction $-\frac{1}{\alpha}h$ to be consistent, which holds only when $h = 0$, i.e. $\nabla\ell(\theta^\infty) = 0$. So the *only* consensus the method can rest at is a global stationary point. The dynamic linear term makes consensus a local stationary point; the server correction makes it a global one; the two fixed points are forced to coincide. Crucially the server reconstructs $-\frac{1}{\alpha}h^t = \frac{1}{m}\sum_k H_k$ from the model deltas it already receives — the correction lives in client-persistent state and server-reconstructed state, *not* in a vector transmitted each round. That is the same direction-alignment SCAFFOLD bought, at *half the bits*, and without a staleness-prone gradient estimate; it removes heterogeneity from the rate with no bounded-dissimilarity assumption (linear for strongly convex $L_k$, $O(1/T)$ non-convex, with an $m/P$ partial-participation factor).

In this task's edit surface the harness instantiates one `Strategy` across all rounds, so I keep server state `h_sum` (the running sum, which is $-\frac{1}{\alpha}h$, the mean over all clients of $H_k$) and a dictionary of per-client $H_k$, persistent on CPU between participations. In `client_local_train` I freeze the broadcast parameters $\theta^{t-1}$, load this client's $H_k$ (zero on first sight), run plain SGD adding $\alpha(\theta - \theta^{t-1} + H_k)$ to each gradient, then accumulate $H_k \mathrel{+}= (\theta_k^t - \theta^{t-1})$ and stash that round's displacement for the server. In `aggregate` I take the plain average of the active clients' models, add $\frac{1}{m}\sum_{\text{active}}$ displacement to `h_sum`, and set the new global model to the active-average plus `h_sum`. The single coefficient $\alpha$ does everything — quadratic-anchor strength and gradient-to-displacement scale at once; larger $\alpha$ convexifies the subproblem and pulls harder to the broadcast model, smaller $\alpha$ lets clients move more but weakens the correction — and for a Dirichlet non-IID image/character setting the canonical choice is $\alpha = 0.1$ (sweep $\{10^{-3}, 10^{-2}, 10^{-1}\}$). The local step must be plain SGD: momentum would break the displacement/gradient-state identity the whole refresh rests on. The harness runs every client a fixed 5 epochs, so as with the earlier rungs the framework's variable-work/straggler surface is unused; this rung is the dynamic regularizer plus the server reconstruction. Because the alignment is structural rather than an aging estimate, CIFAR should hold at or above SCAFFOLD's 0.6309 and its loss at or below 1.0659, at half the per-round communication — the construction whose fixed point *is* the global optimum, so that converging the clients to agreement converges them to the right place by design rather than by maintaining and communicating an estimate of where it is.

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
