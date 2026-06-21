The gentle spring bought almost nothing, and the numbers say so precisely. FedProx with $\mu = 0.01$ landed at 0.5948 on CIFAR-10, 0.8126 on FEMNIST, and 0.4326 on Shakespeare — exactly what "throttle the magnitude, not the direction" predicts. The proximal step is $\nabla F_k(w) + \mu(w - w^t)$: the data-gradient term still points at the client's own optimum $\theta_k^*$, and a gentle $\mu$ only shortens the leash without rotating that direction. On the harsh Dirichlet(0.1) image splits a weak spring can only nibble; on Shakespeare's milder per-speaker heterogeneity it is essentially invisible, even fractionally below where plain averaging should sit because the tether costs a sliver of local progress without removing the bias it was meant to fight. The lesson is sharp — damping the drift's *magnitude* is the wrong knob — and before I reach for a method that fixes the drift's *direction*, I want to strip the spring back out and establish the clean, unembellished reference it was a modification of, so I know exactly what plain weighted averaging scores on this harness and can confirm the spring was a near-no-op. That clean reference is the recipe the whole task is built around, and every later rung is a fill of *its* combine, so deriving it carefully is worth doing in its own right.

I propose **FedAvg** — plain local SGD on each client plus a sample-count-weighted average of the returned models — and I want to build it from the constraint that actually matters rather than just removing the $\mu$ term. The data lives on 100 clients and cannot move; the link is rare and slow, so the only currency worth minimizing is *communication rounds*; on-device compute is comparatively free. The objective is $f(w) = \sum_k (n_k/n)\,F_k(w)$, and I want a good model in as few of the 200 rounds as possible while spending the free local compute to get there. Start with the most obvious correct thing, naive synchronous distributed SGD (FedSGD): each round, each selected client computes the gradient of its loss on its local data, the server sums them weighted by $n_k/n$ and takes one step. Because $f = \sum_k (n_k/n) F_k$, its gradient is $\sum_k (n_k/n)\,\nabla F_k$, so $w_{t+1} = w_t - \eta\sum_k (n_k/n)\,g_k$ is literally a stochastic gradient step on $f$ — correct, defensible, and hopeless on the round budget: one gradient step per communication round, and a CNN needs thousands, so 200 rounds get nowhere. The arithmetic of one-step-per-round is the wall.

What makes FedAvg work is a rewrite of that step that opens a knob the rigid version does not have. The server does $w_{t+1} = w_t - \eta\sum_k (n_k/n)\,g_k$. Let each client take the step *itself* and ship the result: client $k$ computes $w^k = w_t - \eta g_k$ and the server averages the stepped models with the same weights, $w_{t+1} = \sum_k (n_k/n)\,w^k$. Expand it: $\sum_k (n_k/n)(w_t - \eta g_k) = w_t - \eta\sum_k (n_k/n)\,g_k$, since the weights sum to one — *identical* to FedSGD. "Every client sends a gradient, server steps" equals "every client takes one step, server averages the stepped models." That equivalence is the crack, because the second form has slack the first does not: once a client is *stepping and shipping a model* rather than a single gradient, nothing stops it from taking more than one step before it ships. Let it sweep its local data in minibatches for several local epochs and only then send $w^k$. The rigid one-gradient-step-per-round becomes a knob — the number of local updates per round — and on this harness that knob is fixed at 5 epochs, exactly the aggressive local-compute setting that buys round-efficiency. FedSGD is the one-epoch, full-batch corner of this family; the harness sits well inside the interior.

But this is the dangerous step — the same danger FedProx was trying to contain — so it earns a careful look. When each client takes one step, the average of the stepped models is *provably* a gradient step on $f$; the algebra carried me. The moment each client takes many steps, $w^k$ is no longer $w_t$ minus $\eta$ times the gradient at $w_t$: the client has wandered off descending its *own* $F_k$, and I end up averaging models that moved to different places in a non-convex landscape. Averaging parameters of different networks is in general a catastrophe — two excellent networks can have a garbage parameter-space midpoint, because the loss surface is non-convex and the midpoint can sit on a ridge between two basins. So why would averaging 5-epoch-drifted models give me anything usable? Because of the one structural fact the recipe rests on: the server broadcasts *one* model $w_t$ to every selected client at the start of each round, so the clients do not start from independent seeds — they start from the *same* $w_t$ and walk in somewhat different directions over a few epochs. They share their symmetry-breaking, stay in the same basin, and along the straight line between two points in one gentle basin the loss is well-behaved, no ridge to cross. The average sits low, plausibly lower than either parent, because each client's drift descended a bit and the idiosyncratic wanderings partly cancel. Re-broadcasting $w_t$ every round is precisely what keeps each averaging operation an in-basin interpolation — and it is *also* the anchor FedProx's spring was reinforcing, which is the real reason that spring was nearly a no-op: the shared per-round init was already doing the basin-keeping, and a gentle extra tether on top had little left to contribute.

Two design choices carry the combine. First, the **sample-count weighting**. The global objective $f$ is itself the $n_k$-weighted combination of the $F_k$, so if I want the average of the stepped models to behave like a step on $f$, the weights must be $n_k$-proportional — that is what made the one-step algebra collapse onto the gradient-of-$f$ update. Weighting clients equally would implicitly optimize the *unweighted* mean of the $F_k$, over-counting a tiny client as much as a large one and biasing the model toward small clients, a real distortion under the Dirichlet split where client sizes vary. There is one subtlety when I see only 10 of 100 clients: $\sum_{k \in S_t}(n_k/n)\,w^k$ has weights summing to far less than one (the participating fraction of all data), giving a fraction of a model. The fix is to normalize over the clients I *actually have*: weights $n_k/m_t$ with $m_t = \sum_{k \in S_t} n_k$, a proper convex combination,

$$w_{t+1} = \sum_{k \in S_t}\frac{n_k}{m_t}\,w^k.$$

This is exactly what the harness combine does, so I do not add it — I just do not break it. Second, the **shared per-round initialization** above, the thing that makes the average meaningful in the first place.

I land *this task's* version, the minimal fill, rather than the general framework: the general account carries a client fraction $C$, a local-epoch count $E$, a batch size $B$, the one-shot-averaging extreme at $E \to \infty$, and a decay-the-local-computation schedule, but the harness fixes all of them ($C = 10/100$, $E = 5$, $B = 64$, no schedule). So the local trainer is plain SGD for the fixed epochs returning the trained model, its sample count, and its loss; the combine is the $n_k$-weighted average over the received updates, normalized by the total samples among them. Two harness details are load-bearing and I keep them: a model's `state_dict` carries non-floating-point buffers (integer counters, batch-norm tracked-batches) that it makes no sense to average, so I copy one client's value through for those and weight-average only the float tensors; and the aggregation runs on CPU in float32 for numerical safety before casting back. This is just the harness default with the spring removed — the clean reference. It damps neither the magnitude nor the direction of the drift, so the aggregate stays biased toward the average of client optima; the CIFAR number sitting in the high-0.59s while the same architecture trained centrally would clear that easily is the signature of residual drift, and *direction* — rotating each client's local step away from $\theta_k^*$ and back toward $\theta^*$ — is the gap the next rung must close.

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
