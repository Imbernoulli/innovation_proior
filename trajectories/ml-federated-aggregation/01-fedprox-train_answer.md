The harness hands me the obvious recipe — plain local SGD on each client, sample-count-weighted average of the returned models — and that obvious recipe is exactly the one whose disease defines the task. Under the Dirichlet(0.1) split each of the 100 clients holds a wildly skewed slice, I reach only 10 of them per round, and I let each run a full 5 local epochs. Five epochs on a few hundred skewed examples is a lot of local work: each client's model walks a long way toward *its own* optimum $\theta_k^*$, which under this heterogeneity has little to do with the global $\theta^*$. Average ten such drifted models and the result is pulled toward $\frac{1}{10}\sum_k \theta_k^*$, the mean of the participating clients' private solutions, not the place the federation needs to reach. The very lever that makes the recipe communication-efficient — many local steps per round — is the lever that amplifies the drift. I want to attack that with the lightest possible touch first, to measure how much a minimal repair buys before reaching for heavier machinery.

Write down what I am optimizing so the fix has somewhere to attach. The global objective is the sample-weighted average of per-client risks, $f(w) = \sum_k p_k F_k(w)$ with $p_k = n_k/n$. Averaging the returned models works at all only because everyone started this round from the *same* broadcast $w^t$: models that begin at a common point and drift modestly stay in a shared low-loss basin, and their average is a sensible model, whereas averaging models trained from different initializations lands somewhere worse than either parent. The shared per-round initialization is load-bearing. The trouble is that non-IID data makes the $F_k$ genuinely different functions whose minimizers scatter, so the more local epochs I run, the further each client travels toward its own $\theta_k^*$, and the less their average sits in any shared basin. The single knob the floor exposes — the local-epoch count — is fighting itself.

I propose **FedProx**: throttle the drift not by capping epochs (coarse, fixed for everyone, and it throws away the communication efficiency I was buying) but by capping the *displacement* $\|w - w^t\|$ directly, with a tether the optimizer itself respects. Instead of having client $k$ minimize $F_k(w)$, have it minimize

$$h_k(w; w^t) = F_k(w) + \frac{\mu}{2}\,\|w - w^t\|^2.$$

The added term is a quadratic **proximal spring** anchored at the broadcast model $w^t$: it costs the client to move away from where it started, and a stiffer $\mu$ is a stiffer spring. I am not capping epochs and not changing the optimizer — I am changing the *objective*, so however the client optimizes and for however many steps, it is now descending something that penalizes drift. The gradient of the penalty is $\mu(w - w^t)$, so one local SGD step on $h_k$ is

$$w \leftarrow w - \eta\,\big(\nabla F_k(w) + \mu(w - w^t)\big),$$

ordinary local SGD on $F_k$ with an extra restoring force pulling $w$ back toward $w^t$ at every step. When $w$ has drifted far the spring is strong and pulls it back; near $w^t$ it is weak and the step behaves like plain SGD. The whole cost is a frozen copy of $w^t$ and one elementwise term per step. And it is a strict generalization of the floor: set $\mu = 0$ and I recover exactly the plain local SGD the default does, so whatever I land here can differ from the floor only by the strength of the spring.

There is a second, principled reason the quadratic is the right object. $F_k$ is a non-convex neural-net loss, but suppose its negative curvature is bounded, $\nabla^2 F_k \succeq -L_- I$. Then the Hessian of $h_k$ is $\nabla^2 F_k + \mu I \succeq (\mu - L_-) I$, so picking $\mu > L_-$ makes the *local subproblem* strongly convex even though $F_k$ itself is not convex at all. A strongly convex subproblem has a unique minimizer and a clean displacement bound, $\|\arg\min h_k - w^t\| \le \frac{1}{\mu - L_-}\,\|\nabla F_k(w^t)\|$ — the formal statement of "the spring caps how far a client can travel, and a stiffer spring caps it tighter." The same quadratic that tethers the drift convexifies the subproblem; these are one move, not two.

I have to land *this task's* version, not the general framework, because the harness exposes only part of it. The general proximal framework also carries a *systems-heterogeneity* story where slow clients do a variable, inexact amount of local work and have their partial solutions aggregated rather than dropped, quantified by a per-client inexactness. But here every selected client runs the same fixed 5 local epochs — no straggler, no variable epoch count, no inexactness exposed at the edit surface — so that half of the framework has no surface to land on, and the part I build is the spring. Likewise its distributed-Newton relative adds the spring *plus a linear gradient-correction term* using the full global gradient; that needs every client's gradient each round, unaffordable with only 10 of 100 participating, so I deliberately carry **no** gradient-correction term. The bare proximal anchor is the right call for low participation precisely because it needs nothing global beyond the $w^t$ I already broadcast. The server combine stays the sample-count-weighted average exactly as the default has it — this is a client-side-only change.

What $\mu$? It is the half-width of the leash in squared-distance units. Too small and the spring barely engages; too large and it dominates the data gradient, pinning every client at $w^t$ so the round accomplishes nothing. The theory wants $\mu > L_-$ for convexity and *larger* $\mu$ for higher target accuracy, but in practice $\mu$ is a stability knob tuned with the epoch count and the degree of heterogeneity. I take $\mu = 0.01$, the small, conservative default (Li et al. 2020 suggest the range $0.001$–$0.1$) — a gentle tether on top of 5 epochs of SGD at $\text{lr} = 0.01$. The bet of starting here is precisely that a *gentle* spring is enough to take the edge off the drift. I should be honest about its ceiling, though, because that is the point of running it first: the spring throttles the *magnitude* of each client's move, not the *direction*. The corrected step still contains $\nabla F_k(w)$, which points toward $\theta_k^*$; the spring pulls back toward $w^t$ but does not rotate the direction toward $\theta^*$. So at any fixed distance the client still heads to its own optimum, just on a shorter leash — the aggregate stays biased toward the average of client optima, the bias throttled, not removed. If the measured numbers come back essentially on top of plain averaging, the diagnosis for the next rung is already written: damping the *magnitude* of the drift is the wrong knob, and I will have to act on the drift's *direction* instead.

Concretely in the edit surface, the cleanest realization freezes a copy of the broadcast parameters at the start of `client_local_train` and adds the penalty term $\frac{\mu}{2}\sum (w - w_0)^2$ to each minibatch loss through the harness's `loss_aug` hook on its plain-SGD helper — autodiff then contributes exactly $\mu(w - w_0)$ to the gradient. No new optimizer, no server state, no extra communication.

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
