EWC told me in one row that pure regularization does not carry this regime: avg_final_success $0.045$ and auc $0.0487$, with fwt $0.2091$ but nbt $0.174$. Read together the story is unambiguous — fwt of $0.21$ says the policy *does* learn each task as it arrives, so the BC learner works, but avg_final $0.045$ means that by the tenth task almost nothing of the early tasks survives, and the nbt of $0.17$ is the forgetting written explicitly. The decayed weight-spring simply did not hold the early tasks: their curvature was multiplied by $0.9$ each task until it drowned, and an anchor that pins *coordinates* to a frozen scalar never protected the *function* the early policies computed. The target was wrong. I should stop protecting weights and start protecting the old tasks' *behavior*, which means putting some of their data back in front of the optimizer — but I want to understand the geometry before I pick how aggressively to use it.

Go back to the one fact I trust completely. A single step $\theta \leftarrow \theta - \alpha g$, with $g = \nabla_\theta L_k(\theta)$ the current gradient, changes a past task $j$'s loss by, to first order, $L_j(\theta - \alpha g) - L_j(\theta) \approx -\alpha\,\langle g, g_j\rangle$, where $g_j = \nabla_\theta L_j$ is the past task's gradient at the current weights. So the *sign of an inner product* is the forgetting diagnostic: $\langle g, g_j\rangle > 0$ means the step lowers task $j$'s loss (backward transfer, good), and $\langle g, g_j\rangle < 0$ means it raises it (forgetting, happening one negative dot product at a time). EWC never looked at this — it judged a step by how far weights strayed from $\theta^*$, not by what the step actually did to the old tasks, which is exactly why it could clamp the wrong coordinates while the policy's function drifted anyway. The inner-product view says: police the *angle* of each step against the directions the old tasks care about, and address forgetting at its source. To compute $g_j$ I keep a small bounded episodic memory of past examples and approximate each past gradient by the gradient on that memory; the memory measures a *direction*, not a full retraining set, so it can stay small.

I propose **A-GEM** (averaged gradient episodic memory). Phrase the goal as a hard constraint rather than EWC's soft penalty: while learning task $k$, minimize the current loss subject to "do not let any past task's loss go up." The crucial thing this buys over EWC is that the inequality is one-sided — it *permits* a past loss to drop, so positive backward transfer is allowed, exactly the thing EWC's symmetric quadratic anchor threw away. Linearized over the small step, "past loss does not increase" becomes $\langle g, g_j\rangle \geq 0$ for every past task $j$; if all hold, take the step, and if some are violated, do not discard $g$ — most of it is still the right direction — but fix it *minimally*: $\min_{\tilde g} \tfrac{1}{2}\|g - \tilde g\|^2$ s.t. $\langle \tilde g, g_j\rangle \geq 0$ for all $j$. That statement is right, and it is exactly where it gets too expensive. It is a quadratic program over millions of parameters; dualizing helps (the constraint matrix has only $k-1$ rows, so the dual QP is in $k-1$ variables), but the per-step bill still grows with the stream — to build the constraints I need one backward pass through the memory of *each* of the $k-1$ past tasks plus a QP solve, and by task 9 every step pays nine extra backwards plus a QP. The constraint formulation is right; the *number* of constraints is the disease.

Why $k-1$ constraints? Because I demanded that *each individual* past task not get worse — a worst-case promise. But I am graded on the *average* success over the ten tasks, so I care that the *average* past loss not rise, not that every single task holds, and the two genuinely differ: insisting every task improve-or-hold can block a step that raises one task's loss a hair while dropping the others a lot, a step that improves the average. So I weaken the promise on purpose: require only that the average loss over the union of all past memories not increase. Let $g_{\text{ref}} = \nabla_\theta L(\theta, M)$ be the gradient of the loss over the combined memory $M = \bigcup_{j<k} M_j$ — a single reference direction summarizing what the past wants — and the whole constraint set collapses to $\langle \tilde g, g_{\text{ref}}\rangle \geq 0$, one linear inequality, a projection onto one halfspace with a closed form.

I derive it so I am sure of the vector and the scalar. $\min_z \tfrac{1}{2}\|g - z\|^2$ s.t. $z^\top g_{\text{ref}} \geq 0$. Drop the constant $\tfrac{1}{2}g^\top g$, write the constraint as $-z^\top g_{\text{ref}} \leq 0$, form the Lagrangian $\mathcal{L}(z,\alpha) = \tfrac{1}{2}z^\top z - g^\top z - \alpha\, z^\top g_{\text{ref}}$ with $\alpha \geq 0$. Stationarity gives $z - g - \alpha g_{\text{ref}} = 0$, so $z^* = g + \alpha g_{\text{ref}}$ — the answer has the shape "current gradient slid along the reference." Substituting back, the dual is $\theta_D(\alpha) = -\tfrac{1}{2}g^\top g - \alpha\, g^\top g_{\text{ref}} - \tfrac{1}{2}\alpha^2\, g_{\text{ref}}^\top g_{\text{ref}}$, concave in $\alpha$, with $\alpha^* = -\,g^\top g_{\text{ref}} / g_{\text{ref}}^\top g_{\text{ref}}$. The KKT sign check is the entire if/else: when the constraint is *violated*, $g^\top g_{\text{ref}} < 0$, so $\alpha^* > 0$ is feasible and active and $\tilde g = g - \dfrac{g^\top g_{\text{ref}}}{g_{\text{ref}}^\top g_{\text{ref}}}\, g_{\text{ref}}$; when it is already *satisfied*, $g^\top g_{\text{ref}} \geq 0$, the unconstrained $\alpha^* \leq 0$ is clamped to $0$ and $\tilde g = g$. The geometry: the subtracted term is the orthogonal projection of $g$ onto $g_{\text{ref}}$, so in the violating case I remove exactly the component of $g$ pointing against the reference and keep everything else — and checking, $\tilde g^\top g_{\text{ref}} = g^\top g_{\text{ref}} - g^\top g_{\text{ref}} = 0$, so the corrected gradient sits on the boundary of the feasible halfspace, to first order neither raising nor lowering the average past loss while staying as close to $g$ as the constraint allows. Minimal fix, no QP, no matrix of per-task gradients. And $g_{\text{ref}}$ need not be exact: it is a *direction*, so a stochastic estimate from a single random minibatch over the union memory suffices — one forward/backward on a fixed-size batch regardless of how many tasks are stored, the averaging across tasks coming for free because the union memory mixes them. Per-step cost is now flat in the stream: the backward for $g$ I was doing anyway, one extra for $g_{\text{ref}}$, two dot products. I am honest about the trade against the per-task version — I gave up the worst-case "no single task gets worse" guarantee for an average-loss guarantee — but the average is exactly my metric.

Landing it in *this* harness fixes a few concrete details. `__init__` allocates the episodic memory (a list of finished-task datasets) and the flat gradient scratch: `grad_dims = [p.numel() for p in policy.parameters()]` and two preallocated flat buffers `grad_xy` (for $g$) and `grad_er` (for $g_{\text{ref}}$) of total parameter size on device. `start_task`, once past task 0, builds the reference stream — truncate each finished dataset to `n_memories = 1000` sequence windows with `TruncatedSequenceDataset`, `ConcatDataset` them, and wrap a `RandomSampler` `DataLoader` in `cycle(...)`, infinite because the reference stream must outlast the current-task loader, which has a different length. `end_task` just appends the finished dataset to the memory list. The heart is `observe`, and the *sequencing of the two backward passes* is load-bearing because they share the same `.grad` fields: first the current-task forward and backward (`(loss*loss_scale).backward()`) so `.grad` holds $g$; if the buffer exists, flatten `.grad` into `grad_xy` with `_store_grad` *before clobbering it*, then `zero_grad`, pull a reference batch with `next(self.buffer)`, forward/backward it so `.grad` holds $g_{\text{ref}}$, flatten into `grad_er`; take `dot_prod = ⟨grad_xy, grad_er⟩`, and if it is `< 0` project (`_project` computes $g_{xy} - (g_{xy}^\top g_{er}/g_{er}^\top g_{er})\,g_{er}$) and write back with `_overwrite_grad`, otherwise write `grad_xy` back unchanged — necessary because the second backward overwrote `.grad` with $g_{\text{ref}}$. Either branch ends with grad-clip and `optimizer.step()`. The three helpers `_project`, `_store_grad`, `_overwrite_grad` are module-level functions above the class, moving between per-parameter `.grad` tensors and the flat buffer by precomputed slice offsets. On task 0 the buffer is empty, so it is plain BC.

The delta from EWC is precise and addresses its measured failure directly: where EWC anchored *coordinates* to a decayed snapshot and let the early-task function drift (avg_final $0.045$, nbt $0.17$), A-GEM keeps actual old examples and *vetoes* any step whose gradient points against the average old-task direction, protecting behavior at its source and permitting the backward transfer EWC forbade. I expect A-GEM to clear EWC's $0.0487$ auc and $0.045$ avg_final comfortably — real old data in the constraint should hold meaningfully more than a decayed spring, with nbt *lower* than $0.17$ since the projection's whole job is to stop the average old loss from rising. But I can already feel A-GEM's own ceiling: the memory acts only as a *veto*. Most steps have $\langle g, g_{\text{ref}}\rangle \geq 0$ and the memory does literally nothing — I take the plain current-task step; only on a violation does it shave off a component. It never *pushes the old-task loss down*; it guards a ceiling. That caution should leave it underfitting both the new task and the memory, so if avg_final lands respectable but short of what the data could give, the next move is already written: stop vetoing, and just *train on* the old examples by stacking them into the batch and descending their loss directly.

```python
# EDITABLE region of LIBERO/libero/lifelong/algos/custom.py (lines 37-65) — step 2: A-GEM
def _project(gxy, ger):
    """Project gradient gxy so it does not conflict with ger."""
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


def _store_grad(params, grads, grad_dims):
    """Store parameter gradients into a flat vector."""
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


def _overwrite_grad(params, newgrad, grad_dims):
    """Overwrite parameter gradients from a flat vector."""
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[: count + 1])
            this_grad = newgrad[begin:end].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


class Custom(Sequential):
    """A-GEM (Averaged Gradient Episodic Memory) lifelong learning algorithm."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.n_memories = 1000
        self.datasets = []
        self.buffer = None

        self.grad_dims = []
        for pp in self.policy.parameters():
            self.grad_dims.append(pp.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            buffers = [
                TruncatedSequenceDataset(dataset, self.n_memories)
                for dataset in self.datasets
            ]
            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=(self.cfg.train.num_workers > 0),
                )
            )

    def observe(self, data):
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (loss * self.loss_scale).backward()

        if self.buffer is not None:
            _store_grad(self.policy.parameters, self.grad_xy, self.grad_dims)
            buf_data = next(self.buffer)
            self.policy.zero_grad()

            buf_data = self.map_tensor_to_device(buf_data)
            buf_loss = self.policy.compute_loss(buf_data)
            buf_loss.backward()
            _store_grad(self.policy.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = _project(gxy=self.grad_xy, ger=self.grad_er)
                _overwrite_grad(self.policy.parameters, g_tilde, self.grad_dims)
            else:
                _overwrite_grad(self.policy.parameters, self.grad_xy, self.grad_dims)

        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def end_task(self, dataset, task_id, benchmark, env=None):
        self.datasets.append(dataset)
```
