The starting floor is plain sequential fine-tuning, and I already know what it does: while learning task $k$ the objective is $L_k(\theta)$ and nothing else, so AdamW drags every shared weight toward whatever reduces $L_k$ fastest, with no term anywhere saying "do not touch the weights task $j$ relied on." The early tasks slide off whatever ridge held them up, and by task 9 they have collapsed. The cheapest family of fix is the one that stores no past data at all — pure regularization — so I begin there: if a weight penalty already holds the early tasks, I never have to pay for rehearsal. The naive version of that, a uniform L2 anchor to the post-task weights with a single global stiffness $\lambda$, cannot work, and its failure is itself the requirement: one stiffness pays the same protection cost on weights an old task never cared about as on the weights it depended on, so to hold the old task I have to freeze the net and then cannot fit the new one. I need the same quadratic spring — anchor to the post-task weights, penalize departure — but with a *per-parameter* stiffness, large on the weights that mattered and small on the rest. Everything reduces to one number per weight: how important was weight $i$ to the task that just finished?

I propose **online EWC** — elastic weight consolidation with a single decayed Fisher. I do not want to bolt on a heuristic importance, because a wrong importance is worse than none: it would clamp the wrong weights and leave the load-bearing ones free. So I derive it. Read training as MAP estimation: by Bayes, $\log p(\theta \mid D) = \log p(D \mid \theta) + \log p(\theta) - \log p(D)$, where the log-likelihood is the negative of the BC loss (the cross-entropy *is* a negative log-likelihood). Split the data into the finished task $A$ and the current task $B$ and apply Bayes a second time: $\log p(\theta \mid D_A, D_B) = \log p(D_B \mid \theta) + \log p(\theta \mid D_A) - \text{const}$, the constant independent of $\theta$ and gone the moment I take a gradient. The whole of $A$'s contribution to the new objective is carried in that one middle term, the posterior $p(\theta \mid D_A)$ — if I had it, maximizing the $B$-likelihood plus the $A$-posterior would fit $B$ while respecting everything $A$ taught me. The posterior is the compact summary of $A$ I was after, but it is a distribution over millions of weights and intractable as written. I need an approximation that is cheap to store, cheap to add to the per-step loss, and that actually encodes per-weight importance — and a Gaussian gives all three, because its negative log density is a quadratic in $\theta$ (exactly the spring) and its precision becomes per-weight stiffness once I make it diagonal.

The Laplace approximation is the natural way to get that Gaussian. Task $A$ was trained to a local optimum $\theta^*_A$, where the gradient of $-\log p(\theta \mid D_A)$ vanishes; expanding to second order around $\theta^*_A$, the constant drops, the first-order term is zero, and what remains is $\frac{1}{2}(\theta - \theta^*_A)^\top H (\theta - \theta^*_A)$ — the quadratic part of a Gaussian centered at $\theta^*_A$ with precision $H$, the Hessian of the old negative log posterior at the optimum. Since $\theta^*_A$ is a minimum, $H$ is PSD there, as a precision must be. Its diagonal is exactly the per-weight importance: a weight with large curvature is one the $A$-solution is sensitive to (move it and $A$ degrades fast, so make it stiff), and a weight with near-zero curvature is one $A$ does not care about (leave it loose for $B$). But the full Hessian is a millions-by-millions matrix I cannot form. The escape is the **Fisher information**,

$$F = \mathbb{E}_{y \sim p_\theta}\!\left[(\nabla_\theta \log p_\theta(y\mid x))(\nabla_\theta \log p_\theta(y\mid x))^\top\right],$$

the expected outer product of the score, which equals the expected Hessian of the negative log-likelihood. The one-line check: differentiating $\log p$ twice and taking the expectation over $y \sim p$, the $\tfrac{1}{p}\,\tfrac{d^2 p}{d\theta^2}$ piece integrates to $\tfrac{d^2}{d\theta^2}\big(\sum_y p\big) = \tfrac{d^2}{d\theta^2}(1) = 0$, leaving $\mathbb{E}[-d^2\log p/d\theta^2] = \mathbb{E}[(d\log p/d\theta)^2]$. So the likelihood curvature is an average of *squared first-order gradients* — no second derivatives — and it is PSD by construction (a sum of squares), whereas the raw empirical Hessian could be indefinite and hand me meaningless negative stiffnesses. Keeping only the diagonal asserts a factorized Gaussian, one independent quadratic per weight, and the loss I minimize while training $B$ becomes

$$L_B(\theta) + \sum_i \frac{\lambda}{2}\, F_i\,(\theta_i - \theta^*_{A,i})^2,$$

the spring derived rather than guessed, with $\lambda$ absorbing the task-$A$ sample-size scaling and the overconfidence of a diagonal point-estimate Laplace.

What makes this *this task's* EWC, rather than the textbook one, is three concrete commitments the harness pins down. First, the multi-task bookkeeping. The clean derivation says keep one quadratic per past task and sum them — but that grows memory with the number of tasks. Instead I keep a *single running Fisher* with exponential decay: after each task, $\texttt{fish} \leftarrow \gamma \cdot \texttt{fish} + F_{\text{new}}$ with $\gamma = 0.9$, and re-snapshot a single anchor $\texttt{checkpoint}$ at the just-finished weights. That is constant memory regardless of how many of the ten tasks have passed; the decay down-weights the curvature of tasks that finished long ago, so the penalty is dominated by recent tasks. I expect that trade to *cost* retention on the earliest tasks — their curvature is repeatedly multiplied by $0.9$ until it drowns — which matters because the metric averages over all ten and the early tasks are where forgetting hits hardest. Second, and the sharp divergence from the principled Fisher: I compute the *empirical* Fisher from the demonstrated actions, not the model-distribution Fisher. In `end_task` I loop the finished task's dataloader, compute `nll = compute_loss(data, reduction="none")`, backprop `(-nll).mean()`, and accumulate `grads**2`. That is $\mathbb{E}_{(x,y)\sim\text{data}}[(\text{score})^2]$ — squared gradients of the demonstrated-action likelihood — not the $\mathbb{E}_{y\sim p_\theta}$ object that equals the expected Hessian. For BC this is the natural choice (the demonstrations are the only labels there are, and there is no clean categorical to sum over for an exact model-distribution expectation), but I hold it as a known approximation: the importances are measured against the expert actions, exact only when the model already fits the task well. Third, the coefficient: `e_lambda = 50000`, an enormous stiffness, which it must be because the empirical-Fisher magnitudes on a transformer's BC loss are tiny and the spring would otherwise never bite, and the penalty is applied as $\texttt{e\_lambda} \cdot \sum_i F_i (\theta_i - \texttt{checkpoint}_i)^2$ with no $\tfrac{1}{2}$ factor (absorbed into `e_lambda`), added only for `current_task > 0` since on the first task there is no anchor. Two implementation realities make this computable on a transformer: `get_params()` flattens every parameter into one vector so the anchor comparison is a single elementwise op, and `get_grads()` flattens the gradients (substituting zeros for any `None` grad) so the Fisher accumulation is one vector square; the penalty `(fish * (get_params() - checkpoint)**2).sum()` is then a single elementwise pass, cheap enough to evaluate every step.

I expect this floor to be weak, and that expectation is the reason to run it first. EWC stores no old data — its entire defense is a quadratic penalty that pins the coordinates a finished task cared about. But it has no idea which *directions* in weight space preserve the old policy's function, only how far each coordinate has strayed from a frozen scalar, so it protects the old behavior bluntly; and with the running Fisher decayed by $0.9$ each task, the earliest tasks' curvature is drowned out while the `e_lambda` large enough to hold anything risks fighting the new task's gradient hard enough to hurt forward learning. So I expect avg-final-success near the bottom — regularization alone barely beating the plain fine-tune it sits on. If that is what the numbers say, the diagnosis is already pointed at: pinning *weights* is the wrong target, and what I actually need to protect is the *function* on the old tasks, which means putting a few old examples back into the batch. That is the rehearsal camp, and where the next rung goes.

```python
# EDITABLE region of LIBERO/libero/lifelong/algos/custom.py (lines 37-65) — step 1: EWC (online)
class Custom(Sequential):
    """EWC (Elastic Weight Consolidation) lifelong learning algorithm."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.e_lambda = 50000
        self.gamma = 0.9
        self.checkpoint = None
        self.fish = None

    def get_params(self):
        return torch.cat([p.reshape(-1) for p in self.policy.parameters()])

    def get_grads(self):
        return torch.cat(
            [
                p.grad.reshape(-1)
                if p.grad is not None
                else torch.zeros_like(p).reshape(-1)
                for p in self.policy.parameters()
            ]
        )

    def penalty(self):
        if self.checkpoint is None:
            return safe_device(torch.tensor(0.0), self.cfg.device)
        else:
            penalty = (self.fish * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def start_task(self, task):
        super().start_task(task)

    def observe(self, data):
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        forward_loss = loss.item()
        if self.current_task > 0:
            loss += self.e_lambda * self.penalty()
        assert not torch.isnan(loss)
        (loss * self.loss_scale).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return forward_loss

    def end_task(self, dataset, task_id, benchmark, env=None):
        self.policy.train()
        fish = torch.zeros_like(self.get_params())

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
        )

        for data in dataloader:
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=self.cfg.device)
            )
            self.policy.zero_grad()
            nll = self.policy.compute_loss(data, reduction="none")
            (-nll).mean().backward()
            grads = self.get_grads()
            fish += grads ** 2

        fish /= len(dataloader)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.gamma
            self.fish += fish

        self.checkpoint = self.get_params().data.clone()
```
