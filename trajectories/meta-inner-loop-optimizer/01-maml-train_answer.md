The substrate hands me a two-loop meta-learner and asks me to design only the inner adaptation rule, and the honest place to start is the floor: the simplest rule that adapts a CNN4 to a five-way task from one or five labeled images per class and still generalizes to held-out query points. The obstacle is capacity against data — fit a hundred-thousand-parameter net to five points and I have memorized them and learned nothing that transfers. The cheapest fallback, pretrain-and-fine-tune, does not escape this: pooling the meta-training episodes and minimizing the pooled loss builds an initialization that minimizes *average* loss, which for a symmetric task family lands on a compromise (something like the mean target) that adaptation must then fight its way out of. The point is sharp — fine-tuning from a generic pretrain fails not because gradient descent is weak but because the initialization was never *chosen* for adaptability. The fix is to choose it directly for the property I want.

I propose MAML: do not learn an optimizer or a metric, learn an *initialization* $\theta$ such that a few ordinary gradient-descent steps on a new task's support set produce a model that generalizes on that task. The whole construction follows from writing that property down as an objective. For a task $\mathcal{T}_i$, adaptation is one (or a few) gradient steps on the support set,
$$\theta_i' = \theta - \alpha\,\nabla_\theta \mathcal{L}^{\text{sup}}_{\mathcal{T}_i}(f_\theta),$$
and the thing I want small is *not* the support loss — of course the adapted model does well on the points it just trained on, that is exactly the overfitting I am fighting — but the loss on a *held-out query set* of the same task. So the meta-objective is
$$\min_\theta \sum_i \mathcal{L}^{\text{qry}}_{\mathcal{T}_i}(f_{\theta_i'}), \qquad \theta_i' = \theta - \alpha\,\nabla_\theta \mathcal{L}^{\text{sup}}_{\mathcal{T}_i}(f_\theta).$$
The inversion that makes this work is that the variable being optimized is $\theta$, the initialization, but the loss is evaluated at $\theta_i'$, the *post-adaptation* parameters: the held-out query loss after fine-tuning *is* the training signal for the launch point. Instead of "minimize loss and hope fine-tuning works," it is "make fine-tuning work, and that is the loss." The support/query split is load-bearing, not cosmetic — score the meta-loss on the very points I adapted on and I would reward $\theta$ for being a place from which I can *memorize* the support, the exact failure I am avoiding; splitting them means the only way to lower the meta-loss is an initialization from which a support step yields a model that *generalizes*. That is why the loop draws a fresh 15-shot query set every episode.

What makes it trainable is that the outer step is just SGD on this object, $\theta \leftarrow \theta - \beta\,\nabla_\theta \sum_i \mathcal{L}^{\text{qry}}_{\mathcal{T}_i}(f_{\theta_i'})$, so everything reduces to the meta-gradient through $\theta_i'$, which contains $\theta$ twice — once in the leading term and once inside the support gradient. Writing $\theta_i' = \theta - \alpha\,g(\theta)$ with $g(\theta)=\nabla_\theta\mathcal{L}^{\text{sup}}(f_\theta)$, the chain rule gives
$$\nabla_\theta \mathcal{L}^{\text{qry}}(\theta_i') = \big(I - \alpha\,\nabla_\theta^2 \mathcal{L}^{\text{sup}}(f_\theta)\big)^\top \nabla_{\theta'}\mathcal{L}^{\text{qry}}(\theta_i').$$
Two distinct losses live here and must not be confused: the Hessian is of the loss I *adapted on* (support), the gradient is of the loss I *evaluate* (query). The $-\alpha H$ term is doing real work — moving $\theta$ also moves the gradient I subtract, so $\theta_i'$ does not shift by the same amount as $\theta$; the Hessian measures exactly how much the inner gradient bends, and $(I-\alpha H)$ propagates that bending so the outer optimizer accounts for steering the *start* of a gradient step rather than the end. Crucially I never form $H$: I need $H$ times a vector, which is one extra backward pass, and in practice not even that. If I build $\theta_i'$ as a node in the graph (subtract $\alpha g$ while *keeping the subtraction differentiable*), forward the query set, and call backward to $\theta$, reverse-mode autodiff produces $(I-\alpha H)^\top\nabla_{\theta'}\mathcal{L}^{\text{qry}}$ for me — provided the inner gradient was taken with `create_graph=True`. With several inner steps the trajectory simply unrolls, $\theta_i'' = \theta_i' - \alpha\nabla\mathcal{L}^{\text{sup}}(\theta_i')$, the Jacobian chaining into a product of $(I-\alpha H_k)$ factors that autodiff handles through the loop. Nothing about the architecture entered any of this — I only assumed $f_\theta$ is differentiable and trained by gradient descent, which is exactly why the same three lines serve a conv net under cross-entropy here as an MLP under MSE, with no metric, no recurrent optimizer, and no extra parameters. That generality is the point, and it is why this is the honest floor: `meta_parameters()` is empty because a fixed scalar rate carries no learnable state.

There is one harness-specific decision that is not in the generic story and that separates a floor that runs from one that diverges. The fixed loop sets a global `inner_lr = 0.5`, which is fine at 5-shot where five examples per class give a steadier support gradient (and matches the learn2learn benchmark value). But at 1-shot, a single example per class makes the support gradient a high-variance estimate, and stepping *every* parameter — all four conv blocks plus BatchNorm plus the head — by half of it blows the clone into a region where the meta-gradient through five such steps is meaningless. The standard 1-shot MAML recipe is an order of magnitude smaller, $\alpha=0.01$, precisely to keep full-network adaptation stable. So in this harness MAML is shot-aware: $\alpha=0.01$ when `N_SHOT==1`, else $0.5$ — a property of the benchmark's chosen rate, not of the method.

What this floor cannot do is the entire diagnosis for what comes next. MAML adapts *all* parameters with *one fixed scalar* rate and *no learned optimizer state*: the direction is locked to the raw gradient, the magnitude is a single global number, and the only thing meta-learned is where to start. Anything the task family might know about *how* to move — which coordinates should move a lot and which barely, what combined direction generalizes on a fresh episode — cannot be expressed. I expect this to be genuinely competitive on the 5-shot benchmarks, where the larger support set forgives the rigid step and the conv features carry the day, and I expect 1-shot to be the soft spot even at $\alpha=0.01$ — the lowest and most exposed number — which is exactly where a learned, per-coordinate step should later pull ahead.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — step 1: vanilla MAML
class InnerLoopOptimizer:
    """MAML inner-loop optimizer (Finn et al., 2017).

    Vanilla SGD with a fixed learning rate applied uniformly to all
    model parameters. This is the standard MAML inner loop.

    Shot-aware LR override: the global INNER_LR=0.5 destabilizes
    full-network adaptation at 1-shot in the local harness. At 5-shot the larger
    support set buffers gradient noise so 0.5 is fine (matches
    learn2learn benchmark default). Use the common 1-shot recipe
    (0.01) only when N_SHOT=1, keep 0.5 for 5-shot.
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = 0.01 if N_SHOT == 1 else 0.5

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            model = l2l.algorithms.maml.maml_update(
                model, lr=self.inner_lr, grads=grads
            )
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []
```
