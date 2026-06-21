Meta-SGD's numbers split exactly along the seam I flagged. At 1-shot the learned per-parameter rate did what I predicted: mean 0.4760 against MAML's 0.4365, and more tellingly the seed spread collapsed from std 0.0147 to 0.0047 (the three seeds bunching at 0.4693, 0.4794, 0.4793) — the single-shared-rate knife-edge was the 1-shot problem, and giving each coordinate its own rate removed it. But the other half of my worry landed too: miniImageNet 5-shot *slipped* to 0.6237 (from MAML's 0.6379) and CIFAR-FS 5-shot to 0.6885 (from 0.7067), CIFAR's spread actually widening to std 0.0113. The per-parameter rate helped where the support set was thin and hurt where it was not — the signature of *meta-overfitting*: with five examples per class the rigid scalar step was already near the ceiling, so the extra hundred-thousand meta-parameters of $\alpha$ tuned themselves to the meta-training episodes in a way that did not transfer. The lesson is sharp. I have been adding *capacity to the step* — scalar, then per-coordinate vector — and at 5-shot the ceiling is not the step's expressiveness at all. So I stop pouring capacity into *how* the step moves and ask a different question: *which* parameters should the inner loop move at all?

The full-network inner loop nags me from two sides. The cost: it runs per task, inside every one of 60,000 meta-iterations, over every parameter of CNN4, and because the adapted parameters depend on $\theta$ both directly and through the inner gradient, the meta-loss drags in the second-order term $(I-\alpha\nabla_\theta^2\mathcal{L}^{\text{sup}})^\top\nabla_{\theta'}\mathcal{L}^{\text{qry}}$, a Hessian-vector product over the *whole* network. The opacity: I cannot say what that inner loop is *for*. Two stories. *Rapid learning*: the meta-initialization is a conditioned launch point and the inner steps make big, efficient changes to internal representations, genuinely re-learning each task. *Feature reuse*: the meta-initialization already holds good general conv features and the inner loop barely moves them — adaptation is almost a no-op on most of the network. Meta-SGD's 5-shot regression is a hint that the second is true, since a richer step should have helped at 5-shot if the body were doing heavy re-learning.

I have to settle which story holds, because it dictates what I am allowed to cut. CNN4 is not homogeneous: there is a sharp split between the head and everything before it. In a five-way episode the five output neurons get an arbitrary, task-specific class assignment — (dog, cat, frog, cupcake, phone) in one task, (airplane, frog, boat, car, pumpkin) in the next from the same pool — so no fixed last layer can be right for both, and the head *must* move per task no matter which story is true. The conv blocks compute class-agnostic features (edges, textures, parts) that *could* be shared without moving. So the whole question lives in the body. The diagnostics on this exact MAML/MiniImageNet setup are unambiguous: freezing layer 1 leaves 1-shot at 46.5 vs 46.9 unfrozen; freezing layers 1–2, 46.4; 1–3, 46.3; freezing all four conv layers (everything but the head), 46.3 ± 0.4; on 5-shot it slips only from 63.1 to 61.0. Removing body adaptation costs essentially nothing. The sharper measurement compares the body's representation before and after the inner loop directly — CCA/CKA of a layer's activations, 1 for identical functions — and the body conv layers sit at CCA above 0.9, CKA near 1, while the head sits below 0.5. The Euclidean weight movement says the same more starkly: the average distance between initialization and finetuned weights is tiny for every layer except the last, *even though the body has far more parameters*. And this holds from ten thousand iterations in, not just at convergence. Feature reuse, not rapid learning — the body's good features come from the *outer* loop accumulating across tasks; the inner loop on the body is along for the ride. This retro-explains Meta-SGD: a learned per-coordinate rate on body parameters that should barely move can only meta-overfit, exactly the 5-shot regression I saw.

I propose ANIL — Almost No Inner Loop. Partition $\theta = (\theta_1,\dots,\theta_l)$ into body $\theta_1,\dots,\theta_{l-1}$ and head $\theta_l$. The inner loop updates only the head — $\theta_l$ descends the support loss at $\alpha$, the body components pinned at the meta-initialization across all inner steps — while the outer loop is untouched: same meta-loss, query loss at the adapted parameters summed over the meta-batch, and the body parameters still *learned*, just only by the outer loop now. The timing result licenses cutting the body's inner loop at *both* train and test: the inner loop is inert on the body throughout training, so removing it during training does not change what the body learns. This is the whole move — keep the head's inner loop, delete the body's.

The load-bearing subtlety is that this is *not* first-order MAML, which is why I expect it to *match* MAML's accuracy rather than lag like a first-order approximation. First-order MAML keeps the full inner loop (adapts every parameter) but *drops the Hessian*; I do the opposite axis — keep the second-order machinery and shrink *which* parameters get an inner loop down to the head. Take the smallest net with a body and a head, $\hat y = \theta_2(\theta_1 x)$, one inner step. In my rule the body does not move, so the adapted query prediction is $[\theta_2 - \partial\mathcal{L}^{\text{sup}}/\partial\theta_2]\cdot\theta_1\cdot x_2$. The *second* bracket collapsed to a bare $\theta_1$ — the body not adapting — but the *first* bracket still contains $\theta_1$, because the head's inner-loop gradient $\partial\mathcal{L}^{\text{sup}}/\partial\theta_2$ was computed on a forward pass through the body. So differentiating the query loss with respect to $\theta_1$ still differentiates *through* the head update, and a second-order term survives. Removing the body's inner loop does not make this first-order — the curvature that flows through the *head's* inner step is retained, which is exactly the part of the second-order information attached to the only thing still adapting.

The implementation has a sharp gotcha forced by the harness primitive. `l2l.update_module(model, updates=[...])` wants a full-length list — one update per parameter, in order — and swaps each $p$ for $p+u$. So I compute the support cross-entropy, take `torch.autograd.grad` with respect to *only the head parameters* with `create_graph=True` (keeping the surviving second-order term), and build the updates list with $-\alpha g$ for each head parameter and `torch.zeros_like(p)` for every body parameter. The gotcha: `update_module` does *not* mutate in place — it *replaces* the parameter objects with fresh tensors $p+u$ every step. Grab references to the head parameters once before the loop and after the first step those references are stale, the gradient is taken with respect to detached tensors, and every subsequent update silently comes out zero — the method looks like it runs while doing nothing. So I record the head parameter *names* once in `__init__` (names are stable across the object swaps even though the tensors are not — I scan `named_parameters` for the ones containing "classifier"), and inside *each* inner step re-collect the current parameter objects matching those names, build an id-set, and route updates by id. The rule carries no learnable optimizer state of its own — unlike Meta-SGD I add no meta-parameters — so `meta_parameters()` returns `[]`, and the outer Adam optimizes only the model's own weights.

The delta from Meta-SGD is a change of axis, not of capacity, and at a fraction of the per-task cost — at evaluation the body is a single forward pass and only the tiny linear head iterates its ten steps. The falsifiable claims point at where Meta-SGD failed. At **5-shot** ANIL should *recover* what Meta-SGD lost and edge past MAML, because the meta-overfitting came from adapting body parameters that should not move: miniImageNet 5-shot back above 0.6379, CIFAR-FS 5-shot toward or past 0.7067, CIFAR's spread tightening from 0.0113. At **1-shot** ANIL should at least hold Meta-SGD's 0.4760 — the head still gets a genuine second-order-aware inner loop, the part that fixed 1-shot — and may improve on it, since head-only adaptation from a single example is far less prone to the instability that forced MAML's tiny rate. A 1-shot drop would falsify the feature-reuse story for the thin-support regime; the bet is the opposite, that feature reuse is the whole game and removing the body's inner loop fixes 5-shot while holding 1-shot, making ANIL the strongest rung on every benchmark at the lowest cost.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — step 3: ANIL
class InnerLoopOptimizer:
    """ANIL inner-loop optimizer (Raghu et al., 2019).

    Almost No Inner Loop: only adapts the final classification head
    during inner-loop adaptation. The feature extractor backbone is
    frozen, relying on feature reuse from the meta-initialization.
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Identify head parameters (the last linear layer: classifier)
        # CNN4 structure: features (CNN4Backbone) -> classifier (Linear)
        self._head_param_names = set()
        for name, _ in model.named_parameters():
            if "classifier" in name:
                self._head_param_names.add(name)

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            # Re-identify head params each step because l2l.update_module
            # replaces parameter objects (new ids), so stale references
            # from a previous step would cause all updates to be zero.
            head_params = []
            head_ids = set()
            for name, p in model.named_parameters():
                if name in self._head_param_names:
                    head_params.append(p)
                    head_ids.add(id(p))

            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, head_params, create_graph=True
            )
            grad_map = {id(p): g for p, g in zip(head_params, grads)}
            updates = [
                -self.inner_lr * grad_map[id(p)] if id(p) in head_ids
                else torch.zeros_like(p)
                for p in model.parameters()
            ]
            l2l.update_module(model, updates=updates)
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []
```
