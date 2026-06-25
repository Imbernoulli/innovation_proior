Let me start from what actually goes wrong. I have a stream of tasks coming at me one at a time, and I get a single pass over each one — train on task 1, then task 2, and at the very end I'm graded on all of them. The thing that kills me is forgetting: the moment I run gradient descent on task `t`, the parameters slide off whatever ridge was holding up tasks `1..t-1`, and their accuracy falls apart. Of course it does — my objective while learning task `t` is `ℓ(f_θ, D_t)` and nothing else, so the optimizer has no reason to care what happens to the old tasks. So the real problem isn't "learn task t," it's "learn task t *without* letting the old ones rot," under conditions that forbid the cheap escape hatches. I can't just stash every example I've ever seen and retrain on all of it — that's not lifelong learning anymore, that's multitask learning with the whole dataset in hand, and it blows the memory budget and the single-pass premise. And whatever I do, the per-step cost has to stay near plain SGD, because if the cost of one gradient step grows with how many tasks I've already seen, the method dies on a long stream. So: small bounded memory, single pass, per-step compute roughly flat in the number of tasks, forgetting controlled, and — this one is easy to forget — I'd like to *not forbid* old tasks from improving as I learn new ones, since a new task can genuinely sharpen a shared feature.

What's on the table? The regularization family — EWC and its cousins — is the cheapest thing going. After I finish a task, estimate how important each parameter was (EWC uses the diagonal of the Fisher, the expected squared gradient, as the importance `F_i`), and then while learning the next task add `Σ_i (λ/2) F_i (θ_i − θ*_i)²` to pull the important parameters back toward their old values. It's beautiful in its cheapness: one scalar penalty, memory linear in the parameters, no stored examples. But I keep tripping over two things. First, that `λ` is a tightrope — too small and I forget anyway, too large and I can't fit the new task — and it's fragile. Second, and more damning for *my* setting, this whole construction was made for the multi-epoch regime, where the optimizer gets many passes to settle into a basin that satisfies the quadratic anchor and the new task at once. Give me a single pass and the anchor never gets a chance to do its job; the empirical record is clear that the regularizers wilt in single-pass streaming. And there's a flavor problem: a quadratic anchor *forbids* important parameters from moving. It doesn't ask "is this move going to hurt the old task?" — it just clamps. So it can't tell apart a move that wrecks an old task from a move that happens to *help* it, and it gives up positive backward transfer wholesale. I want something that judges moves by their actual effect on past tasks, not by how far they stray from a frozen point.

The other end is the architectural cure — give each task its own column and freeze the rest, like Progressive Networks. Forgetting becomes literally zero because you never touch the old weights. But the memory and parameter count grow with the number of tasks, and you carry every column to test time. On a long stream that's exactly the unbounded-growth I'm trying to avoid. Off the table.

So neither soft-anchor nor grow-the-net is right. Let me go back to the one geometric fact I can lean on. Take a single SGD step `θ ← θ − α g` with `g = ∇_θ ℓ(f_θ, D_t)` the current task's gradient. What does that step do to some past task's loss `ℓ_k`? Taylor-expand around the current `θ`: `ℓ_k(θ − αg) = ℓ_k(θ) − α ⟨g_k, g⟩ + O(α²)`, with `g_k = ∇_θ ℓ_k` the past task's gradient at the current parameters, so to first order `ℓ_k(θ − αg) − ℓ_k(θ) ≈ −α ⟨g, g_k⟩`.

I should check that this first-order story is actually predictive and not just a formal expansion, because the whole method is going to ride on it. Let me make a past loss concrete and measure. Take `ℓ_k(θ) = ½(θ − c)ᵀ A (θ − c)` with `A = [[2, 0.3],[0.3, 1]]`, `c = (0.5, −0.5)`, evaluate at `θ = (1, 1)`, and step along `g = (0.8, −0.4)`. Then `g_k = A(θ − c) = A(0.5, 1.5) = (1.45, 1.65)` and `⟨g, g_k⟩ = 0.8·1.45 − 0.4·1.65 = 1.16 − 0.66 = 0.5`. The first-order prediction for the loss change is `−α·0.5`. The actual change, computed exactly from the quadratic: at `α = 0.1` I get `Δℓ_k = −4.376e-2` against a prediction of `−5.0e-2` (ratio 0.875); at `α = 0.01`, `−4.938e-3` vs `−5.0e-3` (ratio 0.988); at `α = 0.001`, `−4.994e-4` vs `−5.0e-4` (ratio 0.999). So the prediction is not exact — at a coarse step the `O(α²)` term eats an eighth of it — but it converges to the truth as the step shrinks, exactly as a first-order term should, and at the small learning rates I actually use it's a faithful sign-and-magnitude guide. Here `⟨g, g_k⟩ = 0.5 > 0` and indeed the past loss *went down*. So the picture holds up: if `⟨g, g_k⟩ > 0` the step decreases the past task's loss (backward transfer), if `⟨g, g_k⟩ < 0` it increases it (forgetting, one negative inner product at a time). That points somewhere specific — I don't need to anchor parameters; I can instead police the *angle* of each step against the directions that matter for past tasks.

To compute `g_k` I need to evaluate the past task's gradient, which means I need *some* of its examples on hand. Fine — keep a tiny episodic memory `M_k` per task, a handful of stored examples, and approximate the past task's gradient by the gradient of the loss on `M_k`. The memory is small and bounded, which respects the budget, and it's used to *measure* a direction, not to retrain on.

Now I can phrase the goal as a hard constraint instead of a soft penalty. While learning task `t`, I want: minimize the current loss, subject to "don't let any past task's loss go up,"

  minimize_θ  ℓ(f_θ, D_t)   s.t.  ℓ(f_θ, M_k) ≤ ℓ(f_θ^{t-1}, M_k)  for all k < t,

where `f_θ^{t-1}` is the network at the end of task `t-1`. Notice this *allows* the past loss to drop — the inequality is one-sided — so positive backward transfer is permitted, exactly the thing the quadratic anchor threw away. And I don't actually have to store the old predictor `f_θ^{t-1}`: as long as every individual update doesn't raise the past-task losses, the running value stays at or below where it was, so the reference cancels and the condition becomes purely local, on the gradient. Linearizing around the small step, "the loss on `M_k` does not increase" becomes `⟨g, g_k⟩ ≥ 0`. So the per-step constraint set is

  ⟨g, g_k⟩ ≥ 0   for all k < t.

If all of those hold, the proposed `g` harms nobody to first order; take it. If some are violated, I shouldn't throw the step away — most of `g` is still the right direction for the current task — I should *fix* it minimally. Find the closest gradient (in L2) that satisfies all the constraints:

  minimize_{g̃}  (1/2)‖g − g̃‖²   s.t.  ⟨g̃, g_k⟩ ≥ 0  for all k < t.

That's a clean statement of intent. But now stare at what it costs me to actually run it. It's a quadratic program in `g̃`, which lives in `P` dimensions — the number of network parameters, millions. You don't solve a million-variable QP at every gradient step. The standard move is to dualize: the constraint matrix has only `t−1` rows, so the dual QP is in `t−1` variables, with `G = −(g_1,...,g_{t-1})`,

  minimize_v  (1/2) vᵀ G Gᵀ v + gᵀ Gᵀ v   s.t.  v ≥ 0,

and then `g̃ = Gᵀ v* + g`. Much smaller — `t−1` variables instead of `P`. So this is solvable. But look at the bill at *every single training step*: I have to build `G`, which means one backward pass through the memory of *each* of the `t−1` past tasks to get every `g_k`, and then hand a numerical QP solver a `(t−1)×(t−1)` problem. Both pieces grow with `t`. Early in the stream that's fine; by the time I'm on task 20, every step is twenty extra backward passes plus a QP. That is precisely the per-step-cost-scales-with-the-stream pathology I said I couldn't tolerate. Wall.

So the constraint formulation is right and the *number* of constraints is the disease. Let me poke at it. Why do I have `t−1` separate constraints? Because I asked for the loss of *each individual* past task not to rise. That's a strong promise — it's a worst-case promise, no single old task is allowed to get worse on its memory, even by a hair. Do I actually need that? My headline metric is *average* accuracy across all tasks at the end. What I really care about is that the average past-task loss doesn't go up. And those are not the same thing: insisting that *every* task improve-or-hold can block a step that would *raise one task's loss a little while lowering the others a lot* — a step that drops the average. The worst-case promise is leaving average accuracy on the table, and it's the thing making me pay `t−1` constraints. So weaken it on purpose: ask only that the *average* loss over the union of all past memories not increase.

  minimize_θ  ℓ(f_θ, D_t)   s.t.  ℓ(f_θ, M) ≤ ℓ(f_θ^{t-1}, M),   M = ∪_{k<t} M_k.

One task, one constraint. Linearize the same way: let `g_ref = ∇_θ ℓ(f_θ, M)` be the gradient of the loss over the combined memory — a *single* reference direction summarizing "what the past wants." The whole constraint set collapses to

  ⟨g̃, g_ref⟩ ≥ 0.

One constraint. A single linear inequality means the projection problem is a projection onto one halfspace, which I can write down in closed form. No `G` matrix, no QP solver, just a couple of inner products. The per-step cost is now: one backward for `g` (which I was doing anyway), one backward for `g_ref`, two dot products. Flat in `t`. The `t−1`-scaling is gone, killed by replacing "every task" with "the average task."

Before I trust the closed form, let me actually derive it rather than assert it, because I want to be sure I have the right vector and the right scalar. The problem is

  minimize_{g̃}  (1/2)‖g − g̃‖²   s.t.  g̃ᵀ g_ref ≥ 0.

Rename `g̃ = z` and expand the objective: `(1/2)‖g − z‖² = (1/2)zᵀz − gᵀz + (1/2)gᵀg`. The last term `(1/2)gᵀg` is constant in `z`, drop it. Write the constraint as `−zᵀg_ref ≤ 0` to put it in standard form. Lagrangian with multiplier `α ≥ 0`:

  L(z, α) = (1/2)zᵀz − gᵀz − α zᵀg_ref.

Stationarity in `z`: `∂L/∂z = z − g − α g_ref = 0`, so

  z* = g + α g_ref.

So at the optimum `g̃` is `g` plus some multiple of `g_ref` — I'm sliding the current gradient along the reference direction, and `α` will say how far. To pin `α` I plug `z*` back to get the dual function. Compute the pieces: `(1/2)z*ᵀz* = (1/2)(gᵀg + 2α gᵀg_ref + α² g_refᵀg_ref)`, `gᵀz* = gᵀg + α gᵀg_ref`, `α z*ᵀg_ref = α gᵀg_ref + α² g_refᵀg_ref`. So

  θ_D(α) = (1/2)(gᵀg + 2α gᵀg_ref + α² g_refᵀg_ref) − (gᵀg + α gᵀg_ref) − (α gᵀg_ref + α² g_refᵀg_ref).

Collect terms. The `gᵀg` terms: `(1/2)gᵀg − gᵀg = −(1/2)gᵀg`. The `α gᵀg_ref` terms: `α gᵀg_ref − α gᵀg_ref − α gᵀg_ref = −α gᵀg_ref`. The `α² g_refᵀg_ref` terms: `(1/2)α² g_refᵀg_ref − α² g_refᵀg_ref = −(1/2)α² g_refᵀg_ref`. So

  θ_D(α) = −(1/2)gᵀg − α gᵀg_ref − (1/2)α² g_refᵀg_ref.

Maximize over `α ≥ 0` (dual is a max): `∂θ_D/∂α = −gᵀg_ref − α g_refᵀg_ref = 0`, giving

  α* = − gᵀg_ref / (g_refᵀg_ref).

And `θ_D` is concave in `α` (the `−(1/2)α² g_refᵀg_ref` term), so this is the max. Now the KKT sign condition: I need `α* ≥ 0`. When the constraint is *violated*, `gᵀg_ref < 0`, which makes `α* = −gᵀg_ref/g_refᵀg_ref > 0` — feasible, the constraint is active, and I plug `α*` into `z* = g + α* g_ref`:

  g̃ = g − (gᵀg_ref / g_refᵀg_ref) g_ref.

When the constraint is already *satisfied*, `gᵀg_ref ≥ 0`, the unconstrained-optimal `α* = −gᵀg_ref/g_refᵀg_ref ≤ 0` falls outside `α ≥ 0`, so the constraint isn't active, I clamp `α = 0`, and `g̃ = g` — take the raw gradient. That's the if/else, and it dropped out of the KKT conditions rather than being stipulated:

  if gᵀg_ref ≥ 0:  g̃ = g
  else:            g̃ = g − (gᵀg_ref / g_refᵀg_ref) g_ref.

Let me read the geometry of `g̃ = g − (gᵀg_ref / g_refᵀg_ref) g_ref`, because I want to be sure it's doing the *minimal* thing and not overcorrecting. `(gᵀg_ref / g_refᵀg_ref) g_ref` is exactly the orthogonal projection of `g` onto the line spanned by `g_ref` — the component of `g` *along* `g_ref`. When that component is negative (the violating case), I'm subtracting it off, which removes precisely the part of `g` that points *against* the reference, and keeps everything else. Check the result: `g̃ᵀg_ref = gᵀg_ref − (gᵀg_ref/g_refᵀg_ref)(g_refᵀg_ref) = gᵀg_ref − gᵀg_ref = 0`. After projection the corrected gradient is exactly *orthogonal* to `g_ref` — it sits on the boundary of the feasible halfspace. Orthogonal means, to first order, the step neither raises nor lowers the average past-task loss, while staying as close to the original `g` as the constraint permits. I'm not flipping the gradient or zeroing it; I'm shaving off exactly the offending component and no more.

Let me not take any of that on faith and just run the formula on numbers. Pick a clean violating case: `g = (1, −2)`, `g_ref = (1, 1)`. Their dot is `⟨g, g_ref⟩ = 1 − 2 = −1 < 0`, so this is the case where a raw step would raise the average past loss — projection should fire. Then `corr = ⟨g, g_ref⟩/⟨g_ref, g_ref⟩ = −1/2 = −0.5`, and `g̃ = g − corr·g_ref = (1, −2) − (−0.5)(1, 1) = (1.5, −1.5)`. Three things I want to confirm. First, orthogonality: `⟨g̃, g_ref⟩ = 1.5 − 1.5 = 0` — exactly on the boundary, as the symbolic argument said. Second, that this really is a *small* correction and not throwing the step away: `‖g − g̃‖ = ‖(−0.5, −0.5)‖ = 0.707`, whereas the lazy alternative of just dropping the offending step (using `0`) would be a change of `‖g‖ = 2.236` — the projection moves `g` about a third as far, which is the "minimal fix" claim made concrete. Third, that the current task's signal mostly survives: `⟨g̃, g⟩ = 1.5 + 3.0 = 4.5` against `⟨g, g⟩ = 5.0`, and the cosine between `g̃` and `g` is `0.95`, so the corrected step still points 95% of the way toward where the current task wanted to go. That's the L2-closest formulation earning its keep — it keeps the maximum of the current task's signal subject to "do no average harm," rather than sacrificing the step wholesale the way a constraint-violation-means-skip rule would.

Now `g_ref` itself. The exact thing in the constraint is the gradient of the loss over the *entire* union memory `M`. Computing that exactly every step means a backward over all stored examples — cheaper than GEM's per-task backwards but still scaling with total memory. But I don't need it exact. `g_ref` is a *direction* summarizing the past; a stochastic estimate is fine. So draw a random minibatch from the union memory `(x_ref, y_ref) ∼ M` and let `g_ref` be that batch's gradient. One forward/backward on a fixed-size batch, regardless of how many tasks are in the memory. The cost is now genuinely flat in `t` — that's the second place the `t`-scaling could have crept back in, and sampling kills it too. The averaging across tasks comes along for free: the constraint I weakened to was the loss over the *union* memory `M = ∪_k M_k`, and the gradient of an average-over-`M` loss is the average of the per-example gradients in `M`; a uniform random minibatch from `M` has that same quantity as its expectation, so in expectation `g_ref` points along the true union-memory gradient. The estimate is noisy per step, but it's the right direction on average, and noise in the reference direction is far less damaging than the per-step QP cost it buys me out of — provided the buffer is drawn roughly uniformly across tasks, which is why the balanced per-task budget below matters.

I should be honest with myself about what I gave up versus the per-task-constraint version. Enforcing every task's constraint gives a worst-case guarantee: on the memory, no single task is allowed to get worse. Enforcing only the average gives an average guarantee: the mean memory loss won't rise, but some individual task *could* tick up as long as the average holds or falls. So the worst-case-forgetting promise is genuinely weaker. But my metric is average accuracy, and the averaged constraint is tailored to that objective. I'm trading a guarantee I don't directly need for a step rule with no QP, no matrix of per-task gradients, and only one reference-gradient measurement.

There's a quiet assumption underneath all of this I should name: the linearization. "Loss doesn't increase" became "`⟨g, g_ref⟩ ≥ 0`" only because I treated the loss as locally linear over the small step `α g̃`. For the small learning rates used here that's a fine approximation — the first-order term dominates over the step. And the memory has to be *representative* of the past tasks for `g_ref` to point the right way; that's why I keep a balanced per-task budget rather than letting one task dominate the buffer.

So how do I keep the memory? Bounded, balanced across tasks. Fix a total budget and give each task an equal slice — at the end of each task, truncate that task's dataset down to a fixed number of stored sequences (a few hundred to a thousand), and sample the reference batches from the union of all the stored slices. Bounded total memory, equal representation, drawn at random each step.

Now I have to confront the one ugly engineering reality: `g` and `g_ref` are full-parameter gradients, scattered across the `.grad` field of every parameter tensor in the network. The math wants them as single flat vectors so I can take `gᵀg_ref` as one dot product over the whole network. So I need to (a) flatten all the per-parameter gradients into one contiguous vector, (b) do the scalar arithmetic and the projection on that flat vector, and (c) write the projected vector back into the per-parameter `.grad` fields so the optimizer's `step()` applies it. The existing GEM helpers already do the two bookkeeping jobs: precompute `grad_dims = [p.numel() for p in params]` once so I know each parameter's slice; `store_grad` copies each `p.grad` into its slice of a flat buffer; `overwrite_grad` reshapes a flat buffer's slices back into each `p.grad`. The only new helper I need is the closed form itself: `corr = (gxyᵀger)/(gerᵀger); return gxy − corr·ger`.

Let me also make sure I sequence the two backward passes correctly, because they share the same `.grad` buffers. First, compute the current-task loss and `backward()` — now `.grad` holds `g`. Copy it out to the flat buffer `grad_xy` *before I clobber it*. Then `zero_grad` the network, compute the reference-batch loss, `backward()` — now `.grad` holds `g_ref`; copy it to `grad_er`. Take the dot product. If it's non-negative, write `grad_xy` back into `.grad` (restore the original current-task gradient, since the second backward overwrote it) and step. If it's negative, write the projected `grad_xy − corr·grad_er` back into `.grad` and step. Either way the optimizer steps on the right vector. And the whole memory branch only fires once there's something in the buffer — on the very first task there are no past tasks, so it's plain SGD, no reference gradient, no projection.

Let me write it as the code I'd actually ship, filling the open slot in the continual-learning harness. The memory is the bounded buffer owned by the model; `observe` is where the gradient check and possible correction live; `end_task` adds examples from the finished task so later steps can draw a reference batch.

```python
import numpy as np
import torch

from models.gem import overwrite_grad, store_grad
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(ContinualModel):
    NAME = 'agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(AGem, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        loader = dataset.train_loader
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.zero_grad()
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()
        return loss.item()
```

Let me trace the causal chain once more to be sure it holds together. Sequential fine-tuning forgets because the objective ignores the past; I confirmed the harm a step does to a past task is, to first order, `−α⟨g, g_k⟩`, so the *sign of a gradient inner product* is the forgetting diagnostic. Regularization anchors parameters but is fragile, multi-epoch-flavored, and forbids backward transfer; architectural growth has unbounded memory. So I phrased "don't raise past losses" as a hard one-sided inequality constraint on each step — which permits backward transfer and needs only a small episodic memory to measure the past-task gradients. Enforcing every task individually gives the closest-feasible projection, but it's a `P`-variable QP, reducible to a `t−1`-variable dual QP — still requiring `t−1` backward passes and a solver *per step*, which scales with the stream. Weakening "every task" to "the average task" collapses the `t−1` constraints to a single inequality against one reference gradient `g_ref`, and projection onto a single halfspace has a closed form, which I derived from the KKT conditions: take `g` if `gᵀg_ref ≥ 0`, else subtract the component of `g` along `g_ref`, leaving the corrected gradient orthogonal to `g_ref` (the minimal fix). Estimating `g_ref` from a random memory batch makes the per-step cost flat in `t` — one extra backward and two dot products, no `G` matrix, no QP. The price is a worst-case-forgetting guarantee swapped for an average-loss guarantee aligned with the metric I'm graded on. Flatten the gradients into one vector, project only when `dot_prod.item() < 0`, write the chosen vector back, step, and seed a bounded buffer at the end of each task.
