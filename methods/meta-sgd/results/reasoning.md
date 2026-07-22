Let me start from what actually hurts. I have a high-capacity network and a new task with five labeled examples, maybe one. If I do the obvious thing — random init, run SGD on those five points until the support loss is small — I fit the five points and generalize to nothing. The gradient of the support loss is a great direction for *fitting that support set*, which in the few-shot regime is exactly the thing that overfits. And I have a pile of knobs I'm supposed to set by hand with almost no data to set them from: where to start, what learning rate, how many steps, when to stop. None of that is reliable when the data is this thin. So plain learning-from-scratch is out, and the question becomes: I'm not facing one task in isolation, I have a whole distribution `p(T)` of related tasks I can meta-train on first — what is the right thing to *extract* from that distribution so that a new task takes only a few steps and doesn't overfit?

The cleanest framing I know: lift learning from data to tasks. Each task `T_i` comes with a support set and a held-out query set; I meta-train across many tasks and I'm scored, at meta-test, on brand-new held-out tasks. The unit of generalization is the task. And the protocol that makes this honest is to match meta-training to meta-testing — sample `N`-way `K`-shot episodes during training too, with a held-out query set scored each episode, so I'm never optimizing the thing I'll be measured on. Fine. The real design question is *what* the meta-learner should be. An adaptation procedure for a parametric learner `f_θ` is fully specified by three things: where it starts (the initialization), which way it moves (the update direction), and how far (the learning rate). With lots of data the defaults — random start, follow the gradient, small hand-set rate — are fine. With five examples each of those defaults is a liability. So abstractly I want to *learn* all three from the task distribution. The question is how much of the three to learn, and with what machinery, because the machinery is where everything has gone wrong before.

Let me look hard at what already exists, because the two existing answers stake out opposite corners and the tension between them is the whole problem.

The first answer keeps the inner loop trivial and meta-learns only the starting point. The inner update is plain SGD: for task `T_i`, take `θ'_i = θ − α ∇L_{T_i}(f_θ)`, one step (or a few), with `α` a scalar step size I set by hand. Then — and this is the elegant part — I don't train `θ` to fit any task; I train it so that *after* this adaptation step the *query* loss is small, across tasks:

  min over θ of Σ_i L_{T_i}(f_{θ'_i}),   with θ'_i = θ − α ∇L_{T_i}(f_θ),

and I meta-update by differentiating this through the inner step, `θ ← θ − β ∇_θ Σ_i L_{T_i}(f_{θ'_i})`. That `∇_θ` passes through `θ'_i = θ − α ∇L`, so it differentiates a gradient — a gradient-through-a-gradient, which is a Hessian-vector product in the inner loss (and there's a cheaper first-order variant that just pretends `∇L` doesn't depend on `θ`). I like a lot about this. It's architecture-agnostic, it adds no parameters beyond `θ` itself, and the same recipe works whether the inner loss is classification, regression, or a policy-gradient return. And it genuinely works: meta-learning *where to start* so that one gradient step lands somewhere good is a real, scalable idea. But stare at the inner rule. The *only* thing being meta-learned is `θ^0`. The way the learner moves on a new task is frozen: every coordinate steps along the raw gradient `∇L`, all at the one scalar rate `α`. That `α` is a hand-tuned hyperparameter, and it can need wildly different values for different problems and even different steps. So whatever the task distribution knows about *how to move* — which coordinates should move a lot and which barely, and what combined direction is actually good on a fresh task — none of it can be expressed. The update direction is locked to the gradient, the magnitude is one global number. That's the ceiling here: a great starting point, a dumb step.

The second answer goes all the way the other direction: meta-learn the *entire* update rule with a recurrent network. The slick observation is that an LSTM cell-state update,

  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t,

becomes gradient descent if I set the cell state to be the learner's parameters, `c_t = θ_t`, feed the negative gradient as the candidate, `c̃_t = −∇L_t`, let the input gate be the learning rate, `i_t = α_t`, and the forget gate `f_t = 1`. Then `θ_t = θ_{t-1} − α_t ∇L_t`, plain SGD. But now `i_t` and `f_t` are *learned* functions of the loss, the gradient, and the current parameters — `α_t = σ(W_I·[∇L_t, L_t, θ_{t-1}, i_{t-1}] + b_I)`, and a learned `f_t` that can shrink the previous parameters (a learned weight-decay). The initialization `c_0 = θ^0` is learned too. This is the right *ambition*: it learns the initialization, the direction (through the gates), and the rate, all together. So why isn't this just the answer? Because of the machinery. The LSTM has tens of thousands of learner parameters to update, so to avoid an explosion of meta-parameters they share one tiny LSTM *across all coordinates* and feed it coordinates one at a time (with a log-magnitude-and-sign preprocessing of the gradient so the scales are usable). Training is by backprop-through-time over the unrolled inner loop, which means storing every intermediate LSTM state — a memory cost on the order of `T · #states · dim(θ)`, with `T` around 8 and `#states` around 20 and a fat constant from the LSTM internals. On a convolutional learner with tens of thousands of parameters that BPTT graph simply doesn't fit, and the thing is hard to optimize on top of that. And the coordinate-sharing means every parameter is pushed around independently by the same little recurrent rule, ignoring how parameters relate. So this corner has the capacity I want and a cost I can't pay.

There are metric methods too — learn an embedding, classify a query by soft nearest-neighbor to the support embeddings, trained episodically. Clean and effective for classification, but a metric doesn't *train* a parametric learner; it reshapes distances around a non-parametric comparison. It doesn't give me an adapted network, and it doesn't carry over to regression or control in the same shape. So it's not a contender for "learn the optimizer."

So here's the real tension. One corner is simple, scalable, architecture-agnostic — and meta-learns *only the starting point*, leaving a rigid inner step (raw gradient, one hand-set scalar rate). The other corner meta-learns *everything about the step* — and the only reason it's unscalable and hard to train is the *recurrent network* it uses to do so. The recurrence is what costs the BPTT memory; the recurrence is what's hard to optimize; the recurrence is the per-step, per-coordinate machine that doesn't fit. But the recurrence was never the *goal*. The goal was: learn the initialization, the update direction, and the learning rate from the task distribution. I need the smallest object I can fold into the gradient step that still buys me a learned direction and a learned, per-coordinate rate, and that I can train by ordinary backprop with no recurrence at all.

Let me think about what "richer than a scalar rate" even means, concretely, by sitting between the two corners. The simple corner's step is `−α ∇L` with `α` a scalar. The most general linear thing I could put in front of the gradient is a matrix: `−A ∇L`, where `A` is learned. A full matrix `A` is the natural "different direction" object — it can rotate the step arbitrarily off the gradient and rescale every coordinate. But `A` is `dim(θ) × dim(θ)`. That's `|θ|²` parameters to store and meta-learn, and worse, if I wanted it to change step-to-step I'd be carrying a matrix *per step* — which is exactly the kind of per-step, per-coordinate cost that just sank the LSTM route. So a full matrix is out for the same reason the LSTM was out: it doesn't scale. I'm being pushed toward the cheapest preconditioner that still does something a scalar can't: the *diagonal*. Let `A = diag(α)` with `α` a vector the size of `θ`. Then the step is

  −α ∘ ∇L,

an elementwise (Hadamard) product of a learned vector `α` with the gradient. I want to be sure this actually buys both things I claimed I needed, because "diagonal" sounds like nothing more than per-coordinate rates and I dismissed those a moment ago as not a real direction change. Per-coordinate rate is immediate — entry `α_j` is coordinate `j`'s own learning rate, so different coordinates can move at completely different scales, which a single scalar can never do. The direction is the part I'm unsure of, so let me actually compute it rather than wave. Take a gradient `g = (3, 1)` and a non-uniform `α = (0.5, 2.0)`. The step is `α ∘ g = (1.5, 2.0)`. The raw gradient points mostly along the first axis (`3` vs `1`); the rescaled step points *more* along the second (`2.0 > 1.5`) because the second coordinate's rate is four times the first's. The cosine between them is `(1.5·3 + 2.0·1) / (‖(1.5,2.0)‖·‖(3,1)‖) = 6.5 / (2.5·3.162) = 0.822`, i.e. about 34.7° apart. So `α ∘ g` genuinely points in a different direction than `g`. Now the control: set `α = (0.7, 0.7)`, uniform. Then `α ∘ g = (2.1, 0.7) = 0.7·g`, cosine `= 1`, exactly parallel — the rescaling collapses to a pure scalar rescale and changes nothing about orientation. So the tilt is caused *precisely* by the non-uniformity of `α`: equal entries reproduce the scalar step, unequal entries rotate off it. (And a negative entry, e.g. `α = (0.5, −1)`, gives `(1.5, −1)`, cosine `0.61` — it can even pull a coordinate the *opposite* way, which a positive scalar rate never can.) The general statement I can now stand behind: for a nonzero gradient with at least two active coordinates, `α ∘ ∇L` stays parallel to `∇L` only when the active entries of `α` are all equal; otherwise it tilts. So a diagonal already gives me an off-gradient update direction, not just per-coordinate rates. The norm of `α ∘ ∇L` is the effective step size and its orientation is the effective update direction, both rolled into the one vector `α` — both ingredients the simple corner couldn't reach, and I got them without a matrix and without a recurrence, since `α` is just a tensor the same shape as `θ`.

Now, is `α` something I hand-design or something I learn? The hand-designed adaptive optimizers — AdaGrad, RMSProp, Adam, AdaDelta — also put a per-coordinate rescaling on the gradient, and they set it from the *gradient history* by a fixed formula (divide by the running `L2` norm of past gradients, roughly). That tells me the *shape* — per-coordinate gradient rescaling — is the right shape; people have found it works. But their machinery needs a history to accumulate, and I don't have one: few-shot, and I want adaptation in *one* step. There's no long stream of gradients to take a running norm over. So I can't get `α` from history. What I *do* have is the task distribution. So instead of computing `α` from a gradient history with a hand-set rule, I'll *learn* `α` across tasks — let it be a free vector of meta-parameters that meta-training tunes so that, on a new task, one step of `−α ∘ ∇L` lands somewhere that generalizes. `α` becomes a learned encoding of "for this family of tasks, how far and which combined way should each coordinate move when the support gradient points this way." That's the same ambition as the LSTM's learned step, distilled into a single static vector with no recurrence to pay for.

And `α` is differentiable — that's the whole point of choosing it this shape. So I fold it straight into the same outer objective the simple corner used, except now I meta-learn `θ` *and* `α` together:

  θ' = θ − α ∘ ∇L_T(θ),
  min over θ, α of  E_{T ~ p(T)} [ L_test(T)(θ − α ∘ ∇L_train(T)(θ)) ].

The query loss after adaptation is differentiable in both `θ` and `α`. For one task, write `g = ∇_θ L_train(θ)`, `H = ∇^2_θ L_train(θ)`, `θ' = θ − α ∘ g`, and `v = ∇_{θ'} L_test(θ')`. Differentiating `L_test(θ')` through `θ' = θ − α ∘ g`: `α` enters only through the `−α ∘ g` term and linearly, so `∂θ'_j/∂α_j = −g_j` and the chain rule gives `∂L_test/∂α = v ∘ (∂θ'/∂α) = −v ∘ g`. For `θ`, `θ'` depends on `θ` both directly (the leading `θ`) and through `g(θ)`, whose derivative is `H`; so `∂θ'/∂θ = I − diag(α) H` and `∂L_test/∂θ = (I − diag(α) H)^T v`. So:

  ∂L_test(θ')/∂α = − v ∘ g,
  ∂L_test(θ')/∂θ = (I − diag(α) H)^T v.

I don't trust hand-derived chain rules through a gradient step until I've checked them, because the `diag(α)H` term is exactly where a sign or a transpose slips. So I'll pin both lines against autograd on a tiny quadratic task where `H` is constant and nonzero: `L_train(θ) = ½ θᵀAθ + bᵀθ` (so `g = Aθ + b`, `H = A`) and a *different* quadratic `L_test(θ) = ½ θᵀCθ + eᵀθ`, in `d = 3` with random SPD `A, C` and random `b, e, θ, α`. I form `θ' = θ − α ∘ g` with the graph retained, take autograd's `∂L_test/∂α` and `∂L_test/∂θ`, and compare to my closed forms `−v ∘ g` and `(I − diag(α)A)ᵀv` with `v = ∇_{θ'}L_test`. The max absolute discrepancies come out `0.0` for the `α` line and `4.4e-16` for the `θ` line — machine zero, so both derivations are right, transpose and sign included. The `θ` line is the same through-the-inner-gradient path as before: the `diag(α) H` term is a Hessian-vector product, exactly the cost of asking how the post-adaptation query loss changes when I perturb the pre-adaptation initialization. The `α` line is cleaner because `α` enters `θ'` linearly through the elementwise product; if the first-order approximation is needed, I detach `g`, keep the identity path through `θ`, and drop the `diag(α)H` term. I can optimize the whole thing by ordinary SGD over batches of tasks; meta-update `(θ, α) ← (θ, α) − β ∇_{(θ,α)} Σ_i L_test(T_i)(θ'_i)`. No BPTT through an LSTM, no stored recurrent states, no coordinate-sharing hack — just two sets of meta-parameters and backprop through one (or a few) inner steps. The thing that made the LSTM route unscalable is simply gone, because I replaced "a recurrent network that emits the step" with "a learned vector that scales the gradient."

Let me check that this is the *right* generalization and not a third unrelated gadget, by seeing whether the methods I came from actually line up with it. Algebraically, freeze `α` to a single constant value `c` in every coordinate and don't learn it: then `α ∘ ∇L = c ∇L` and `θ' = θ − c ∇L`, which is the simple corner — plain SGD with a scalar rate, meta-learning only `θ`. I'd rather not leave that as algebra on paper, since the implementation re-routes parameter tensors in a slightly fiddly way and I want to know the *code* reduces, not just the formula. So I run the inner step on a small net (a `4→8→3` MLP) two ways from the same cloned weights and the same six support points: once with my optimizer's `α` left uniform at `inner_lr`, once with a hand-written `p ← p − inner_lr · g` scalar-SGD step. The adapted parameters agree to a max difference of `0.0` across every tensor — the uniform-`α` path *is* the scalar inner loop, exactly, through the real `adapt`. Then I reset `α` to random per-coordinate values and rerun: now the adapted parameters differ from the scalar step by up to `0.045`. So the generalization is genuine, not cosmetic — uniform `α` collapses onto the scalar method and non-uniform `α` moves the learner somewhere a single rate cannot reach. That places the scalar-rate method at the point in my space where `α` is uniform and fixed, with the non-uniform learned `α` a strict enlargement of it. And the LSTM corner is the expensive way to pursue the same three learned ingredients: instead of a static vector, a recurrent network emits per-coordinate gates at each step and pays BPTT for the privilege. So my method sits between the two corners: more capacity than the scalar-rate method (a learned, non-uniform `α` that changes the direction), and far cheaper than the recurrent one (a static vector, no BPTT-over-LSTM).

I should pin down whether the *diagonal* really is the right rank and not a cop-out, since I waved at it. A per-layer scalar (one rate per layer rather than per parameter) is cheaper still, but it can only rescale a whole layer's gradient uniformly — within a layer it's a single shared entry, which by the parallelism test above means *no* tilt within that layer, and within-layer is exactly where the interesting per-coordinate structure lives. The full matrix is the other extreme and the *most* expressive, but it costs `|θ|²` and, if it varied per step, reintroduces the per-step-per-coordinate cost I'm trying to escape. The diagonal is the middle that is (i) linear in `dim(θ)` to store and meta-learn, (ii) capable of a distinct rate for every single coordinate, and (iii) capable of an off-gradient direction — the off-gradient part being the thing I just verified numerically that the scalar and the per-layer scalar both lack. It's the cheapest object that clears the bar the scalar couldn't, so among these three it's the one I'll take.

Now a couple of decisions the construction forces. How do I initialize `α`? At the start of meta-training I have no reason to favor any coordinate over another, and I'd like to begin from behavior I trust — which is the simple corner, uniform small steps. So initialize every entry of `α` to a common small constant (around the scalar rate I'd have hand-picked), and let meta-training pull the entries apart as it discovers which coordinates want bigger or differently-signed steps. Concretely, set `α` to a constant tensor the shape of each parameter — something like `ones_like(p) * inner_lr` — or draw the common starting value from a small range; either way it begins MAML-like and differentiates from there. How many inner steps? The goal is as few as possible, and the whole point of learning `α` is that one step can carry much more structure than the rigid scalar rule — `α` can absorb into a single update what the fixed rule would otherwise try to approximate with several careful steps and a well-tuned scalar. If I do want several inner steps, I just iterate the same rule, reusing the one learned `α` at each step (it's a property of the optimizer, not of the step index), carrying the differentiable graph across the steps so the outer gradient still flows. And the outer optimizer: any SGD variant on the meta-objective; Adam is the practical choice for the meta-update. I keep the second-order path (differentiate through the inner gradient) when I can afford it, since that's what tells `θ` and `α` how the post-adaptation loss responds to the pre-adaptation move; the first-order approximation that ignores the inner gradient's dependence on the meta-parameters is the cheap fallback.

So the picture is settled, and it came straight out of the tension: the scalar-rate method gave me a scalable, architecture-agnostic frame that learns only where to start; the recurrent method showed that learning the init, direction, and rate together is the real prize but made me pay for a recurrence I don't need; and the resolution is to keep that prize and drop the recurrence by replacing "a network that emits the step" with the cheapest learnable preconditioner that still rotates and rescales the gradient — a diagonal, i.e. a single learned vector `α`, meta-trained jointly with `θ`, with the step `θ' = θ − α ∘ ∇L`. Per-parameter rate and a learned off-gradient direction, linear memory, ordinary backprop, with the scalar-rate method recovered when `α` is fixed and uniform.

Let me write it as the inner-loop optimizer it actually is, using the harness's generic optimizer object: create the learnable per-parameter `α` as optimizer state, implement the differentiable `θ ← θ − α ∘ ∇L` update in the empty update slot, and hand `α` to the outer loop as meta-parameters:

```python
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

INNER_LR = 0.5  # initial value for every entry of alpha


class InnerLoopOptimizer(nn.Module):
    """Inner-loop rule: a learned per-parameter learning-rate VECTOR alpha,
    meta-trained jointly with the initialization. Inner step: theta <- theta - alpha (.) grad."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR,
                 first_order: bool = False):
        super().__init__()
        self.inner_lr = inner_lr
        self.first_order = first_order
        # alpha: one learnable tensor per parameter, same shape, initialized uniform
        # (= the scalar-rate start). Each entry is that coordinate's own learnable rate;
        # because alpha is non-uniform after training, alpha (.) grad also tilts the
        # update OFF the raw gradient direction. These are meta-parameters.
        self.lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * inner_lr)
            for p in model.parameters()
        ])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE; keep everything differentiable so the outer loop can
        # backprop through the inner step into both theta and alpha.
        model.train()
        second_order = not self.first_order
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            # second_order=True keeps the Hessian-vector path through the inner
            # gradient; first_order=True is the FOMAML-style approximation.
            grads = torch.autograd.grad(
                loss,
                model.parameters(),
                retain_graph=second_order,
                create_graph=second_order,
            )
            model = self._apply_update(model, grads)
        return model

    def _apply_update(self, model: nn.Module, grads: List[Tensor]) -> nn.Module:
        # Attach the gradient and its learned rate, then re-route parameter tensors
        # as p - lr * grad so the outer graph remains differentiable.
        for p, lr, g in zip(model.parameters(), self.lrs, grads):
            p.grad = g
            p._lr = lr

        def reroute(module: nn.Module) -> nn.Module:
            for name in module._parameters:
                p = module._parameters[name]
                if p is not None and p.grad is not None:
                    module._parameters[name] = p - p._lr * p.grad
                    p.grad = None
                    p._lr = None

            for name in module._buffers:
                buff = module._buffers[name]
                if buff is not None and buff.grad is not None and getattr(buff, "_lr", None) is not None:
                    module._buffers[name] = buff - buff._lr * buff.grad
                    buff.grad = None
                    buff._lr = None

            for name in module._modules:
                module._modules[name] = reroute(module._modules[name])
            return module

        return reroute(model)

    def meta_parameters(self) -> List[Tensor]:
        # hand alpha to the outer loop so it is meta-optimized alongside theta.
        return list(self.lrs.parameters())
```
