Let me start from what actually bugs me about the two-loop meta-learner I'm running. The setup is clean and it works: an outer loop learns an initialization $\theta$ across a distribution of tasks, and for each new task an inner loop takes a few gradient steps on its support set, $\theta^{(b)}_m = \theta^{(b)}_{m-1} - \alpha \nabla \mathcal{L}^{\text{sup}}(f_{\theta^{(b)}_{m-1}})$, and I evaluate the adapted model on held-out query points, summing that query loss into a meta-loss and stepping $\theta$ to minimize it. The thing that nags me is the cost and the opacity, and they're the same complaint from two sides. The cost: that inner loop runs *per task, inside every meta-iteration*, over every single parameter of the network, and because the adapted parameters depend on $\theta$ both directly and through the inner gradient, differentiating the meta-loss through the inner step drags in a second-order term — the meta-gradient is $(I - \alpha \nabla^2_\theta \mathcal{L}^{\text{sup}})^\top \nabla_{\theta'} \mathcal{L}^{\text{qry}}$, a Hessian-vector product over the *whole* network, one extra backward pass per task on top of the inner forward/backward. The opacity: I can't actually say what that inner loop is *for*. There are two stories I could tell and I have no idea which is true. Story one, rapid learning: the meta-initialization is a cleverly conditioned launch point, and the few inner steps make big, efficient changes to the network's internal representations, genuinely re-learning each task. Story two, feature reuse: the meta-initialization already holds good, general features, and the inner loop barely moves them — adaptation is almost a no-op on most of the network. These predict completely different things about where the work happens, and I'm paying full price for an inner loop without knowing whether it's doing heavy lifting or coasting.

So before I try to make it cheaper, I should figure out which story is true, because the answer dictates what I'm even allowed to cut. And the first thing to get straight is that the network isn't homogeneous — there's a sharp structural difference between the last layer and everything before it, and it matters here more than anywhere. Think about an $N$-way episode. The $N$ output neurons get mapped to an arbitrary, task-specific set of classes. In one task the five outputs mean (dog, cat, frog, cupcake, phone); in the next, drawn from the same held-out pool, they mean (airplane, frog, boat, car, pumpkin). There is no fixed last layer that can be right for both at once — the assignment from feature space to *these particular five output slots* is a fresh, essentially random relabeling per task. So the final layer, the head, has to be free to move per task no matter what story is true; that's forced by the problem, not a design decision. The earlier layers, the body, compute features that are class-agnostic — edges, textures, parts — and those *could* be shared across tasks without moving. Which means the rapid-learning-versus-feature-reuse question isn't really about the head at all. The head trivially must adapt. The whole question lives in the body: does the inner loop substantively change the body's representations per task (rapid learning), or does it leave them essentially fixed (feature reuse)? That's the thing to settle.

How would I tell? I want to measure whether the body's *function* changes during the inner loop. Two ways come to mind. The blunt one: just don't let the body adapt and see if anything breaks. After training the model normally with both loops, at test time freeze a contiguous block of body layers — give them no inner-loop update, force them to reuse the meta-initialization's features — and read off accuracy. If rapid learning is the story, freezing the body should hurt: those layers were supposed to re-learn per task and now they can't. If feature reuse is the story, freezing should barely matter, because the inner loop wasn't changing them anyway. And it's well documented that this is exactly what happens — freezing layer 1 of a trained MiniImageNet model leaves 5-way-1-shot accuracy at $46.5$ against $46.9$ with no freezing; freezing layers 1–2, $46.4$; 1–3, $46.3$; freezing *all four* convolutional layers, everything except the head, $46.3 \pm 0.4$. On 5-shot it slips from $63.1$ to $61.0$. Removing the inner-loop adaptation of the entire body costs essentially nothing. That already tilts me hard toward feature reuse, but freezing is indirect — it tells me the body *can* be frozen at test time, not that the inner loop wasn't moving it.

So the sharper measurement: directly compare the body's representation before and after the inner loop. I have CCA and CKA sitting around for exactly this — take a layer's activation vectors over a set of inputs before adaptation, the same after adaptation, and get a similarity score in $[0,1]$, $1$ for identical functions. Applied per layer, pre versus post inner loop, the established picture is unambiguous: the body conv layers sit at CCA above $0.9$, CKA near $1$ — the inner loop induces almost no functional change in the body — while the head sits below $0.5$, moving a lot, as it must. And the Euclidean weight movement tells the same story even more starkly: the average distance between initialization and finetuned weights is tiny for every layer except the last, *even though the body layers have far more parameters than the head*. More parameters, less movement. The inner loop is, functionally, a near-no-op on the body. And this isn't a quirk of the converged model — the same pattern holds from early in training, by ten thousand iterations in. Feature reuse, not rapid learning. The body's good features come from the *outer* loop accumulating across tasks; the inner loop, on the body, is along for the ride.

If the inner loop leaves the body's function essentially unchanged, then computing inner-loop updates for the body is computing a correction that rounds to zero and then paying full second-order price to backpropagate through it. The freezing experiment already showed I can drop body adaptation *at test time* with no accuracy loss. The obvious next thought: why run it at *training* time either? Here I have to be careful, because freezing-at-test and not-adapting-at-train are different interventions, and I could talk myself into a wrong one. At test time, freezing the body of an already-trained model just says "don't bother adapting features that won't move." At training time, if I stop adapting the body in the inner loop, I'm changing the *meta-objective* — the body parameters would now be learned purely by the outer loop, never task-adapted. Would the body still learn good features without ever being adapted per task during training? The timing result is what reassures me: the high body-similarity and the freezing-robustness are present from ten thousand iterations in, meaning the inner loop is inert on the body *throughout* training, not just at the end. If the inner loop isn't doing anything to the body's representations during training, then removing it during training shouldn't change what the body learns either. So I can cut the body's inner loop at *both* training and testing, and almost no inner loop survives — only the head, which genuinely has to adapt for the per-task class alignment, keeps its inner-loop updates.

Let me write that down as the adaptation rule. Partition $\theta = (\theta_1, \dots, \theta_l)$, body $\theta_1, \dots, \theta_{l-1}$ and head $\theta_l$. The inner loop updates only the head:
$$\theta^{(b)}_m = \Big(\theta_1, \dots, \theta_{l-1},\; (\theta_l)^{(b)}_{m-1} - \alpha \nabla_{(\theta_l)^{(b)}_{m-1}} \mathcal{L}^{\text{sup}}_{S_b}(f_{\theta^{(b)}_{m-1}})\Big).$$
The body components are pinned at the meta-initialization across all inner steps; only the last component descends the support loss. The outer loop is untouched — same meta-loss, the query loss at the adapted parameters summed over the task batch, same $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{meta}}$. And the body parameters are still *learned* — they're just learned only by the outer loop now, never task-adapted. That's the whole move: keep the head's inner loop, delete the body's, train and test alike.

I want to count what this saves and, more importantly, make sure I'm not quietly throwing away the second-order signal that made MAML strong, because that's the trap. Per task, the inner loop now computes a gradient with respect to the head only — a handful of parameters — instead of the whole net, and the backward pass through that inner step only has to carry curvature for the head. At inference, where I'd take ten inner steps, I'm doing ten head-only updates instead of ten full-network updates and the body is a single forward pass — that's where the biggest speedup should come from, since adaptation collapses to forwarding the support set through a frozen body once and then iterating on a tiny linear head. I'd expect roughly a couple-fold speedup at training and a larger one at inference, and I'd want to confirm that on the standard benchmarks.

Have I just reinvented first-order MAML? First-order MAML keeps the *full* inner loop — it adapts every parameter — but *drops the Hessian*, using the query gradient at the adapted point as if it were the gradient at $\theta$. What I'm doing is the opposite axis: I keep the second-order machinery but I shrink *which parameters* get an inner loop down to the head. These are different cuts, and I should check that mine keeps the curvature. Take the smallest example that still has a body and a head: a two-layer linear net $\hat{y}(x;\theta) = \theta_2(\theta_1 x)$, where $\theta_2$ is the head and $\theta_1$ is the body, one-shot regression, one inner step. In full MAML both parameters get the inner update:
$$\theta_1^{(t)} = \theta_1 - \frac{\partial L(\hat{y}(x_1^{(t)};\theta), y_1^{(t)})}{\partial \theta_1}, \qquad \theta_2^{(t)} = \theta_2 - \frac{\partial L(\hat{y}(x_1^{(t)};\theta), y_1^{(t)})}{\partial \theta_2},$$
using the support point $(x_1^{(t)}, y_1^{(t)})$, and the adapted parameter vector is $\theta^{(t)}_{\text{MAML}} = [\theta_1^{(t)}, \theta_2^{(t)}]$. In my rule the body doesn't move, so $\theta^{(t)}_{\text{ANIL}} = [\theta_1, \theta_2^{(t)}]$ — same head update, body pinned. Now the outer update for the body parameter $\theta_1$ uses the query point $(x_2^{(t)}, y_2^{(t)})$:
$$\theta_1 \leftarrow \theta_1 - \sum_t \frac{\partial L(\hat{y}(x_2^{(t)}; \theta^{(t)}), y_2^{(t)})}{\partial \theta_1}.$$
I have to differentiate $\hat{y}(x_2^{(t)}; \theta^{(t)})$ with respect to $\theta_1$. For full MAML the prediction at the query point is
$$\hat{y}(x_2^{(t)}; \theta^{(t)}_{\text{MAML}}) = \Big[\theta_2 - \tfrac{\partial L(\hat{y}(x_1^{(t)};\theta), y_1^{(t)})}{\partial \theta_2}\Big] \cdot \Big[\theta_1 - \tfrac{\partial L(\hat{y}(x_1^{(t)};\theta), y_1^{(t)})}{\partial \theta_1}\Big] \cdot x_2,$$
and both bracketed factors contain $\theta_1$ — the second one explicitly, and the first one because the head's inner gradient $\partial L^{\text{sup}}/\partial \theta_2$ depends on $\theta_1$ through the forward pass on $x_1$. Differentiating either bracket with respect to $\theta_1$ pulls down a second derivative; that's the second-order content. For my rule the prediction is
$$\hat{y}(x_2^{(t)}; \theta^{(t)}_{\text{ANIL}}) = \Big[\theta_2 - \tfrac{\partial L(\hat{y}(x_1^{(t)};\theta), y_1^{(t)})}{\partial \theta_2}\Big] \cdot \theta_1 \cdot x_2.$$
The second bracket has collapsed to a bare $\theta_1$ — that's the body not adapting. But look at the first bracket: it's still there, and it still contains $\theta_1$, because the head's inner-loop gradient $\partial L^{\text{sup}}/\partial \theta_2$ is computed on a forward pass through the body $\theta_1$. So when I differentiate the query loss with respect to $\theta_1$, I still differentiate *through* that inner head update, and a second-order term survives. This is the thing I almost missed: removing the body's inner loop does *not* make this first-order. The curvature that flows through the *head's* inner step is retained. So my rule is not first-order MAML restricted to the head — it keeps the second-order signal through the head while spending almost nothing, whereas first-order MAML spends on the full inner loop and keeps no curvature at all. That difference is exactly why I'd expect mine to match full MAML's accuracy where first-order MAML sometimes lags: I kept the part of the second-order information that's attached to the only thing still adapting.

That tells me precisely how to implement the inner step so the curvature isn't lost. I compute the support loss, take its gradient with respect to *only the head parameters*, and — critically — keep that gradient in the autodiff graph (`create_graph=True`), so the later outer backward pass can differentiate through it and recover the surviving second-order term. Then I apply the update: head parameters move by $-\alpha g$, body parameters move by zero. The differentiable in-place update primitive I have, `update_module`, wants a full-length list of per-parameter updates — one entry per parameter, in order — and swaps each $p$ for $p + u$ while preserving differentiability. So I build the updates list with a zero tensor for every body parameter and $-\alpha g$ for every head parameter, and hand it over. There's a sharp implementation gotcha hiding here: `update_module` doesn't mutate the parameters in place, it *replaces* the parameter objects with new ones ($p + u$ is a fresh tensor) every step. That means if I grab references to the head parameters once before the loop, after the first step those references are stale — they point at the old objects that are no longer in the module — and on the next step my gradient would be taken with respect to detached tensors and every update would silently come out zero. So I have to re-identify the head parameters by name *inside* each inner step, from the current module, right before I take the gradient. Get that wrong and the method looks like it's running while doing nothing.

How do I know which parameters are the head? By name. In the standard four-conv-block backbone the body is the convolutional feature extractor and the head is the final linear classifier, so I scan the named parameters and mark the ones whose name contains the classifier/head tag as the head set; everything else is body. I record those names once at construction (the names are stable across the `update_module` swaps even though the tensor objects aren't), and inside each step I re-collect the current parameter objects matching those names. The adaptation rule itself carries no learnable state of its own — unlike a per-parameter-learning-rate scheme, I'm not adding any meta-parameters; the only things the outer loop optimizes are the model's own weights. So the rule exposes an empty list of meta-parameters.

Let me write it as the inner-loop optimizer that drops into the harness, head-only inner updates with the second-order path through the head kept:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

import learn2learn as l2l

INNER_LR = 0.5


class InnerLoopOptimizer:
    """Inner loop adapts only the final classification head; the body is
    frozen during adaptation and reuses the meta-initialization's features.
    Body weights are still learned, but only by the outer loop."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Head = the final linear classifier; body = everything else.
        # Record head parameter *names* (stable across update_module swaps).
        self._head_param_names = set()
        for name, _ in model.named_parameters():
            if "classifier" in name:
                self._head_param_names.add(name)

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            # Re-identify head params each step: update_module replaces the
            # parameter objects (new ids), so stale references from a prior
            # step would make every update silently zero.
            head_params, head_ids = [], set()
            for name, p in model.named_parameters():
                if name in self._head_param_names:
                    head_params.append(p)
                    head_ids.add(id(p))

            loss = F.cross_entropy(model(support_x), support_y)
            # Grad w.r.t. the head only; create_graph=True keeps the inner
            # step in the meta-graph so the surviving second-order term that
            # flows through the head's update reaches the outer backward pass.
            grads = torch.autograd.grad(loss, head_params, create_graph=True)
            grad_map = {id(p): g for p, g in zip(head_params, grads)}

            # Full-length update list: -alpha*g for head params, zero for body.
            updates = [
                -self.inner_lr * grad_map[id(p)] if id(p) in head_ids
                else torch.zeros_like(p)
                for p in model.parameters()
            ]
            l2l.update_module(model, updates=updates)  # differentiable p <- p + u
        return model

    def meta_parameters(self) -> List[Tensor]:
        # No learnable optimizer state of its own.
        return []
```

Having pushed the inner loop down to the head, I can't help asking how far this logic goes, because if it's really feature reuse all the way then maybe the head isn't load-bearing at *test* time either. At test time the head exists only to map frozen features to the five task-specific output slots. But there's a parameter-free way to do that mapping that I already have lying around in the metric-methods line: don't train a head at all, just compare. Drop the head entirely at test, push the support set through the frozen body to get penultimate representations, and for a query example compute its cosine similarity to each support representation, then weight the support labels by those similarities — exactly the Matching-Networks rule $\hat{y} = \sum_j a(g(\hat{x}), g(x_j)) y_j$ with $a$ a softmax over cosine similarities. No inner loop anywhere — no inner loop at all, at test. If the few-shot performance is really determined by feature quality, this should match the head-adapting versions, and it's worth checking that it does, because if it does it nails down that the body's features, reused as-is, are the whole game and the head's test-time role is incidental.

But that's a test-time move and I should be honest about a limit it doesn't cross: the head can be dropped at *test*, not at *training*. The head's job during training is to provide the per-task target that forces the body to learn task-discriminative features in the first place — the task specificity at training is what teaches the body good features. Strip the per-task head during training (train one shared classifier over all tasks at once) and the body learns markedly worse features; that's the price of removing task specificity from the training objective, and it's why I keep the head's inner loop during training even though I can throw the head away at test. So the asymmetry is real and it's the right one: cut the body's inner loop everywhere, keep the head's inner loop at train (it disciplines the body) and at test (it aligns to each task's labels), and optionally even cut the head at test with a cosine rule once the features are good.

For the standalone form, away from the harness's clone-and-update plumbing, the same logic is a fast-weights dictionary where only the head entries get gradient steps:

```python
def adapt_head_only(model, loss_fn, x_s, y_s, alpha, n_steps,
                    head_keys, first_order=False):
    # fast_weights: body keys pinned at meta-init; only head keys descend.
    fast_weights = model.init_weights()  # dict of name -> meta-init parameter
    for _ in range(n_steps):
        pred = model.functional_forward(x_s, fast_weights)
        loss = loss_fn(pred, y_s)
        head_vals = [fast_weights[k] for k in head_keys]
        # Grad w.r.t. head only; keep the graph for the (surviving) 2nd-order term.
        grads = torch.autograd.grad(loss, head_vals, create_graph=not first_order)
        g = dict(zip(head_keys, grads))
        fast_weights = {
            k: (v - alpha * (g[k].detach() if first_order else g[k])) if k in g else v
            for k, v in fast_weights.items()
        }  # body keys re-bound unchanged; head keys take the step
    return fast_weights
```

The causal chain, start to finish: the two-loop meta-learner pays to run a full-network inner loop with second-order backprop per task, and I couldn't say whether that inner loop was doing heavy lifting on the body or coasting — so I leaned on what's documented about the existing system, where freezing body layers leaves accuracy essentially unchanged even with all four conv layers frozen and representations pre- and post-adaptation are highly similar in the body (CCA above $0.9$) but not the head (below $0.5$), and the answer was feature reuse: the inner loop barely moves the body's function, from early in training onward. That made the body's inner loop a correction that rounds to zero and is expensive to differentiate, so I removed it at both training and testing, keeping the inner loop only for the head, which must adapt because each task relabels the output neurons. Checking the smallest two-layer example showed the cut keeps the second-order term that flows through the head's inner update — so this is not first-order MAML, and that retained curvature is why it should match full MAML's accuracy at a fraction of the per-task cost — which dictated the implementation: gradient with respect to the head only, kept in the graph, head parameters stepped and body parameters zeroed via the differentiable update, re-identifying the head each step because the update replaces parameter objects. And pushing the same feature-reuse logic one step further, at test time the head can be replaced by a parameter-free cosine-similarity rule over the frozen features entirely, while at training the head must stay to teach the body its features.
