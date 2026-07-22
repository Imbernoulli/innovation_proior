Let me start from what I actually want and what I'm allowed to use. I have a trained, frozen feedforward classifier `F: R^M -> R^N`, white-box — I know the weights, I can run it forward and I can backpropagate through it. I hold a clean input `X` it labels correctly as `Y`, and I want a perturbed `X*` it labels as some chosen target `Y* != Y`. The clean statement of the goal is argmin over δ of ‖δ‖ subject to F(X+δ)=Y*. Two things make this hard. The constraint F(X+δ)=Y* has no inverse I can write down — F is a stack of non-linear layers, non-convex, so I can't just solve for δ. And the choice of norm ‖·‖ isn't innocent; it decides what kind of perturbation I'm even looking for. The pressure I care about is the *number of features I touch*. I'd rather flip the label by setting a handful of pixels — each possibly all the way to its extreme — than by smearing an invisible nudge across all 784 of them. That's an L0 objective: count the nonzero entries of δ, don't measure their size.

So let me look hard at what's already on the table, because I want to understand precisely where each existing attack stalls for *this* objective. Szegedy's method takes the program literally: minimize ‖r‖₂ subject to f(x+r)=l and x+r in [0,1]^m, approximated by line-searching a constant c and running box-constrained L-BFGS on c·‖r‖ + loss_f(x+r,l) until the minimizer actually lands in the target class. It works, it finds imperceptible examples, it's where transferability was first seen. But stare at what it optimizes: an L2 penalty on r. The minimizer of an L2 penalty is *dense* — L2 has no incentive to zero anything out; it would rather make a thousand coordinates each a little bit nonzero than make ten coordinates large, because spreading the same squared budget thin lowers ‖r‖₂. So this method, by its very objective, produces perturbations spread over essentially the whole image. It has no term that counts touched features. And operationally it runs an iterative second-order optimization *per example*, which is heavy. Wall, on both counts: wrong objective for sparsity, expensive.

Goodfellow's fast gradient sign method is the other pole. Linearize the training loss J around the input and step η = ε·sign(∇_x J(θ,x,y)). One backprop, closed form, fast, and there's a beautiful argument for why it's the right step under an L∞ budget: for a linear unit, w^T(x+η) = w^T x + w^T η, and to maximize w^T η subject to ‖η‖_∞ ≤ ε you put η = ε·sign(w), giving w^T η = ε·‖w‖₁ = ε·m·n, which grows linearly with the input dimension n. That's the deep insight — many tiny coordinated changes sum to a large logit shift. But notice what that means for *my* objective: the sign step writes a nonzero perturbation into *every single coordinate*. It is, by construction, the densest possible perturbation under the L∞ ball — exactly the opposite of what L0 wants. And it's organized around the loss gradient and the untargeted "raise the loss" direction: it's great at pushing a sample off its true class, but it's not built to drive the network into one *specific* target class while keeping the support small. Wall: maximally dense, and aimed at the loss rather than at a chosen output.

There's a third thing worth pulling apart, Simonyan's class-saliency idea, because it's the closest to "which pixels matter." Take a class score S_c, backpropagate it to the input, and you get a vector ∂S_c/∂I; rearrange it to image shape and take the per-pixel magnitude, M_ij = max_c |(∂S_c/∂I)_{ijc}|. That highlights pixels that matter to class c. But it's a *visualization*: single class, magnitude-only — it throws away the sign of the derivative — and it has no sense of "target versus everyone else." It tells me where a class lives; it doesn't tell me which pixels to push, in which direction, to *convert* one class into another. Still, it plants a seed: the derivative of an output score with respect to an *input feature* is a per-feature, per-class quantity, and that's a much richer object than one loss gradient.

Now let me notice the thing all three share, because that's the crack I want to pry open. Szegedy and Goodfellow both go output → input through the gradient of the *loss*: they compress the whole output vector and the label into one scalar J, take ∂J/∂x, and that collapses everything I might want to know into a single direction over all pixels. Simonyan differentiates one class score, which is richer but still one class and sign-blind. What I actually have available, and nobody is using directly, is the full sensitivity of *every output* to *every input feature*: the matrix

  ∂F_j(X)/∂x_i  for i in 1..M, j in 1..N.

This is the Jacobian of the network's output map — call it the forward derivative, because it answers "if I push input feature x_i, how does each output F_j move?", which is a forward-direction question (input → output), whereas training backprop answers "how does the cost move if I push a weight?". Two deliberate differences from training: I differentiate the network's *outputs*, not its cost; and with respect to the *input features*, not the weights. The reason this is the right object: it's per-feature *and* per-class, and it keeps the sign. With it I can ask, for each individual pixel, the targeted question — does raising this pixel raise my target class, and what does it do to the others — which is exactly the trade-off a targeted attack lives or dies on, and exactly what a single loss gradient or a sign-blind magnitude map cannot express.

Let me make sure I can actually compute this matrix from white-box access, end to end, because if I can't it's a fantasy. Take one entry ∂F_j/∂x_i. Write the network layer by layer: H_0 = X; H_k is the output of hidden layer k, H_k = f_k(W_k H_{k-1} + b_k) componentwise; and F_j = f_{n+1,j}(W_{n+1,j}·H_n + b_{n+1,j}). I want the derivative of an output with respect to an input component, and the chain rule threads it through every layer. Start at the first hidden layer and differentiate forward: for neuron p of layer k,

  ∂H_k,p/∂x_i = f'_{k,p}(W_{k,p}·H_{k-1} + b_{k,p}) · ( W_{k,p} · ∂H_{k-1}/∂x_i ),

i.e. the activation derivative at that neuron times the weighted sum of the previous layer's input-derivatives. This is just the chain rule, and it's recursive in k: to get ∂H_k/∂x_i I need ∂H_{k-1}/∂x_i, with the base case ∂H_0/∂x_i = ∂X/∂x_i = e_i (one in slot i, zero elsewhere). Push it all the way up and apply the same rule once more at the output layer:

  ∂F_j/∂x_i = f'_{n+1,j}(W_{n+1,j}·H_n + b_{n+1,j}) · ( W_{n+1,j} · ∂H_n/∂x_i ).

Every term here is known under my threat model — the weights, the biases, the activations at X (one forward pass gives me all the W_{k}·H_{k-1}+b), and the activation derivatives are elementary. So ∇F is computable by propagating these derivatives forward, layer by layer, requiring only differentiable activations — which is no extra assumption, since backprop already requires it. In practice I won't hand-roll the forward recursion; automatic differentiation gives me the same matrix more conveniently. Each row j of the Jacobian is ∂F_j/∂x for all i at once, and that is exactly one backward pass with the output seeded at neuron j. So I compute the M×N (or here N×M, one row per class) Jacobian with N backward passes — one per output neuron — zeroing the input gradient between them. N is the number of classes, ten on these problems; that's cheap.

Good — I have the right object and I can compute it. Now the harder design question: given this matrix, how do I turn it into a sparse, targeted attack? The L0 objective is discrete by nature — I'm choosing a *subset* of features to touch — so a continuous norm-minimizing optimizer is the wrong shape. What fits a subset-selection problem is a greedy iterative search: pick the single most useful feature right now, modify it, recompute, repeat, and stop the moment the network flips or I've spent my feature budget. Greedy is natural here precisely because each touched feature is a discrete commitment, and best-first selection over which coordinate to commit is how you keep the support small.

So I need a per-feature *score* that ranks "how useful is touching x_i, right now, for getting to target t?" And here the forward derivative pays off, because a useful feature for a *targeted* attack has to do two things at once when I increase it: raise the target output F_t, and lower the others. Let me write that as two conditions on the derivative. For increasing x_i to raise F_t I need ∂F_t/∂x_i > 0. For increasing x_i to not-help the rest I need the others to go down, i.e. Σ_{j≠t} ∂F_j/∂x_i < 0. A feature that fails either test is useless or counterproductive for this direction, so it should score zero. Among the features that pass both tests, I want a single number that's large when the target-help is large *and* the others-hurt is large. The product captures "both must be big" in a way a sum wouldn't — a sum would let a huge target-derivative paper over a tiny others-effect. So:

  S(X,t)[i] = 0,  if ∂F_t/∂x_i < 0  or  Σ_{j≠t} ∂F_j/∂x_i > 0,
            = (∂F_t/∂x_i) · |Σ_{j≠t} ∂F_j/∂x_i|,  otherwise.

The absolute value on the others-sum is so that a *more negative* others-derivative — more helpful — gives a *higher* score; since I've already gated on that sum being negative, |·| just turns "more negative" into "more positive score." High S means: increasing this feature pushes hard toward the target, drags the rest down, or both. This is the adversarial saliency map, and it's the part Simonyan's visualization couldn't be: signed, target-aware, and pitting one class against the rest.

One subtlety I should pin down before I trust the two conditions: they implicitly assume that the target-derivative and the others-derivative carry independent information — that raising F_t doesn't automatically force Σ_{j≠t} F_j down. If I differentiate the *softmax probabilities*, that independence is false: the probabilities sum to one, so by construction raising one drops the rest, and worse, the softmax's normalization produces extreme, saturated derivatives that flatten out the differences between features and ruin the ranking. So I should compute the forward derivative on the *pre-softmax* outputs — the logits, or equivalently the last layer before the probability normalization. On the logits the outputs are *not* constrained to sum to one, so ∂F_t/∂x_i > 0 genuinely does not imply Σ_{j≠t} ∂F_j/∂x_i < 0, which is exactly why enforcing *both* conditions is meaningful rather than redundant. This also tells me the gate isn't bureaucratic: it's selecting the rare features that happen to move target and rest in opposite, favorable directions.

Now let me actually try ranking single features and see if it works, because I suspect it's too strict. The score is nonzero only when ∂F_t/∂x_i > 0 AND Σ_{j≠t} ∂F_j/∂x_i < 0 — a single feature has to be favorable on both axes simultaneously. In a real network most features are mixed: a pixel might strongly raise the target but also slightly raise one competing class, so its others-sum is positive and it gets gated to zero, even though it's a great target-pusher. Picking one feature at a time throws away every such pixel. Concretely, suppose pixel p has target-derivative 5 but others-sum +0.1, and pixel q has target-derivative −0.5 but others-sum −6. Individually p fails the others-test (its others-sum is positive) and q fails the target-test (its target-derivative is negative), so both score zero and the single-feature search finds almost nothing usable. Wall: the both-signs gate, applied per feature, is so restrictive that very few features survive.

But look at p and q *together*. Their combined target-derivative is 5 + (−0.5) = 4.5 > 0, and their combined others-sum is 0.1 + (−6) = −5.9 < 0. As a pair they pass both gates handsomely — q's strongly-negative others-sum compensates p's slightly-positive one, and p's strong target-push compensates q's slight negative. So the fix is to search over *pairs* of features and modify two at a time: one pixel can cover the other's flaw. The score generalizes by summing each derivative over the pair before taking the product:

  argmax over (p1,p2) of ( Σ_{i in {p1,p2}} ∂F_t/∂x_i ) · | Σ_{i in {p1,p2}} Σ_{j≠t} ∂F_j/∂x_i |,

subject to the left factor being positive and the right (inside |·|) being negative. Why stop at pairs and not triples or larger groups? Cost. Searching pairs is O(M²) per iteration; each extra feature in the group multiplies the number of combinations by roughly M again, and the gain from a third compensating feature is small once a pair already clears both gates. Two is the sweet spot — enough slack to satisfy the joint sign condition, cheap enough to evaluate exhaustively.

Next, how much do I change the selected features by? This is where the L0 objective directly shapes the choice. Features live in [0,1]. If I want to touch as *few* features as possible, then each feature I do touch should be used to the hilt in one shot, not nudged a little and revisited. So set θ = +1: push each selected pixel straight to its maximum, 1. A small θ would mean re-selecting the same pixel over many iterations to accumulate effect, wasting iterations and, worse, not actually reducing the count of distinct features needed. Saturating means one visit per feature. And it pays a second dividend: once a pixel is at its extreme (0 or 1), modifying it again does nothing, so I can *remove it from the search domain* Γ permanently. That shrinks the candidate set every iteration and makes the pair search cheaper as it goes — a best-first heuristic search with no backtracking, which is fine because a saturated pixel has nothing left to give.

Should θ increase or decrease? If I drive pixels down toward 0 with θ = −1, the meaning of a favorable derivative reverses because the input change is negative. To make the target output rise under a negative step, I now want ∂F_t/∂x_i < 0; to make the other outputs fall under that same negative step, I want Σ_{j≠t} ∂F_j/∂x_i > 0. The decreasing saliency score is therefore |∂F_t/∂x_i| · (Σ_{j≠t} ∂F_j/∂x_i) after those flipped gates. It works, but increasing is the better default: pushing pixels up tends to flip the label with fewer features touched. Driving pixels to zero removes information from the image — it lowers the entropy of the input — and a network has less to latch onto, so the desired *target* behavior is harder to induce reliably; larger absolute intensity *additions* are more confidently misclassified. So increasing intensities is the primary strategy, decreasing the counterpart.

Now the loop and the budget. The L0 budget is "modify at most a given fraction of features," call it Υ (or equivalently a hard count). Since I spend two features per iteration, the iteration cap is

  max_iter = floor( (number of features · Υ) / 2 )

— half as many iterations as the allowed feature changes, with Υ written as a fraction of the feature count. In a count-style interface, I can let `pixels` denote the allowed number of feature changes and use the torchattacks convention `ceil(pixels/2)`, the same as `ceil(num_features·gamma/2)` when `gamma` is a fraction. Each iteration: recompute the forward derivative at the current X*, build the pairwise saliency map, pick the best pair, push both by θ, clamp to [0,1], drop saturated pixels from Γ. Halt when the network's predicted class equals the target, or max_iter is reached, or Γ is empty. The recompute-every-iteration matters: the network is non-linear, so after I move two pixels the whole Jacobian shifts, and the next-best pair must be chosen against the updated sensitivities, not the original ones. That's what makes the greedy search actually track the moving decision boundary instead of committing to a stale ranking.

Let me also be careful that "feature" and "spatial pixel" aren't the same on color images. The forward derivative is per *channel-feature* — ∂F_j/∂x_i with i ranging over C·H·W. For an L0 budget that counts *spatial pixels* (a pixel modified if any of its channels changes), bounding the number of touched features bounds the number of touched spatial pixels from above: if I touch at most `pixels` features, I touch at most `pixels` distinct spatial locations. So setting the feature budget equal to the spatial-pixel budget is a safe, valid way to respect the constraint.

Now let me turn all of this into code, grounded in how it's actually implemented. There are three pieces: compute the Jacobian (forward derivative) by one backward pass per class; build the pairwise saliency map with broadcasting so I can score all O(M²) pairs at once; and the outer greedy loop. For the saliency map, the trick to vectorize the pair search is to form, from the per-feature target-derivatives, an M×M matrix alpha whose (i,j) entry is the pair-sum target-derivative ∂F_t/∂x_i + ∂F_t/∂x_j — that's just a broadcast add of the vector to its own transpose — and likewise beta from the others-derivatives. Then the gate masks (alpha > 0 and beta < 0 for increasing, alpha < 0 and beta > 0 for decreasing), the diagonal is zeroed out (a feature can't pair with itself), out-of-domain features are knocked out with the same `increase_coef` biasing pattern used in torchattacks, the score is the positive product of the two favorable factors over the whole grid, and argmax over the flattened M² grid recovers the pair (p, q) by integer-divide and modulo.

```python
import numpy as np
import torch
import torch.nn as nn


def compute_jacobian(model: nn.Module, x: torch.Tensor, n_classes: int,
                     device: torch.device) -> torch.Tensor:
    # Forward derivative: row c is dF_c/dx for all input features.
    # One backward pass per output neuron, differentiating the OUTPUTS (logits),
    # not the loss, w.r.t. the INPUT.
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)                                  # pre-softmax outputs
    num_features = int(np.prod(x.shape[1:]))           # C*H*W
    jac = torch.zeros(n_classes, num_features, device=device)
    for c in range(n_classes):
        if x.grad is not None:
            x.grad.zero_()
        logits[0, c].backward(retain_graph=True)       # seed output neuron c
        jac[c] = x.grad.reshape(-1).clone()            # dF_c/dx_i for every feature i
    return jac


@torch.no_grad()
def saliency_pair(jac: torch.Tensor, target: int, increasing: bool,
                  search_domain: torch.Tensor, num_features: int):
    # alpha[i] = dF_t/dx_i ; beta[i] = sum_{j!=t} dF_j/dx_i
    all_sum = jac.sum(dim=0, keepdim=True)             # sum over classes of dF_j/dx_i
    target_grad = jac[target]                          # target sensitivity
    others_grad = (all_sum - target_grad).squeeze(0)   # rest sensitivity

    # knock out features already out of the search domain with the torchattacks
    # increase_coef bias: for increasing, target is pushed negative and others positive.
    out = (search_domain == 0).float()
    increase_coef = (2.0 if increasing else -2.0) * out
    target_grad = target_grad - increase_coef * target_grad.abs().max()
    others_grad = others_grad + increase_coef * others_grad.abs().max()

    # pair-sums over all (i,j): broadcast a vector against its own transpose
    alpha = target_grad.view(1, -1) + target_grad.view(-1, 1)   # sum_{i in pair} dF_t/dx_i
    beta = others_grad.view(1, -1) + others_grad.view(-1, 1)    # sum_{i in pair} dF_others/dx_i

    if increasing:                                     # need target up, others down
        mask = (alpha > 0) & (beta < 0)
        scores = alpha * beta.abs()
    else:                                              # decreasing flips both gates
        mask = (alpha < 0) & (beta > 0)
        scores = alpha.abs() * beta
    eye = torch.eye(num_features, dtype=torch.bool, device=jac.device)
    mask = mask & ~eye                                 # a feature cannot pair with itself

    scores = scores * mask.float()
    flat = scores.view(-1).argmax()
    p = int(flat // num_features)                      # first feature of the best pair
    q = int(flat % num_features)                       # second feature of the best pair
    return p, q


def attack_one(model, x, target, pixels, n_classes, device, theta=1.0):
    increasing = theta > 0
    num_features = int(np.prod(x.shape[1:]))
    shape = x.shape
    max_iters = int(np.ceil(pixels / 2.0))             # 2 features per iteration

    # only features not yet saturated stay in the search domain
    if increasing:
        search_domain = (x < 0.99).reshape(num_features).clone()
    else:
        search_domain = (x > 0.01).reshape(num_features).clone()

    pred = model(x).argmax(1)
    it = 0
    while it < max_iters and pred.item() != target and search_domain.sum() != 0:
        jac = compute_jacobian(model, x, n_classes, device)      # recompute: F is non-linear
        p, q = saliency_pair(jac, target, increasing, search_domain, num_features)
        flat = x.reshape(-1).clone()
        flat[p] += theta                               # saturate both selected features
        flat[q] += theta
        x = flat.clamp(0.0, 1.0).reshape(shape)        # stay in valid range
        search_domain[p] = 0                           # saturated -> drop from search
        search_domain[q] = 0
        pred = model(x).argmax(1)
        it += 1
    return x
```

And the wrapper picks a target per image and runs the loop per image (the Jacobian is per-sample, so batching it gets unwieldy; one image at a time keeps it clean):

```python
def run_attack(model, images, labels, pixels, device, n_classes):
    model.eval()
    images = images.to(device)
    adv = []
    for x, y in zip(images, labels.to(device)):
        x = x.unsqueeze(0)
        target = int((y + 1) % n_classes)              # a chosen target != true class
        adv.append(attack_one(model, x, target, pixels, n_classes, device).clamp(0, 1))
    return torch.cat(adv, 0)
```
