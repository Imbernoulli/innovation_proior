I have a trained, frozen feedforward classifier $F: \mathbb{R}^M \to \mathbb{R}^N$ with white-box access — I know the weights, I can run it forward, and I can backpropagate through it. I hold a clean input $X$ that it labels correctly as $Y$, and I want a perturbed $X^*$ that the very same network reads as a chosen target $Y^* \neq Y$. The clean statement of the goal is $\arg\min_{\delta} \lVert \delta \rVert$ subject to $F(X+\delta) = Y^*$. Two things make this hard. The constraint $F(X+\delta) = Y^*$ has no inverse I can write down — $F$ is a stack of non-linear, non-convex layers, so I cannot solve for $\delta$ in closed form. And the choice of norm is not innocent; it decides what kind of perturbation I am even looking for. The pressure I care about is the *number of features I touch*: I would rather flip the label by saturating a handful of pixels — each possibly driven all the way to its extreme — than by smearing an invisible nudge across all 784 of them. That is an $L_0$ objective: count the nonzero entries of $\delta$, never mind their size.

The existing attacks all stall on exactly this objective. Szegedy's box-constrained L-BFGS takes the program literally, line-searching a constant $c$ and minimizing $c\,\lVert r \rVert_2 + \mathrm{loss}_f(x+r, l)$ until the minimizer lands in the target class. It works and it finds imperceptible examples, but its objective is an $L_2$ penalty, and the minimizer of an $L_2$ penalty is *dense*: $L_2$ has no incentive to zero anything out, and prefers to make a thousand coordinates each slightly nonzero rather than ten coordinates large, because spreading a fixed squared budget thin lowers $\lVert r \rVert_2$. So by construction it returns a perturbation spread across the whole image, with no term that counts touched features, and it runs a heavy second-order optimization per example. Goodfellow's fast gradient sign method is the opposite pole: linearize the training loss $J$ and step $\eta = \varepsilon \cdot \mathrm{sign}(\nabla_x J(\theta, x, y))$. The argument for it is genuinely deep — for a linear unit, $w^\top(x+\eta) = w^\top x + w^\top \eta$, and maximizing $w^\top \eta$ under $\lVert \eta \rVert_\infty \le \varepsilon$ gives $\eta = \varepsilon \cdot \mathrm{sign}(w)$ and $w^\top \eta = \varepsilon \lVert w \rVert_1 = \varepsilon \, m \, n$, a logit shift that grows linearly with the input dimension $n$, so many tiny coordinated changes sum to a large effect. But that single sign step writes a nonzero perturbation into *every* coordinate — it is the densest possible perturbation under the $L_\infty$ ball, the exact opposite of $L_0$ — and it is organized around the untargeted "raise the loss" direction, not around steering the net into one specific class. Simonyan's class-saliency idea is the closest to "which pixels matter": backpropagate a class score $S_c$ to the input and take per-pixel magnitudes $M_{ij} = \max_c |(\partial S_c/\partial I)_{ijc}|$. But it is a visualization — single-class, magnitude-only, sign-blind, with no sense of target-versus-rest. It tells me where a class lives; it does not tell me which pixels to push, in which direction, to *convert* one class into another. What all three share is the crack I want to pry open: Szegedy and Goodfellow both collapse the whole output and label into one loss scalar $J$ and read back one gradient $\partial J/\partial x$ over all pixels; Simonyan differentiates one class score and discards the sign. None of them keeps the object that a targeted, sparse attack actually needs.

I propose JSMA, the Jacobian-based Saliency Map Attack. The object nobody is using directly is the full sensitivity of *every output* to *every input feature* — the Jacobian of the network's output map,
$$\frac{\partial F_j(X)}{\partial x_i}, \quad i = 1,\dots,M,\; j = 1,\dots,N,$$
which I call the *forward derivative*, because it answers the forward-direction question "if I push feature $x_i$, how does each output $F_j$ move?", as opposed to training backprop, which answers "how does the cost move if I push a weight?". Two deliberate differences from training: I differentiate the network's *outputs*, not its cost, and with respect to the *input features*, not the weights. This is the right object precisely because it is per-feature *and* per-class *and* signed, so for each individual pixel I can ask the targeted question — does raising this pixel raise my target class, and what does it do to the others — which is exactly the trade-off a single loss gradient or a sign-blind magnitude map cannot express. It is fully computable under white-box access: writing the network layer by layer with $H_0 = X$ and $H_k = f_k(W_k H_{k-1} + b_k)$, the chain rule threads forward as $\partial H_{k,p}/\partial x_i = f'_{k,p}(W_{k,p}\!\cdot\! H_{k-1} + b_{k,p})\,(W_{k,p}\!\cdot\! \partial H_{k-1}/\partial x_i)$ with base case $\partial H_0/\partial x_i = e_i$, and one more application at the output layer gives $\partial F_j/\partial x_i$. Every term — weights, biases, the activations at $X$ from one forward pass, the elementary activation derivatives — is known, so $\nabla F$ requires only differentiable activations (no extra assumption, since backprop already requires it). In practice I do not hand-roll the recursion: each row $j$ of the Jacobian is exactly one backward pass with the output seeded at neuron $j$, so I get the $N \times M$ matrix in $N$ backward passes, $N$ being the number of classes (ten here), which is cheap.

Given this matrix, the question is how to turn it into a sparse, targeted attack. $L_0$ is discrete by nature — I am choosing a *subset* of features — so a continuous norm-minimizing optimizer is the wrong shape; what fits is a greedy iterative search that picks the single most useful feature now, modifies it, recomputes, and repeats, stopping the instant the network flips or the budget is spent. Greedy is natural exactly because each touched feature is a discrete commitment and best-first selection keeps the support small. For that I need a per-feature *score* ranking "how useful is touching $x_i$ right now for reaching target $t$?", and a useful feature for a *targeted* attack must do two things at once when increased: raise the target output $F_t$, and lower all the others. As two conditions on the derivative, raising $F_t$ needs $\partial F_t/\partial x_i > 0$, and not-helping the rest needs $\sum_{j \neq t} \partial F_j/\partial x_i < 0$. A feature failing either test is useless or counterproductive and scores zero. Among the survivors I want a number that is large when both effects are large, and the *product* captures "both must be big" in a way a sum would not — a sum lets a huge target-derivative paper over a tiny others-effect. This is the adversarial saliency map:
$$S(X,t)[i] = \begin{cases} 0, & \text{if } \dfrac{\partial F_t}{\partial x_i} < 0 \ \text{ or }\ \sum_{j\neq t}\dfrac{\partial F_j}{\partial x_i} > 0,\\[2mm] \left(\dfrac{\partial F_t}{\partial x_i}\right)\left|\sum_{j\neq t}\dfrac{\partial F_j}{\partial x_i}\right|, & \text{otherwise.}\end{cases}$$
The absolute value on the others-sum turns "more negative, hence more helpful" into "higher score"; since I have already gated that sum to be negative, $|\cdot|$ is just a sign fix. One subtlety pins down why enforcing *both* conditions is meaningful rather than redundant: the two derivatives must carry independent information, i.e. raising $F_t$ must not automatically force $\sum_{j\neq t} F_j$ down. If I differentiated the *softmax probabilities* that independence would be false — they sum to one, so raising one mechanically drops the rest, and the normalization produces saturated, extreme derivatives that flatten the ranking. So the forward derivative is computed on the *pre-softmax logits*, where the outputs are not constrained to sum to one and $\partial F_t/\partial x_i > 0$ genuinely does not imply $\sum_{j\neq t}\partial F_j/\partial x_i < 0$. The gate is then not bureaucratic; it selects the rare features that move target and rest in opposite, favorable directions.

But applied per single feature the both-signs gate is too strict. In a real network most features are mixed — a pixel might strongly raise the target yet also slightly raise a competitor, making its others-sum positive and gating it to zero despite being a great target-pusher. Concretely, if pixel $p$ has target-derivative $5$ but others-sum $+0.1$, and pixel $q$ has target-derivative $-0.5$ but others-sum $-6$, then individually $p$ fails the others-test and $q$ fails the target-test, so both score zero and the single-feature search finds almost nothing. Taken *together*, though, their combined target-derivative is $5 + (-0.5) = 4.5 > 0$ and their combined others-sum is $0.1 + (-6) = -5.9 < 0$: as a pair they clear both gates handsomely, $q$'s strongly-negative others-sum covering $p$'s slightly-positive one and $p$'s strong target-push covering $q$'s slight negative. So I search over *pairs* and modify two features at a time, one compensating the other's flaw, summing each derivative over the pair before the product:
$$\arg\max_{(p_1,p_2)}\ \left(\sum_{i\in\{p_1,p_2\}}\frac{\partial F_t}{\partial x_i}\right)\left|\sum_{i\in\{p_1,p_2\}}\sum_{j\neq t}\frac{\partial F_j}{\partial x_i}\right|,$$
subject to the left factor being positive and the term inside $|\cdot|$ being negative. I stop at pairs for cost: searching pairs is $O(M^2)$ per iteration, each extra feature in the group multiplies the combinations by roughly $M$ again, and a third compensating feature buys little once a pair already clears both gates. Two is the sweet spot — enough slack to satisfy the joint sign condition, cheap enough to evaluate exhaustively.

The step size follows directly from the $L_0$ objective. Features live in $[0,1]$; to touch as *few* as possible, each one I touch should be used to the hilt in one shot, so I set $\theta = +1$ and push each selected pixel straight to its maximum. A small $\theta$ would mean re-selecting the same pixel over many iterations, wasting them and never reducing the count of distinct features needed. Saturation means one visit per feature, and pays a second dividend: once a pixel sits at an extreme ($0$ or $1$), modifying it again does nothing, so I permanently remove it from the search domain $\Gamma$, shrinking the candidate set and cheapening the pair search every iteration — a best-first heuristic with no backtracking, which is fine because a saturated pixel has nothing left to give. Increasing is the better default over decreasing ($\theta = -1$, which flips both gates to $\partial F_t/\partial x_i < 0$ and $\sum_{j\neq t}\partial F_j/\partial x_i > 0$): driving pixels to zero removes information from the image, lowering its entropy and giving the network less to latch onto, so inducing the *target* behavior is less reliable, whereas larger intensity *additions* are more confidently misclassified. The budget closes the loop: an $L_0$ cap of "modify at most fraction $\Upsilon$ of features" becomes $\mathrm{max\_iter} = \lfloor (\text{num\_features}\cdot\Upsilon)/2 \rfloor$ since I spend two features per iteration, or in a count-style interface where `pixels` is the allowed number of feature changes, $\lceil \text{pixels}/2 \rceil$. Each iteration recomputes the forward derivative at the *current* $X^*$ — the network is non-linear, so after I move two pixels the whole Jacobian shifts and the next-best pair must be chosen against the updated sensitivities, which is what makes the greedy search track the moving decision boundary instead of committing to a stale ranking — builds the pairwise saliency map, picks the best pair, pushes both by $\theta$, clamps to $[0,1]$, and drops the saturated pixels from $\Gamma$; it halts when the predicted class equals the target, the iteration cap is reached, or $\Gamma$ is empty. One last care: on color images the forward derivative is per channel-feature, $i$ ranging over $C\cdot H\cdot W$, but bounding the touched *features* by `pixels` bounds the touched *spatial pixels* from above (a spatial pixel counts as modified if any channel changes), so the feature budget safely respects a spatial-pixel constraint.

```python
import numpy as np
import torch
import torch.nn as nn


def compute_jacobian(model: nn.Module, x: torch.Tensor, n_classes: int,
                     device: torch.device) -> torch.Tensor:
    """Forward derivative: row c is dF_c/dx over all input features (one backward pass per class)."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)                                  # pre-softmax outputs
    num_features = int(np.prod(x.shape[1:]))           # C*H*W
    jac = torch.zeros(n_classes, num_features, device=device)
    for c in range(n_classes):
        if x.grad is not None:
            x.grad.zero_()
        logits[0, c].backward(retain_graph=True)
        jac[c] = x.grad.reshape(-1).clone()
    return jac


@torch.no_grad()
def saliency_pair(jac, target, increasing, search_domain, num_features):
    """argmax over feature pairs of the adversarial saliency score (Eqs. above)."""
    all_sum = jac.sum(dim=0, keepdim=True)
    target_grad = jac[target]
    others_grad = (all_sum - target_grad).squeeze(0)

    out = (search_domain == 0).float()                 # features outside the search domain
    increase_coef = (2.0 if increasing else -2.0) * out
    target_grad = target_grad - increase_coef * target_grad.abs().max()
    others_grad = others_grad + increase_coef * others_grad.abs().max()

    alpha = target_grad.view(1, -1) + target_grad.view(-1, 1)   # pair-sum of dF_t/dx
    beta = others_grad.view(1, -1) + others_grad.view(-1, 1)    # pair-sum of dF_others/dx

    if increasing:
        mask = (alpha > 0) & (beta < 0)
        scores = alpha * beta.abs()
    else:
        mask = (alpha < 0) & (beta > 0)
        scores = alpha.abs() * beta
    eye = torch.eye(num_features, dtype=torch.bool, device=jac.device)
    mask = mask & ~eye                                  # no self-pairs

    scores = scores * mask.float()
    flat = scores.view(-1).argmax()
    return int(flat // num_features), int(flat % num_features)


def attack_one(model, x, target, pixels, n_classes, device, theta=1.0):
    increasing = theta > 0
    num_features = int(np.prod(x.shape[1:]))
    shape = x.shape
    max_iters = int(np.ceil(pixels / 2.0))             # 2 features committed per iteration

    if increasing:
        search_domain = (x < 0.99).reshape(num_features).clone()
    else:
        search_domain = (x > 0.01).reshape(num_features).clone()

    pred = model(x).argmax(1)
    it = 0
    while it < max_iters and pred.item() != target and search_domain.sum() != 0:
        jac = compute_jacobian(model, x, n_classes, device)     # recompute: F is non-linear
        p, q = saliency_pair(jac, target, increasing, search_domain, num_features)
        flat = x.reshape(-1).clone()
        flat[p] += theta                               # saturate the two selected features
        flat[q] += theta
        x = flat.clamp(0.0, 1.0).reshape(shape)
        search_domain[p] = 0                           # drop saturated features
        search_domain[q] = 0
        pred = model(x).argmax(1)
        it += 1
    return x


def run_attack(model, images, labels, pixels, device, n_classes):
    model.eval()
    images = images.to(device)
    adv = []
    for x, y in zip(images, labels.to(device)):
        x = x.unsqueeze(0)
        target = int((y + 1) % n_classes)              # a chosen target class != true label
        adv.append(attack_one(model, x, target, pixels, n_classes, device).clamp(0, 1))
    return torch.cat(adv, 0)
```
