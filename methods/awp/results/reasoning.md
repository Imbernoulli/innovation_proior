Let me start from the thing that actually frustrates me. The ordinary PGD adversarial-training baseline is already learning the training set: after 200 epochs, a PreAct ResNet-18 on CIFAR-10 classifies about 84% of the *training* set correctly under a 10-step PGD attack. On the *test* set, under the same attack, it manages about 43%. Forty-one points of gap. The same network trained normally, on clean images, keeps its train/test accuracy gap under ten points. So robustness, specifically, refuses to generalize, and it refuses by a margin that dwarfs anything I see in standard training. And it gets worse across the training curve: test robustness peaks early — right around the first learning-rate decay — and then it *decays* while training robustness keeps climbing. That is overfitting, but overfitting of a strange kind, because the thing being overfit is the worst-case loss, not the clean loss. People have noticed that you can recover almost all the apparent gains of fancier adversarial-training losses just by stopping early, at that peak. But early stopping is a confession, not a cure: it closes the gap only by quitting before the model has actually learned to be robust, so the peak it stops at is capped by a mediocre training robustness. I want both — a small gap *and* high training robustness — which means I need to understand *why* the robustness overfits, and attack that cause directly.

So what does ordinary adversarial training actually do? It solves a saddle point. For each example, find the worst input inside an eps-ball, then descend on the loss of those worst inputs:

  min_w (1/n) Σ_i max_{‖x'_i − x_i‖_p ≤ eps} ℓ(f_w(x'_i), y_i),

with the inner max approximated by PGD — random start, then repeatedly x' ← Π_eps(x' + η₁·sign(∇_{x'}ℓ)), projecting back into the ball each step. Geometrically I know exactly what this buys me: it flattens the loss as a function of the *input*. Around each training point, the loss barely moves when you push the input around inside the ball. That is the whole point of training on the worst input — you make the worst input not much worse than the clean one. Call that the input loss landscape, and adversarial training flattens it by construction.

But flattening the loss in input space says nothing about how the loss behaves in *weight* space, and the gap I'm chasing is a generalization gap, a property of the weights I land on, not of any single input. The connection I keep circling back to is the ordinary, non-robust flat-minima story: if the loss surface as a function of the weights is flat around your solution — you can jiggle the weights and the loss barely rises — that solution generalizes better than one sitting in a sharp, narrow valley. Keskar, Neyshabur, Li and others have hammered on this. The object is the weight loss landscape: hold the data fixed, move the weights, watch the loss. Sharp is bad, flat is good. Nobody has cleanly established whether that same link survives under adversarial training, and the couple of attempts I've seen botched the measurement — they probed the surface using a fixed set of adversarial examples that were generated once, on the *unperturbed* model, and then reused for every perturbed weight. That's wrong: an adversarial example crafted for f_w is a weak attack on f_{w+v}, so reusing it makes every perturbed model look artificially good, i.e. the landscape looks artificially flat. If I want to see the real weight loss landscape under adversarial training I have to regenerate the adversarial examples on-the-fly for each perturbed weight.

The measurement I need is clear. I want to plot

  g(α) = ρ(w + α·d) = (1/n) Σ_i max_{‖x'_i − x_i‖ ≤ eps} ℓ(f_{w+α·d}(x'_i), y_i)

as I slide the weights along a direction d by amount α — and crucially I recompute the PGD attack for each f_{w+α·d}, not reuse a cached set. One subtlety bites immediately: which direction d, and at what scale? A ReLU network is scale-invariant — multiply one layer's weights by c, divide the next by c, and the function is literally unchanged — so a fixed-size random perturbation means wildly different things depending on how the weights happen to be scaled, and two networks' landscapes aren't comparable. The fix is the filter-normalization technique: sample d ~ N(0, I), then rescale each filter of the direction to the norm of the matching weight filter, d_{l,j} ← (d_{l,j}/‖d_{l,j}‖_F)·‖w_{l,j}‖_F. Now α is measured in units relative to the weights themselves, and the comparison is fair; repeating this over several random directions is the sanity check that the shape is not just one lucky slice.

The diagnostic evidence has a very specific shape. During training, before the "best" epoch, the weight loss landscape is *flat* and the robust generalization gap is small. After the best epoch — as test robustness starts to fall — the landscape gets visibly *sharper*, in lockstep with the gap widening. The same ordering shows up across vanilla AT, the KL-regularized variant, the misclassification-aware variant, the semi-supervised one, and early-stopped AT: the smaller-gap methods also have flatter weight loss landscapes. So the flat-minima story does survive adversarial training, and — this is the part that reframes everything — all those fancier methods are, whatever else they're doing, *implicitly* flattening the weight loss landscape. They each found a different indirect route to the same geometric property.

That observation is the lever. If flatness of the weight loss landscape is the thing that controls the robust generalization gap, and every successful method is only flattening it as a side effect, then I should stop being coy and flatten it *directly* — make it an explicit term in the objective rather than hope it emerges. So write down what "flat at w" even means as a quantity. The honest measure of sharpness is: how much does the adversarial loss go up if I move the weights to a nearby w + v? That's ρ(w + v) − ρ(w). I want the loss low *and* I want that bump small, so:

  min_w { ρ(w) + ( ρ(w + v) − ρ(w) ) }.

Stare at that for a second — the ρ(w) and the −ρ(w) cancel, and the whole thing collapses to

  min_w ρ(w + v).

Huh. Minimizing "the loss plus its flatness penalty" is just minimizing the loss *at the perturbed point*. That's clean and a little surprising: I don't need to carry a separate scalar regularizer. The update should descend from a nearby point where the loss is deliberately made large. If I keep doing that while returning to the center after each step, then sharp rises around the center become expensive directions for training, which is exactly the geometry I want to suppress.

But which v? If I pick v badly, this is vacuous. The shallow choice is a *random* v — sample a direction, perturb, train. That's the spirit of the PAC-Bayes expectation E_v[ρ(w+v)], and people have tried injecting random weight noise during training. The problem is efficiency. A random direction mostly points along directions the loss doesn't care about; to make a random perturbation actually probe the steep directions, I would have to crank its magnitude up until the temporary model is no longer a small local probe and training itself becomes harder. A random probe is the wrong tool. I want the *worst* direction — the one along which the adversarial loss rises fastest — because that's the direction that, if I flatten it, does the most good, and I can do it with a tiny magnitude. So make v adversarial, exactly the way x' is adversarial:

  min_w max_{v ∈ V} ρ(w + v) = min_w max_{v ∈ V} (1/n) Σ_i max_{‖x'_i − x_i‖ ≤ eps} ℓ(f_{w+v}(x'_i), y_i).

This is a double perturbation: the inputs are perturbed adversarially (inner, per-example) and the weights are perturbed adversarially (outer over v). One thing I have to be careful about — the two maximizations don't commute. The input perturbation is per-example: each x'_i is the worst input for *its own* example. The weight perturbation acts on the shared weights, so it has to be chosen to make the *whole-batch* loss worst, not example-by-example; v is one direction in weight space that the entire batch's loss responds to jointly. So I can't fold the v-max inside the per-example sum; it sits outside, over the averaged loss.

Before I worry about how to solve this, is there a reason to believe worst-case is the *right* thing and not just a heuristic that happens to be efficient? The PAC-Bayes flatness bound gives me the shape I need. If I let Q be the randomized predictor obtained by drawing a weight perturbation ν around w, and let P be a data-independent prior, then with probability 1−δ over the training draw,

  E_ν[L(f_{w+ν})] ≤ L̂(f_w) + { E_ν[L̂(f_{w+ν})] − L̂(f_w) } + 4·sqrt( ( KL(Q‖P) + ln(2n/δ) ) / n ).

The middle braced term is exactly the expected sharpness — how much the empirical loss rises on average under weight jitter — and the last term is the KL complexity of Q relative to P. Now make the standard Gaussian instantiation: let P be zero-mean spherical with variance σ², and let Q be the same spherical Gaussian shifted to w. The KL part is ‖w‖²/(2σ²). If I tie the variance to the weight scale, σ² = a·‖w‖², the KL contribution becomes 1/(2a); here a is the relative variance scale, so I am not accidentally squaring the constant. Once I fix that relative noise scale, the term that still depends on the learned surface is the expected sharpness. Translate this from the clean empirical loss L̂ to the adversarial loss ρ — the same PAC-Bayes shape applies because ρ is a bounded 0-1 robust loss — and I read it as: the robust generalization gap is controlled by the expected flatness of the weight loss landscape. If the random perturbation ν is supported on the same feasible region V, then subtracting the same ρ(w) from both sides gives E_ν[ρ(w+ν)] − ρ(w) ≤ max_{v∈V} ρ(w+v) − ρ(w), because an expectation over V cannot exceed the maximum on V. So if I drive down the worst-case sharpness, I drive down a sufficient upper bound on the expected sharpness term. That's the justification: minimizing max_v ρ(w+v) is a conservative way to squeeze the PAC-Bayes flatness term without having to sample many random perturbations.

Notice the bound also told me, for free, how to *size* v: the perturbation scale should move with ‖w‖, a *relative* magnitude. That matches what filter normalization already taught me. So I won't constrain v by a fixed value the way I constrain x' by eps. The numeric scale of weights differs from layer to layer — a fixed v that's tiny for one layer is enormous for another — and the ReLU scale invariance means absolute weight magnitudes are meaningless anyway. So I constrain v *per layer, relative to that layer's weight norm*:

  ‖v_l‖ ≤ γ·‖w_l‖,

with γ a single small dimensionless knob shared across all layers. This is the weight-space analogue of eps, but expressed as a fraction of each layer's own scale.

Now solve the outer max_v. By exact analogy with PGD on inputs, do projected gradient *ascent* on v. I want to push ρ(w+v) up, so I step v along the gradient of the loss with respect to v and then project back into the per-layer ball. The gradient ∇_v of the batch-averaged loss points in the steepest-ascent direction; I take a step in that direction. But there's a scaling question — for inputs I used sign(∇), which is the right steepest-ascent direction under an L_inf budget. For weights my budget is an L_2 (Frobenius) ball per layer, ‖v_l‖ ≤ γ‖w_l‖, so the steepest-ascent direction under that budget is the *normalized* gradient, ∇_v/‖∇_v‖, and I scale it to the budget radius. So the v update is:

  v_l ← Π_γ( v_l + η₂ · ( ∇_{v_l} (1/m)Σ_i ℓ(f_{w+v}(x'_i), y_i) / ‖∇_{v_l} (1/m)Σ_i ℓ(f_{w+v}(x'_i), y_i)‖ ) · ‖w_l‖ ),

where the normalized gradient gives the direction and ‖w_l‖ supplies the per-layer relative scale. The projection Π_γ enforces the ball: for each layer, if ‖v_l‖ > γ‖w_l‖ then rescale v_l ← γ·(‖w_l‖/‖v_l‖)·v_l, else leave it. If I take a single step of size η₂ = γ (and start from v=0 so there's nothing to project), the step lands right on the surface of the γ‖w_l‖ ball, which is what I want for a one-step ascent. More generally, with A alternations and K₂ inner steps, the default η₂ = γ/(A·K₂) sets the nominal total ascent radius and the projection keeps every layer inside ‖v_l‖ ≤ γ‖w_l‖ as directions change.

I should also alternate the two maximizations: regenerate x' on the *perturbed* model f_{w+v} (since the inputs should be worst-case for the model I'll actually evaluate), then update v on those x'. Call A the number of alternations, K₁ the PGD steps for x', K₂ the steps for v. The natural worry is cost — every extra step is another pair of forward/backward passes. I want the cheapest version that still implements the math, so I take the one-shot setting: A=1, K₁=10 (the usual for the inner attack), K₂=1. With a single alternation there's no second pass, so v=0 the first and only time I craft x' — meaning x' is generated on the plain model f_w, and v is computed afterward on those x'. That gives one extra ascent step per minibatch, and the worst-case direction is doing the work that a random perturbation would need a much larger radius to probe.

Now the outer min_w has a trap I have to be careful about. I perturbed the weights to w+v to *measure and flatten* the landscape, but the perturbation is a temporary probe — I do not want to actually leave the weights sitting at w+v. The thing I'm optimizing is the network at its center w; v is scaffolding. So the gradient I descend should be the gradient of the loss evaluated at the perturbed point (that's where the flatness information lives), but the step has to bring the parameters back to a center, then move that center. Concretely:

  w ← (w + v) − η₃ · ∇_{w+v} (1/m)Σ_i ℓ(f_{w+v}(x'_i), y_i) − v.

Read the three pieces: start from the perturbed weights w+v (where I have the model and its gradient), subtract the SGD step computed at that perturbed point, then subtract v to undo the perturbation and return to center. The net effect is: the SGD update direction is the gradient *at the perturbed point*, but it's applied to the center weights. That's how I get a flat-minimum-seeking step without drifting the weights toward the adversarial perturbation.

There's a practical wrinkle in actually computing v that I have to get right, and it's about batch-normalization statistics. To compute ∇_v I need forward passes through f_{w+v}, and those forward passes update BN's running mean/variance. But those are throwaway forward passes on a deliberately corrupted model — I don't want them polluting the BN statistics of the real model I'm training. The clean way is to do the v-computation on a *proxy*: a copy of the model. Load the current weights into the proxy, do the ascent step there to find v, read off the perturbation, then apply v to the real model only for the actual training forward/backward, and restore afterward. The proxy carries the BN-stat damage; the real model stays clean.

Let me make the one-step case completely concrete, because that's the default and it's where the implementation lives. With K₂=1, A=1, the entire v-computation is a single ascent step. I copy the model into the proxy, and I want the proxy to take one step that *maximizes* the loss. A gradient-descent optimizer minimizes, so to ascend I just feed it the *negated* loss: minimize −ρ over the proxy's weights for one SGD step. After that step the proxy's weights are w_proxy = w − lr·∇(−ρ) = w + lr·∇ρ — moved in the ascent direction. The raw difference Δw = w_proxy − w is therefore proportional to +∇ρ, the ascent direction, exactly what I want for v's direction. The magnitude of that proxy step (its learning rate) doesn't matter, because I'm about to renormalize: I only use Δw for its *direction*, and I rescale it to the relative size γ‖w‖ myself. So I don't need the proxy learning rate to set the perturbation radius; the magnitude is governed by γ and the renormalization.

The renormalization is the per-layer relative-size constraint, implemented exactly. For each weight tensor I want the perturbation to have norm equal to ‖w_l‖ (the relative scale, before multiplying by γ). So I take the raw direction Δw, normalize it to unit norm by dividing by ‖Δw‖, and multiply by ‖w_l‖:

  diff_l = ( ‖w_l‖ / ‖Δw_l‖ ) · Δw_l,

with a tiny epsilon (1e-20) added to the denominator to avoid dividing by zero when the proxy step happened to leave a layer unchanged. After this, ‖diff_l‖ = ‖w_l‖ exactly. Then the actual perturbation applied is v_l = γ·diff_l, so ‖v_l‖ = γ‖w_l‖ — sitting right on the boundary of the per-layer ball, which is the one-step ascent landing exactly on the constraint surface. The projection Π_γ is automatically satisfied by construction; I don't need a separate clamp.

Which parameters do I perturb? The scale-invariance and relative-size argument is about weight *matrices/tensors* — convolutional and linear weights — not about one-dimensional parameters like BN scales and biases. So I perturb only tensors with more than one dimension whose name marks them as a weight; I skip BN parameters and biases. That keeps the perturbation in the part of weight space where "relative to ‖w_l‖" is meaningful.

Now, this whole construction is generic — it sits on top of *any* inner adversarial loss, because all that changes is which ℓ I plug into ρ. The version I'll write lands on the KL-regularized surrogate, because that is one of the main adversarial-training bases this wrapper is meant to extend. That base loss has two parts: cross-entropy on the clean inputs to hold natural accuracy, plus a β-weighted KL between clean and adversarial predictions to push robustness,

  ℓ_base(x, x', y) = CE(f_w(x), y) + β·KL(f_w(x) ‖ f_w(x')),

with β=6 the usual trade-off weight, and the adversarial example for this base crafted by maximizing the KL between clean and perturbed predictions rather than the cross-entropy:

  x' ← Π_eps( x' + η₁·sign(∇_{x'} KL(f_w(x) ‖ f_w(x'))) ).

So the weight perturbation's loss — the thing I ascend on v to maximize — is this same base loss: I want the worst v for the actual training objective. In the one-alternation code, x' has already been generated before the proxy step, which matches the formal loop's first pass where v is still zero. Mathematically the KL term is KL(proxy(clean) ‖ proxy(adv)). In PyTorch, `F.kl_div(input, target)` expects the first argument in log-probability form and computes `target * (log(target) - input)`, so the call that implements KL(clean ‖ adv) is `F.kl_div(log_softmax(proxy(adv)), softmax(proxy(clean)), reduction='batchmean')`. I compute CE(proxy(clean), y) plus β times that KL, negate it, and take one SGD step on the proxy. Then I read off the difference, perturb the real model by +γ·diff, compute the same base loss under the perturbed weights with the same PyTorch argument order, backward, step the real optimizer, and restore by −γ·diff. That last detail — restore by subtracting exactly the perturbation I added — is the "come back to center" piece from the w-update written above: optimizer.step() moved the perturbed weights, and then I subtract γ·diff to land the center back where it should be.

Let me also be honest about the size of γ. It can't be too small or it doesn't flatten anything — the perturbation is too gentle to probe the steep directions. It can't be too large or the temporary model w+v becomes a poor training point instead of a local flatness probe. The useful band for the relative size is small, around γ ∈ [1e-3, 5e-3], and the KL-regularized implementation uses γ = 5e-3. The norm used for the relative size is the ordinary tensor norm, so for convolutional and linear weights I am using the Frobenius norm.

Let me write it as the per-step training procedure, filling the empty slot in the harness. The model, the SGD optimizer, the learning-rate schedule, and the data live outside; my job is one train_step: craft x' on the current (KL-maximizing) attack, compute the worst-case weight perturbation on a proxy, apply it, compute the base loss under the perturbed weights, step, and restore.

```python
import torch
import torch.nn.functional as F
from collections import OrderedDict

EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (_, new_w) in zip(model_state_dict.items(),
                                          proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        # PyTorch computes KL(target || input_distribution), so this is KL(clean || adv).
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss = -1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        return diff_in_weights(self.model, self.proxy)

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


def perturb_input_trades(model, images, eps, step_size, perturb_steps):
    model.eval()
    adv_images = images.detach() + 0.001 * torch.randn_like(images)
    adv_images = torch.clamp(adv_images, 0.0, 1.0)
    for _ in range(perturb_steps):
        adv_images.requires_grad_()
        loss_kl = F.kl_div(F.log_softmax(model(adv_images), dim=1),
                           F.softmax(model(images), dim=1),
                           reduction='sum')  # KL(clean || adv), maximized over adv
        grad = torch.autograd.grad(loss_kl, [adv_images])[0]
        adv_images = adv_images.detach() + step_size * torch.sign(grad.detach())
        adv_images = torch.min(torch.max(adv_images, images - eps), images + eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images.detach()


def train_step(model, images, labels, optimizer, awp_adversary,
               eps, step_size, perturb_steps, beta=6.0,
               epoch=0, awp_warmup=0):
    x_adv = perturb_input_trades(model, images, eps, step_size, perturb_steps)

    model.train()
    awp = None
    if epoch >= awp_warmup:
        awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                     inputs_clean=images,
                                     targets=labels,
                                     beta=beta)
        awp_adversary.perturb(awp)

    optimizer.zero_grad()
    logits_adv = model(x_adv)
    loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                           F.softmax(model(images), dim=1),
                           reduction='batchmean')  # KL(clean || adv)
    logits = model(images)
    loss_natural = F.cross_entropy(logits, labels)
    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()

    if awp is not None:
        awp_adversary.restore(awp)
    return {'loss': loss.item()}
```

Let me retrace the causal chain. Adversarial training flattens the input loss landscape but leaves a huge robust generalization gap that grows late in training — robust overfitting — and early stopping only papers over it by quitting with low training robustness. Measuring the *weight* loss landscape properly, with on-the-fly attacks and filter-normalized directions, shows that flatter weight landscapes track smaller robust gaps, and that the successful adversarial-training variants are already flattening this surface indirectly. So I made the flattening explicit: minimizing the loss-plus-its-flatness-penalty collapses to minimizing the loss at a perturbed weight, min_w ρ(w+v); a random v is too weak unless huge, so I take the worst-case v, which is a conservative upper bound on the expected sharpness in the PAC-Bayes flatness term. Sizing v relatively per layer, ‖v_l‖ ≤ γ‖w_l‖, respects ReLU scale invariance; one normalized-gradient ascent step on a proxy (to spare BN statistics) gives the direction, renormalized to γ‖w_l‖; I take the SGD step at w+v and then subtract v to keep the center weights honest. In the KL-regularized version, both the proxy ascent and the final descent use CE(clean) plus β·KL(clean ‖ adv), implemented with PyTorch's `kl_div(log_softmax(adv), softmax(clean))`, so the code follows the same objective I derived.
