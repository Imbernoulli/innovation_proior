Let me start from what actually goes wrong when I train a deep net on labels I don't fully trust. I have a softmax classifier `f(x)` with `f_j` the predicted probability of class `j`, and I train it by descending a per-example loss summed over minibatches. Some fraction of my training labels are wrong — flipped to another class — and I don't know which ones. The thing that makes this dangerous, and not just annoying, is capacity: these networks can drive training error to zero even on completely random labels, so they have no built-in reluctance to memorize garbage. And what people have watched happen on partially-noisy data is a two-phase process: early on the network picks up the genuinely predictive structure, which mostly comes from the clean majority, and is implicitly *correct* on many of the corrupted examples; then, if I keep training, it goes back and memorizes the wrong targets, and clean-test accuracy falls off. So the enemy is the second phase. I want an objective that still learns the real structure but refuses to be dragged into fitting the corrupted labels.

The standard objective is cross entropy, `L = -log f_y`. Why does it overfit noise? I shouldn't just assert it; let me look at the gradient, because that's what actually moves the weights. For a softmax output the per-sample parameter gradient of `-log f_y` is

  `∂(-log f_y)/∂θ = -(1/f_y) ∇_θ f_y`.

There's the `1/f_y` factor staring at me. A sample where the model assigns low probability to the given label — a sample it currently "disagrees" with — gets a gradient blown up by `1/f_y`. So cross entropy is implicitly a hard-example weighter: it pours effort into the points it's getting wrong. On clean data that's a feature; the hard points are where the signal is. But on noisy data it's the exact mechanism of the disease. A poisoned example, where the true content of the image disagrees with the flipped label, is precisely a point the model is confidently *wrong* about relative to that label, so `f_ỹ` is small, so `1/f_ỹ` is large, so cross entropy yanks hard to memorize the corrupted target. The very weighting that helps on clean labels is what burns me on noisy ones. So if I want robustness, I have to kill, or at least tame, that `1/f_y` amplification of low-confidence samples.

Is there a loss that doesn't have it? Yes — there's a clean theoretical story I can lean on. Call a loss *symmetric* if, for some constant `C`,

  `sum_{j=1}^c L(f(x), j) = C`   for every `x` and every `f`.

The reason to care: under uniform noise (a wrong label is equally likely to be any of the other `c-1` classes, with rate `η`), I can expand the noisy risk by conditioning on the true label `y`,

  `R^η_L(f) = E_x E_{y|x}[ (1-η) L(f,y) + (η/(c-1)) sum_{i≠y} L(f,i) ]`.

Now `sum_{i≠y} L(f,i) = sum_{i=1}^c L(f,i) - L(f,y)`, and if the loss is symmetric that first piece is just the constant `C`. Substitute:

  `R^η_L(f) = E_x E_{y|x}[ (1-η) L(f,y) + (η/(c-1))(C - L(f,y)) ] = Cη/(c-1) + (1 - ηc/(c-1)) R_L(f)`.

So the noisy risk is nothing but an affine function of the clean risk: a constant, plus a coefficient `1 - ηc/(c-1)` times the clean risk. As long as `η < (c-1)/c` that coefficient is positive, the transform is order-preserving, and the minimizer of `R_L` is also the minimizer of `R^η_L`. That's noise tolerance, and it fell out with no estimation of the noise, no clean data, no extra network — purely from a property of the loss. This is exactly the lever I want, and it's distribution-free.

Which usable loss is symmetric? Mean absolute error, on a softmax output. With one-hot target `e_j`, `L_MAE = ||e_j - f||_1 = 2 - 2 f_j` (using `sum_k f_k = 1`), and summing over classes, `sum_{j=1}^c (2 - 2 f_j) = 2c - 2`, a constant — symmetric. (Up to a constant this is the unhinged loss `1 - f_j`.) So MAE is provably noise-tolerant. Look at its gradient too:

  `∂(2 - 2 f_y)/∂θ = -2 ∇_θ f_y`,

up to the constant just `-∇_θ f_y`. No `1/f_y`. Every sample is weighted equally, regardless of how confident the model is. That's *why* it's robust — there's no amplification of the low-confidence (and therefore likely-noisy) samples.

So I should just train with MAE? I want to believe it, but the symmetry that buys robustness is exactly what wrecks the learning dynamics, and I should be honest about that before I commit. With `-∇_θ f_y` and no `1/f_y`, MAE gives the same gradient magnitude to a sample the model already nails and to a sample it's badly wrong about. The hard, informative points — the ones where there's actually something to learn — get no extra pull. On a small clean problem that's survivable; on a large, hard dataset trained with stochastic minibatches, it's a disaster. Without the implicit emphasis on difficult samples, the gradient signal that matters gets averaged into noise, convergence crawls, and the final accuracy lands well below cross entropy. And this isn't speculation — it's what's observed: on CIFAR-10, MAE converges far more slowly than CCE; on CIFAR-100 it's far worse, topping out around 38% test accuracy even after thousands of epochs, long after cross entropy has converged to a higher number in a handful of epochs. So I'm wedged between two failures. Cross entropy learns hard examples beautifully but memorizes noise through its `1/f_y` weighting. MAE refuses to memorize noise but won't learn hard examples because it has no weighting at all. Wall.

Stare at the two gradients side by side, because the contrast is the whole problem in one line:

  CCE:  `-(1/f_y) ∇_θ f_y`     (weight on sample = `1/f_y`, huge for low-confidence)
  MAE:  `-      ∇_θ f_y`        (weight on sample = `1`, flat).

The per-sample weight is `1/f_y` for one and `1` for the other. These are the two endpoints of a knob I haven't named yet. `1/f_y` is `f_y^{-1}`; `1` is `f_y^{0}`. So both are `f_y` raised to a power — power `-1` versus power `0`. What if I let that exponent be a free parameter? If the per-sample gradient weight were `f_y^{q-1}` for some `q`, then `q = 0` gives weight `f_y^{-1}`, recovering cross entropy, and `q = 1` gives weight `f_y^{0} = 1`, recovering MAE. Everything strictly between would be a partial down-weighting of the low-confidence samples — robustness dialed somewhere between "full hard-example weighting" and "none." That's exactly the tradeoff I'm trying to hit. The question is what *loss* has gradient weight `f_y^{q-1}`.

Work backwards from the gradient. I want `∂L/∂θ = - f_y^{q-1} ∇_θ f_y`. Treat `L` as a function of the scalar `p = f_y`: I need `dL/dp · ∇_θ f_y` to equal `- p^{q-1} ∇_θ f_y`, so `dL/dp = -p^{q-1}`. Integrate: `L(p) = -∫ p^{q-1} dp = - p^q / q + const`. Pin the constant so that a perfect prediction `p = 1` gives zero loss: `L(1) = -1/q + const = 0`, so `const = 1/q`, and

  `L_q(f(x), e_j) = (1 - f_j(x)^q) / q`.

That's it — the loss whose gradient weight is `f_y^{q-1}` by construction. And it's a recognizable object: it's the negative Box-Cox transformation of `f_j` (Box & Cox 1964), the same power-transform family used in robust statistics; Ferrari & Yang (2010) used it to define a maximum-`L_q`-likelihood estimator, where pushing `q` away from zero deliberately discounts low-probability observations for robustness — exactly the flavor I'm after, just transplanted from likelihood estimation onto a classifier's softmax. So I'm not inventing a strange one-off; I'm borrowing a known robustification knob.

Let me check the endpoints directly on the loss, not just the gradient, because I want to be sure this really *is* a generalization and not just something that happens to match at the derivative. At `q = 1`: `L_1 = (1 - f_j)/1 = 1 - f_j`. That's the unhinged loss, which is MAE up to a constant. Good. At `q = 0` the formula is `0/0`, so I take the limit. By L'Hôpital in `q`,

  `lim_{q→0} (1 - f_j^q)/q = lim_{q→0} d/dq(1 - f_j^q) / d/dq(q) = lim_{q→0} (-f_j^q log f_j)/1 = -log f_j`,

which is exactly categorical cross entropy. So `L_q` continuously interpolates: `q→0` is cross entropy, `q=1` is MAE, and the parameter `q ∈ (0,1]` slides between learnability and robustness. One loss, one knob, both endpoints I care about recovered as limits. That's the sign I have the right family rather than a third competitor bolted on beside them.

Now I want to nail the gradient interpretation cleanly, because the *why* of the knob lives there. Differentiate `L_q = (1 - f_y^q)/q` with respect to the parameters:

  `∂L_q/∂θ = (1/q)·(-q f_y^{q-1}) ∇_θ f_y = - f_y^{q-1} ∇_θ f_y = f_y^q · (-(1/f_y) ∇_θ f_y)`.

I wrote it two ways on purpose. The last form, `f_y^q · (CCE gradient)`, says: `L_q` is cross entropy with each sample's gradient multiplied by `f_y^q`. Since `f_y ∈ [0,1]` and `q > 0`, `f_y^q ∈ [0,1]`, so this is a *down-weighting* — and it bites hardest exactly where `f_y` is small, i.e. on the low-confidence samples, which are the likely-noisy ones. That's the robustness, relative to cross entropy. The middle form, `f_y^{q-1} · ∇_θ f_y` with `q-1 < 0`, compares it to MAE: relative to MAE's flat weight of `1`, `L_q` *up-weights* low-confidence samples by `f_y^{q-1} > 1`, restoring some of cross entropy's attention to hard points. So `L_q` sits between them on both readings simultaneously: more robust than CCE because it discounts uncertain samples, more trainable than MAE because it still leans into the hard ones. Larger `q` discounts the uncertain samples more aggressively — more robust, but the optimization gets harder as I approach MAE's flat-gradient pathology; smaller `q` keeps the dynamics close to cross entropy — easier to train, less robust. The knob *is* the robustness-versus-learnability tradeoff, made continuous.

I've argued robustness from the gradient, but I claimed cross entropy isn't noise-tolerant and MAE is, via the symmetry theorem. Where does `L_q` fall on that spectrum? It's not symmetric for general `q` — only the `q=1` endpoint is — so I can't just invoke the theorem. I need to see how *close* to symmetric it is, because the theorem's whole leverage was that `sum_j L(f,j)` is constant; if that sum is merely close to constant, the argument should degrade gracefully and I'd get an approximate tolerance with an error I can size. So let me bound `sum_{j=1}^c L_q(f, e_j) = sum_j (1 - f_j^q)/q` and see how far it ranges as `f` varies over the simplex.

Upper bound first. Since each `f_j ≤ 1` and `q ≤ 1`, I have `f_j ≤ f_j^q` (raising a number in `[0,1]` to a smaller positive power increases it), so `1 - f_j^q ≤ 1 - f_j`. Sum:

  `sum_j (1 - f_j^q)/q ≤ sum_j (1 - f_j)/q = (c - sum_j f_j)/q = (c - 1)/q`,

using `sum_j f_j = 1`. Lower bound: I want to minimize `sum_j (1 - f_j^q) = c - sum_j f_j^q`, i.e. maximize `sum_j f_j^q` over the simplex. Since `t ↦ t^q` is concave for `q ∈ (0,1]`, by Jensen (or the power-mean inequality) `sum_j f_j^q` is maximized at the uniform point `f_j = 1/c`, giving `sum_j (1/c)^q = c · c^{-q} = c^{1-q}`. So `sum_j f_j^q ≤ c^{1-q}`, hence `sum_j(1 - f_j^q) ≥ c - c^{1-q}`, and

  `(c - c^{1-q})/q ≤ sum_{j=1}^c (1 - f_j^q)/q ≤ (c - 1)/q`.

So the symmetry defect — the range of that class-sum — is the gap between `(c - c^{1-q})/q` and `(c-1)/q`. At `q = 1` it's `(c - 1)/q` on both ends (`c^{1-q} = c^0 = 1`), the sum is exactly constant, MAE is perfectly symmetric, consistent. As `q → 0` the gap widens (`c^{1-q} → c`, lower end `→ 0`-ish while upper stays `(c-1)/q → ∞`), cross entropy is far from symmetric. Good — the bound tracks the intuition: `q` near 1 means near-symmetric means near-tolerant.

Now push the affine-risk argument through with these bounds instead of an exact constant. Repeat the uniform-noise expansion, but keep the class-sum term instead of replacing it by `C`:

  `R^η_{L_q}(f) = (1 - ηc/(c-1)) R_{L_q}(f) + (η/(c-1)) E_x E_{y|x}[ sum_{i=1}^c L_q(f(x), i) ]`.

Substitute the two-sided bound on the inner sum. The lower-bound side gives `sum_i L_q ≥ (c - c^{1-q})/q`, the upper-bound side `sum_i L_q ≤ (c-1)/q`:

  `(1 - ηc/(c-1)) R_{L_q}(f) + η[c - c^{1-q}]/(q(c-1)) ≤ R^η_{L_q}(f) ≤ (1 - ηc/(c-1)) R_{L_q}(f) + η/q`.

Now I can invert this to bracket the *clean* risk in terms of the noisy risk. Let `m = 1 - ηc/(c-1) > 0` (so I need `η < (c-1)/c` for the inversion). From the two inequalities,

  `(R^η_{L_q}(f) - η/q)/m ≤ R_{L_q}(f) ≤ (R^η_{L_q}(f) - η[c - c^{1-q}]/(q(c-1)))/m`.

Let `f*` minimize the clean risk and `f̂` minimize the noisy risk. Compare them. Since `f̂` minimizes `R^η`, `R^η(f*) ≥ R^η(f̂)`, so `R^η(f*) - R^η(f̂) ≥ 0`. Apply the bracketing at both `f*` and `f̂` and difference. The cleanest statement that comes out, after the bookkeeping, is

  `0 ≤ R^η_{L_q}(f*) - R^η_{L_q}(f̂) ≤ A`,  with  `A = η[c^{1-q} - 1]/(q(c-1)) ≥ 0`,

and equivalently, on the clean risk,

  `A' ≤ R_{L_q}(f*) - R_{L_q}(f̂) ≤ 0`,  with  `A' = η[1 - c^{1-q}]/(q(c - 1 - ηc)) < 0`.

Let me make sure I read these right. `f*` is the classifier I *want* (clean optimum); `f̂` is what minimizing the noisy loss actually gives me. The second line says the clean risk of `f̂` exceeds that of `f*` by at most `|A'|` — the noisy-optimum is within `|A'|` of the true optimum *measured on clean data*. So `|A'|` is the price of noise, and I want it small. Look at how `q` controls it. The factor `1 - c^{1-q}`: at `q = 1`, `c^{1-q} = 1`, so `A = A' = 0` — exact noise tolerance, recovering the MAE/symmetric result as a special case, which is the consistency check I need. As `q → 0`, `c^{1-q} → c`, so `|A'|` grows — the bound loosens toward cross entropy, which has no tolerance guarantee. So the larger I make `q`, the tighter the noise-tolerance bracket. This is the *same* tradeoff the gradient tells me, now from the risk side: large `q` is more robust. And it confirms `L_q` is a graded relaxation of the symmetric-loss tolerance — exact at `q=1`, degrading smoothly as `q` decreases. (For class-dependent noise the analogous noisy-risk gap bound holds when the clean optimum is perfect, `R_{L_q}(f*) = 0`, and each correct label is more likely than any particular wrong label: `B = (c^{1-q} - 1)/q · E[1 - η_y] ≥ 0`, with the same `c^{1-q} - 1` factor vanishing at `q = 1`. The label-flip corruption I face is a structured class-dependent noise, so this condition is the relevant one.)

So both views agree the choice of `q` is a real tension and not a free win: crank `q` toward 1 and the noise-tolerance bound tightens but the optimization slides toward MAE's flat-gradient stall; pull `q` toward 0 and training is easy but the tolerance bound blows up. There's no `q` that's best on both axes, which means `q` has to be picked in the middle, where the loss is robust *enough* while the gradient still has enough `f_y^{q-1}` curvature to drive learning. Empirically the sweet spot on these image benchmarks sits around `q = 0.7` — robust enough that overfitting to the noise is suppressed across the noise rates that matter, while convergence stays close to cross entropy's. It's genuinely a hyperparameter (one could tune it by watching validation accuracy, and noisier data wants a larger `q`), but `0.7` is the value that hits the compromise on this kind of data, so I'll take it as the default.

There's a sharper move available if I want even more robustness, and it's worth deriving because it shows where the symmetry argument was leaving slack. The tolerance bound was driven by the symmetry *defect* — the gap in `sum_j L_q(f,j)`. A tighter class-sum would give a tighter bound. What inflates that sum is the contribution of classes where `f_j` is small: a wrong class with tiny probability still contributes nearly `1/q` to the per-class loss. So cap the loss: once `f_j` drops below a threshold `k`, stop letting it matter, by flattening the loss to a constant there. Define

  `L_trunc(f, e_j) = L_q(k)` if `f_j ≤ k`, else `L_q(f, e_j)`,   with `L_q(k) = (1 - k^q)/q`.

Below the threshold the loss is constant, so its gradient is zero — that sample stops contributing to the update entirely. In effect, a sample on which the model is very unconfident about its given label (very likely a poisoned one) gets *pruned* from this step. As `k → 0` this collapses back to plain `L_q`. The reason this is more robust is the same symmetry-defect logic: assuming `k ≥ 1/c`, truncation bounds the class-sum between `d L_q(1/d) + (c-d)L_q(k)` and `c L_q(k)`, where `d = max(1, (1-q)^{1/q}/k)`. Its range is therefore `d[L_q(k) - L_q(1/d)]`, and that is smaller than the plain `L_q` range `(c^{1-q}-1)/q` whenever `d[L_q(k) - L_q(1/d)] < (c^{1-q}-1)/q`, which holds comfortably for, say, `k ≥ 0.3` across all `q` and `c` — so the truncated loss is strictly more noise-tolerant.

But I have to be careful using truncation directly, and seeing why tells me how to handle it. At the very start of training every softmax is near-uniform, so `f_y` is around `1/c`, well below a threshold like `k = 0.5` — which would prune almost the entire dataset on the first epoch, and pruning by confidence before the model knows anything is exactly backwards. The fix is to notice that minimizing the truncated loss is equivalent to a weighted problem with binary weights `v_i ∈ {0,1}`,

  `argmin_θ sum_i L_trunc(f_i, y_i) = argmin_θ sum_i [ v_i L_q(f_i, y_i) + (1 - v_i) L_q(k) ]`,

with `v_i = 1` exactly when `f_{y_i} > k` (equivalently `L_q(f_i) ≤ L_q(k)`), and then relax `v` to a free `w ∈ [0,1]^n`, giving `argmin_{θ, w} sum_i w_i L_q(f_i) - L_q(k) sum_i w_i`. If the model-side loss is convex in `θ`, this objective is biconvex because it is linear in `w`, so alternating convex search is the clean idealization: hold `θ`, set each `w_i` to its optimum (the binary pruning rule), then hold `w`, take gradient steps on `θ`. For the DNN I actually care about, the same alternation is only a local training procedure, so I initialize `w_i = 1` for everyone and let pruning kick in only after the model has learned enough to make `f_y > k` meaningful. That's a self-paced-style pruning loop riding on top of the same `L_q`. It's the right thing when I want to squeeze out maximum robustness, but it adds a per-step pruning pass and an alternating schedule.

For a strict drop-in objective — one scalar loss per minibatch, nothing touching the training loop — the plain `L_q` is what I want to land on, and the truncated version is the optional, heavier extension on top of it. So the contribution that fills the loss slot is `L_q` itself: take the softmax, read off the probability the model gives to the (possibly corrupted) label, raise it to the power `q`, and average `(1 - f_y^q)/q` over the batch. One subtlety to get right numerically: `f_y` can be essentially zero for a confidently-wrong sample, and `0^q` is fine but the gather can underflow, so I floor the probability at a tiny epsilon before the power, which costs nothing in the regime that matters and keeps the loss finite. The whole thing is a handful of tensor ops with no state and no schedule.

```python
import torch


class RobustLoss:
    """Generalized cross entropy (L_q) loss for label noise.

    L_q(f(x), e_y) = (1 - f_y^q) / q, averaged over the minibatch, where f_y is the
    softmax probability assigned to the given (possibly corrupted) label. q in (0,1]
    interpolates: q -> 0 is categorical cross entropy, q = 1 is MAE/unhinged loss.
    Relative to CCE the per-sample gradient is scaled by f_y^q, down-weighting
    low-confidence (likely-noisy) samples; relative to MAE it is f_y^{q-1}, restoring
    attention to hard samples so the network still trains.
    """

    def __init__(self):
        self.q = 0.7  # robustness <-> learnability tradeoff; ~0.7 is the compromise on image data

    def compute_loss(self, logits, labels, epoch):
        probs = torch.softmax(logits, dim=1)                 # f(x): class probabilities
        p = probs.gather(1, labels[:, None]).clamp_min(1e-8)  # f_y: prob of the given label, floored
        return ((1.0 - p.pow(self.q)) / self.q).mean()        # (1 - f_y^q)/q, mean over the batch
```

Let me retrace the chain. I need a loss that learns real structure but won't memorize flipped labels, and the gradients tell me why neither default works: cross entropy's `1/f_y` factor over-weights low-confidence samples — perfect for hard clean examples, fatal because poisoned examples look exactly like hard examples and get memorized; MAE drops that factor entirely, which buys provable noise tolerance through symmetry but flattens the gradient so hard examples get no pull and deep training stalls. The two per-sample weights, `f_y^{-1}` and `f_y^{0}`, are the two ends of a single exponent, so I let the gradient weight be `f_y^{q-1}` and integrate back to the loss `(1 - f_y^q)/q`, the negative Box-Cox transform, which recovers cross entropy as `q→0` and MAE at `q=1`. Its gradient is `f_y^q` times cross entropy's — a down-weighting of uncertain samples that gives robustness without killing the hard-example pull. Bounding the symmetry defect `sum_j L_q(f,j)` between `(c - c^{1-q})/q` and `(c-1)/q` and pushing it through the same affine-risk expansion that proves symmetric losses tolerant gives a graded noise-tolerance bracket whose width is governed by `c^{1-q} - 1` — zero at `q = 1` (exact tolerance, MAE recovered) and widening as `q → 0`, the same robustness-versus-learnability tradeoff the gradient shows, so `q` lands in the middle, around `0.7`. Truncating the loss below a confidence threshold tightens that bracket further for extra robustness, at the cost of a self-paced pruning loop, so I keep it as an optional extension and let the drop-in objective be plain `L_q`: softmax, gather the label's probability, clamp it for stability, `(1 - f_y^q)/q`, mean over the batch.
