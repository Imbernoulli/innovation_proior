Let me start from the class-wise diagnostics for classifiers trained on corrupted labels, because the standard story and those curves don't quite agree. The standard story, from Zhang et al. and Arpit et al., is that a high-capacity network first fits the clean, easy patterns and only later memorizes the wrong labels — so label-noise damage is overfitting, and the cure is to stop the late-stage memorization. Fine. But let me not take the aggregate accuracy curve at face value; let me break it out by class, because cross entropy never treated all classes alike even on clean data. And there it is: with clean labels the per-class test-accuracy curves fan out into a wide band — some classes ("easy") shoot up fast, others ("hard") crawl. That's intrinsic; some classes just have more separable patterns. Under 40% uniform label noise the band doesn't just shift down uniformly. The easy classes climb high and then start to *droop* — that's the overfitting everyone talks about. But the hard classes? They plateau far below where they sit on clean data and they never recover. Class 3 ends around 60% while class 6 sits above 90%. On the *clean portion* of a hard class — the examples whose labels were never flipped — the network's confidence on the correct class hovers around 0.5, with a few percent leaking onto visually-similar classes. So on examples it should find easy to get right, it's barely committing.

That reframes the whole problem for me. The dominant cost here isn't the easy-class overfitting — that accuracy drop is a few points. The dominant cost is that the hard classes are *under-learned*: cross entropy, under noise, isn't pushing them hard enough to ever reach their clean-label ceiling. So a fix that only suppresses memorization — label smoothing, say — is attacking the smaller leak; the label-smoothing diagnostic eases the easy-class droop but still leaves hard classes stuck low. So whatever I add to CE has to do two things at once: supply *extra learning pressure* aimed at the hard, half-learned classes, and be *noise-tolerant* itself, so that in supplying that pressure it doesn't just turn around and memorize the flipped labels. Those two requirements pull in opposite directions — "push harder" and "don't overfit" — and that tension is the real design constraint.

So the dumb first idea: if CE under-learns hard classes, just turn CE up. Use `2*CE`, `5*CE`. Bigger gradients, harder push. Let me think about what that does. CE's gradient on the true-class logit is `p_y - 1`, and scaling CE by `c` scales every gradient by `c`. But scaling is indiscriminate — it amplifies the pull toward the *observed* label whether that label is clean or flipped. On a flipped example the observed label is wrong, and `5*CE` pulls five times harder toward the wrong class. So upscaling CE accelerates exactly the memorization I'm trying to avoid; it makes overfitting *worse*, not better, and it doesn't selectively help the hard clean examples. Wall. The extra push cannot come from more CE, because CE has no notion of "this target might be a lie." I need a different, *robust* source of learning pressure.

What does "robust" even mean precisely? I want the formal version, not a vibe. Ghosh, Kumar and Sastry pinned it down: a loss `L` is noise-tolerant if the global minimizer of the risk under noisy labels is also a global minimizer of the clean risk. And they gave a clean sufficient condition — a loss is robust if it is *symmetric*, meaning `sum_{i=1}^K L(f(x), i) = C` for some constant `C`, the same for every `x` and every classifier. Let me re-derive why that condition buys robustness, because the algebra is going to be my template. Under uniform noise at rate `eta`, each clean label `y` is, with probability `1-eta`, kept, and with probability `eta` flipped uniformly to one of the other `K-1` classes. So the noisy risk is

```
R^eta(f) = E_{x,yhat} L(f(x), yhat)
         = E_x E_{y|x} [ (1-eta) L(f(x), y) + (eta/(K-1)) sum_{k != y} L(f(x), k) ].
```

The inner sum over wrong classes is `sum_{k != y} L = sum_{k=1}^K L - L(f(x), y) = C - L(f(x), y)` precisely *because* the total over all classes is the constant `C`. Substitute:

```
R^eta(f) = (1-eta) R(f) + (eta/(K-1)) ( C - R(f) )
         = C*eta/(K-1) + ( 1 - eta*K/(K-1) ) R(f).
```

So the noisy risk is just an affine function of the clean risk: a constant plus a positive multiple of `R(f)`, where the multiplier `1 - eta*K/(K-1)` is positive as long as `eta < (K-1)/K = 1 - 1/K`. A positive affine map preserves argmin. So whoever minimizes `R^eta` minimizes `R` — robustness, with no dependence on the data distribution, only on the noise rate staying below `1 - 1/K`. The property to engineer is now precise: the loss summed over all classes must be a constant.

Now check CE against that test. `sum_{k=1}^K -log p_k` — that's not constant; it depends on `p`, and it's unbounded as any `p_k -> 0`. So CE flatly fails the symmetry condition, which is the formal reason it has no robustness guarantee and memorizes noise. What *does* satisfy it? Ghosh's answer is mean absolute error. For a one-hot label, `ell_mae = sum_k |p_k - q_k| = (1 - p_y) + sum_{k != y} p_k = 2(1 - p_y)`, and summing over which class you call the target, `sum_i ell_mae(f(x), i) = 2 sum_i (1 - p_i) = 2(K-1)`, a constant. So MAE is symmetric, hence robust. Tempting — just use MAE then? But I know why nobody does: look at MAE's gradient on the true-class logit. `ell_mae = 2(1 - p_y)`, and `d p_y / d z_y = p_y(1 - p_y)`, so `d ell_mae / d z_y = -2 p_y(1 - p_y)`. That vanishes both as `p_y -> 0` and as `p_y -> 1`. The `p_y -> 1` part is fine — a well-learned example needs no push. But `p_y -> 0` is a disaster: a hard example the network is currently getting *wrong* (`p_y` small) gets an almost-zero gradient. MAE refuses to push exactly the under-learned examples I'm trying to rescue. That's why MAE trains painfully slowly and under-learns hard classes even worse than CE. Wall. So I'm caught: CE has the gradients but not the robustness; MAE has the robustness but not the gradients.

GCE tried to thread this — `ell_q = (1 - p_y^q)/q`, a Box-Cox interpolation that is CE-like as `q -> 0` and MAE-like at `q = 1`, so one knob `q` slides between fit and robustness. It's a real improvement, but notice the shape of the solution: it's one scalar trading off two things on a single axis, and the whole framing is still "make CE more like MAE." On a dataset that's intrinsically hard to fit, I might need CE's convergence *and* a lot of robustness, and a single `q` can't give me both — push `q` toward robustness and convergence suffers, push it toward fit and robustness suffers. I want two *separate* forces I can dial independently, not one dial sliding between a robust loss and a fitting loss. Let me hold that thought: keep CE for what it's good at (fast, informative gradients, convergence), and *add* a robust term for what CE lacks — instead of deforming CE into MAE.

So the question becomes: what robust term do I add to CE? It has to be symmetric (sum-over-classes constant, by the Ghosh argument), and ideally it should be a *cross-entropy-shaped* object so it sits naturally next to CE. Let me go back to what CE is, structurally, and ask whether there's a sibling hiding in plain sight. CE is `H(q, p) = -sum_k q_k log p_k`, the cross entropy of the observed label distribution `q` under the predictive distribution `p`. Through `KL(q||p) = H(q,p) - H(q)` with `H(q)` constant for a fixed label distribution, minimizing CE *is* minimizing `KL(q||p)` — fitting the prediction `p` toward the observed label `q`. And that is precisely the thing that hurts under noise: when `q` is a flipped label, `KL(q||p)` drags `p` toward a lie. The KL divergence is asymmetric — this direction measures the cost of using `p` to code samples drawn from `q`, so it treats `q` as ground truth. But under noise `q` is *not* ground truth; if anything, a competent `p` reflects the truth better than a corrupted `q` does. So why am I only measuring discrepancy in the direction that trusts `q`? Measure it the other way too: `KL(p||q)`, which penalizes coding samples drawn from `p` using a code built for `q`; now the prediction is the reference distribution. Symmetrize:

```
SKL = KL(q||p) + KL(p||q).
```

Now transfer that symmetry from KL to cross entropy. `KL(q||p)` corresponds to `H(q,p) = CE`. The mirror, `KL(p||q)`, corresponds to `H(p, q) = -sum_k p_k log q_k` — the same cross-entropy formula with `p` and `q` swapped. Call it Reverse Cross Entropy:

```
ell_rce = - sum_{k=1}^K p_k log q_k.
```

And the combined objective is `SCE = CE + RCE = H(q,p) + H(p,q)`, the cross-entropy analogue of symmetric KL. I like that this *adds* a term rather than deforming CE — exactly the two-forces shape I wanted, CE pulling `p` toward `q`, RCE pulling in the mirror direction.

But wait — RCE has `q` *inside* the logarithm, and `q` is the one-hot label. For a single true label, `q_y = 1` and `q_k = 0` for `k != y`. So RCE has `log q_k = log 0` for every wrong class. That's `-infinity`. The whole term is ill-defined. Wall. Most deep-learning frameworks clip probabilities to avoid `log 0`; let me do the analogous thing here and just *define* `log 0 := A`, where `A` is some finite negative constant. Now is this a hack I'll have to apologize for, or does it do real work? Let me write RCE out with the definition. With `q_y = 1` (so `log q_y = log 1 = 0`) and `q_{k != y} = 0` (so `log q_k = A`):

```
ell_rce = - p_y * log 1 - sum_{k != y} p_k * A = -A sum_{k != y} p_k = -A (1 - p_y),
```

using `sum_{k != y} p_k = 1 - p_y`. So RCE collapses to `-A(1 - p_y)` — a clean, finite quantity. And now the crucial check: is RCE *symmetric* in Ghosh's sense? Sum it over which class plays the role of the target. `ell_rce(f(x), i) = -sum_k p_k log q^{(i)}_k`, where `q^{(i)}` is the one-hot at class `i`, so `log q^{(i)}_i = 0` and `log q^{(i)}_{k != i} = A`; thus `ell_rce(f(x), i) = -A sum_{k != i} p_k = -A(1 - p_i)`. Summing over `i`:

```
sum_{i=1}^K ell_rce(f(x), i) = -A sum_{i=1}^K (1 - p_i) = -A (K - sum_i p_i) = -A(K - 1),
```

a constant, independent of `x` and of `f`. So RCE satisfies the symmetry condition with `C = -(K-1)A`, and by the Ghosh argument I re-derived above, RCE is noise-tolerant for `eta < 1 - 1/K`. The `log 0 := A` definition wasn't a numerical band-aid — it's the very thing that makes the term symmetric and therefore robust. Let me even run the robustness proof for RCE explicitly to be sure the constant flows through. Plug `C = -(K-1)A` into the affine identity:

```
R^eta(f) = C*eta/(K-1) + (1 - eta*K/(K-1)) R(f)
         = -A*eta + (1 - eta*K/(K-1)) R(f),
```

so `R^eta(f^*) - R^eta(f) = (1 - eta*K/(K-1))(R(f^*) - R(f)) <= 0` for `eta < 1 - 1/K` and `f^*` the clean minimizer — RCE's clean minimizer is its noisy minimizer too. Robust, confirmed, with the algebra closing on the constant the `log 0 := A` choice supplied.

The class-dependent case needs stronger assumptions, so I should not let the uniform-noise proof pretend to cover it. Let the true class `y` stay correct with probability `1 - eta_y` and flip to a particular wrong class `k` with probability `eta_yk`, with `sum_{k != y} eta_yk = eta_y`. Then the noisy RCE risk is

```
R^eta(f) = E_{x,y} [ (1 - eta_y) ell_rce(f(x), y) + sum_{k != y} eta_yk ell_rce(f(x), k) ].
```

Use the class-sum identity to rewrite the true-label term as `ell_rce(f(x), y) = -(K-1)A - sum_{k != y} ell_rce(f(x), k)`. Then

```
R^eta(f) = -(K-1)A * E_{x,y}(1 - eta_y)
           - E_{x,y} sum_{k != y} (1 - eta_y - eta_yk) ell_rce(f(x), k).
```

Now compare the noisy minimizer `f_eta^*` with the clean minimizer `f^*`: since `R^eta(f_eta^*) <= R^eta(f^*)`,

```
E_{x,y} sum_{k != y} (1 - eta_y - eta_yk)
  (ell_rce(f^*(x), k) - ell_rce(f_eta^*(x), k)) <= 0.
```

If the transition matrix is diagonally dominant in the sense `eta_yk < 1 - eta_y`, every coefficient is positive. If also `R(f^*) = 0`, then on clean labels RCE can be zero only with `p_y = 1`, so for every wrong label `k`, `ell_rce(f^*(x), k) = -A`. Any classifier has `ell_rce(f(x), k) = -A(1 - p_k) <= -A` because `A < 0`. Each bracket is therefore nonnegative, and a positive weighted sum of nonnegative terms can be `<= 0` only if every bracket is zero. That forces `p_k = 0` for all `k != y`, hence `p_y = 1`, so `f_eta^* = f^*`. The asymmetric guarantee is real, but it rests on diagonal dominance and zero clean RCE risk; the uniform-noise guarantee is the cleaner distribution-free one.

I should sanity-check what `A` *is*, because right now it's a free knob and free knobs make me nervous. RCE for a sample is `-A(1 - p_y)`. Compare to MAE, which I showed is `2(1 - p_y)`. These are the *same function of `p_y`* up to the multiplier: `-A(1-p_y)` versus `2(1-p_y)`. So when `A = -2`, RCE is *exactly* MAE. That's a lovely landmark — RCE isn't some exotic new loss, it's a one-parameter generalization of MAE, where `A` is a steepness knob MAE doesn't expose. Pushing `A` more negative makes the robust term steeper (larger penalty per unit of `1 - p_y`); `A = -2` recovers plain MAE. So I've recovered a known robust loss as a special case, which tells me I'm on the right one-parameter family rather than off in the weeds.

Now, is adding RCE actually going to fix the *under-learning* I started from, or just add robustness? The robustness story is settled; the under-learning story needs the gradient. Let me derive the gradient of the combined loss with respect to the logits and read off its effect. Take the simplified case with both terms weighted equally first; I'll add coefficients after. `ell_sl = ell_ce + ell_rce = -sum_k q_k log p_k - sum_k p_k log q_k`. Differentiate with respect to logit `z_j`:

```
d ell_sl / d z_j = - sum_k q_k (1/p_k) (d p_k / d z_j) - sum_k (d p_k / d z_j) log q_k.
```

I need the softmax Jacobian. With `p_k = e^{z_k} / sum_m e^{z_m}`, the quotient rule gives, for `k = j`, `d p_k/d z_j = (e^{z_k} (sum_m e^{z_m}) - (e^{z_k})^2)/(sum_m e^{z_m})^2 = p_k - p_k^2 = p_k(1 - p_k)`; and for `k != j`, the numerator's first term drops (`e^{z_k}` doesn't depend on `z_j`), leaving `d p_k/d z_j = -e^{z_k} e^{z_j}/(sum_m e^{z_m})^2 = -p_k p_j`. Substitute these into both sums. The first (CE) sum: `-sum_k q_k (1/p_k)(d p_k/d z_j) = -[ q_j (1/p_j) p_j(1-p_j) + sum_{k!=j} q_k (1/p_k)(-p_j p_k) ] = -q_j(1-p_j) + p_j sum_{k!=j} q_k = -q_j + q_j p_j + p_j(1 - q_j) = p_j - q_j`, using `sum_k q_k = 1`. Good — that's the familiar `p_j - q_j` CE gradient. The second (RCE) sum: `-sum_k (d p_k/d z_j) log q_k = -[ p_j(1-p_j) log q_j + sum_{k != j}(-p_j p_k) log q_k ] = -p_j(1-p_j)log q_j + p_j sum_{k!=j} p_k log q_k = p_j( sum_{k} p_k log q_k - log q_j )`. Adding the two:

```
d ell_sl / d z_j = (p_j - q_j) + p_j ( sum_{k=1}^K p_k log q_k - log q_j ).
```

Now read it in the two cases that matter. Case `q_j = q_y = 1` (the labeled class). Then `log q_j = log 1 = 0`, and `sum_k p_k log q_k = p_y log 1 + sum_{k != y} p_k log 0 = sum_{k != y} p_k * A = (1 - p_y) A = (1 - p_j) A` (since `j = y`). So

```
d ell_sl / d z_j = (p_j - 1) + p_j ( (1 - p_j) A - 0 ) = (p_j - 1) - (A p_j^2 - A p_j),
```

where `(p_j - 1)` is exactly the CE gradient on the true-class logit. The RCE contribution is `A p_j(1 - p_j)`, equivalently `-(A p_j^2 - A p_j)` because `A p_j^2 - A p_j = -A p_j(1 - p_j)`. Since `A < 0` and `p_j(1-p_j) >= 0` on `[0,1]`, the contribution added to the gradient is `<= 0` — it adds to the downhill push on the true-class logit, so gradient descent raises `p_y` faster. How much? The magnitude is `(-A) p_j(1 - p_j)`, an upside-down parabola in `p_j` on `[0,1]`, zero at the ends and maximal at `p_j = 0.5`. So the extra acceleration is largest precisely when the network is half-sure of the true class — a hard, half-learned example sitting at `p_y ~ 0.5` — and it tapers off as `p_y -> 1`, where the example is already learned and needs no extra shove. That's exactly the adaptive pacing the under-learning problem called for: speed up the classes stuck around `0.5`, ease off the ones already near `1` so they don't keep getting hammered into overfitting. And it answers my earlier worry about `2*CE` — CE upscaling pushed everything harder including toward wrong labels; this term is gated by `p_j(1-p_j)`, so it concentrates on uncertain labeled-class probabilities.

Case `q_j = 0` (a wrong class). Then `log q_j = log 0 = A`, and the `sum_k p_k log q_k` is over the labeled class `y` (`log q_y = 0`) and the other zero classes (`log = A`): `sum_k p_k log q_k = sum_{k != y} p_k A = (1 - p_y)A`. So

```
d ell_sl / d z_j = (p_j - 0) + p_j ( (1 - p_y) A - A ) = p_j + p_j ( -A p_y ) = p_j - A p_j p_y,
```

where `p_j` is the CE gradient on a non-target logit. The RCE contribution is `-A p_j p_y`, which is `>= 0` since `A < 0`; under gradient descent, a positive derivative lowers the wrong-class logit, and its size scales with `p_y`, the confidence on the labeled class. So if the network is confident the labeled class is right (`p_y` large), suppress the residual wrong-class mass hard; but if `p_y ~ 0` — the network isn't buying the label at all, which is exactly what a *flipped* label looks like from the inside — then `-A p_j p_y ~ 0`, no suppression. The robust term *self-gates*: it only sharpens predictions the network already half-believes, and it goes quiet on examples the network implicitly flags as mislabeled. That's robustness emerging directly from the gradient, on top of the population-level guarantee from the symmetry proof.

So RCE earns its place twice over: symmetric ⇒ provably noise-tolerant, and its gradient adaptively accelerates the under-learned (`p_y` near 0.5) classes while declining to push on likely-mislabeled examples. CE earns its place because, although it's not robust, it gives the dense, well-conditioned gradients that make the network *converge* in the first place — RCE/MAE alone converge slowly, as the saturating `p_y(1-p_y)` gradient showed. Two complementary forces. Now I should give myself the freedom to weight them, because the right balance won't be one-size-fits-all. Write

```
ell_sl = alpha * ell_ce + beta * ell_rce.
```

Why two separate coefficients rather than a single mixing `alpha` and `1-alpha`? Because the two knobs do genuinely different jobs and I want them decoupled. `alpha` scales CE, which is the source of overfitting — so *lowering* `alpha` is how I ease memorization of noise. `beta` scales RCE, the robust learning pressure — so *raising* `beta` is how I add noise-tolerant signal. On a dataset that converges easily, like CIFAR-10 at moderate noise, I can afford a small `alpha` (say 0.1) to suppress overfitting and lean on RCE. But on a dataset notorious for hard convergence — CIFAR-100, many classes, fewer examples each — I actually need *more* CE to converge at all, so I want a large `alpha`. A single `alpha`/`(1-alpha)` trade-off couldn't express "lots of CE for convergence *and* meaningful robust signal" the way two free coefficients can. This is exactly the independence that GCE's single `q` couldn't provide.

One more thing to settle about `A`, and its relationship with `beta`. RCE for a sample is `-A(1 - p_y)`, so `beta * ell_rce = beta * (-A) * (1 - p_y)` — the product `beta * (-A)` is what sets the RCE strength, so `beta` and `A` overlap in effect, and tuning one can stand in for tuning the other. That's why in practice one fixes `A` to a sensible negative constant and tunes the coefficients: if CE's overfitting is already controlled by a small `alpha`, the result is fairly insensitive to the exact `A`; only when CE overfitting is poorly controlled does `A` start to matter. And I should *not* use label smoothing to soften `q` instead of clamping it — smoothing puts `epsilon/(K-1)` mass on every class, which biases the model at *every* point including the clean `q_y = 1` ones; clamping `q_k` from `0` up to a tiny floor introduces a controlled bias only at the `q_k = 0` points (finite, and the very thing that makes the term symmetric) and *no* bias where `q_y = 1`. Less distortion, and it's what gives me the constant `A`.

Now, how do I realize `log 0 := A` in actual tensor code, where `q` is a one-hot vector? I don't literally special-case `log 0`; I clamp the one-hot label up to a small positive floor before taking its log. If I floor `q` at, say, `1e-4`, then `log(1e-4) ~ -9.21`, so choosing the floor chooses `A`. (A floor of `e^{-4} ~ 0.018` realizes `A = -4`; a floor of `1e-4` realizes `A ~ -9.21`.) I also clamp the prediction `p` away from `0` before its own log inside CE, the standard guard every framework uses, so neither cross entropy blows up. In the Keras loss factory the model already supplies class probabilities `y_pred` and one-hot labels `y_true`; CE is `-sum y_true*log(y_pred)`, RCE is `-sum y_pred*log(clipped_y_true)`, and the full objective is `alpha * mean(CE) + beta * mean(RCE)`.

Let me write it into the one open slot — the loss object — keeping the structure minimal and faithful to the Keras implementation: clip the predicted probabilities for the CE logarithm, clip the one-hot labels to instantiate `A`, then return the weighted sum.

```python
import tensorflow as tf


def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred
        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)  # A = log(1e-4)

        ce = tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis=-1))
        rce = tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis=-1))
        return alpha * ce + beta * rce

    return loss
```

I broke CE's behavior out by class and found the real damage under noise isn't only easy-class overfitting but persistent *under-learning* of hard classes, so the fix had to both push hard classes and stay noise-tolerant. Upscaling CE failed because CE is non-robust — it just memorizes flipped labels harder. Ghosh's criterion told me robustness means the loss summed over all classes is constant; CE fails that, MAE passes it but its gradient vanishes on the very hard examples I need to push, and GCE's single knob couldn't keep CE's convergence and add robustness independently. Reading CE as minimizing `KL(q||p)` and noticing `q` is the untrustworthy quantity under noise, I symmetrized — `KL(q||p)+KL(p||q)` — and carried that symmetry to cross entropy, getting Reverse-CE, `-sum_k p_k log q_k`. The `log 0` from one-hot labels forced a finite constant `A := log 0`, and that very constant makes RCE's class-sum equal `-(K-1)A`, constant, so RCE is symmetric and — by Ghosh's affine-risk argument with `C = -(K-1)A` — provably noise-tolerant for `eta < 1 - 1/K`; `A = -2` recovers MAE exactly, placing RCE in a known robust family. The gradient derivation showed RCE adds `A p_y(1-p_y) <= 0` to the true-class logit gradient, largest in magnitude when `p_y ~ 0.5`, and adds `-A p_j p_y >= 0` to each wrong-class logit gradient, silent when the model puts almost no mass on the labeled class. Keeping CE for convergence and RCE for robust pressure, with two decoupled coefficients `alpha` and `beta`, gives `alpha*CE + beta*RCE`, dropped into the Keras loss slot with `A` realized by clamping the one-hot label to a small floor.
