Let me start from what the attacker actually sees, because that's the only thing I can fight. I have a classifier trained on a private set, and an adversary who hands me a sample and asks: was this in your training data? I want to drive that adversary down to a coin flip, and I want to do it without wrecking test accuracy. So the first question isn't "what defense" — it's "what signal is the adversary reading?" If I knew exactly the quantity it keys on, I'd know exactly what to flatten.

There's a clean answer to that, and it's worth getting precise about because it tells me where to aim. Sablayrolles and colleagues modeled the trained parameters as a posterior, `P(theta | z_1, ..., z_n) ∝ exp(-(1/T) sum_i m_i ell(theta, z_i))` — theta is whatever (approximately) minimizes the training risk, with a temperature `T` for the stochasticity of the optimizer. Now single out one sample `z_1` and ask for the Bayes-optimal guess of its membership bit. Bayes' rule gives the posterior probability as a sigmoid of a log-ratio; when you push the algebra through, the dependence on theta collapses. The score comes out as `s(z_1, theta, p) = (1/T)(tau_p(z_1) - ell(theta, z_1))`, where `tau_p(z_1) = -T log ∫ exp(-(1/T) ell(t, z_1)) p_T(t) dt` is just the typical loss `z_1` incurs under models that *didn't* train on it. Stare at that. The optimal attacker, even with the full white-box model in hand, reads off exactly one number: `ell(theta, z_1)`, the sample's loss. Everything else — the weights, the activations, the gradients — adds nothing once you have the loss. The optimal rule is "call it a member if its loss is below a threshold."

That is enormously clarifying. I don't need to defend against a thousand attack architectures. There is essentially one attack — threshold the per-sample loss — and every metric-based or learned attacker is an approximation of it. So my entire job reduces to a statement about *distributions*: let `P` be the distribution of per-sample loss over members, `Q` over non-members. If a threshold on the loss separates `P` from `Q`, the attacker wins; if `P` and `Q` overlap, it can't. Defense = make `P` and `Q` indistinguishable.

And there's a second result that tells me this is the *right* target and not a side issue. Yeom and colleagues defined membership advantage as `Pr[A=0|b=0] - Pr[A=0|b=1]` and built the simplest possible attacker: bounded loss `ell <= B`, output "non-member" with probability `ell/B`. Its advantage works out to `R_gen / B`, where `R_gen = E_{z~D}[ell] - E_{z~S}[ell]` is the generalization gap — expected test loss minus expected train loss. With 0-1 loss that's literally "membership advantage = generalization gap." So the gap between the typical member loss and the typical non-member loss is, by itself, a sufficient lever for an attack. Two papers, same conclusion from different directions: it's the *loss gap* that leaks.

Now where does that gap come from? Plain cross-entropy training is the culprit, and I can see exactly why from the gradient. With `ell_CE = -sum_c y^c log p^c` and `p = softmax(logits)`, the gradient with respect to the logits is `p - y`. That gradient vanishes as `p → y`, i.e. as the loss goes to zero. So nothing stops the optimizer from grinding member losses all the way down to ~0. Members pile up at zero; non-members, never optimized, sit at a higher loss. Two well-separated humps. On CIFAR-10 with a ResNet that's a loss-thresholding AUC around 0.84. The training procedure is *manufacturing* the very signal the attacker needs.

So the most obvious move: stop driving the loss to zero. Don't ask the model to be perfectly confident on its members. If member loss never collapses to ~0, the member hump sits closer to the non-member hump, the gap shrinks, the attacker's threshold has less to work with. Fine — but how do I *stop* at a nonzero loss? Cross-entropy's whole nature is to keep descending.

First instinct: regularize the confidence, which is exactly what the existing toolkit does. Label smoothing replaces the one-hot target with `(1-eps)` on the truth plus `eps/C` everywhere, equivalently adds `KL(uniform || p)` to the objective. Confidence penalty subtracts the predictive entropy, `L = L_CE - beta*H(p)`, whose logit gradient `p_i(-log p_i - H)` is a steady push toward higher entropy on every sample. Both lift the floor on member loss; both shrink the mean gap. So why aren't they enough? Let me think about what they do to the *shape* of `P`, not just its location. Both apply the *same* pressure to every sample: pull every output toward uniform by the same amount. They translate the member-loss hump to a higher mean, but they translate it more or less rigidly — the spread of `P` barely changes. And the non-member hump moves too, in the same direction. The two humps slide together; they don't *overlap*. A threshold attacker doesn't care about absolute loss level, it cares about separability, and rigidly shifting both distributions hardly changes how separable they are. That's the wall: mean-shifting regularizers leave the distributions just as distinguishable as before, only relocated.

So shrinking the mean gap is necessary but not sufficient. Let me look at the distributions again and ask what *else* makes two humps hard to tell apart. There's the distance between their means, yes — but there's also their widths. Two distributions with the same means but tiny variances are perfectly separable if they're offset at all; the same two distributions blurred into wide overlapping smears are not. And here's a fact about how these losses actually sit: the non-member distribution `Q` already has the larger variance — it's never been optimized, so its losses are all over the place — while the member distribution `P` is tight near zero. The member hump is narrow *and* low; that's a doubly easy target. So I want two things at once: bring `P`'s mean up toward `Q`'s, *and* spread `P` out so it stops being a tight spike. If I can do both, the humps don't just relocate — they bleed into each other.

The mean part I can simplify drastically. Trying to manipulate the full distributions is hopeless in practice — I'd need a large hold-out set of non-members just to estimate `Q`, and I'm assuming a knowledge-limited defender with no such luxury. So forget the full distribution and target the *mean*. Pick a target loss level `alpha > 0` — a value that is *achievable for non-members too*, not the unreachable zero — and aim the average member loss at `alpha` instead of at 0. Concretely: as long as the batch's mean loss is above `alpha`, the model is genuinely undertrained, so just do ordinary gradient descent. But once the mean loss drops *below* `alpha`, I don't want to keep descending — that would re-open the gap. I want to push it back *up* to `alpha`.

Pushing the loss up means gradient *ascent*. When the batch mean `L < alpha`, step `theta ← theta + tau * grad L` instead of `theta - tau * grad L`. That's a thermostat: descend when above `alpha`, ascend when below, settle around `alpha`. I can fold both directions into one differentiable objective the fixed optimizer can just minimize: minimize `|L - alpha|`. When `L > alpha`, `d|L-alpha|/dtheta = +grad L`, ordinary descent; when `L < alpha`, `d|L-alpha|/dtheta = -grad L`, which under the minimizer's "subtract the gradient" rule becomes `theta ← theta + tau*grad L`, ascent. One absolute-value loss encodes the whole thermostat, and its magnitude `|L-alpha|` even tapers the correction as I approach the target. Good.

But wait — is the ascent step doing anything *beyond* mean control? I claimed I also wanted to spread `P` out, and I haven't done that yet; I've only steered the mean. Let me actually look at what gradient ascent does sample-by-sample, because I suspect it does the spreading for free. Take one sample's loss change under an ascent step and Taylor-expand it. With the batch ascent `theta^{(t+1)} = theta^{(t)} + tau*grad L`,
```
ell(theta^{(t+1)}, z_i) = ell(theta^{(t)}, z_i) + tau <grad ell(theta, z_i), grad L(theta)> + o(tau),
```
so the per-sample loss change is
```
Delta ell_i = (tau/B) ||grad ell_i||^2 + (tau/B) sum_{j != i} <grad ell_i, grad ell_j> + o(tau).
```
Write the cross terms with the cosine of the angle between sample gradients, `<grad ell_i, grad ell_j> = ||grad ell_i|| ||grad ell_j|| cos(a_ij)`. To turn that Taylor term into a simple proportionality, I need the local batch gradients to have comparable norms, non-negative alignment, and roughly stable alignment factors across batches; under the same-norm version of that approximation this is
```
Delta ell_i ≈ (tau/B) ||grad ell_i||^2 (1 + sum_{j != i} cos(a_ij)) + o(tau) = c_1 ||grad ell_i||^2 + c_2,
```
with `c_1 = (tau/B)(1 + sum cos(a_ij)) > 0` when the gradients are non-negatively aligned, while the higher-order term is locally absorbed into `c_2`. So the loss *increase* a sample experiences under ascent is proportional to its squared gradient norm. Now connect that norm back to the loss itself. For cross-entropy, `nabla_theta ell_CE = J_theta^T(p - y)`, with `J_theta` the parameter-to-logit Jacobian. As a sample's loss shrinks, `p → y`, so `||p - y|| → 0`, so `||nabla_theta ell_CE||^2 → 0` under a bounded local Jacobian. The gradient norm is small for low-loss samples and large for high-loss samples — formally `Cov(||grad ell||^2, ell) > 0`. Chain the two: `Delta ell_i = c_1 ||grad ell_i||^2 + c_2` is linear in the squared norm with `c_1 > 0`, and the squared norm is positively correlated with the loss, so
```
Cov(ell, Delta ell) = Cov(ell, c_1 ||grad ell||^2 + c_2) = c_1 Cov(ell, ||grad ell||^2) > 0.
```
High-loss samples get pushed up *more* than low-loss ones. And that's exactly a variance-increasing operation, because
```
Var(ell + Delta ell) = Var(ell) + Var(Delta ell) + 2 Cov(ell, Delta ell),
```
and with `Var(Delta ell) >= 0` and the covariance strictly positive, `Var(ell + Delta ell) > Var(ell)`. The ascent step *spreads the member-loss distribution out* — it doesn't just shift the mean, it stretches the tail. That's the second lever, and it falls out of the same ascent I introduced for the mean. This is the thing the mean-shifting regularizers can't do: label smoothing and confidence penalty push every sample's loss up by roughly the same amount, leaving `Var(P)` essentially untouched; ascent pushes the already-high losses up *harder*, fattening `P` until it overlaps `Q`.

Let me make sure this actually buys lower attack AUC and isn't just a nice-sounding side effect. Treat the attack as a binary hypothesis test, member (`P`) vs non-member (`Q`). There's a chain of bounds. From the composition results, the optimal attacker's true- and false-positive rates obey `TP <= FP + min{D_TV(P, Q), 1 - FP}`, and integrating the ROC gives an AUC bound `AUC <= -1/2 D_TV(P,Q)^2 + D_TV(P,Q) + 1/2`. So shrinking the total-variation distance between `P` and `Q` shrinks the AUC ceiling. `D_TV` is awkward, so bound it by the Hellinger distance, `D_TV <= sqrt(2) D_H`, which for Gaussians has a closed form. With `P ~ N(mu_1, sigma_1^2)`, `Q ~ N(mu_2, sigma_2^2)` and `c = sigma_2/sigma_1`,
```
D_H^2(P, Q) = 1 - sqrt( 2c / (1 + c^2) ) * exp( -1/4 * (mu_1 - mu_2)^2 / ((1 + c^2) sigma_1^2) ).
```
Now read off what my two levers do. Shrinking the mean gap drops `(mu_1 - mu_2)^2`, which raises the exponential factor toward 1 and lowers `D_H`. Raising the member variance `sigma_1^2`, with `sigma_2` fixed, raises the denominator `sigma_1^2 + sigma_2^2`, so the exponential factor rises too. The ratio term needs a little care. Since `Q` starts wider than `P`, `c >= 1`; as I fatten `P`, `c` drops toward 1, and `sqrt(2c/(1+c^2))` is maximized at `c = 1` (its derivative `(1-c^2)/(sqrt(2c)(c^2+1)^{3/2})` is zero at 1 and negative beyond), so that prefactor rises toward its maximum. If I varied `c` downward while artificially freezing `sigma_1`, the exponential term would move the other way; the useful comparison is the combined one I actually have: the mean gap shrinks, the total variance denominator grows, and the variance-ratio prefactor moves toward equality. That shrinks the Hellinger distance in the intended regime, `D_TV <= sqrt(2) D_H` then shrinks the total-variation bound, and `-D_TV^2/2 + D_TV + 1/2` is increasing for `D_TV in [0,1]`, so the AUC ceiling drops. So: relax the mean toward `alpha` *and* let ascent fatten the tail, and the optimal attack's ceiling goes down.

Now the cost. I've stopped driving member losses to zero, which means the predicted probability of the ground-truth class, `p_gt`, is no longer pushed toward 1. That's the whole point for privacy — but it threatens utility in a specific way. If I let `p_gt` sit lower and the leftover mass `1 - p_gt` happens to concentrate on one competing class — which is exactly what happens on hard samples near a decision boundary between two classes — then some non-ground-truth class can end up with a probability *larger* than `p_gt`, and the argmax flips to the wrong class. I've relaxed confidence into misclassification. So relaxing the loss is good for privacy and bad for accuracy unless I do something about *where the non-ground-truth mass goes*.

The fix should preserve the privacy gain — I must not re-sharpen `p_gt` back toward 1, or I undo the whole defense — while protecting the argmax. The argmax is safe as long as `p_gt` beats every competitor, i.e. as long as there's a margin between `p_gt` and the largest non-ground-truth probability. So at a *fixed* `p_gt`, I want to maximize that margin, which means making the largest competitor as small as possible. With `1 - p_gt` mass to distribute over `C - 1` non-ground-truth classes, the way to minimize the maximum competitor is to spread that mass perfectly evenly: each gets `(1 - p_gt)/(C - 1)`. Any uneven allocation leaves some class higher, shrinking the margin. So construct a target distribution that keeps the current `p_gt` on the truth and flattens the rest:
```
t^c = p_gt                  if c is the ground-truth class
t^c = (1 - p_gt)/(C - 1)    otherwise
```
and train the prediction toward `t`. This is "posterior flattening": don't change how confident I am in the truth, just equalize my doubt across the alternatives so no single rival can overtake.

A couple of details matter for this to do the right thing. The target `t` is built from the model's *own* current `p_gt` — it's a moving target read off the forward pass. I must treat it as a *constant target*, not something to differentiate through: apply a stop-gradient to `t`. Otherwise the gradient would also flow back through the `p_gt` that sits inside `t` and start nudging it, which is exactly the confidence I'm trying to leave alone. So `t = stopgradient(...)`, and I match the prediction to it with a soft cross-entropy `ell_soft = -sum_c sg[t^c] log p^c`.

And there's the question of *which* samples to flatten. If a sample is already correctly classified, its argmax is fine; flattening it spends effort equalizing rivals that are not currently winning and can perturb a settled prediction. The samples that actually need the margin protection are the *misclassified* ones. So gate the soft-CE term by `(1 - correct_i)`, where `correct_i = 1` if the prediction matches the label — apply flattening only to the samples currently getting the wrong answer. Meanwhile I still want to keep pushing the loss up (the privacy direction), so subtract the per-sample cross-entropy: the per-sample objective becomes
```
loss_i = (1 - correct_i) * ell_soft_i - ell_CE_i,
```
and the batch loss is its mean. The `-ell_CE_i` term is ascent on the ordinary loss (minimizing `-ell_CE` maximizes `ell_CE`), keeping the loss elevated and the variance spreading even during this step; the `(1 - correct_i) ell_soft_i` term fixes the argmax of exactly the samples that need it, at fixed `p_gt`.

So I now have two operations near the target: the variance-spreading absolute-value thermostat, and the margin-fixing posterior flattening. I don't want the flattening branch to run every time the loss is small, because then it can swamp the ascent step. Alternate by epoch parity, but implement the even branch as the absolute-value objective directly. On an even epoch, minimizing `|L - alpha|` gives ordinary descent when `L > alpha` and ascent when `L < alpha`. On an odd epoch, if `L > alpha`, just do ordinary cross-entropy descent; only when `L <= alpha` does the posterior-flattening branch take over.

Let me assemble the whole per-batch rule. Compute the per-sample cross-entropy `ell_CE_i` and the batch mean `L`. If `epoch % 2 == 0`, use `loss = |L - alpha|`; above `alpha` this has the same descent direction as ordinary cross-entropy, and below `alpha` it ascends `L` and spreads the variance. If the epoch is odd and `L > alpha`, the model is still undertrained, so use ordinary descent, `loss = L`. If the epoch is odd and `L <= alpha`, use posterior flattening: build `t_i` from the detached `p_gt`, keep `p_gt` on the truth, spread the rest as `(1 - p_gt)/(C-1)`, and set `loss = mean_i [ (1 - correct_i) ell_soft_i - ell_CE_i ]`.

One leftover knob. The procedure carries an `upper` parameter that clamps `p_gt` into `[0, upper]` before building `t`. With `upper = 1.0` it does nothing — it can't clamp a probability below 1 — so it leaves the posterior unchanged. And `alpha` itself is the single hyperparameter trading privacy for utility: a value reachable by non-members. For this benchmark I can fix it from the class count, `alpha = 0.5` when `C == 100` and `alpha = 1.0` otherwise.

The causal chain in one breath: the optimal membership attack reads only the per-sample loss, so it sees only how the member-loss distribution `P` differs from the non-member `Q`; standard cross-entropy crushes `P` to a tight spike near zero, well-separated from `Q`, which is exactly the leak; so I relax the target mean to `alpha > 0` and use gradient ascent to hold the loss there, which simultaneously stretches `P`'s variance because high-loss samples move up faster — shrinking the mean gap and fattening the tail, the two changes that lower the optimal-attack AUC ceiling in the Gaussian/Hellinger analysis; and because relaxing confidence can flip the argmax of hard samples, I flatten the non-ground-truth posterior to maximize the margin at fixed `p_gt`, applied only to the misclassified samples, with the target detached so it never re-sharpens the confidence I just relaxed. Privacy and utility, from one relaxed objective.

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Training rule: relax the target loss to alpha, hold it there
    with gradient ascent (which also spreads the member-loss variance), and
    flatten the non-ground-truth posterior to keep the argmax correct."""

    def __init__(self):
        # upper-bounds p_gt before building the soft target; 1.0 is a no-op
        # (kept for faithfulness; matters only when p_gt is suppressed).
        self.upper = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.size(1)
        # single hyperparameter: a target loss level reachable by non-members
        alpha = 0.5 if num_classes == 100 else 1.0

        loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
        loss_ce = loss_ce_full.mean()                          # batch mean loss L

        if epoch % 2 == 0:
            # thermostat: descend |L - alpha|. Above alpha -> descent;
            # below alpha -> ascent toward alpha, which also fattens Var(P).
            return (loss_ce - alpha).abs()

        # odd epoch
        if loss_ce > alpha:
            # still undertrained -> ordinary cross-entropy descent
            return loss_ce

        # posterior flattening: keep p_gt, spread the rest evenly to maximize
        # the margin between the truth and the largest competitor.
        probs = torch.softmax(logits, dim=1)
        p_gt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_gt = torch.clamp(p_gt, min=0.0, max=self.upper)
        p_else = (1.0 - p_gt) / (num_classes - 1)

        onehot = F.one_hot(labels, num_classes=num_classes).float()
        soft_targets = onehot * p_gt.unsqueeze(1) + (1.0 - onehot) * p_else.unsqueeze(1)
        soft_targets = soft_targets.detach()                   # target is a constant

        log_probs = F.log_softmax(logits, dim=1)
        ce_soft = -(soft_targets * log_probs).sum(dim=1)       # soft cross-entropy

        correct = logits.argmax(dim=1).eq(labels).float()
        # flatten only the misclassified (fix their argmax); subtract per-sample
        # CE on all (ascent on the loss, keep it elevated and spreading).
        loss = (1.0 - correct) * ce_soft - loss_ce_full
        return loss.mean()
```
