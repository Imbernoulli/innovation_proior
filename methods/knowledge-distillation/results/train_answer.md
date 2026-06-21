The setting is large-scale supervised classification with deep networks, where the best accuracy comes either from a single very large, heavily-regularized net or from an ensemble of separately trained nets whose predictive distributions are averaged. Both are accurate and both are too cumbersome to deploy: the ensemble multiplies inference cost by the number of members, and the giant net carries a heavy memory and latency footprint. Training and deployment have opposite constraints — training can spend unlimited offline compute, deployment is latency- and resource-bound — yet the same model is forced to serve both. We need to move what the cumbersome model has learned into a single small, fast net while preserving most of its accuracy. The conceptual obstacle is identifying *what* the knowledge of a trained model even is. If we equate knowledge with the specific learned weights, then changing the architecture destroys it and the whole enterprise is hopeless; that framing is useless. The right abstraction is that the knowledge is the learned *mapping from inputs to a distribution over classes*, which is architecture-free and could in principle be reproduced by a net of a completely different shape. Compression is then not "copy the weights" but "reproduce the function."

The existing options each fall short of this. Training the small net directly on hard one-hot labels gives it only the single correct answer per example, so with its limited capacity it never sees the rich inter-class similarity structure the big model discovered and must rediscover it from scratch; it generalizes worse. Strong regularization of the small net alone — dropout, weight constraints, input jitter — improves it but cannot inject the big model's learned invariances, which came from transformations the small net never sees. The closest prior recipe, Caruana-style compression, trains the small net to match the big model's *logits* under a squared-error loss; this is motivated by a real problem (more on that below) and it works, but it is an ad-hoc target with no clear connection to a probabilistic transfer objective, and it is rigid: every logit is fitted with equal weight, including very negative logits that the teacher's own training barely constrains and that may simply be noise, with no knob to vary how much they count.

Look concretely at the function the big model implements. Given an image of a BMW it does not just say "BMW"; it emits a full distribution in which BMW takes almost all the mass, garbage-truck gets a tiny sliver, and carrot a far tinier one. The crucial fact is that garbage-truck is *many times* more probable than carrot — that ratio is not noise, it is the model reporting that a BMW looks much more like a garbage truck than like a carrot. This relative-probability structure over the wrong classes — the dark knowledge — is the most valuable thing to transfer, it is a free side effect of training to maximize the correct-class log-probability, and the hard label discards all of it. So the small net should be trained to match the big net's output *distribution* on a transfer set; soft targets carry far more information per case than a one-hot label and their gradient varies less across cases, so the student can learn from less data and at a higher learning rate.

The method I propose is **knowledge distillation**. The naive realization — minimize the cross-entropy $-\sum_i p_i \log q_i$ between the student's softmax $q$ and the teacher's softmax $p$ — fails for a sharp reason. On a task like MNIST the teacher is essentially always right with enormous confidence, so the soft target for a "2" is roughly $0.99999$ on "2" and the genuinely interesting information lives in probabilities down at $10^{-6}$ (looks a bit like a 3) and $10^{-9}$ (not at all like a 7). In the cross-entropy the term for "3" is weighted by $p_3 \approx 10^{-6}$, so it contributes essentially nothing to the loss or the gradient: the very information we wanted is invisible because the softmax has squashed it to the floor. This is exactly why logit matching surfaced it — logits are unbounded, a logit producing a $10^{-6}$ probability is an ordinary finite number, and squared error weights it on par with the big ones — but I want to recover that benefit from inside the probabilistic story rather than from an arbitrary target. The fix is to un-squash the probabilities before forming the target by introducing a temperature $T$ into the softmax,
$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}.$$
At $T=1$ this is the ordinary softmax; as $T$ rises the logits are divided down, the exponentials are pulled toward each other, and the relative mass on the small-logit classes grows. A high temperature lifts the dark knowledge out of the floor into probabilities large enough to matter. So I produce soft targets $p_i = \mathrm{softmax}(v/T)_i$ from the frozen teacher's logits $v$, and train the student's softmax at the *same* temperature $T$ to match them; after training the student reverts to $T=1$ for deployment.

When true labels are available I want to use them too, since the student cannot match the soft targets perfectly and a nudge toward the correct answer helps. I deliberately do *not* fold the correct label into the soft targets (which would blend two different signals into one distribution); instead I keep the teacher's softened distribution intact and add an ordinary hard-label cross-entropy beside it, letting a coefficient decide how much the hard label is allowed to pull. The soft loss is $L_{\text{soft}} = KL(p_T \| q_T)$, which equals the soft-target cross-entropy up to the teacher's fixed entropy, evaluated at temperature $T$; the hard loss $L_{\text{hard}}$ is ordinary cross-entropy against the true labels at $T=1$; and the objective is
$$L = w_{\text{soft}}\, T^2\, L_{\text{soft}} + w_{\text{hard}}\, L_{\text{hard}}.$$

Two derivation steps make this construction load-bearing. First, take one transfer case with student logit $z_i$, teacher logit $v_i$, $p_i = \mathrm{softmax}(v/T)_i$, $q_i = \mathrm{softmax}(z/T)_i$, and soft cross-entropy $C = -\sum_i p_i \log q_i$. The cross-entropy gradient with respect to the input of a softmax is (output $-$ target), and here that input is $z_i/T$, so the chain rule pulls out a $1/T$:
$$\frac{\partial C}{\partial z_i} = \frac{1}{T}(q_i - p_i).$$
Now push to high temperature. With $T$ large relative to the logit magnitudes, $\exp(x/T) \approx 1 + x/T$, so each softmax fraction becomes $(1 + z_i/T)/(N + \sum_j z_j/T)$ with $N$ the number of classes. Subtracting a constant from all logits leaves the softmax unchanged, so I am free to zero-mean each case's logits, $\sum_j z_j = \sum_j v_j = 0$; both denominators collapse to $N$, the $1$'s cancel, and
$$\frac{\partial C}{\partial z_i} \approx \frac{1}{N T^2}(z_i - v_i),$$
which is, up to a positive scalar, the gradient of $\tfrac12\sum_i (z_i - v_i)^2$ — exactly Caruana's squared logit-matching objective. So that target was never ad hoc: it is precisely what temperature-$T$ distillation becomes as $T \to \infty$. And the knob I wanted falls out of the same expansion. At intermediate $T$ the approximation $\exp(x/T) \approx 1 + x/T$ breaks for logits that are large and negative: such a logit makes $\exp(z_i/T)$ much smaller than $1 + z_i/T$ would suggest, the soft target $p_i$ there is even nearer zero, and the gradient pays *much less* attention to matching it. Lowering $T$ from infinity is therefore a dial that down-weights the very negative, possibly-noisy logits, while high $T$ matches all of them equally. Whether ignoring them helps or hurts is a genuine empirical question — they are nearly unconstrained by the teacher's training so they could be noise, but they could encode real structure — and distillation is what gives me the dial; logit matching is stuck at one end of it. For a very small student, intermediate $T$ is often the better capacity allocation, since limited capacity is better spent on the salient structure than on chasing every extremely negative logit, and the sweet-spot $T$ drops as the student shrinks.

Second, the $T^2$ factor. The soft-target gradient just derived scales as $1/T^2$, whereas the hard-target term runs at $T=1$ and so does not shrink with $T$. If I added the two with fixed weights and then swept $T$, the soft contribution would silently collapse relative to the hard one as $T$ grows and my balance would drift out from under me. Multiplying the soft loss (equivalently its gradients) by $T^2$ cancels the $1/T^2$, so the soft contribution is roughly scale-invariant in $T$ and the relative weight of hard and soft stays fixed while I sweep temperature. In practice I compute $KL(p_T \| q_T)$ at temperature $T$, scale it by $T^2$, and add the optional small-weighted hard cross-entropy at $T=1$, with the teacher frozen throughout. This also predicts what the mechanism should do: because the softened targets carry cross-class similarity, invariances the teacher learned from jittered inputs can ride along even when the student sees only unjittered cases, and in the stress test where an entire digit class is withheld from the transfer set, that class is still represented through other digits' softened targets, so any failure should appear as an output-bias problem rather than as the class vanishing.

```python
import torch
import torch.nn.functional as F

def transfer_loss(small_logits, large_logits, hard_targets=None,
                  temperature=1.0, soft_weight=1.0, hard_weight=0.0):
    T = float(temperature)

    # Soft transfer: KL(p_T || q_T), same gradient as soft-target CE.
    with torch.no_grad():
        p = F.softmax(large_logits / T, dim=1)
    log_q = F.log_softmax(small_logits / T, dim=1)
    soft = F.kl_div(log_q, p, reduction="batchmean") * (T * T)

    loss = soft_weight * soft
    if hard_targets is not None and hard_weight:
        # Optional hard-label CE at T=1.
        hard = F.cross_entropy(small_logits, hard_targets)
        loss = loss + hard_weight * hard
    return loss

# Training (large model frozen; transfer set = train set or a separate set)
large_model.eval()
opt = torch.optim.SGD(small_model.parameters(), lr=0.1, momentum=0.9)
T, hard_weight = 20.0, 0.0                                # set hard_weight > 0 when labels should also pull
for x, y in transfer_loader:
    with torch.no_grad():
        large_logits = large_model(x)
    small_logits = small_model(x)
    loss = transfer_loss(small_logits, large_logits, y, temperature=T,
                         soft_weight=1.0, hard_weight=hard_weight)
    opt.zero_grad(); loss.backward(); opt.step()
# Deployment: small_model runs an ordinary T=1 softmax.
```
