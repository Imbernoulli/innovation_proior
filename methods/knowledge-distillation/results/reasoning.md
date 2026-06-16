OK, let me think this through from scratch. The thing I actually have is a model that's too expensive to deploy — either an ensemble of nets whose predictions I average, or one giant net regularized to the teeth with dropout. It's accurate. But at deployment I want something small and fast. So the question is: how do I move what the big thing learned into a small net?

My first instinct is the obvious thing people already do: take the big model, run it on a pile of data, and train the small model to copy its answers. Caruana and his collaborators did exactly this — compress an ensemble into one small net by labeling a transfer set with the ensemble and fitting the small net to those labels. So the recipe exists. The question that's nagging me is *what* the small net should copy, and why copying works at all.

I keep hitting a conceptual block. I'm tempted to think the "knowledge" of a trained net *is* its weights — the specific parameter values it settled into. But if that's the knowledge, then changing the architecture throws it all away, and the whole enterprise is hopeless. So that framing is wrong, or at least useless. Let me reframe: the knowledge is the *mapping* the model has learned, from an input to a distribution over outputs. That's architecture-free. A small net of a totally different shape could in principle implement the same mapping. Now compression isn't "copy the weights," it's "reproduce the function."

So what does the function look like, concretely? The big model, given an image, doesn't just say "BMW." It puts out a full distribution: BMW gets almost all the mass, but garbage-truck gets a tiny sliver, and carrot gets an even tinier one. And — this is the part that matters — garbage-truck is *many times* more probable than carrot. That ratio isn't noise; it's the model telling me it thinks a BMW looks a lot more like a garbage truck than like a carrot. That's a rich similarity structure over the classes, and it's a *side effect* of training to maximize the correct-class log-prob. The hard label "BMW" throws all of that away. If I train the small net only on hard labels, it never sees that a BMW is truck-ish and not carrot-ish; it has to rediscover all of that from limited capacity and limited data. But the big model already worked it out. So the soft outputs — the full distributions — are the thing to transfer. They carry far more information per training case than a one-hot label, and the gradient varies less from case to case, so the small net should be trainable on much less data and with a higher learning rate.

Good, so: train the small net to match the big net's output *distribution* on a transfer set. Let me just write the cross-entropy between the small net's softmax and the big net's softmax and call it a day.

But wait — stare at this. Take MNIST. The big net is essentially always right with enormous confidence. So for a given "2", the soft target is like 0.99999something on "2", and the genuinely interesting information — this 2 looks a bit like a 3 ($10^{-6}$) and not at all like a 7 ($10^{-9}$) — lives in probabilities down at $10^{-6}$ and $10^{-9}$. Now in the cross-entropy $-\sum_i p_i \log q_i$, the term for "3" is weighted by $p_{\text{3}} \approx 10^{-6}$. It contributes essentially nothing to the loss and essentially nothing to the gradient. So the very information I said was valuable has almost no influence on training. The near-1 mass on the correct class drowns everything out. So naive probability-matching mostly transfers "it's a 2," which I already knew from the hard label. The dark knowledge is there in the targets but it's invisible to the objective.

This is exactly why Caruana et al. matched *logits* instead of probabilities, with a squared error between the student's logits and the teacher's logits. Logits are unbounded; a logit that produces a $10^{-6}$ probability is a perfectly ordinary, finite number, sitting maybe 14 below the top logit, and squared error treats it on equal footing with the big ones. So logit matching surfaces the small-probability structure. It works. But it bugs me as a target. It's pulled out of nowhere relative to the probabilistic story — why squared error on logits? And it forces the student to match *every* logit equally, including logits that are very negative. Those very negative logits are almost completely unconstrained by the teacher's own training (the teacher's loss barely cares whether a logit is $-20$ or $-30$, both give negligible probability), so they could be mostly noise. Maybe matching them is harmful; maybe it's helpful because even noise-ish logits encode something. I'd like a method with a knob that lets me dial how much attention to pay to those.

Let me go back to the softmax and ask: the reason the small probabilities are invisible is that the softmax has *squashed* them down near zero. What if I un-squash them before computing the target? Introduce a temperature into the softmax:

  q_i = exp(z_i / T) / Σ_j exp(z_j / T).

At T=1 this is the ordinary softmax. As I raise T, the distribution gets softer — the logits get divided down, the exponentials get pulled toward each other, and the relative mass on the small-logit classes goes up. So a high temperature *lifts the dark knowledge out of the floor* and into probabilities big enough to matter in a cross-entropy. I can produce the soft targets by running the teacher's softmax at a high temperature, and train the student's softmax at the *same* high temperature to match them. After training, the student goes back to T=1 for deployment.

So the simplest form of distillation: soft target $p_i = \mathrm{softmax}(v/T)_i$ from teacher logits $v$, student trains to minimize cross-entropy $-\sum_i p_i \log q_i$ with $q_i = \mathrm{softmax}(z/T)_i$, same $T$ on both sides.

Now I want to also use the true hard labels when I have them — the student can't perfectly match the soft targets, and nudging it toward the actually-correct answer should help. One option is to fold the correct label into the soft targets, by bumping the true class. But that mixes two different signals into one target distribution. Cleaner: keep the large model's softened distribution intact, add an ordinary hard-label objective beside it, and let a coefficient decide how much the hard label is allowed to pull. So

  L = w_soft · CE_soft(T) + w_hard · CE_hard(T=1),

Before I trust those weights, I need to look at the gradients, because the temperature is going to scale them.

Take one transfer case. The student logit is $z_i$, the teacher logit is $v_i$, soft target $p_i = \mathrm{softmax}(v/T)_i$, student soft output $q_i = \mathrm{softmax}(z/T)_i$. The soft cross-entropy is $C = -\sum_i p_i \log q_i$. I need $\partial C / \partial z_i$. The standard softmax-cross-entropy gradient with respect to the *logit-before-temperature-scaling* picks up a $1/T$ from the chain rule, because $q$ depends on $z_i/T$. Working it out, the gradient of cross-entropy with respect to the input of a softmax is (output − target), and here the softmax input is $z_i/T$, so

  ∂C/∂z_i = (1/T) (q_i − p_i) = (1/T) ( exp(z_i/T)/Σ_j exp(z_j/T) − exp(v_i/T)/Σ_j exp(v_j/T) ).

Now take the high-temperature limit. If $T$ is large compared to the magnitudes of the logits, then $z_i/T$ is small, and $\exp(z_i/T) \approx 1 + z_i/T$. Substitute into both fractions:

  ∂C/∂z_i ≈ (1/T) ( (1 + z_i/T)/(N + Σ_j z_j/T) − (1 + v_i/T)/(N + Σ_j v_j/T) ),

where $N$ is the number of classes (the sum of the $1$'s in each denominator). This is messy because of the denominators. But suppose I've zero-meaned the logits separately for each transfer case — i.e. $\sum_j z_j = 0$ and $\sum_j v_j = 0$. Subtracting a constant from all logits doesn't change the softmax, so this is a free normalization. Then both denominators collapse to $N$, and

  ∂C/∂z_i ≈ (1/T) ( (1 + z_i/T)/N − (1 + v_i/T)/N ) = (1/(NT)) ( z_i/T − v_i/T ) = (1/(N T²)) ( z_i − v_i ).

The $1$'s cancel. So in the high-$T$ limit, the gradient of the soft cross-entropy is proportional to $(z_i − v_i)$. That means minimizing the soft cross-entropy at high temperature is equivalent, up to a positive scalar, to minimizing $\tfrac{1}{2}\sum_i (z_i − v_i)^2$ — exactly Caruana's logit-matching squared error. The ad-hoc squared-error target isn't ad hoc after all; it is what temperature-$T$ distillation becomes as $T \to \infty$.

And the knob I wanted is right here. At *intermediate* temperature I'm not in that limit, and the approximation $\exp(z/T)\approx 1+z/T$ breaks for logits that are large and negative — a logit far below the average makes $\exp(z_i/T)$ much smaller than $1+z_i/T$ would suggest, so the soft target $p_i$ for that class is even closer to zero, and the gradient pays *much less* attention to matching very negative logits. So lowering $T$ from infinity is exactly a dial that *down-weights* the very negative, possibly-noisy logits, while high $T$ matches all of them. Whether ignoring the big negative logits helps or hurts is genuinely an empirical question — they're nearly unconstrained by the teacher's training so they could be noise, but they could also encode real structure. The point is distillation gives me the dial; logit matching is stuck at one end of it.

Now the weighting of hard vs soft. The soft-target gradient I just derived scales as $1/T^2$ (look at the $1/(NT^2)$). The hard-target term runs at $T=1$, so its gradient does *not* shrink with $T$. If I just add the two with fixed weights and then change $T$ while tuning, the soft term's contribution silently collapses relative to the hard term as $T$ grows, and my soft/hard balance drifts out from under me. I need the coefficient to mean the same thing as I sweep temperature, so I multiply the soft-target gradients, equivalently the soft loss, by $T^2$. Then the soft contribution is roughly scale-invariant in $T$, and the relative weight of hard and soft stays put while I sweep temperature. Concretely I'll compute the KL divergence, or equivalently the cross-entropy up to the teacher's fixed entropy, at temperature $T$, multiply it by $T^2$, and add an optional small-weighted hard loss at $T=1$.

Let me also sanity-check what this mechanism would have to explain before coding. MNIST is useful because the large regularized net can learn invariances from dropout, weight constraints, and jittered inputs, while the transfer set for the small net can be the original unjittered cases. If matching softened outputs really transfers the input-to-distribution mapping, then some of that invariance can ride along in the targets even when the student never sees the transformed images directly. A sharper stress test is to remove a digit class from the transfer set: the student has no hard examples of that class, but every other digit's softened target still assigns some probability to it in proportion to visual similarity. If the story is right, the missing class is represented through those cross-class similarities, and any failure should show up partly as an output-bias problem rather than as complete absence of the class.

One more practical point for very small students. If the student is far too small to absorb everything, very high $T$ (matching all the negative logits) may be worse than an intermediate $T$ — limited capacity is better spent on the salient structure than on chasing nearly-unconstrained negative logits. So I expect a sweet spot in $T$ that gets lower as the student shrinks.

At implementation time the small model forward gives logits; I compute $KL(p_T\|q_T)$ between the large model's temperature-softmax targets and the small model's temperature-softmax outputs, scale it by $T^2$, optionally add ordinary cross-entropy against hard labels at $T=1$, and optimize. The large model is frozen.

```python
import torch
import torch.nn.functional as F

def transfer_loss(small_logits, large_logits, hard_targets=None,
                  temperature=1.0, soft_weight=1.0, hard_weight=0.0):
    T = float(temperature)

    # Soft term: KL(p_T || q_T), same gradient as soft-target CE.
    with torch.no_grad():
        p = F.softmax(large_logits / T, dim=1)
    log_q = F.log_softmax(small_logits / T, dim=1)
    soft = F.kl_div(log_q, p, reduction="batchmean") * (T * T)

    loss = soft_weight * soft
    if hard_targets is not None and hard_weight:
        # Hard term: ordinary cross-entropy at T=1.
        hard = F.cross_entropy(small_logits, hard_targets)
        loss = loss + hard_weight * hard
    return loss
```

Training loop (large model frozen, small model learns; transfer set can be the train set or a separate set):

```python
large_model.eval()
opt = torch.optim.SGD(small_model.parameters(), lr=0.1, momentum=0.9)
T, hard_weight = 20.0, 0.0                                # set hard_weight > 0 when labels should also pull
for x, y in transfer_loader:
    with torch.no_grad():
        large_logits = large_model(x)                    # large model logits, fixed
    small_logits = small_model(x)
    loss = transfer_loss(small_logits, large_logits, y, temperature=T,
                         soft_weight=1.0, hard_weight=hard_weight)
    opt.zero_grad(); loss.backward(); opt.step()
# deployment: small_model runs an ordinary T=1 softmax.
```

So the chain is: the knowledge is the input→distribution mapping, not the weights; that mapping's most valuable part is the relative probabilities of the wrong classes, the dark knowledge; ordinary probability-matching can't see it because the softmax buries it near zero; raising the softmax temperature lifts it into the loss; training the student at the same high temperature to match these soft targets transfers it; this generalizes logit matching, which is exactly the $T\to\infty$ limit (after zero-meaning the logits), and lowering $T$ gives a dial that down-weights the noisy very-negative logits; and rescaling the soft loss by $T^2$ keeps the hard/soft balance stable as $T$ varies.
