The first thing I need is not another way to memorize the sample. I need a way to make the learned function answer for the regions empirical risk leaves silent. If the loss only checks the training points, then a high-capacity model can be perfect on those points and still have arbitrary shape nearby. Random labels expose the memorization side of that failure, and adversarial perturbations expose the local-smoothness side.

So I start from vicinal risk rather than from a new weight penalty. The old idea is to replace each empirical point mass by a distribution around the point, then train on virtual samples from that neighborhood. That helps when I know a label-preserving move: translate an image a little, add harmless noise, move along a tangent direction. But that story keeps each virtual example tied to one original label. It does not say what should happen as one training example turns into another.

Interpolation is already nearby, so the line segment itself cannot be the whole discovery. A same-class interpolation method can thicken a minority class region, and feature-space interpolation can do something similar after an encoder. Those choices avoid the hard case by staying inside one class or a chosen local neighborhood. If I move between two arbitrary examples, especially two different classes, a fixed hard label is no longer an honest target.

My first tempting shortcut is to average the inputs and keep one endpoint's label. The obvious rule is: for `x_tilde = lambda x_i + (1 - lambda) x_j`, assign the class of whichever endpoint has the larger weight. Let me actually trace that target across the segment, walking `lambda` from `1` down to `0`:

```text
lambda=0.95  ->  closer endpoint is i  ->  target class_i
lambda=0.55  ->  closer endpoint is i  ->  target class_i
lambda=0.51  ->  closer endpoint is i  ->  target class_i
lambda=0.49  ->  closer endpoint is j  ->  target class_j
lambda=0.45  ->  closer endpoint is j  ->  target class_j
lambda=0.05  ->  closer endpoint is j  ->  target class_j
```

Two things jump out when I read this table. First, `lambda=0.95` and `lambda=0.55` produce different inputs — a 95-5 blend and a 55-45 blend look very different — yet they get the identical target `class_i`. The loss cannot tell those two mixtures apart, so all the supervision in the interior of the segment collapses onto two flat plateaus. Second, between `lambda=0.51` and `lambda=0.49` the target flips `class_i -> class_j` while the input barely moved. That is a discontinuity sitting right at the middle of the chord. So this rule has a cliff and two plateaus, which is exactly the opposite of the smooth, position-aware behavior I am trying to buy. The averaged input is a fine place to ask a question; the hard nearest-endpoint label is a bad answer.

My second shortcut is ordinary label smoothing. It reduces overconfidence, which is useful, but it is blind to the direction of the synthetic input. It assigns the same small probability mass no matter which example I moved toward, and no matter how far I moved. So it cannot fix the table above: at `lambda=0.95` and at `lambda=0.55` it would still hand back the same target, just a softened version of it. I need the target to know both the partner example and the amount of movement.

Both failures point at the same missing ingredient: the target has to be a function of `lambda` and of *both* endpoints, varying as the input varies. The simplest such function — and the one that removes the cliff, because it is continuous in `lambda` — is to let the target slide linearly from one endpoint's label to the other as the input slides. So I take two examples `(x_i, y_i)` and `(x_j, y_j)`, draw `lambda` in `[0, 1]`, and form `x_tilde = lambda x_i + (1 - lambda) x_j` together with `y_tilde = lambda y_i + (1 - lambda) y_j`. The same coefficient `lambda` controls the input and the target. I want to keep an eye on whether that single shared coefficient is doing real work or is just notation; I will come back to it when I check the regularization claim.

I have to keep the claim modest. I am not claiming that an averaged image is a natural photograph. The classifier is already a function on the ambient input vector space, so it already has some output on that averaged vector. The question is what target should constrain that output. The averaged input gives me a place to ask; the averaged target gives the answer that makes the constraint meaningful.

Now this becomes a two-example vicinal distribution. I draw one example, draw a partner example, draw `lambda ~ Beta(alpha, alpha)`, and train on the paired feature-target convex combination. I want the sampling to interpolate smoothly between "ordinary training" and "lots of interior questions," and the symmetric Beta gives me one knob for that. To see it concretely I sample two million draws and measure how much mass lands within `0.01` of an endpoint:

```text
alpha=2.00 -> P(|lambda - endpoint| < 0.01) = 0.001
alpha=1.00 -> P(|lambda - endpoint| < 0.01) = 0.020
alpha=0.40 -> P(|lambda - endpoint| < 0.01) = 0.188
alpha=0.10 -> P(|lambda - endpoint| < 0.01) = 0.641
alpha=0.01 -> P(|lambda - endpoint| < 0.01) = 0.955
```

So as `alpha` tends toward zero the draws pile up at the endpoints — at `alpha=0.01` over 95% of samples are essentially `0` or `1` — and the virtual examples approach ordinary empirical examples. Larger positive values flatten the distribution and put more mass in the interior, so the training rule asks more questions between examples. That matches the behavior I wanted from the knob.

Now the regularization claim, which is where I most need to be careful, because a coupled target sounds nice but I should check it actually buys smoothness. I define the averaged predictor `tilde f(x) = E_{x'', lambda} hat f(lambda x + (1 - lambda) x'')` and measure a Lipschitz constant only over real training inputs. Suppose `hat f` has fit the virtual examples, so at a paired virtual location I can replace the learned output by the mixed target: `hat f(lambda x' + (1 - lambda)x'') = lambda f(x') + (1 - lambda)f(x'')`, and similarly with `x` in place of `x'`. Subtract the two:

```text
hat f(lambda x' + (1-lambda)x'') - hat f(lambda x + (1-lambda)x'')
  = [lambda f(x') + (1-lambda)f(x'')] - [lambda f(x) + (1-lambda)f(x'')]
  = lambda(f(x') - f(x)),
```

and the shared `(1 - lambda)f(x'')` term cancels. I do not fully trust an algebra step I want to lean on, so I check it with a concrete linear `f(x) = W x` (linear so that `hat f` on a mix is exactly the mix of `hat f`, which is the assumption the step uses). With random `W`, `x`, `x'`, `x''` and `lambda = 0.3`:

```text
hat f(lambda x' + (1-lambda)x'') - hat f(lambda x + (1-lambda)x'') = [-0.3106, -0.4613]
lambda * (f(x') - f(x))                                            = [-0.3106, -0.4613]
```

The two vectors agree, so the cancellation is real and not an artifact of how I grouped terms. The partner `x''` genuinely drops out, which is the whole point: it means the bound does not depend on which partner I happened to draw. Taking norms, the leftover `lambda(f(x') - f(x))` gives `hat Lip(tilde f) <= E[lambda] hat Lip(f)`. So `tilde f` is provably smoother than `f` by a factor `E[lambda]`, and now I can see the shared coefficient was load-bearing after all — it is what makes the `x''` terms identical so they cancel.

It is worth pinning down `E[lambda]`, since a reader could imagine `alpha` tunes the regularization strength through this mean. For the symmetric `Beta(alpha, alpha)` the mean is `alpha / (alpha + alpha) = 1/2` for every `alpha > 0`. Checking numerically across a range of `alpha`:

```text
alpha=0.1 -> E[lambda] ~ 0.500
alpha=0.4 -> E[lambda] ~ 0.500
alpha=1.0 -> E[lambda] ~ 0.500
alpha=2.0 -> E[lambda] ~ 0.500
```

It is flat at `1/2` regardless of `alpha`. So the factor in the Lipschitz bound is just `1/2`; this proof is evidence for the cancellation produced by paired targets, not an alpha-dependent constant explaining the whole method. The way `alpha` changes regularization strength comes from *where* the virtual points concentrate — the endpoint-vs-interior table above — not from the mean of `lambda`.

That same cancellation tells me why the alternatives fail algebraically, not just intuitively. If I add Gaussian noise, I do not know the target value at the perturbed point, so I cannot make the substitution `hat f = lambda f(x') + (1-lambda)f(x'')` in the first place. If I average inputs but keep one hard label, the output I am told to match is `f(x')` (a single endpoint), not the convex combination, so the `(1-lambda)f(x'')` terms are not shared and nothing cancels. If I use fixed label smoothing, the target softness is detached from the partner and the coefficient, so again the substitution has the wrong right-hand side. The cancellation is specific to coupling the target to the input with the same `lambda`.

The implementation should preserve exactly that coupling without extra machinery. In a minibatch, I sample one scalar `lambda`, shuffle the batch to choose partners, form `mixed_x = lambda x + (1 - lambda) x_shuffled`, and want to train against the dense target `lambda y + (1 - lambda) y_shuffled`. The convenient claim is that I can skip building that dense target and instead compute `lambda * CE(pred, y) + (1 - lambda) * CE(pred, y_shuffled)`. That rests on cross-entropy being linear in the target distribution, which I should verify rather than assume. Take a softmax output `p = [0.584, 0.149, 0.266]`, endpoints `a=0`, `b=2`, and `lambda=0.3`:

```text
CE(p, lambda*e_a + (1-lambda)*e_b) [dense soft target] = 1.08717
lambda*CE(p, e_a) + (1-lambda)*CE(p, e_b)              = 1.08717
```

They match to five places, so the two-term weighted loss is exactly cross-entropy against the dense soft target. Good — that means the implementation needs no custom loss, just two ordinary cross-entropy calls with the original and shuffled labels.

The design choices now fit together. Random partners cover more of the span of high-dimensional training data than a nearest-neighbor rule and avoid choosing a metric. Raw input interpolation constrains the classifier where it is actually evaluated, even when the virtual input is not perceptually natural. Two-example mixtures keep the procedure cheap, while adding more examples with Dirichlet weights increases cost without changing the core prior. If `alpha <= 0` in code, setting `lambda = 1` turns the routine back into ordinary minibatch training, consistent with the `alpha -> 0` concentration I measured.

There is still a limitation inside the same prior. If an interpolated point lands on the true data manifold with a soft target that conflicts with the real label there, the constraint can cause underfitting. That does not invalidate the construction; it names the assumption. I am imposing local linear behavior between sampled examples unless the data geometry makes that bias wrong.

So what survives all the checks is a coupled vicinal rule: supervise both the synthetic input position and the synthetic target with the same interpolation coefficient. It is not input averaging alone — that gave the cliff and the plateaus. It is not target smoothing alone — that ignored the partner and the distance. It is the demand that predictions vary linearly along data-dependent chords, made enforceable because the target attached to each chord varies linearly too, and made smoothing-provable because that shared coefficient is exactly what lets the partner term cancel.
