Let me start from the gap, because the gap is embarrassing. The best class-conditional ImageNet generator I have produces samples that score about 52 on Inception Score; real ImageNet data scores about 233. Stare at that for a second. The model is more than four times short of real images on a metric that is itself fairly forgiving — IS doesn't even penalize lack of variety within a class, it just wants each image to look confidently like *some* class and the marginal over classes to be spread out. So I'm losing badly on a generous yardstick. And when I look at the samples, the failure is in both directions at once: the good-looking ones are good-looking because they're stereotyped, and when I push for diversity the fidelity falls apart. There's no dial. I'd like a dial.

Why is this so hard? Supervised image classifiers got better more or less by getting bigger — more data, more layers, more channels, and the loss keeps going down. Adversarial training doesn't obviously work that way. The thing I'm optimizing isn't a loss with a floor; it's a two-player game, min_G max_D E_x[log D(x)] + E_z[log(1−D(G(z)))], whose intended resting point is a Nash equilibrium, not a minimum. The dynamics are coupled and famously twitchy: change the architecture, the optimizer, the relative learning rates, and the whole thing can diverge. So the first real question isn't "what clever module do I add," it's blunter: does this game even survive being scaled up, and if not, why not?

Let me set the toolbox out, because I'm not building from nothing. I have a ResNet-style convolutional G and D. I have spectral normalization, which divides each weight matrix by its largest singular value σ₀(W) — estimated cheaply by a single power-iteration step that reuses the left singular vector from last time — so that each layer is 1-Lipschitz and the discriminator can't produce arbitrarily steep, exploding gradients. The point of bounding D's Lipschitz constant is that D then hands G a bounded, usable gradient *everywhere* instead of a spiky one. Prior work found that putting spectral norm in G too, not just D, helps stability and lets me take fewer D steps. I have the non-local self-attention block: a softmax over pairwise feature similarities so each spatial location can pull from every other location, which fixes the thing stacked 3×3 convolutions are bad at — long-range structure, the overall layout of an object rather than its local texture. I have the hinge objective: D pushes real scores above +1 and fake below −1, with losses E[relu(1−D(x))] + E[relu(1+D(G(z)))] for D and −E[D(G(z))] for G; it plays nicely with spectral norm because once the output scale is constrained, the margin actually means something. And I have two ways to tell the model the class.

The class-conditioning question is worth slowing down on, because the obvious thing is wrong. The obvious thing — AC-GAN style — is to concatenate a one-hot class vector to z at the input and bolt an auxiliary classifier onto D. But the auxiliary classifier rewards samples that are easy to classify, and "easy to classify" is exactly "stereotyped," so it actively pushes against variety. That's the wrong incentive for the problem I have. A cleaner channel into G is through normalization: instead of BatchNorm's single learned gain and bias, produce per-class gain γ(c) and bias β(c) and modulate the normalized features as BN(h)·γ(c) + β(c). The class doesn't fight for space in the activations; it just sets the affine knobs of every normalization layer. That's how the class gets into G.

For D, I want the principled way to use the label rather than another concatenation, so let me work from the optimal discriminator and see what the label *has* to look like. The optimal discriminator for this game is a sigmoid of the log density ratio: D*(x,y) = σ(f(x,y)) with f(x,y) = log q(x,y)/p(x,y), where q is data and p is the generator's joint. Factor each joint as conditional times marginal: f = log q(y|x)/p(y|x) + log q(x)/p(x). The marginal ratio log q(x)/p(x) depends only on x, so set it aside; the question is what the *conditional* term log q(y|x) − log p(y|x) reduces to.

Now model the class posteriors, for both real and generated distributions, as log-linear in some shared feature φ(x): q(y=c|x) ∝ exp(v_c^q·φ(x)) and p likewise with v_c^p — just saying "a linear softmax classifier on features is a reasonable model of p(class|image)." Write out one log-posterior fully, log q(y|x) = v_y^q·φ(x) − log Σ_c exp(v_c^q·φ(x)). The partition term is a sum over *all* classes c, so it does not depend on the particular label y — it depends only on x. Subtract the two:

  log q(y|x) − log p(y|x) = (v_y^q − v_y^p)·φ(x) + [log Z_p(x) − log Z_q(x)],

where the bracket is x-only. I want to be sure I'm not fooling myself that the only y-dependence is that first inner product, so let me check it numerically. Pick φ ∈ ℝ⁵, four classes, random v^q and v^p; compute the full log-softmax difference for each y and subtract (v_y^q − v_y^p)·φ. If the algebra is right the leftover should be *constant across y*. I get residual = [−4.1302, −4.1302, −4.1302, −4.1302] — identical to 15 digits — and that constant equals log Σ exp(v^p·φ) − log Σ exp(v^q·φ) exactly. So the only place y enters the conditional term is through (v_y^q − v_y^p)·φ(x); everything else is a function of x alone.

That settles the form. Fold the per-class difference vector v_y^q − v_y^p into a single learned embedding row V[y], and dump everything x-only — the two partition terms and the marginal ratio log q(x)/p(x) — into one scalar function ψ(φ(x)):

  f(x,y) = yᵀ V φ(x) + ψ(φ(x)).

The label enters as an *inner product* between a learned class embedding and the discriminator's feature vector, added on top of an ordinary unconditional critic. No concatenation, no auxiliary classifier — and the auxiliary classifier was the thing whose incentive I distrusted. In code this is just: pool D's features to a vector h, compute an unconditional score linear(h), and add ⟨embed(y), h⟩. That's the discriminator-side conditioning.

So my baseline is all of that assembled: ResNet G/D, attention in both, spectral norm in both, hinge loss, conditional BatchNorm in G, projection in D, different learning rates for G and D. Now the actual test: scale it, and see whether the pieces hold together or come apart.

Start with the batch: increase it by a factor of eight. Why would batch size alone matter here, when in supervised learning a bigger batch mostly buys you a steadier gradient that you then have to retune the learning rate to cash in? Because each minibatch is a sample of the *modes* of the data, and the gradient each network gets is an expectation over that sample. ImageNet has a thousand classes and enormous intra-class diversity; a small batch sees a thin, high-variance slice of that, so both players are chasing a noisy, mode-starved estimate of the game. Eight times the batch should cover far more modes per step, so both the discriminator's notion of "real" and the generator's gradient toward it should be far less biased. The prediction: a real jump in quality from this change alone, holding architecture fixed, and — if the mode-coverage story is the right explanation — fewer iterations needed to reach a given quality bar, since the per-step gradient is simply less noisy rather than qualitatively different. A matched-budget comparison at fixed architecture and hyperparameters, batch size the only thing varied, is what would tell the mode-coverage story apart from a null result.

But scaling the batch this far is not a free lunch, and the risk is worth stating before taking it. The game is already coupled and twitchy at small scale — it can diverge from architecture or learning-rate choices alone, as noted above — and pushing batch size to an extreme is exactly the kind of large step that could tip a marginally-stable setup into outright divergence rather than a merely slower climb. If that risk is real, the predicted failure mode is not a gentle plateau but something sharper: quality climbing faster than before and then collapsing over a comparatively short window, since a coupled dynamical system pushed harder tends to fail abruptly rather than gracefully. That risk has to be checked for, not assumed away, and it's a separate question from whether scale helps at all — so the plan is to keep measuring the benefit of scale up to the point just before any collapse (checkpointing near best-observed quality) while treating the instability itself as a diagnosis problem to solve on its own, rather than letting an uncharacterized failure contaminate the read on whether scale works.

Next, width: add 50% more channels everywhere, which roughly doubles the parameter count. The straightforward hypothesis is that the dataset is complex and the model is capacity-limited relative to it, so more channels means more room to represent it — predicting a further quality gain on top of the batch-size gain, at the cost of the extra compute. The other obvious axis is depth — double it by stacking an extra residual block after each up/down block. There's a competing hypothesis worth stating rather than assuming: at a matched parameter budget, added depth could help for the same capacity reason, or it could hurt if the extra blocks make the already-twitchy optimization harder to push gradient through without buying comparable representational benefit. Comparing width-scaling against depth-scaling at matched parameter count is how to tell which hypothesis is right, and it's cheap to check before committing to one axis over the other; naive depth is not a safe default just because width is expected to work. (A different, bottleneck-style way of adding depth is worth returning to separately — it changes the parameter/compute trade-off enough that it isn't a fair test of the same hypothesis.)

Now two architecture changes that aren't about raw size but about how the conditioning is wired, and both of them I can motivate from a cost visible before ever training anything. The class-conditional BatchNorm needs, per layer, a vector of class embeddings to project into gains and biases. With many conditional-BN layers, each holding its own embedding table over a thousand classes, that's a mountain of weights doing nearly-redundant work — a parameter count computable directly from the layer widths and class count, no run required. So share one class embedding and linearly project it to each layer's gains and biases: fewer parameters, less memory, by construction. There's a further prediction worth stating, beyond the parameter accounting: forcing every layer to read from a single, coherent class representation, rather than letting a thousand little per-layer tables drift independently, should also make optimization easier and so reach a given quality in fewer iterations than the unshared version — a claim about optimization dynamics, not just parameter count, and the part a matched-iteration-budget comparison against the unshared baseline would have to confirm.

The second wiring change starts from a question about z. I'm feeding the latent only into the very first layer; from there on the network has to carry whatever it needs about z up through every resolution. But z is supposed to be the handle on all the factors of variation, and those factors live at different scales — pose and layout are coarse, texture is fine. Why force all of that to squeeze through the bottom layer and survive the climb? Give the generator direct access to z at every resolution: split z into equal chunks, one per residual block, and at each block concatenate its chunk to the shared class embedding before projecting to that block's BN gains and biases. Now the latent can directly modulate features at every level of the hierarchy, which predicts some combination of better quality (each resolution gets its own latent signal instead of an attenuated, indirect one) and faster training (less has to be learned about routing z information upward through unrelated layers) — the two effects a matched comparison against the single-injection-point version would have to separate. Splitting also trims the first linear layer, since the bottom block only consumes its own chunk of z rather than all of it — a parameter saving that follows directly from the split, not from any run. I'll call this giving z skip connections into the blocks.

That settles the scaling and conditioning story. Now the dial I said I wanted — the fidelity/variety knob.

Here's a freedom that GANs have and most generative models don't. A VAE or a normalizing flow has to *backpropagate through its latents* or evaluate a density in latent space, so its prior is load-bearing and you're stuck with it. A GAN never inverts z; G just consumes whatever z I hand it. So I'm free to choose the prior — and, more to the point, I'm free to sample z at test time from a *different* distribution than the one I trained with. I trained with z ∼ N(0, I). What if at test time I draw z from a *truncated* normal — resample any coordinate whose magnitude exceeds some threshold so it lands back inside? This only changes the sampling distribution at evaluation time, not the trained weights, so it's a nearly free thing to check: score a single already-trained model under ordinary N(0, I) sampling and under truncated sampling, everything else held fixed. If the prior's shape matters the way I suspect, truncated sampling should improve fidelity metrics relative to the untruncated baseline from the same checkpoint — a prediction the comparison would either confirm or kill outright, with no retraining involved.

Why on earth would that help? Think about where the generator is actually any good. During training, z came from N(0, I), so the network saw latents overwhelmingly from the high-density core of the Gaussian and almost never from the far tails. It's best-fit exactly where it got the most supervision: near the mode of the prior. The tails are the under-trained regions, the places where G has to extrapolate. Truncating throws away the tails and keeps z near the high-density core, so every sample comes from a region the generator models well — fidelity should go up.

Let me make the "as I shrink the threshold, z heads for the mode" claim concrete rather than hand-wave it, since the whole variety story rides on it. Sample z = t·TruncNorm(−2, 2) and measure how spread the coordinates stay as I shrink t. At t = 1.0 I get mean|z| ≈ 0.72, std ≈ 0.88; at t = 0.5, mean|z| ≈ 0.36, std ≈ 0.44; at t = 0.2, mean|z| ≈ 0.14; at t = 0.04, mean|z| ≈ 0.029 with every coordinate inside ±0.08. The cloud contracts smoothly and monotonically onto 0 as t → 0, with no coordinate ever escaping ±t. So shrinking the threshold really does drive z toward the prior's mode — that part is a fact about the sampling distribution alone, provable without touching G. The natural prediction is that G's outputs follow z toward the mode of G's per-class distribution, the single most canonical image for that class, so variety should go down as fidelity goes up. Whether the outputs actually do that is a question about G, not about z, and needs checking on a trained model rather than assumed from the sampling math.

The slider should read as a curve, and its rough shape is predictable from what each metric penalizes rather than from having plotted it. Sweeping the threshold and tracking IS against FID should trace something like a precision/recall curve. IS doesn't punish missing intra-class variety, so it should behave like precision: monotonically rising as truncation gets harder, since harder truncation only ever pushes samples toward the canonical, confidently-classified region. FID punishes both bad fidelity and dropped variety, so it should behave like a mix: improving at first as fidelity climbs, then turning over and getting worse once truncation starts eating variety faster than it's buying fidelity. Actually sweeping the threshold and plotting the two curves against a trained model is the direct, cheap way to check both predictions.

But there's no reason to expect this to be safe for every trained G, and the failure mode is worth reasoning out in advance rather than discovering it by accident. Truncation concentrates evaluation-time z near the prior's mode, away from the shape of the full training distribution; whether that constitutes a harmful distribution shift depends entirely on whether G's response to z is smooth. If nearby latents map to nearby, sensible images — if the whole z-space, not just the cloud of training samples, maps to good outputs — truncation is safe by construction, since every truncated z stays close to some well-covered z. But a jagged, ill-conditioned G could have pockets where a perfectly reasonable truncated z lands on nonsense: the predicted failure signature would be saturation artifacts, blown-out garbage, and — plausibly — a failure that's worse for larger, less-constrained generators, since more capacity gives more room for exactly this kind of pocket to form. So the truncation trick is really demanding a property of G — smoothness, good conditioning — that nothing proposed so far explicitly enforces. If truncation is to be reliably safe rather than safe by luck on some models and not others, that smoothness has to be built in rather than hoped for.

How do I enforce smoothness? I want the linear maps inside G to be well-conditioned, not stretching some directions enormously while crushing others — that kind of anisotropy is exactly what makes a small change in z explode into a large change in output. The clean way to ask a weight matrix to be norm-preserving is to push it toward orthonormality. Treat each output filter as a row of `W`, so the relevant Gram matrix is `WWᵀ`. The standard orthogonal regularizer is

  R_β(W) = β‖WWᵀ − I‖²_F,

which says: make the Gram matrix of the filters the identity — unit norm and mutually orthogonal. That's more than conditioning actually needs, and the problem is visible directly in the algebra rather than needing a run to surface: forcing WWᵀ = I constrains the norms of the filters as well as their directions, since the diagonal of WWᵀ is exactly the squared norms. The network plausibly has legitimate reasons to want filters of varying magnitude, so a penalty that also pins every norm to one is fighting the network on something unrelated to conditioning, and predicts a needlessly costly regularizer. So relax it. What conditioning actually needs is that the filters point in *different* directions — that they're decorrelated — not that they all have unit length. The off-diagonal entries of WWᵀ are the pairwise inner products of the filters; the diagonal entries are their squared norms. So drop the diagonal and penalize only the off-diagonal part:

  R_β(W) = β‖WWᵀ ⊙ (1 − I)‖²_F,

where (1 − I) is the all-ones matrix with the diagonal zeroed. This minimizes the pairwise cross terms between filters — decorrelating their directions — while leaving their norms completely free. That's the smoothness I wanted without the strangling.

I'd rather not materialize this loss and backprop through it every step; I want the gradient in closed form so I can just add it to the weight's gradient. Let A = (WWᵀ) ⊙ (1 − I); note A is symmetric since the mask is symmetric and WWᵀ is symmetric. R = ‖A‖²_F = ⟨A, A⟩. Differentiating, dR = 2⟨A, d((WWᵀ) ⊙ (1−I))⟩ = 2⟨A ⊙ (1−I), d(WWᵀ)⟩ = 2⟨A, dW·Wᵀ + W·dWᵀ⟩ (the mask is idempotent and already folded into A). The two terms ⟨A, dW·Wᵀ⟩ and ⟨A, W·dWᵀ⟩ are equal because A is symmetric, so each contributes A·W and they add: ∇_W R = 4·(WWᵀ ⊙ (1 − I)) W.

A factor of *four*. That's worth pausing on, because the implementation I'm going to write uses `2 * mm(mm(w, w.t()) * (1 - eye), w)` — a coefficient of 2, not 4. Did I drop a factor, or did the code? Let me settle it with finite differences rather than re-deriving anxiously. Random W ∈ ℝ^{4×6}, compute R(W) by hand and its numerical gradient, compare against 4·A·W and against 2·A·W. The result: max|4·A·W − numeric| ≈ 2·10⁻⁸ (machine zero), while max|2·A·W − numeric| ≈ 26 — nowhere close. So the literal derivative of β‖WWᵀ ⊙ (1−I)‖²_F is unambiguously 4β·A·W. The coefficient-2 form is the gradient of *(β/2)*‖·‖²_F, i.e. the same penalty with the conventional ½ out front, which just rescales what "strength" means. Since I'm hand-tuning the strength by sweep anyway, the ½ convention is harmless — but I'm glad I checked, because had I written 4 with a strength tuned against 2, I'd have been off by 2× on a knob I'm choosing by eye. I'll keep the coefficient-2 gradient and remember the penalty it corresponds to carries the ½. (The full-orthogonal version is identical with (WWᵀ − I) in place of WWᵀ ⊙ (1 − I); I verified its 4·(WWᵀ−I)W gradient the same way and it matched to 10⁻⁸ too.)

The strength β is the one knob left to set, and it should be small — this is meant to nudge conditioning, not dominate the main objective — so the plan is to sweep it over a few orders of magnitude starting from something tiny like 1e-4, and pick by the criterion that actually matters here rather than by the headline metric directly. The criterion is whether the penalty does the one job it's for: train matched pairs of models with and without it (same architecture, same scale, same budget), then run each through the truncation sweep and check how many are truncation-amenable — sharpen cleanly at low truncation values rather than producing artifacts — versus how many aren't. The regularizer earns its place only if it substantially raises that fraction; if it doesn't move the fraction, or costs meaningfully more untruncated quality than it buys in truncation-amenability, it isn't worth keeping. The two pieces of the truncation story — sampling near the prior's core, and conditioning G so that's safe — only make sense evaluated together, which is why the validation has to be this joint one rather than checking either piece in isolation.

If the predicted collapse risk from scaling the batch does materialize, checkpointing before it is not an answer on its own — it treats the symptom and leaves the cause uncharacterized, which means no way to know whether it gets worse at the next scale step or whether early stopping keeps working indefinitely. And any diagnosis has to happen at the scale where the risk is expected to show up: a game this coupled and twitchy could behave differently under different batch/data-mode coverage, so toy-scale intuitions aren't guaranteed to transfer, and the only honest test is on the large-scale configuration itself. The plan is to monitor broadly — weight, gradient, and loss statistics — while hunting for something that reliably precedes collapse rather than merely accompanies it. The natural first candidates are the top few singular values of each weight matrix, σ₀, σ₁, σ₂: spectral quantities are a sensible place to look because an exploding or collapsing operator norm is exactly the kind of thing that would make a network's output blow up or degenerate, and they're cheap to track with an extended power iteration — the same machinery as spectral norm, pushed to recover a couple more singular vectors.

Looking at G's layers first, the natural place trouble might show up is the very first layer — the over-complete, non-convolutional linear that turns z into a 4×4 feature map — since it's the one layer that has to absorb the full z-to-features mapping with no spatial structure to constrain it, unlike every downstream convolutional layer. The hypothesis: if instability originates in G's conditioning, this layer's σ₀ should behave differently in runs heading toward collapse than in ones that aren't — for instance, growing steadily through training and then spiking right around the point of collapse, rather than staying flat. That's a correlational signature, though, and correlation isn't enough to act on; the discriminating test is to intervene directly on σ₀ and see whether the intervention changes the outcome, not just the readout. Two ways to attack it: regularize σ₀ toward a fixed target, or toward a ratio of the second singular value, r·sg(σ₁), with a stop-gradient on σ₁ so the penalty can't cheat by inflating σ₁ instead of shrinking σ₀; or clamp it surgically with a partial SVD — take the top singular triple (σ₀, u₀, v₀) and subtract off the excess:

  W ← W − max(0, σ₀ − σ_clamp) v₀ u₀ᵀ,

with σ_clamp set to that fixed value or to r·sg(σ₁). Both interventions, if applied, should at minimum stop σ₀ (or the ratio σ₀/σ₁) from creeping up and exploding — a check on whether the intervention does what it's engineered to do, separate from whether it helps. The real test is whether constraining σ₀ this way also prevents collapse. Two outcomes distinguish the hypotheses cleanly: if collapse goes away once σ₀ is held down, G's conditioning is close to the whole story and the fix is to keep this constraint; if collapse still happens with σ₀ demonstrably held in check, the σ₀ explosion is at most a symptom riding along with the real cause rather than the cause itself, conditioning G is necessary at best and not sufficient, and the disease isn't purely in G — the search has to continue elsewhere.

D deserves the same kind of scrutiny, under a distinct hypothesis: if G's conditioning isn't sufficient to explain collapse, D's dynamics are the other place to look. The predicted pattern is qualitatively different from G's story, because D's role in the game is different — D isn't producing an unconstrained mapping to fit, it's tracking a rapidly-moving target (G) that fights back. So rather than a smooth, steady climb in σ₀ like the one hypothesized for G, the pattern to look for in D is something noisier and more punctuated: spikes concentrated in the top singular directions rather than a smooth drift, on the theory that if the instability is adversarial in nature — G occasionally producing a batch that strongly perturbs D along its leading directions — the signature would look like an impulse response, a sudden jump followed by a decaying oscillation back down, rather than a monotone trend. Checking this means tracking D's σ₀/σ₁ ratio and Frobenius norms through training and looking specifically for that spike-and-decay shape concentrated in the leading singular directions, distinct from a smooth trend spread across the whole matrix.

If that spectral noise is what's destabilizing things, the textbook counter is to regularize D's Jacobian directly — a zero-centered gradient penalty on real data, R₁ = (γ/2) E_{q_data}‖∇D(x)‖²_F, at some strength γ. The prediction, if the adversarial-perturbation story is right, is that turning this on at a large enough γ should smooth and bound both networks' spectra and eliminate collapse; the competing prediction is that it costs quality, since constraining D's Jacobian directly limits how sharp a boundary D can draw, which is presumably part of what makes D useful in the first place. The way to map that trade-off rather than guess at it is to sweep γ from something small up through a large, textbook-suggested value like 10, at each point recording whether collapse still occurs and what quality metric the surviving run reaches. The decision rule is a threshold search: find the smallest γ that reliably prevents collapse across seeds, and treat the quality metric at that γ as the price of stability via this route. If the same pattern — buy stability, pay quality, at every strength — holds for other ways of constraining D (orthogonal reg on D, dropout in D's final features, L2), that would be evidence the trade is a property of constraining D per se, not an artifact of the R₁ penalty specifically, which matters for how much to try to buy out of it versus accepting it as a cost and looking for a cheaper fix elsewhere.

There's a second thing about D worth building a hypothesis around, this time starting from a question rather than an observed pattern. Is D overfitting outright — memorizing the training set rather than learning a real/fake boundary that generalizes? The way to check is the obvious one: evaluate D's classification accuracy on real vs. fake separately on the training set it saw and on a held-out validation split it didn't. If D is generalizing, the two accuracies should be close; if D is memorizing, training accuracy should sit far above validation accuracy, with validation hovering near chance regardless of which regularizer is in use — that last part being how to tell memorization apart from D simply being weak, since a weak-but-generalizing D would do similarly badly on both splits rather than well on one and near-chance on the other.

Suppose memorization turns out to be what's happening. That wouldn't automatically be a bug: D's job in this game isn't to generalize, it's to distill the training data into a useful gradient signal for G, and memorization is consistent with that role rather than opposed to it. It also predicts something specific and checkable about the spikes. Both the hinge loss and the ordinary log loss give exactly zero gradient on an example D already scores confidently and correctly — a fact about the loss functions' definitions, not about any run. So the causal chain the memorization hypothesis predicts is: as D approaches confident-correct classification of the real examples (memorization), the real-data gradient attenuates toward zero; D keeps receiving the fake-side gradient, which pushes its outputs negative with nothing on the real side to balance it; that drift continues until the negative bias is large enough that D starts misclassifying a batch of real images, at which point it takes a large corrective gradient back toward positive outputs — and that corrective jump is the candidate explanation for the impulse spikes hypothesized in D's spectra. This is a mechanism that, if right, should be checkable by watching D's per-class gradient magnitude on real examples decay over training in lockstep with the memorization gap widening, with spikes in σ₀ coinciding with the corrective jumps.

If the mechanism holds up, it also points at a menu of candidate fixes, each with its own predicted trade-off to test rather than assumed to work: an unbounded loss like Wasserstein removes the zero-gradient-on-correct problem entirely, but needs checking against whether it trains stably here at all, even with gradient penalties, since removing one failure mode doesn't guarantee avoiding others; widening the hinge margin pulls more examples inside the margin so they keep contributing gradient longer, but predicts a sweet spot rather than a monotonic fix — a large margin should still cost stability past some point, and a small margin should cost accuracy — so the shape of the margin-vs-collapse curve is the thing to actually sweep and read off; shrinking D to limit its capacity to memorize should reduce the effect directly, at the predictable cost of a weaker gradient signal for G. None of these looks like a clean win from mechanism alone, which is exactly why each needs the sweep rather than a single point estimate before picking one.

So is it G's fault, D's fault, or something that lives only in how they're coupled? The cleanest way to separate the three is a freeze test: hold one network fixed while the other keeps training, which isolates each network's dynamics from the other's ability to adapt — something a normal joint run never lets you observe directly. Two matched conditions: freeze G and keep training D; freeze D and keep training G.

The prediction follows from what each network's gradient is actually valid for. D's gradient signal to G is only trustworthy for the instantaneous G that produced it — it's a local critique of the *current* generator, not a fixed target G can be judged against forever. So freezing D and continuing to train G removes the one thing keeping G honest: D's ongoing adaptation. The predicted failure mode is that G, trained against a discriminator that can no longer respond, should be able to find directions in its own parameter space that drive D's frozen output arbitrarily far from where it started, without those directions corresponding to any real gain in image quality — a generator racing to exploit a target that can't chase back. Freezing G and continuing to train D is the mirror condition, and if D's loss-driving-toward-zero behavior in ordinary training reflects D adapting to a moving G rather than needing G to move, this arm should look comparatively uneventful: D optimizing against a fixed target the way any ordinary discriminative model would.

An asymmetric outcome between the two arms — one staying orderly, the other running away, with the failing arm's loss departing wildly from anything seen in ordinary joint training — would pin the cause on the G–D interaction rather than either network's conditioning in isolation: it would mean D has to stay adapted to the instantaneous G at every step, or G runs away, which is a claim about the coupling and not about either network's spectral properties alone. That prediction also connects back to the R₁ sweep: if stability really is about keeping D adapted to G's current state rather than about D's raw capacity to constrain gradients, then favoring D harder — a bigger D learning rate, more D steps per G step — should help only up to the point where D can actually keep pace with G's instantaneous updates, and should stop helping, or actively hurt, beyond that point, rather than trading off smoothly the way the R₁ strength does. Confirming that non-monotonic shape, on top of the freeze-test asymmetry, is what would justify the conclusion that stability is a property of the *interaction* rather than of either network's conditioning — and the decision rule that follows from it: neither conditioning G alone nor conditioning D alone is a sufficient fix, so the honest choice is between strongly constraining D at a real, quantifiable performance cost, or keeping conditioning light, accepting a collapse that these diagnostics can characterize and anticipate, and stopping training just before it.

That leaves an honest, slightly uncomfortable engineering conclusion built into the plan itself, rather than a tidy fix promised in advance. If the diagnostics above bear out — collapse traceable to the G–D interaction rather than either network's conditioning, and stability purchasable from D-side constraints only at a real performance cost — then there's no free variant that gets both perfect stability and full quality; the two are predicted to trade against each other, not to be independently solvable. Given that trade-off, the proposed default is the cheaper side of it: keep the conditioning light, accept that collapse is expected to arrive late in training rather than trying to eliminate it, and stop just before it — validated by checking that quality at the stopping point, chosen via the monitored spectral and loss diagnostics, is competitive with what heavier D-side constraints would buy at their own quality cost. Scale, the conditioning changes, and the truncation dial are what the quality is expected to come from; treating collapse as a characterized, anticipated event that early stopping handles — rather than a pathology to eliminate outright — is the proposed way to live with an instability that the diagnostics above predict is not cheaply removable.

Let me write down the proposal in full: the discriminator's projection output, the conditional-BN with shared embedding and skip-z, the modified orthogonal penalty as a direct gradient, the hinge losses, the EMA, and the truncated sampler:

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Class-conditional BatchNorm: one shared embedding, projected per layer to
# (gain, bias). gain centered at 1, bias at 0. cond = [class_embed ; z_chunk].
class ConditionalBN(nn.Module):
    def __init__(self, num_features, cond_dim, which_linear):
        super().__init__()
        self.gain = which_linear(cond_dim, num_features, bias=False)
        self.bias = which_linear(cond_dim, num_features, bias=False)
        self.bn   = nn.BatchNorm2d(num_features, affine=False)  # cross-replica / standing in practice
    def forward(self, x, cond):
        gain = (1 + self.gain(cond)).view(x.size(0), -1, 1, 1)
        bias = self.bias(cond).view(x.size(0), -1, 1, 1)
        return self.bn(x) * gain + bias

# Generator residual block: conditioned on cond = [shared_class_embed ; this block's z chunk].
class GResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, which_conv, which_bn):
        super().__init__()
        self.bn1, self.bn2 = which_bn(in_ch, cond_dim), which_bn(out_ch, cond_dim)
        self.conv1 = which_conv(in_ch, out_ch)
        self.conv2 = which_conv(out_ch, out_ch)
        self.learnable_sc = (in_ch != out_ch)
        if self.learnable_sc:
            self.conv_sc = which_conv(in_ch, out_ch, kernel_size=1, padding=0)
    def forward(self, x, cond):
        h = F.relu(self.bn1(x, cond))
        h = F.interpolate(h, scale_factor=2)            # nearest-neighbour upsample
        x = F.interpolate(x, scale_factor=2)
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h, cond)))
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x

class Generator(nn.Module):
    def __init__(self, dim_z=120, n_classes=1000, ch=96, shared_dim=128):
        super().__init__()
        which_conv  = SNConv2d                          # spectral norm in G
        which_lin   = SNLinear
        self.n_slots = 5                                # one z chunk per res block (128px)
        self.chunk   = dim_z // (self.n_slots + 1)      # 20-D chunks
        cond_dim     = shared_dim + self.chunk          # class embed concatenated with a z chunk
        self.shared  = nn.Embedding(n_classes, shared_dim)   # ONE shared class embedding
        self.linear  = which_lin(self.chunk, 16*ch * 4*4)    # first layer takes only its chunk
        which_bn = lambda c, d: ConditionalBN(c, d, which_lin)
        chans = [16*ch, 16*ch, 8*ch, 4*ch, 2*ch, ch]
        self.blocks = nn.ModuleList(
            [GResBlock(chans[i], chans[i+1], cond_dim, which_conv, which_bn)
             for i in range(self.n_slots)])
        self.attn = SelfAttention(2*ch, which_conv)     # one non-local block at 64x64
        self.attn_after = 3                             # after the block whose output is 2*ch
        self.out  = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(),
                                  which_conv(ch, 3))
    def forward(self, z, y):
        zs = torch.split(z, self.chunk, 1)              # skip-z: split z into per-block chunks
        emb = self.shared(y)
        conds = [torch.cat([emb, zs[i+1]], 1) for i in range(self.n_slots)]
        h = self.linear(zs[0]).view(z.size(0), -1, 4, 4)
        for i, blk in enumerate(self.blocks):
            h = blk(h, conds[i])
            if i == self.attn_after:                    # attention after the 64x64 block
                h = self.attn(h)
        return torch.tanh(self.out(h))

class Discriminator(nn.Module):
    def __init__(self, n_classes=1000, ch=96):
        super().__init__()
        which_conv = SNConv2d
        self.blocks = nn.ModuleList([...])              # ResBlock-down stack + SelfAttention at 64x64
        self.linear = SNLinear(16*ch, 1)
        self.embed  = SNEmbedding(n_classes, 16*ch)     # projection embedding
        self.activation = nn.ReLU()
    def forward(self, x, y):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = torch.sum(self.activation(h), [2, 3])        # global sum pooling -> feature vector
        out = self.linear(h)                             # unconditional critic  psi(phi(x))
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)   # + <V[y], phi(x)>  (projection)
        return out

# Hinge objective
def loss_hinge_dis(d_fake, d_real):
    return F.relu(1. - d_real).mean(), F.relu(1. + d_fake).mean()
def loss_hinge_gen(d_fake):
    return -d_fake.mean()

# Modified orthogonal regularization, applied as a direct gradient on G's weights.
# grad of beta * || W Wᵀ ⊙ (1 - I) ||²_F   is   2 (W Wᵀ ⊙ (1 - I)) W.
def ortho(model, strength=1e-4, blacklist=()):
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim < 2 or any(p is b for b in blacklist):
                continue                                 # skip the shared embedding
            w = p.view(p.shape[0], -1)
            grad = 2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w)
            p.grad.data += strength * grad.view(p.shape)

# Truncation trick: at sampling time draw z from a truncated normal, then scale.
# Smaller `truncation` -> z nearer the prior's mode -> higher fidelity, less variety.
from scipy.stats import truncnorm
def truncated_z_sample(batch, dim_z, truncation=0.5):
    values = truncnorm.rvs(-2, 2, size=(batch, dim_z))   # standard normal, clipped to [-2, 2]
    return truncation * values

def train_step(G, D, GD, real_x, real_y, z, y, ema, cfg):
    for _ in range(cfg['num_D_steps']):                  # D must keep up with G: 2 steps per G step
        D.optim.zero_grad()
        d_fake, d_real = GD(z.sample_(), y.sample_(), real_x, real_y, train_G=False)
        d_loss_real, d_loss_fake = loss_hinge_dis(d_fake, d_real)
        (d_loss_real + d_loss_fake).backward()
        D.optim.step()
    G.optim.zero_grad()
    d_fake = GD(z.sample_(), y.sample_(), train_G=True)
    loss_hinge_gen(d_fake).backward()
    ortho(G, cfg['G_ortho'], blacklist=list(G.shared.parameters()))   # light smoothness penalty
    G.optim.step()
    ema.update()                                         # average G's weights for sampling
```
