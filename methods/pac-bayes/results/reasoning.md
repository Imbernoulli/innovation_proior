I begin with the old Occam certificate because it already contains half of the idea. If I have a countable hypothesis class and a prior mass `P(h)` fixed before the data, I can spend failure probability in proportion to `P(h)`. For each fixed `h`, concentration tells me that an unusually optimistic training error is unlikely. After I weight those bad events by the prior and union-bound over `h`, I get a high-probability event on which every hypothesis is safe to choose after seeing the data. When the learner finally returns `h`, the complexity cost is `ln(1/P(h))`.

This is a useful theorem, but I keep running into the same mismatch. The theorem certifies a single classifier, while the algorithms I want to justify often produce a distribution, a mixture, or a randomized prediction rule. If I force the algorithm to name one exact classifier, I throw away information. A dense region of many similar good classifiers should be easier to certify than one isolated exact point, yet the singleton bound only sees the mass of the final point. In a continuous parameter space, the mismatch becomes sharper: an exact parameter value has prior mass zero, so the old code-length term becomes the wrong object.

The Bayesian vocabulary suggests the missing object. Instead of selecting one classifier `h`, I let the learner output a distribution `Q` over classifiers. The predictor is stochastic: draw `h` from `Q` and use it. Its empirical loss is not the loss of a named classifier but the posterior average `hat L(Q,S) = E_{h~Q} hat L(h,S)`. Its true loss is `L(Q) = E_{h~Q} L(h)`. I do not assume that the data distribution is generated from the prior. I only use the prior as a pre-sample reference measure, a way to say which departures from the initial bias are simple and which are costly.

Once I make `Q` the learned object, the old complexity term tells me what it must become. A singleton choice paid `ln(1/P(h))`, which is the information needed to name `h` under the prior. A distributional choice should pay the information needed to move from the prior `P` to the posterior `Q`. That quantity is `KL(Q||P) = E_{h~Q} log(dQ/dP)`. If `Q` collapses to a point mass on a discrete hypothesis, this reduces to the old Occam code length. If `Q` spreads over a prior-heavy region, the cost reflects the whole region instead of a zero-mass or tiny-mass point.

Now I need to prove that this is not just an analogy. I start from fixed-hypothesis concentration, but I do not union-bound individual bad events anymore. Instead, I build a prior-side exponential moment. For each classifier, let the deviation statistic measure how implausible the gap between `hat L(h,S)` and `L(h)` is. Hoeffding or the Bernoulli-kl moment tells me that, for a fixed classifier, the expectation of an exponential of this deviation is controlled. I then average this exponential under the prior `P`. Fubini lets me exchange the sample expectation and the prior expectation, and Markov turns the expected control into a high-probability statement over samples:

`E_{h~P} exp(f_S(h))` is not too large.

This statement is the continuous replacement for the union-bound budget. It is not tied to a finite list of hypotheses. It says that, on the good sample event, the prior does not assign much exponential mass to overly optimistic classifiers.

The remaining step is to transfer the prior-side statement to any data-dependent posterior. The change-of-measure inequality gives exactly the bridge I need:

`E_{h~Q} f_S(h) <= KL(Q||P) + log E_{h~P} exp(f_S(h))`.

This inequality is just the variational form of relative entropy, but here it plays the role of a continuous union bound. The posterior `Q` can be chosen after seeing the sample because the high-probability event controls the whole function `f_S` before I name `Q`. The price for choosing `Q` is precisely `KL(Q||P)`.

After the change of measure, I use Jensen's inequality to move from a posterior average of deviations to a deviation between posterior-averaged losses. This is where the stochastic predictor becomes essential. I am no longer asking whether each sampled classifier has small true loss by itself. I am certifying the expected true loss of the randomized rule. The proof naturally produces a statement that is uniform over all posteriors:

`L(Q) <= hat L(Q,S) + sqrt((KL(Q||P) + ln(1/delta) + ln m + 2) / (2m - 1))`.

This is McAllester's square-root form. It has exactly the shape the problem demands. The first term is the empirical performance of the learned distribution. The second term is the cost of moving from the prior to the posterior, scaled by sample size and confidence. No assumption says that nature draws classifiers from `P`; the sample is still arbitrary IID data. The prior is a reference measure for the learner's bias, not a generative truth claim.

I also see why Seeger's and Maurer's kl-form is the cleaner final expression. For zero-one or bounded Bernoulli-style losses, the natural deviation is not always best expressed as a square root. It is sharper to bound the binary relative entropy between empirical and true losses:

`kl(hat L(Q,S), L(Q)) <= (KL(Q||P) + confidence_term) / m`.

Seeger expresses the theorem in this form for Gibbs classifiers, and Maurer tightens the exponential moment so that the sample-count logarithm becomes smaller. The proof skeleton remains the same: prove a prior exponential moment, apply Markov, change measure from `P` to `Q`, then use convexity to aggregate losses under the posterior.

The theorem also tells me how to train. If I fix a temperature parameter and minimize a bound-shaped objective, I minimize empirical posterior loss plus a KL penalty. The variational identity says that the optimizer is a Gibbs distribution:

`dQ_beta(h) = Z_beta^{-1} exp(-beta * hat L(h,S)) dP(h)`.

This is the Bayesian-looking object, but its justification is now PAC. The exponential tilt prefers low empirical loss, and the density is measured relative to the prior, so it cannot concentrate far from the prior without paying KL. McAllester's stochastic model-selection theorem derives the same optimizer through finite-dimensional Kuhn-Tucker conditions, while the later expositions recognize it as the general variational identity behind generalized Bayesian learning.

The distinctive insight is therefore not merely to add a prior to a PAC theorem. The older Occam bound already does that. The insight is to replace the selected classifier itself with a learned posterior distribution and to replace the singleton prior code length with `KL(Q||P)`. That one substitution changes the theorem's object. It makes continuous classes possible, it makes randomized classifiers first-class, it rewards broad regions of good predictors, and it keeps the guarantee distribution-free.

I can now read the proof as an upgraded union bound. The finite union bound says: if each hypothesis receives a failure budget proportional to `P(h)`, no data-dependent singleton can cheat the bound. The posterior version says: if the prior exponential moment is controlled, no data-dependent distribution can cheat the bound except by paying the relative entropy required to concentrate on the tempting region. The algebra is different, but the role is the same: protect against adaptive choice after the sample.

This also clarifies what the result does not claim. It does not certify the deterministic majority vote directly unless an additional argument relates the vote to the stochastic classifier. It does not say the prior is true. It does not make the bound automatically non-vacuous for every learning system. Its artifact is more precise: choose a prior before data, learn a posterior after data, compute posterior empirical loss and `KL(Q||P)`, then output a high-probability certificate for the stochastic predictor.

The final method is the bound and its variational training rule together. The certificate is meaningful because it is simultaneous over posteriors. The optimizer is meaningful because it is the distribution that trades empirical error against the information distance from the prior. The original single-classifier PAC story becomes a posterior-distribution PAC story, and the price of flexibility is exactly relative entropy.
