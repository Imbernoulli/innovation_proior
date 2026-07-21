I have trees that are weak learners and the constructive promise that a weak learner can be combined
into a strong one by a *weighted sequence* — each new learner trained on a reweighting of the data that
emphasizes what the committee gets wrong. That promise is a recipe with two holes in it: I have to
decide *how much* to reweight a misclassified example each round, and *how much* each weak learner's
vote should count in the final committee. Before I try to fill those holes I want to lay the honest
options side by side, because "reweight the mistakes" is not yet an algorithm and there is more than one
way to make it into one.

The oldest way is *filtering*. Run the weak learner, then draw the next learner's training set from a
filtered stream so that the first learner is at chance on the new distribution, and combine the
sub-learners by majority. It works, but the price is steep and specific: to size the filtered samples
and decide how many sub-learners to stack, I have to know the weak learner's *edge* γ — how far below ½
its error sits — *before* I run it, because the sample sizes and the number of stacked hypotheses scale
like a power of 1/γ. If I guess γ too optimistically the schedule under-samples and the majority is
unreliable; if I guess it too pessimistically I waste enormous samples. And filtering literally throws
data on the floor to reshape the distribution, which is wasteful when data is the scarce thing. So the
filtering route couples the whole construction to a number I do not have. A second option is a fixed
*reweighting schedule*: keep a distribution over examples, and after each round multiply the weights by
some constant factor on the examples the learner got right, with the *same* constant every round. This
at least stops throwing data away. But I will show in a moment that the round-by-round bookkeeping wants
a factor that depends on how well *that* round's learner did, and a single frozen constant leaves slack
on every round where the realized error differs from whatever the constant was tuned for — and to tune
it I would again need to know the error sequence in advance. The third option is the one I actually
want: reweight multiplicatively, but let the factor *adapt* to the error the learner just delivered, so
nothing has to be fixed ahead of time. Let me build that one and let the bookkeeping tell me exactly
what the factor should be.

Set it up concretely. I keep a distribution over the n training examples — weights w₁,…,wₙ that sum to
one, call the normalized version p. Round t: hand the weak learner the data under distribution pᵗ, get
back a hypothesis hₜ, and measure its weighted error εₜ = Σᵢ pᵢᵗ·[hₜ(xᵢ) ≠ yᵢ]. If the weak learner is
doing its job, εₜ < ½ — it beats random guessing on the *current* distribution. Now the two questions:
how to update the weights for round t+1, and how to weight hₜ's vote.

Start with the weight update. I want the next round's distribution to make hₜ look useless — to move
mass onto exactly the examples hₜ got wrong, so the next weak learner is forced to attend to them. The
cleanest multiplicative move: pick a factor βₜ ∈ (0,1) and *multiply down* the weight of every example
hₜ classified **correctly**, leaving the misclassified ones alone:

  wᵢᵗ⁺¹ = wᵢᵗ · βₜ^(1 − [hₜ(xᵢ) ≠ yᵢ])

so the exponent is 1 on a correct example (weight shrinks by βₜ) and 0 on an incorrect one (weight
unchanged), and after renormalizing, the misclassified examples carry more of the distribution. Why
multiplicative and not additive — say wᵢᵗ⁺¹ = wᵢᵗ + c·[wrong]? Because I want the whole construction to
*telescope*, and only a product telescopes. Watch what the multiplicative rule does to the
normalization. Write the per-round normalizer Zₜ = Σᵢ pᵢᵗ·βₜ^(1 − [hₜ(xᵢ) ≠ yᵢ]). Split the sum into
the correctly-classified mass (fraction 1−εₜ, each multiplied by βₜ) and the wrong mass (fraction εₜ,
untouched):

  Zₜ = (1 − εₜ)·βₜ + εₜ.

Now track the *unnormalized* weights across all T rounds starting from uniform wᵢ¹ = 1/n. Each round
multiplies wᵢ by βₜ^(1 − lossₜ(i)) where lossₜ(i) = [hₜ(xᵢ) ≠ yᵢ] ∈ {0,1}, and the total weight after a
round is the previous total times Zₜ (because summing wᵢ·βₜ^(1−loss) over i is exactly (Σwᵢ)·Zₜ once
you factor out the normalization). So after T rounds the total unnormalized weight is the clean product
Σᵢ wᵢᵀ⁺¹ = ∏ₜ Zₜ, and each individual example's final weight is wᵢᵀ⁺¹ = (1/n)·∏ₜ βₜ^(1 − lossₜ(i)). An
additive bump would give a sum-of-terms here, not a product — the rounds would not compose, and the
bound I am about to write would fall apart. That is the concrete reason the update must be
multiplicative.

Now what should βₜ be? I want it tied to how good hₜ was. If εₜ is tiny — hₜ was almost perfect — then
I should reweight aggressively, because the few examples it missed are genuinely hard and the next
learner must focus on them; βₜ should be small. If εₜ is near ½ — hₜ barely beat random — I should
reweight gently, because hₜ told me almost nothing. There are two independent ways to pin βₜ, and the
fact that they *agree* is what convinces me it is the right choice. The first is a neutrality condition:
I want the next distribution to erase this learner's own advantage, so that the learner I just used has
weighted error exactly 1/2 after the update — it becomes uninformative under the distribution its
successor sees. Under pᵗ⁺¹ the mass on the examples hₜ got wrong is εₜ/Zₜ (their weights were untouched)
and I want that to equal ½; since the wrong mass εₜ and the down-weighted correct mass (1−εₜ)βₜ are the
only two pieces of Zₜ, the condition εₜ/Zₜ = ½ is the same as εₜ = (1−εₜ)βₜ, giving

  βₜ = εₜ / (1 − εₜ).

Check the limits: εₜ → 0 gives βₜ → 0 (correct examples crushed, all mass to the few errors); εₜ → ½
gives βₜ → 1 (barely any reweighting). That is the adaptivity I wanted — βₜ is computed *after* I see
εₜ, so I never need to know the weak learner's edge in advance, which is exactly the move the filtering
route could not make.

The second way to pin βₜ is to minimize the training error directly, and this is what tells me the
*same* factor is not a coincidence but forced. Trace the error of the final weighted-vote classifier.
The final classifier predicts by weighted majority: give hypothesis hₜ vote weight αₜ = log(1/βₜ) (I
will justify this scalar in a moment) and predict the label with the largest total vote. An example i
that the *vote* misclassifies must have been gotten wrong on a heavily-weighted subset of rounds — its
weighted-loss share Σₜ αₜ·lossₜ(i) reaches at least half the total vote Σₜ αₜ. Exponentiate that
inequality: with αₜ = log(1/βₜ), the condition Σₜ αₜ·lossₜ(i) ≥ ½ Σₜ αₜ rearranges to
∏ₜ βₜ^(1 − lossₜ(i)) ≥ ∏ₜ √βₜ. So every misclassified example carries final unnormalized weight at
least (1/n)·∏ₜ √βₜ. But all the final weights sum to ∏ₜ Zₜ, and if a fraction ε_train of the n examples
are misclassified, their combined weight is at least ε_train·∏ₜ √βₜ, which cannot exceed the total:

  ε_train ≤ ∏ₜ Zₜ / √βₜ = ∏ₜ ( (1 − εₜ)βₜ + εₜ ) / √βₜ.

Each factor depends on its own βₜ, so I minimize them one at a time. Differentiate
f(β) = (1 − εₜ)√β + εₜ·β^(−1/2) and set f'(β) = (1−εₜ)/(2√β) − εₜ/(2β^(3/2)) = 0; this is (1−εₜ)β = εₜ,
i.e. β = εₜ/(1−εₜ) — the *same* βₜ the neutrality condition gave. Substituting it back, the per-round
factor collapses to 2√(εₜ(1−εₜ)): with βₜ = εₜ/(1−εₜ) I get Zₜ = 2εₜ and √βₜ = √(εₜ/(1−εₜ)), so
Zₜ/√βₜ = 2εₜ·√((1−εₜ)/εₜ) = 2√(εₜ(1−εₜ)). Writing εₜ = ½ − γₜ, this is √(1 − 4γₜ²) ≤ exp(−2γₜ²), so

  ε_train ≤ ∏ₜ 2√(εₜ(1−εₜ)) = ∏ₜ √(1 − 4γₜ²) ≤ exp(−2 Σₜ γₜ²).

The training error drops geometrically as long as every learner clears the random bar (γₜ > 0). So βₜ
is not two independent choices dressed up as one — the ratio εₜ/(1−εₜ) is simultaneously what
neutralizes the old learner and what minimizes the bound, and that coincidence is the reason to trust
it over any hand-set schedule.

The fixed-schedule option I set aside earlier settles against this bound as a yardstick. With realized
errors that differ across rounds — say ε₁ = 0.1 and ε₂ = 0.4 — the round-optimal minimizers of the bound
factor ((1−ε)β + ε)/√β are β₁* = 1/9 and β₂* = 2/3, a nearly sixfold spread, so any single frozen β must
compromise and pays pure slack on every round whose realized error differs from whatever it was tuned
for; to place it well I would again need the ε-sequence in advance. The adaptive βₜ = εₜ/(1−εₜ) is
exactly the per-round minimizer, so it leaves no slack and needs nothing known ahead of time — the payoff
of computing the factor after seeing εₜ instead of before.

One consistency point about the update matters for the implementation: instead of multiplying the *correct*
examples by βₜ, I can multiply the *wrong* ones by exp(αₜ) = 1/βₜ, and after renormalizing the two give
an identical distribution. So the implementation is free to phrase the update as "amplify the
misclassified examples by exp(αₜ)."

The vote weight is the scalar αₜ = log(1/βₜ) = log((1 − εₜ)/εₜ) I already used in the bound, and I
should say why it is *that* function of the error and not some other increasing one — 1/εₜ, or (1 −
2εₜ), or √((1−εₜ)/εₜ) would all be "bigger when the learner is better." The bound forces it. The
weighted-majority threshold contributes the factor 1/√βₜ to each per-round term precisely when the vote
weight is log(1/βₜ); any other scalar would break the rearrangement Σₜ αₜ·lossₜ(i) ≥ ½ Σₜ αₜ ⟺ ∏ₜ
βₜ^(1−lossₜ(i)) ≥ ∏ₜ √βₜ, and the clean ∏ Zₜ/√βₜ bound would no longer be what I am minimizing. So αₜ
and βₜ are locked together: the log-inverse of the reweighting factor *is* the vote weight, both fixed
by the same telescoping product. αₜ is large when εₜ is small and → 0 as εₜ → ½, and the final
classifier is the weighted vote

  H(x) = argmax_y Σₜ αₜ · [hₜ(x) = y].

There is one more thing the multiclass case forces. With K > 2 labels, "better than random" is no
longer εₜ < ½ — random guessing already gets error (K−1)/K, so the bar is εₜ < (K−1)/K. If I keep the
binary vote weight log((1−εₜ)/εₜ), a weak learner with εₜ just under (K−1)/K — useful in the multiclass
sense — could get a *negative* vote weight, which is wrong; it beat random and should count positively.
The fix is to shift the vote weight by the log of the number of "wrong" classes:

  αₜ = log((1 − εₜ)/εₜ) + log(K − 1).

The shift is forced by requiring αₜ > 0 exactly when the learner beats the K-class random bar:
log((1−εₜ)/εₜ) + log(K−1) > 0 rearranges to (1−εₜ)(K−1) > εₜ, i.e. εₜ < (K−1)/K, no looser and no
tighter. At K = 2 the shift vanishes and I recover the binary form with its guard εₜ < ½; at K = 3
(random error 2/3) a learner with εₜ = 0.6 — useful, since 0.6 < 2/3 — would score log(0.4/0.6) = −0.405
from the binary term alone and be wrongly discarded, and the +log 2 shift lifts it to +0.288. The same αₜ
then drives the reweighting: scale up the misclassified examples by exp(αₜ).

So the per-round step is: fit the weak learner on the weighted data; compute the weighted error; turn it
into the vote weight αₜ with the log(K−1) multiclass term; multiply the weights of the misclassified
examples by exp(αₜ); renormalize. Guard the degenerate cases — if εₜ ≤ 0 the learner is perfect (give it
a large finite vote and stop), and if εₜ is at least as bad as random ((K−1)/K) the learner is useless
and I discard it. (The full `_boost_discrete` step is in the answer.)

Before I close, I want to make the nature of this loss explicit rather than leave it as a feeling,
because it is the thing that will limit the whole scheme. Track the weight that example i carries into
round T+1 under the up-weight-wrong convention: starting from uniform, it is proportional to ∏ₜ
exp(αₜ·lossₜ(i)) = exp(Σₜ αₜ·lossₜ(i)). In the {±1} binary form write the running vote as
F_T(xᵢ) = Σₜ αₜ hₜ(xᵢ) with hₜ(xᵢ) ∈ {−1,+1}, and note that the 0/1 loss is lossₜ(i) = (1 −
yᵢhₜ(xᵢ))/2. Substituting, Σₜ αₜ·lossₜ(i) = ½ Σₜ αₜ − ½ yᵢ F_T(xᵢ), and the first term is a constant
across examples that the renormalization absorbs, so the weight is proportional to exp(−½·yᵢ F_T(xᵢ)).
The reweighting is not descending "the classification error" in some vague sense — it is putting weight
exp(−margin/2) on each example, which is exactly the gradient of the **exponential loss** exp(−yF) of
the margin. AdaBoost is stagewise minimization of one specific objective, and I derived it without ever
naming that objective; the algorithm found it for me. That is reassuring and confining at once: the
reweighting is tied to this loss and no other, because the margin yᵢ F_T(xᵢ) is a *sign*-based quantity
— it only knows whether the committee is on the right side, weighted by confidence, never how far a
continuous prediction sits from a continuous target.

So this is **AdaBoost**: adaptive reweighting by βₜ = εₜ/(1−εₜ) and voting by αₜ = log((1−εₜ)/εₜ)
(+ log(K−1) for K classes), with no hand-tuned schedule, and a training-error bound ∏ₜ 2√(εₜ(1−εₜ)) that
drops geometrically while every learner clears the random bar. This is a foundational construction that
predates the tabular speed/accuracy benchmarks, so what I can honestly claim is the algorithm and its
bound, not a figure on those datasets; the falsifiable prediction is the geometric decrease, and it lives
in the training-error curve, not a held-out benchmark number.

But look at what I had to assume to get here. The whole derivation is built around *misclassification*
— a 0/1 indicator [hₜ(xᵢ) ≠ yᵢ] — and the exponential-style reweighting that crushes correctly-labeled
examples. The neutrality condition, the telescoping product, the bound: every one of them speaks the
language of "right or wrong," never "how wrong." That ties me to classification with one particular
implicit loss; the reweighting is hard-wired to the *sign* of an error, not its magnitude. If I want to
predict a continuous target, or minimize a Huber or Poisson or logistic-deviance loss rather than the
one this reweighting implicitly descends, I have no knob: the machine only knows how to push weight
around based on a binary correctness flag. The next question is whether the reweighting can be replaced
by something that works for *any* differentiable loss — so the boosting recipe stops being a
classification trick and becomes a general descent.
