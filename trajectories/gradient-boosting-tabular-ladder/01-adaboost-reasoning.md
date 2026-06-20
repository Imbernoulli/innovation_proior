I have trees that are weak learners and the constructive promise that a weak learner can be combined
into a strong one by a *weighted sequence* — each new learner trained on a reweighting of the data that
emphasizes what the committee gets wrong. That promise is a recipe with two holes in it: I have to
decide *how much* to reweight a misclassified example each round, and *how much* each weak learner's
vote should count in the final committee. The early boosting constructions answered these with a
filtering scheme that needed the weak learner's error to be bounded in advance and shuffled the data
through multiple sub-learners — heavy, and it forces me to know the edge of the weak learner before I
run it. I want a scheme that adapts to whatever error the weak learner actually delivers, round by
round, with no advance knowledge.

Let me set it up concretely. I keep a distribution over the n training examples — weights w₁,…,wₙ that
sum to one, call the normalized version p. Round t: hand the weak learner the data under distribution
pᵗ, get back a hypothesis hₜ, and measure its weighted error εₜ = Σᵢ pᵢᵗ·[hₜ(xᵢ) ≠ yᵢ]. If the weak
learner is doing its job, εₜ < ½ — it beats random guessing on the *current* distribution. Now the two
questions: how to update the weights for round t+1, and how to weight hₜ's vote.

Start with the weight update. I want the next round's distribution to make hₜ look useless — to move
mass onto exactly the examples hₜ got wrong, so the next weak learner is forced to attend to them. The
cleanest multiplicative move: pick a factor βₜ ∈ (0,1) and *multiply down* the weight of every example
hₜ classified **correctly**, leaving the misclassified ones alone:

  wᵢᵗ⁺¹ = wᵢᵗ · βₜ^(1 − [hₜ(xᵢ) ≠ yᵢ])

so the exponent is 1 on a correct example (weight shrinks by βₜ) and 0 on an incorrect one (weight
unchanged), and after renormalizing, the misclassified examples carry more of the distribution. Why
multiplicative and not additive? Because the whole construction is going to telescope — the product of
the per-round normalizers is what bounds the training error — and a multiplicative update makes that
product clean. An additive bump would not compose.

Now what should βₜ be? I want it tied to how good hₜ was. If εₜ is tiny — hₜ was almost perfect — then
I should reweight aggressively, because the few examples it missed are genuinely hard and the next
learner must focus on them; βₜ should be small. If εₜ is near ½ — hₜ barely beat random — I should
reweight gently, because hₜ told me almost nothing. I also want the next distribution to erase this
learner's own advantage, so the learner I just used has weighted error 1/2 after the update. That
condition says the wrong mass εₜ and the down-weighted correct mass (1−εₜ)βₜ must match, so

  βₜ = εₜ / (1 − εₜ).

Check the limits: εₜ → 0 gives βₜ → 0 (correct examples crushed, all mass to the few errors); εₜ → ½
gives βₜ → 1 (barely any reweighting). That is the adaptivity I wanted — βₜ is computed *after* I see
εₜ, so I never need to know the weak learner's edge in advance. This is the move the older filtering
constructions couldn't make: they fixed the reweighting schedule ahead of time; here it adapts to the
realized error.

The vote weight falls out of the same βₜ. A weak learner that achieved low error εₜ should count for a
lot in the final committee; one near ½ should count for almost nothing. The natural scalar is the
log-inverse of βₜ:

  αₜ = log(1/βₜ) = log((1 − εₜ)/εₜ),

large when εₜ is small, → 0 as εₜ → ½. So the final classifier is a weighted vote, each hypothesis
weighted by how much it beat random:

  H(x) = argmax_y Σₜ αₜ · [hₜ(x) = y].

Why is this the *right* αₜ and not some other increasing function of (1−εₜ)/εₜ? Trace the training
error. After T rounds the unnormalized weight of example i is the product over rounds of βₜ raised to
the (correct-indicator) exponent; an example that the *weighted vote* misclassifies must have been
gotten wrong often enough that its surviving weight is large, and summing those surviving weights gives
an upper bound on the training error of H. The per-round normalizer is
Zₜ = (1−εₜ)βₜ + εₜ, while the vote threshold contributes a matching factor 1/√βₜ in that bound. The
quantity that gets minimized is therefore Zₜ/√βₜ, and differentiating
(1−εₜ)√βₜ + εₜ/√βₜ gives βₜ = εₜ/(1−εₜ). So αₜ and βₜ are not two independent choices; the same ratio
both makes the old learner neutral under the next distribution and minimizes the bound factor tied to
its vote.

There is one more thing the multiclass case forces. With K > 2 labels, "better than random" is no
longer εₜ < ½ — random guessing already gets error (K−1)/K, so the bar is εₜ < (K−1)/K. If I keep the
binary vote weight log((1−εₜ)/εₜ), a weak learner with εₜ just under (K−1)/K — useful in the multiclass
sense — could get a *negative* vote weight, which is wrong; it beat random and should count positively.
The fix is to shift the vote weight by the log of the number of "wrong" classes:

  αₜ = log((1 − εₜ)/εₜ) + log(K − 1),

which is positive exactly when εₜ < (K−1)/K, recovering the right notion of "beats random" for K
classes (and reducing to the binary form at K = 2, where log(K−1) = 0). The same αₜ then drives the
multiplicative reweighting: scale up the misclassified examples by exp(αₜ).

Let me write the per-round step. Fit the weak learner on the weighted data; compute the weighted error;
turn it into the vote weight αₜ with the log(K−1) multiclass term; multiply the weights of the
misclassified examples by exp(αₜ); renormalize. Guard the degenerate cases — if εₜ ≤ 0 the learner is
perfect (give it a large finite vote and stop), and if εₜ is at least as bad as random ((K−1)/K) the
learner is useless and I discard it.

```python
def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
    estimator = self._make_estimator(random_state=random_state)
    estimator.fit(X, y, sample_weight=sample_weight)
    y_predict = estimator.predict(X)

    incorrect = y_predict != y
    # weighted error on the current distribution
    estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

    if estimator_error <= 0:                  # perfect on the weighted data
        return sample_weight, 1.0, 0.0

    n_classes = self.n_classes_
    if estimator_error >= 1.0 - (1.0 / n_classes):   # no better than random
        self.estimators_.pop(-1)
        return None, None, None

    # vote weight: log((1-err)/err) with the +log(K-1) multiclass term
    estimator_weight = self.learning_rate * (
        np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
    )
    # up-weight the misclassified examples
    if not iboost == self.n_estimators - 1:
        sample_weight = np.exp(
            np.log(sample_weight) + estimator_weight * incorrect * (sample_weight > 0)
        )
    return sample_weight, estimator_weight, estimator_error
```

What does this give me, and where does it stop? It gives a boosting algorithm with no free hand-tuned
reweighting schedule — βₜ and αₜ are both computed from the realized error εₜ — and a training-error
bound that drops geometrically as long as every weak learner clears the random bar. That is the answer
to "how do I reweight, and how do I vote": **AdaBoost**, adaptive reweighting by βₜ = εₜ/(1−εₜ) and
voting by αₜ = log((1−εₜ)/εₜ) (+ log(K−1) for K classes).

But look at what I had to assume to get here. The whole derivation is built around *misclassification*
— a 0/1 indicator [hₜ(xᵢ) ≠ yᵢ] — and the exponential-style reweighting that crushes correctly-labeled
examples. That ties me to classification with a particular implicit loss; the reweighting is hard-wired
to the sign of "right or wrong," not to *how wrong*. If I want to predict a continuous target, or
minimize a different loss than the one this reweighting implicitly descends, I have no obvious knob: the
algorithm only knows how to push weight around based on a binary correctness flag. The next question is
whether the reweighting can be replaced by something that works for *any* differentiable loss — so the
boosting recipe stops being a classification trick and becomes a general descent.
