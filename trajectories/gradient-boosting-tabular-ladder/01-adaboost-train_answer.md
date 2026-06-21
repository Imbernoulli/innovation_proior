I start from the constructive promise that a weak tree learner — one that is only slightly better than random guessing — can be combined into a strong committee by training each new learner on a *reweighting* of the data that emphasizes the examples the committee currently gets wrong. That promise is a recipe with two holes in it: I have to decide *how much* to reweight a misclassified example each round, and *how much* each weak learner's vote should count in the final committee. The early boosting constructions filled those holes with a filtering scheme that needed the weak learner's error bounded *in advance* and shuffled the data through multiple sub-learners — heavy, and it forces me to know the edge of the weak learner before I run it. I want a scheme that adapts to whatever error the weak learner actually delivers, round by round, with no advance knowledge.

I propose **AdaBoost** — adaptive boosting — and what makes it work is that *both* the reweighting and the vote weight are computed from the realized weighted error each round, so nothing has to be fixed ahead of time. I keep a distribution over the $n$ training examples, weights $w_1,\dots,w_n$ that sum to one. Round $t$: hand the weak learner the data under the current distribution $p^t$, get back a hypothesis $h_t$, and measure its weighted error

$$\varepsilon_t = \sum_i p_i^{\,t}\,\big[h_t(x_i)\neq y_i\big].$$

If the weak learner is doing its job, $\varepsilon_t < \tfrac12$ — it beats random guessing on the *current* distribution. Now the two questions. For the weight update I want the next round's distribution to make $h_t$ look useless — to move mass onto exactly the examples $h_t$ got wrong. The cleanest move is multiplicative: pick a factor $\beta_t\in(0,1)$ and *multiply down* the weight of every example $h_t$ classified correctly, leaving the misclassified ones alone,

$$w_i^{\,t+1} = w_i^{\,t}\cdot\beta_t^{\,1-[h_t(x_i)\neq y_i]},$$

so the exponent is $1$ on a correct example (its weight shrinks by $\beta_t$) and $0$ on an incorrect one (weight unchanged); after renormalizing, the misclassified examples carry more of the distribution. It must be multiplicative rather than additive because the whole construction telescopes — the product of the per-round normalizers is what bounds the training error — and an additive bump would not compose into a clean product.

The value of $\beta_t$ is tied to how good $h_t$ was. If $\varepsilon_t$ is tiny, $h_t$ was almost perfect, the few examples it missed are genuinely hard, and I should reweight aggressively so $\beta_t$ is small; if $\varepsilon_t$ is near $\tfrac12$, $h_t$ told me almost nothing and I should barely reweight. I also want the next distribution to erase this learner's own advantage — to leave $h_t$ with weighted error exactly $1/2$ after the update — which forces the wrong mass $\varepsilon_t$ and the down-weighted correct mass $(1-\varepsilon_t)\beta_t$ to match, giving

$$\beta_t = \frac{\varepsilon_t}{1-\varepsilon_t}.$$

The limits check: $\varepsilon_t\to0$ gives $\beta_t\to0$ (correct examples crushed, all mass to the few errors), and $\varepsilon_t\to\tfrac12$ gives $\beta_t\to1$ (almost no reweighting). That is the adaptivity I wanted — $\beta_t$ is computed *after* I see $\varepsilon_t$, so I never need the weak learner's edge in advance, which is exactly the move the older filtering constructions could not make.

The vote weight falls out of the same $\beta_t$. A learner with low error should count for a lot; one near $\tfrac12$ should count for almost nothing. The natural scalar is the log-inverse,

$$\alpha_t = \log\frac{1}{\beta_t} = \log\frac{1-\varepsilon_t}{\varepsilon_t},$$

large when $\varepsilon_t$ is small and $\to0$ as $\varepsilon_t\to\tfrac12$, so the final classifier is the weighted vote $H(x)=\arg\max_y\sum_t \alpha_t\,[h_t(x)=y]$. This is the *right* $\alpha_t$ rather than some other increasing function of $(1-\varepsilon_t)/\varepsilon_t$ because it is what the training-error bound minimizes. Tracing the bound, the per-round normalizer is $Z_t=(1-\varepsilon_t)\beta_t+\varepsilon_t$, and the vote threshold contributes a matching factor $1/\sqrt{\beta_t}$; the quantity that gets minimized is $Z_t/\sqrt{\beta_t}$, and differentiating $(1-\varepsilon_t)\sqrt{\beta_t}+\varepsilon_t/\sqrt{\beta_t}$ gives back $\beta_t=\varepsilon_t/(1-\varepsilon_t)$. So $\alpha_t$ and $\beta_t$ are not two independent choices: the same ratio both neutralizes the old learner under the next distribution and minimizes the bound factor tied to its vote.

The multiclass case forces one more term. With $K>2$ labels, "better than random" is no longer $\varepsilon_t<\tfrac12$ — random guessing already errs at rate $(K-1)/K$ — so the bar is $\varepsilon_t<(K-1)/K$. If I keep the binary vote weight, a learner with $\varepsilon_t$ just under $(K-1)/K$, which is genuinely useful, could get a *negative* vote, which is wrong. The fix is to shift the vote weight by the log of the number of wrong classes,

$$\alpha_t = \log\frac{1-\varepsilon_t}{\varepsilon_t} + \log(K-1),$$

which is positive exactly when $\varepsilon_t<(K-1)/K$ and reduces to the binary form at $K=2$ where $\log(K-1)=0$. This is the **SAMME** multiclass form. The same $\alpha_t$ then drives the multiplicative reweighting: scale up the misclassified examples by $\exp(\alpha_t)$.

So the per-round step is: fit the weak learner on the weighted data; compute the weighted error; turn it into the vote weight $\alpha_t$ with the $\log(K-1)$ term; multiply the weights of the misclassified examples by $\exp(\alpha_t)$; renormalize. Two degenerate cases are guarded — if $\varepsilon_t\le0$ the learner is perfect (give it a large finite vote and stop), and if $\varepsilon_t$ is at least as bad as random, $(K-1)/K$, the learner is useless and is discarded. What this buys is a boosting algorithm with no hand-tuned reweighting schedule and a training-error bound that drops geometrically as long as every weak learner clears the random bar. Its limit is built into the very thing that made it work: the whole derivation rests on a $0/1$ misclassification indicator $[h_t(x_i)\neq y_i]$ and the exponential-style reweighting that crushes correctly-labeled examples. That ties it to classification with one implicit loss; the reweighting knows the *sign* of an error, never *how wrong* a prediction is, so there is no handle for a continuous target or any other differentiable loss.

```python
def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
    """Implement a single boost using the SAMME discrete algorithm."""
    estimator = self._make_estimator(random_state=random_state)
    estimator.fit(X, y, sample_weight=sample_weight)
    y_predict = estimator.predict(X)

    if iboost == 0:
        self.classes_ = getattr(estimator, "classes_", None)
        self.n_classes_ = len(self.classes_)

    # Instances incorrectly classified
    incorrect = y_predict != y

    # Error fraction (weighted by the current distribution)
    estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

    # Stop if classification is perfect
    if estimator_error <= 0:
        return sample_weight, 1.0, 0.0

    n_classes = self.n_classes_

    # Stop if the error is at least as bad as random guessing
    if estimator_error >= 1.0 - (1.0 / n_classes):
        self.estimators_.pop(-1)
        if len(self.estimators_) == 0:
            raise ValueError(
                "BaseClassifier in AdaBoostClassifier ensemble is worse than random, "
                "ensemble can not be fit."
            )
        return None, None, None

    # Boost weight using multi-class AdaBoost SAMME alg
    estimator_weight = self.learning_rate * (
        np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
    )

    # Only boost the weights if it will fit again
    if not iboost == self.n_estimators - 1:
        # Only boost positive weights
        sample_weight = np.exp(
            np.log(sample_weight) + estimator_weight * incorrect * (sample_weight > 0)
        )

    return sample_weight, estimator_weight, estimator_error
```
