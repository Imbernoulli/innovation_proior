## A Risk With Two Quantifiers

The statistical task is not only to design an estimator. It is also to know when no estimator can do better. For a model family, a target parameter, and a loss, the lower-bound object has the form

$$
\inf_{\hat\theta}\sup_{\theta\in\Theta} \mathbb E_\theta[\ell(\theta,\hat\theta)].
$$

The outer infimum is the obstacle. A proof cannot defeat one proposed estimator at a time; it must constrain every measurable rule that turns data into an estimate.

## Why Upper Bounds Do Not Settle It

An estimator with a small analyzed risk proves that a certain accuracy is attainable. It does not prove that a sharper accuracy is unattainable. To certify optimality, the argument has to be algorithm-independent: it must talk about what the data can reveal, not about the mechanics of a particular procedure.

This is why ordinary estimator analysis has the wrong direction for lower bounds. It starts from a rule and asks how well it works. A lower bound must start from the model and ask how much uncertainty remains after observing the sample.

## Geometry Of Wrong Answers

Loss is not just a number attached after estimation. It defines which parameter values count as meaningfully different. If two parameter values are close under the loss geometry, confusing them is not a serious failure; if they are far apart, confusing them forces visible risk.

A useful lower-bound proof therefore needs a finite set of possible truths that are well separated in the parameter metric. The separation scale will later become the estimation error scale.

## Information In The Sample

Separated parameters need not produce separated data distributions. In hard problems, many possible truths lead to observation laws that are close in KL divergence, total variation, Hellinger distance, or mutual information. The sample may carry far less information about the truth than the parameter geometry suggests is needed.

This creates a tension: the parameter space can contain many well-separated answers, while the data channel may transmit only a small amount of information about which answer generated the sample.

## The Missing Bridge

The remaining gap is conceptual. A lower bound must connect two statements that live in different languages: geometric estimation error and information-theoretic decision error. If accurate estimation would imply reliable identification of a hidden alternative, then any impossibility theorem for identification would immediately become an impossibility theorem for estimation.

The method begins once that bridge is made precise.
