## A Risk With Two Quantifiers

A statistical task is not only to design an estimator. It is also to characterize the best accuracy any estimator can achieve. For a model family, a target parameter, and a loss, the relevant object has the form

$$
\inf_{\hat\theta}\sup_{\theta\in\Theta} \mathbb E_\theta[\ell(\theta,\hat\theta)].
$$

The outer infimum ranges over every measurable rule that turns data into an estimate; the inner supremum is the worst case over the model family.

## Upper Bounds And Lower Bounds

An estimator with a small analyzed risk shows that a certain accuracy is attainable. A statement about optimality is of a different kind: it concerns what accuracy is attainable by any rule whatsoever, and is algorithm-independent — it speaks to what the data can reveal rather than to the mechanics of a particular procedure. Estimator analysis starts from a rule and asks how well it works; a worst-case accuracy statement starts from the model and asks how much uncertainty remains after observing the sample.

## Geometry Of Wrong Answers

Loss is not just a number attached after estimation; it defines which parameter values count as meaningfully different. If two parameter values are close under the loss geometry, confusing them carries little penalty; if they are far apart, confusing them produces visible risk. This suggests working with a finite set of possible truths that are well separated in the parameter metric, with the separation scale setting the scale of estimation error under consideration.

## Information In The Sample

Separated parameters need not produce separated data distributions. Many possible truths can lead to observation laws that are close in KL divergence, total variation, Hellinger distance, or mutual information. So two quantities are at play: the parameter space can hold many well-separated answers, while the data channel may transmit only a small amount of information about which answer is in force.

## Two Languages

A worst-case accuracy statement involves two kinds of quantity in different languages: geometric estimation error on the parameter side, and information-theoretic decision error on the data side. Information theory supplies inequalities — Fano's inequality among them — that bound the error of identifying a hidden index from observations in terms of the mutual information between index and data. The setting is thus a finite, well-separated family of candidate parameters together with the information their data laws carry about which candidate is active.
