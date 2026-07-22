I start with the part of the attack that is deliberately invisible. The model does not look broken on ordinary validation data. That is not an accident; the adversary wants the clean behavior to survive, because a model that loses ordinary accuracy is easy to reject. The only behavior that changes is conditional on a trigger I do not know. So I cannot wait for a validation failure, and I cannot compare against a known-clean reference set. I have the suspect model and the suspect training set, and I need to point to rows.

The poison is not just arbitrary noise. It has structure. A source-class example is modified by a repeated trigger and labeled as the target class. If the attack succeeds, the model must learn to treat that trigger as evidence for the target label. That gives me a handle: the poisoned samples and the legitimate target samples arrive at the same output, but they should not arrive there through the same internal evidence. A genuine target example is supported by target-class features. A triggered source example is supported by source-class features plus a trigger feature that the model has learned to route to the target. The difference may be tiny in pixels, but it need not be tiny inside the network. If that is right, then somewhere in the representation a poisoned target class is two populations wearing one label, and I want to find where that split is visible.

The first tempting move is to search in input space anyway. Images vary too much naturally: pose, lighting, background, handwriting style, crop, and color can swamp a small trigger. A text signature or pixel patch is not reliably separated by raw norms or raw correlations. If I want the trigger to be loud, I should look after the model has had a reason to amplify it. The late hidden layer is the closest representation to the decision and the least dominated by generic low-level features; earlier layers mostly add edge and texture variation that is not specific to this label-mismatch problem. So I will work on late-layer activations rather than pixels, and I will keep raw-input geometry as a negative control to test that premise later.

I also cannot pool all classes. If I throw every activation into one cloud, the largest structure is ordinary class separation — exactly what the representation was trained to create. The poison is a subpopulation inside a target class, so I have to remove the between-class geometry first. I split the activations by class and analyze each class separately. There are two ways to bin: by the model's predicted class `F(s)`, or by the stored training label `y_train`. For a successful relabeling backdoor these line up on the poisoned examples, because the whole point of the attack is that the model predicts the target label on triggered inputs. I keep the distinction explicit anyway, because the two choices give different code behavior on examples where the model and the label disagree.

Now I have one activation cloud per class. Before I reach for any Euclidean partition I should check whether distance even means anything here, because flattened late-layer activations can be tens of thousands of coordinates. I have a vague memory that distances concentrate in high dimensions, but I do not want to lean on a slogan, so I measure it. I draw 200 standard-Gaussian points and look at the contrast `(max - min) / min` of the distances from one point to the rest, for growing dimension:

```
d=     2  min=0.12 max=3.67  (max-min)/min = 29.44
d=    10  min=2.13 max=7.05  (max-min)/min =  2.31
d=  1000  min=42.34 max=47.26 (max-min)/min = 0.116
d= 50000  min=314.11 max=318.33 (max-min)/min = 0.013
```

At two dimensions the farthest point is thirty times farther than the nearest; at fifty thousand dimensions the nearest and farthest are within about one percent of each other. So a distance-based partition run directly on raw flattened activations would be cutting noise — every point is nearly equidistant from every other. That settles it: I must compress the activation vectors before I ask for geometric structure. PCA is the obvious baseline because it preserves the largest-variance directions. But the largest variance within a class may be ordinary variation among clean examples, not the clean-versus-poison direction. ICA is attractive for a different reason: a distinct injected population is a non-Gaussian component of the class mixture, which is exactly what independent-component analysis is built to surface. I do not want the component count to be a delicate knob; around ten components should be enough room for the geometry I am after, and I would want to confirm the result is stable across nearby counts rather than knife-edge.

With each class reduced to a handful of components, what shape do I expect? If a target class is poisoned, the simplest picture is one large legitimate population and one smaller trigger-bearing source population. Two groups suggests two-means. I do not need a density model first; two-means is fast, matches the two-compact-groups hypothesis, and returns memberships rather than a continuous outlier score — and membership is exactly the form a training-set cleaner needs.

But here is the problem I cannot wave away: two-means always returns two clusters. Run it on a perfectly clean class and it still cuts the cloud in half. So clustering is not detection. Whatever I build needs a second layer that decides whether a split means poison at all, and if so which side is poison.

Let me see how bad the bare clustering is, by running it on a clean class and a poisoned one and comparing. I make a clean class as a single Gaussian blob of 200 points in ten dimensions, and a poisoned class as 170 legitimate points plus 30 points shifted well away (the trigger subpopulation). Two-means on each:

```
clean blob:    silhouette = 0.0804   sizes = [102, 98]
poisoned mix:  silhouette = 0.7728   sizes = [170, 30]
```

This is informative in two ways at once. The clean blob does get split — into a near-even 102/98 — but the silhouette is only 0.08, meaning points are barely closer to their own half than to the other; the boundary is essentially arbitrary. The poisoned mix splits cleanly into 170/30 with silhouette 0.77, points firmly inside their own cluster. So two signals fall out of the same experiment. First, size: the clean split is roughly even, the poisoned split is lopsided with a clear minority. Second, silhouette: an artificial bisection scores low, a real two-population split scores high. Neither is announced in advance — both just dropped out of running the clustering on a known-clean and a known-poisoned case.

The size signal has an attacker-side justification too. To keep the target class mostly legitimate and the attack hidden, the poison has to stay a minority of the class. So when a split reflects a true clean/poison division, the poison is the smaller cluster, and "mark the smaller cluster" is a very cheap detector. The cheapness has a cost, and I should be honest about it: on the clean blob above, this rule would have marked the 98-point half as poison, because the smaller cluster exists whether or not there is any poison. So the bare smaller-cluster rule is only defensible as a deliberately aggressive default, or when a downstream budget or human review filters the report. It is not by itself a clean-class test.

To get a clean-class test I need a threshold, and the size signal already gives me one. If the smaller cluster is not actually small, the split is probably the arbitrary bisection of a clean class. With a defender-set upper bound on poison rate, I flag only clusters below that fraction. Let me trace the rounded-fraction rule from the implementation on concrete splits to see exactly where the line falls, with the default `size_threshold = 0.35`, fractions rounded to two decimals, strict `<`:

```
49/51 -> pct [0.49, 0.51]  flagged: {}      (nothing below 0.35)
15/85 -> pct [0.15, 0.85]  flagged: {0}      (0.15 < 0.35)
35/65 -> pct [0.35, 0.65]  flagged: {}      (0.35 is NOT < 0.35)
34/66 -> pct [0.34, 0.66]  flagged: {0}      (0.34 < 0.35)
```

So the near-even clean split (49/51) is correctly left alone, the lopsided 15/85 is flagged, and the threshold is genuinely strict: a cluster sitting exactly at 0.35 is not flagged, while one at 0.34 is. That last detail matters for matching the implementation — the comparison rounds first and then uses `<`, not `<=`, so the boundary case is a non-flag. This turns the minority assumption into a class-level decision instead of an always-on accusation.

Silhouette gives a second, partly independent test, and I already have the numbers from the clustering experiment above: the clean blob scored 0.08 and the poisoned mix scored 0.77. With a default `silhouette_threshold` around 0.1, the clean blob falls below it and the poisoned mix sits far above it. So I can require both conditions — the candidate cluster is small enough and the two-cluster fit is strong enough. If both hold, the smaller cluster is poison and its sibling is clean. If the silhouette is low, I treat the class as clean even though two-means produced two labels. That directly repairs the over-aggressiveness of the bare smaller-cluster rule, and on my two synthetic classes it would have made the right call on each.

The most discriminating check costs more but commits the least to a heuristic. I remove a suspicious cluster, train a fresh copy of the model on the remaining data, and then classify the held-out cluster with that new model. The logic is causal: if the held-out cluster is legitimate target data, the retrained model still has plenty of target examples and should keep predicting the target label on it. If the held-out cluster is poison, the trigger-bearing source images are gone from training, so the new model has no reason to map them to the target — they should revert to the source class. I count `l`, the number of held-out examples predicted as the cluster's own label, and `p`, the number predicted as the most common other class, and look at the ratio. Let me trace the decision rule on the two cases it is meant to separate, with threshold `T = 1.0` and the clear-iff `p == 0 or l/p > T` rule:

```
legit  cluster:  l=80 p=5   ->  l/p = 16.0   -> clear (stays target)
poison cluster:  l=8  p=72  ->  l/p = 0.111  -> keep poison (reverts to source)
boundary:        l=10 p=10  ->  l/p = 1.0    -> keep poison (1.0 is NOT > 1.0)
```
