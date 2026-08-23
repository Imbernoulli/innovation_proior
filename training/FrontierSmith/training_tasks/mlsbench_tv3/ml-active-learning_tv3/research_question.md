A 26-class recognizer seeded with 100 labels starts life having seen roughly
four examples per class — and, at the mercy of the random seed, essentially
zero for several classes. Whatever those starved classes look like in feature
space, the model cannot claim test accuracy on them, and with per-class test
mass near 1/26 each invisible class puts a hard ceiling of several accuracy
points on every subsequent round until labels reach it. Confidence-driven
selection is slow to repair this: a class the model barely represents does
not generate confident-looking confusion, it generates silence.

The design problem of this variant is label allocation across classes when
the class inventory itself is only partially observed. The acquisition rule
should track which labels it has actually collected, form a view of which
classes are under-supplied relative to the pool — using the model's own
posteriors, embedding structure, or clustering as a proxy for where missing
classes hide — and route a controlled share of each batch toward candidates
most likely to fill the thinnest parts of the labeled histogram. Two failure
modes bound the design from either side: chasing phantom rarity wastes the
budget on outliers, while ignoring rarity reproduces plain uncertainty
sampling. The rule must also degrade gracefully into ordinary informativeness
seeking on spambase, whose two classes are both well supplied from the seed —
class-coverage logic should activate in proportion to measured starvation,
not run open-loop on every dataset.

The metrics do not change: test accuracy at the final budget and the area
under the accuracy-per-labels curve on letter, spambase, and splice. The
claim to defend is that on class-rich pools, repairing labeled-set coverage
early buys accuracy that pointwise informativeness cannot reach at the same
budget, without measurable cost on the class-poor pools.
