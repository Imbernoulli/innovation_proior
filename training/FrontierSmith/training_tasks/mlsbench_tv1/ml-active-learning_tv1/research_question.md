Every label bought in the first few rounds is bought by a model that has barely
seen the problem: after the seed batch, the letter task's network holds roughly
four labels per class, and its softmax confidences, dropout committees, and
gradient embeddings are still largely artifacts of initialization. Classical
uncertainty sampling trusts those artifacts immediately; random sampling never
trusts the model at all. Both leave value on the table, and the
area-under-the-learning-curve number sees it: accuracy at every intermediate
budget counts toward that area, and competing strategies differ most at small
budgets, so the cold-start rounds move it more than the endgame does.

The question this variant asks: when does an active learner earn the right to
use its own model, and what should it buy before then? Design a batch
acquisition rule that behaves like a label-free coverage or representativeness
sampler while the model is untrustworthy, transitions to model-driven selection
as evidence accumulates, and drives that transition from a measured statistic —
committee disagreement, confidence stability between rounds, labeled-set size
relative to the apparent class count — rather than a hand-tuned round schedule.
One rule with one set of constants must serve all three datasets, which
deliberately span the cold-start spectrum: 26 classes seeded with 100 labels,
a binary problem seeded with 50, and a small 3-class pool seeded with 50.

You are judged on the same scoreboard as before — per-dataset final accuracy
and learning-curve area, combined multiplicatively across datasets — but the
target is different: raise the curve where it is currently cheapest to raise,
the first handful of rounds, without giving back final accuracy once the model
matures. A strategy that wins the endgame while starting slower than random
sampling has answered the wrong question here.
