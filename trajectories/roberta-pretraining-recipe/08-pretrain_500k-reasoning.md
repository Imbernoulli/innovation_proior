Extending from 100,000 to 300,000 steps at the fixed 160GB corpus moved every metric: SQuAD from
94.0/87.7 to 94.4/88.7, MNLI-m from 89.3 to 90.0, SST-2 from 95.6 to 96.1. The SQuAD 2.0 gain (+1.0) and
the MNLI-m gain (+0.7) are both noticeably larger than what the data-only rung produced at fixed steps
(+0.4 SQuAD 2.0, +0.3 MNLI-m) — a threefold increase in step count is doing more work than a tenfold
increase in data did, on the two hardest metrics in the suite. That answers, at least directionally, the
question I posed going into that rung: step count, not data volume, was the tighter constraint at 100K
steps. It also directly addresses the risk I flagged before running it. If the model were beginning to
overfit the 160GB corpus at 300K steps, I would expect to see downstream numbers stall or reverse
relative to 100K, particularly on the harder, more overfitting-sensitive metrics — instead every single
metric moved further in the same direction, and by a larger margin than the previous lever produced.
That is not proof overfitting is impossible at some larger step count still ahead, but it is direct
evidence against the specific worry that 300K steps over 160GB had already crossed into memorization
territory: a model quietly overfitting would show exactly the opposite signature from what I just
measured.

So the two questions this rung was designed to settle are both answered, and cleanly: duration was (and
apparently still is) the binding lever relative to data volume at this corpus size, and there's no sign
yet of the point where more steps stops helping or starts hurting. Given that, the natural next move
isn't to switch levers back to data, or to stop here and call 300K steps the final answer — it's to keep
pulling the lever that's still visibly working. The question is how far.

I want to reason carefully about where to stop rather than just picking a round number. The original
BERT-large recipe used one million steps. I've already learned, decisively, that a properly corrected
recipe at 100K steps over 16GB beat that fully-trained million-step original outright — so "match the
original's step count" was never really the target here; the recipe corrections and the 160GB corpus
have already made the original's step budget somewhat beside the point as a benchmark. What matters
now is a narrower, more honest question: given the trajectory just observed — continued improvement from
100K to 300K, with no overfitting signal — does that trajectory keep going if I extend further, or was
300K itself already close to wherever the curve bends?

I don't have a way to answer that except by extending again and checking, which is exactly the same
methodological posture every rung in this study has taken: propose the next increment based on the
trend the last rung established, and let the next measurement either confirm the trend continues or
reveal where it stops. Doubling from 300K to 600K would be a large, expensive step to take on a
hypothesis I haven't yet tested even once — I've only established that the curve is still rising between
two points, not how far past the second point it keeps rising. A smaller, more conservative extension —
roughly 500K steps, about 1.7x the 300K figure and still barely half of the original recipe's million-step
budget — lets me test whether the upward trend from 100K to 300K is a stable, continuing slope or
whether it was already flattening by 300K and I simply hadn't yet extended far enough to see the bend.
If the gain from 300K to 500K looks similar in size to the gain from 100K to 300K on a per-step basis,
that says the curve genuinely hasn't found its ceiling yet and a still-longer run would likely help
further. If the gain from 300K to 500K is much smaller than the 100K-to-300K gain, even though it's
still an improvement, that's the first sign of the returns starting to diminish — informative on its own,
without needing to have pushed all the way to a full million steps to see it.

There's also a resource argument for stopping the search here rather than continuing to escalate step
count indefinitely as this study's only remaining lever. Every one of the eight rungs in this study,
taken together, has already established a clear, compounding picture: the recipe corrections (masking,
input format, batch size, vocabulary) are real and additive; the combined recipe at modest data and
duration already beats the original fully-trained model; more data helps modestly; more duration helps
more, and hasn't yet shown a ceiling. At some point, continuing to escalate step count with diminishing
marginal information about *why* it helps — as opposed to *that* it helps — stops teaching me much more
about the design-choice questions this study set out to answer, and starts just becoming "spend more
compute for a somewhat better number," which is a different kind of question than the one this study has
been asking rung to rung. 500K, roughly half the original recipe's budget but three separate steps past
where this recipe started (100K), gives one more clean data point on the duration curve while keeping
this as a study of design choices rather than an open-ended compute-scaling exercise.

Concretely: hold the 160GB combined corpus, BERT-large architecture, 8K batch, and every recipe element
locked in through the previous rungs exactly fixed, and extend training from 300,000 to 500,000 steps.
Evaluate on the same fixed protocol as every prior rung — SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2
accuracy — against rung 7's 300K-step numbers, and read the size of the 300K-to-500K gain, relative to
the size of the 100K-to-300K gain, as the signal for whether the duration lever is still producing
returns of roughly the same per-step magnitude or has started to taper.
