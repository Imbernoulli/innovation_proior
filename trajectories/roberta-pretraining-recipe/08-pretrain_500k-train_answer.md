Extending from 100,000 to 300,000 steps at the fixed 160GB corpus moved every metric — SQuAD from
94.0/87.7 to 94.4/88.7, MNLI-m from 89.3 to 90.0, SST-2 from 95.6 to 96.1. The SQuAD 2.0 gain (+1.0) and
the MNLI-m gain (+0.7) are both noticeably larger than what the data-only rung produced at fixed steps
(+0.4 SQuAD2.0, +0.3 MNLI-m) — a threefold increase in step count did more work than a tenfold increase
in data, on the two hardest metrics in the suite. That directionally answers the question I posed going
into that rung: step count, not data volume, was the tighter constraint at 100K. It also directly
addresses the overfitting risk I flagged before running it. A model beginning to overfit the 160GB
corpus at 300K steps would show downstream numbers stalling or reversing, particularly on the harder,
more overfitting-sensitive metrics — instead every metric moved further in the same direction, by a
larger margin than the previous lever produced. That's not proof overfitting is impossible at some
larger step count still ahead, but it's direct evidence against the specific worry that 300K steps over
160GB had already crossed into memorization territory.

So both questions that rung was designed to settle are answered, and cleanly: duration is the binding
lever relative to data volume at this corpus size, and there's no sign yet of the point where more steps
stops helping or starts hurting. Given that, the natural move isn't to switch back to data, or to stop
here and call 300K the final answer — it's to keep pulling the lever that's still visibly working. The
open question is how far.

I want to reason carefully about where to stop rather than just pick a round number. The original
BERT-large recipe used one million steps, but I already know a properly corrected recipe at 100K steps
over 16GB beat that fully-trained million-step original outright — so "match the original's step count"
was never really the target; the recipe corrections and the larger corpus have already made the
original's step budget somewhat beside the point as a benchmark. The narrower, more honest question is:
given the trajectory just observed — continued improvement from 100K to 300K, with no overfitting
signal — does that trajectory keep going if I extend further, or was 300K already close to wherever the
curve bends?

I don't have a way to answer that except by extending again and checking — the same posture every rung
in this study has taken: propose the next increment based on the trend the last rung established, then
let the next measurement confirm or reveal where it stops. Doubling straight from 300K to roughly 600K
would bet a large, expensive step on a hypothesis I haven't tested even once — I've only established
that the curve was rising between two points, not how far past the second point it keeps rising. A more
conservative extension, to roughly 500K steps — about 1.7x the 300K figure, still barely half the
original recipe's million-step budget — lets me test whether the 100K-to-300K slope is a stable,
continuing rate or was already flattening by 300K, without yet having extended far enough to see the
bend. If the 300K-to-500K gain looks similar in size, normalized per step, to the 100K-to-300K gain,
that says the curve genuinely hasn't found its ceiling and a still-longer run would likely help further.
If it's noticeably smaller, even while still an improvement, that's the first sign of diminishing
returns — informative on its own, without needing a full million-step run to see it.

There's also a reason to stop escalating duration after this point regardless of which way the
comparison lands. Every rung in this study, taken together, has already built a clear, compounding
picture: the recipe corrections — masking, input format, batch size, vocabulary — are real and additive;
the combined recipe at modest data and duration already beats the original fully-trained model; more
data helps modestly; more duration helps more, and hasn't shown a ceiling yet. Past some point,
continuing to escalate step count with diminishing marginal information about *why* it helps, as opposed
to just *that* it helps, stops answering the design-choice questions this study set out to ask and starts
becoming an open-ended compute-scaling exercise — a different kind of question than the one this ladder
has been asking rung to rung. 500K — three steps past where this recipe started, still short of the
original's million — gives one more clean data point on the duration curve while keeping this a study of
design choices rather than a race to spend more compute.

Concretely: hold the 160GB combined corpus, BERT-large architecture, 8K batch, and every recipe element
locked in through the previous rungs exactly fixed, and extend training from 300,000 to 500,000 steps.

```
steps:          500,000 (up from 300,000)
[unchanged]:    combined recipe (dynamic masking, full-sentences/no-NSP, 8K batch,
                byte-level BPE), BERT-large architecture, 160GB combined data
```

I'll evaluate the same fixed protocol as every prior rung — SQuAD 1.1/2.0 dev F1, MNLI-m accuracy,
SST-2 accuracy — against rung 7's 300K-step numbers, and read the size of the 300K-to-500K gain,
relative to the size of the 100K-to-300K gain, as the signal for whether the duration lever is still
producing returns of roughly the same per-step magnitude or has started to taper.
