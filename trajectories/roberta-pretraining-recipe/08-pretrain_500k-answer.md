**Problem.** Extending 100K to 300K steps at fixed 160GB data moved every metric, and the SQuAD 2.0
(+1.0) and MNLI-m (+0.7) gains were larger than what the tenfold data increase produced at fixed steps
(+0.4, +0.3) — step count, not data volume, was the tighter constraint at 100K. This also argues against
the overfitting risk flagged before that rung ran: a model beginning to overfit 160GB at 300K steps
would show downstream numbers stalling or reversing, and instead every metric moved further in the same
direction, by a larger margin than the previous lever produced.

**Proposal.** Keep pulling the lever that's still visibly working, but conservatively rather than by
jumping straight toward the original recipe's million-step budget. Extend from 300,000 to 500,000 steps
— roughly 1.7x the 300K figure, still about half of the original — holding the 160GB combined corpus,
BERT-large architecture, 8K batch, and every locked-in recipe element exactly fixed.

**Why 500K and not a bigger jump.** The trend from 100K to 300K establishes that the curve was still
rising between those two points, not how far past 300K it keeps rising. Doubling straight to ~600K would
bet a large, expensive step on a hypothesis (the trend continues at the same rate) that hasn't been
tested even once yet. A more conservative extension to 500K tests specifically whether the 100K-to-300K
slope continues at a similar per-step rate or was already flattening by 300K — if the 300K-to-500K gain
looks similar in size (normalized per step) to the 100K-to-300K gain, the curve hasn't found a ceiling
yet; if it's noticeably smaller, that's the first sign of diminishing returns, informative without
needing a full million-step run to see it.

**Why stop escalating duration after this point regardless of outcome.** This study's eight rungs have
already established a compounding, additive picture across masking, input format, batch size,
vocabulary, data volume, and duration. Continuing to escalate step count indefinitely stops answering
design-choice questions and starts becoming an open-ended compute-scaling exercise, a different kind of
question than the one this ladder has been asking rung to rung. 500K — three steps past this recipe's
starting point of 100K, still short of the original's 1M — is treated as the last rung on the duration
axis regardless of which way the 300K-to-500K comparison lands.

**Configuration under test (rung 8, delta from rung 7):**
```
steps:          500,000 (up from 300,000)
[unchanged]:    combined recipe (dynamic masking, full-sentences/no-NSP, 8K batch,
                byte-level BPE), BERT-large architecture, 160GB combined data
```

**Evaluation.** SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy, against rung 7's 300K-step
numbers; the size of the 300K-to-500K gain, relative to the 100K-to-300K gain, is read as the signal for
whether the duration lever is still producing returns of similar per-step magnitude or has started to
taper.
