Dynamic masking landed where I expected: a wash-to-mild-win at this scale, SQuAD 2.0 F1 up 0.4 points
to 78.7, SST-2 up 0.4 to 92.9, MNLI-m down 0.3 to 84.0 — small movements in both directions, no dramatic
effect yet, but a free efficiency win and a mask-diversity ceiling removed before it can confound the
much larger scaling experiments this study is heading toward. I'm keeping it as the default going
forward. That leaves input format and the NSP loss, and this one is a genuinely contested question
rather than a mechanical tweak, so I want to be careful about how I set it up.

The original format concatenates two segments — each possibly spanning several sentences, delimited by
`[CLS]`/`[SEP]`/`[EOS]`, combined length under 512 tokens — trained jointly with a Next Sentence
Prediction loss, a binary classifier on whether the two segments are really contiguous. The people who
built this recipe reported NSP as important: removing it, in their own ablation, hurt performance, with
notable degradation on QNLI, MNLI, and SQuAD 1.1. If that's right, I should keep it. But it's exactly
the kind of claim I flagged as suspect going into this study — several concurrent lines of work outside
this reimplementation have separately questioned whether NSP is necessary at all, and either they're all
wrong or something *else* in the original recipe is doing the work NSP got credited for. I have a
concrete candidate confound: as far as I can tell, the original ablation dropped the NSP loss term while
keeping the two-segment input structure intact. If that's what happened, "removing NSP hurts" might
really mean "removing the loss while leaving a structurally two-segment input, whose second half now has
no training signal explaining why it's there, actively confuses the model" — a statement about one
specific, avoidable implementation choice, not about NSP's inherent value. I can't settle this by
argument. Input structure and the loss have to be tested as genuinely separate axes, not as one bundled
flip.

So instead of a single before/after swap, I want four points that disentangle segment length from the
NSP loss. `segment-pair+NSP` is the original — the anchor, already running as rungs 1 and 2.
`sentence-pair+NSP` keeps the NSP loss but shrinks each segment to a single natural sentence; since
that's far shorter than 512 tokens, batch size goes up to keep total tokens-per-batch matched to
segment-pair, so the comparison isn't contaminated by a throughput difference — this isolates segment
length while holding the loss fixed. If sentence-pair+NSP is clearly worse than segment-pair+NSP despite
keeping the loss, that's evidence the model needs long contiguous spans for its own sake, not evidence
about NSP. `full-sentences` drops NSP entirely: sentences packed contiguously up to 512 tokens, allowed
to cross document boundaries with an extra separator token at the seam. `doc-sentences` is structurally
the same but confined to a single document — a sequence sampled near a document's end can land short of
512 tokens, so batch size is dynamically bumped in those cases to keep total tokens comparable to
full-sentences. Both drop NSP; the difference between them is only whether packing may cross a document
boundary.

Running all four together, rather than one flip, is the only way to answer the question I actually have.
`sentence-pair+NSP` against `segment-pair+NSP` isolates length while holding NSP fixed.
`full-sentences`/`doc-sentences` against `segment-pair+NSP` isolate the loss. If NSP were genuinely
essential the way the original ablation suggests, both NSP-free formats should underperform
segment-pair+NSP by a comparable margin, and sentence-pair+NSP — which keeps NSP even though it's
shorter — should hold up reasonably well by comparison. If instead span length is the real driver, I'd
expect the opposite: sentence-pair should be the weak point despite keeping NSP, because chopping
segments down to single sentences starves the model of the long-range contiguous text it needs, while
the two NSP-free formats — which still pack up to the full 512-token budget — should hold up fine or
better.

I also want two reference points in the same comparison that I'm not choosing between: BERT-base's
published numbers, and a published base-scale model trained under a different, non-masked-LM
pretraining objective. Neither is a candidate; they're there to calibrate how large a spread among my
four format configurations would actually be interesting relative to an objective-level gap I'm not
trying to close at this rung.

And I want to commit to one methodological decision now, independent of what the numbers show:
whichever of `full-sentences` or `doc-sentences` scores best, I'm carrying `full-sentences` forward for
the rest of this study, not `doc-sentences`, even if doc-sentences comes out slightly ahead. The reason
is architectural, not empirical: doc-sentences produces variable batch sizes by construction, and the
very next thing on this ladder is a controlled batch-size ablation, where I need batch size to be a
clean, fixed, comparable knob — not a quantity the input-packing scheme is already adjusting for
unrelated reasons. So this isn't "pick whichever number is highest"; it's weighing a possibly small
accuracy difference against a real methodological cost to everything after this rung, and
full-sentences is very likely the pragmatic choice on that basis alone.

```
input format:  4-way comparison — segment-pair+NSP / sentence-pair+NSP /
               full-sentences(no NSP) / doc-sentences(no NSP)
               batch size matched for total tokens per format
[unchanged]:   dynamic masking, BERT-base architecture, ~1M-step-equivalent
               training budget, BookCorpus+Wikipedia 16GB
[commit]:      full-sentences carried forward as the format for subsequent rungs
               regardless of the doc-sentences comparison, for batch-size-ablation
               cleanliness
```

Evaluation is the same fixed protocol as before, run across all four configurations: SQuAD 1.1/2.0 dev
F1, MNLI-m accuracy, SST-2 accuracy, RACE accuracy.
