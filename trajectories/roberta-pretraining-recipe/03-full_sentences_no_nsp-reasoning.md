Dynamic masking is a wash-to-mild-win at the current training scale — SQuAD 2.0 F1 moved from 78.3 to
78.7, SST-2 from 92.5 to 92.9, MNLI-m dipped 0.3 points from 84.3 to 84.0 — small movements in both
directions, consistent with what I expected going in: no dramatic effect yet, but a free efficiency win
(no tenfold data duplication) and a mask-diversity ceiling removed before it can confound the
much-larger-scale experiments later in this study. I'm keeping it as the default for everything from
here on. That leaves the input format and the NSP loss as the next thing worth interrogating, and this
one is a genuinely contested question, not a mechanical tweak.

BERT's original format concatenates two segments — `[CLS] x_1...x_N [SEP] y_1...y_M [EOS]` — where each
segment can itself span multiple natural sentences, sampled either contiguously from the same document
(with probability 0.5) or from two different documents, and trained jointly with a Next Sentence
Prediction loss: a binary classifier on whether the two segments are truly contiguous. The original
authors reported this loss as important — removing it, in their own ablation, hurt performance, with
particularly significant degradation on QNLI, MNLI, and SQuAD 1.1. That's a strong claim, and if it's
right I should keep NSP. But it's exactly the kind of claim I flagged at the outset of this study as
suspect: several concurrent lines of work outside this reimplementation have separately questioned
whether NSP is actually necessary, which means either those other efforts are wrong, or something about
how NSP interacts with the *rest* of the original recipe — not the loss itself — is what's actually
doing the work the original authors attributed to it. I already have one candidate mechanism for a
confound: the original ablation, as far as I can tell from its description, appears to have removed the
NSP loss term while *keeping* the segment-pair input structure — two segments concatenated as one input,
still up to two-per-instance. If that's what happened, then "removing NSP hurts" could really mean
"removing the loss while leaving a structurally two-segment input, whose second half no longer has any
training signal telling it whether it belongs there, actively confuses the model" — which is a
statement about a specific, avoidable implementation choice, not about the NSP objective's inherent
value. I can't resolve this by argument; I have to test format and loss as genuinely separate axes.

So instead of a single before/after swap, I want to compare four points that disentangle input
structure from the NSP loss:

- **segment-pair+NSP** — the original: two segments, each possibly multi-sentence, combined length
  under 512, NSP loss retained. This is what rungs 1 and 2 have already been running; it's the anchor
  every alternative below is compared against.
- **sentence-pair+NSP** — same NSP loss, but each segment is now a single natural sentence rather than
  a multi-sentence block. Since single sentences are far shorter than 512 tokens, batch size has to
  increase to keep total tokens-per-batch roughly matched to segment-pair, so this comparison isn't
  contaminated by a token-throughput difference. This isolates *segment length* as a variable while
  holding the NSP loss fixed — if this configuration is clearly worse than segment-pair+NSP, that's
  evidence the model benefits from longer contiguous spans regardless of what the auxiliary loss is
  doing, i.e. that long-range dependency modeling matters and short segments starve it.
- **full-sentences** — no NSP loss at all. Each input is packed with full sentences sampled
  contiguously, up to 512 tokens total, and is allowed to cross document boundaries — when one document
  runs out, sampling continues into the next, with an extra separator token marking the seam.
- **doc-sentences** — structurally identical to full-sentences but *not* allowed to cross document
  boundaries; a sequence sampled near the end of a document may come in shorter than 512 tokens, so
  batch size is dynamically increased in those cases to keep total tokens-per-batch comparable to
  full-sentences. Both of these drop NSP.

Running all four together, rather than a single flip, is the only way to answer the actual question I
have: is it the *loss* or the *input structure* that matters? sentence-pair+NSP against segment-pair+NSP
isolates length while holding NSP fixed; full-sentences and doc-sentences against segment-pair+NSP
isolate the loss (and, incidentally, the cross-document-boundary question against each other). If NSP
were genuinely essential the way the original ablation suggested, I'd expect both full-sentences and
doc-sentences — which drop it — to underperform segment-pair+NSP by a comparable margin, and I'd expect
sentence-pair+NSP, which keeps NSP but shortens segments, to hold up reasonably well by comparison. If
instead the real driver is *span length*, not the loss, I'd expect the opposite signature: sentence-pair
should be the weak point despite retaining NSP, because chopping segments down to single sentences
starves the model of exactly the long-range contiguous text it needs to learn dependencies from, while
full-sentences and doc-sentences — which pack up to the full 512-token budget even without NSP — should
hold up fine or better.

I also want a second reference point in the same table, independent of my own reimplementation:
BERT-base's own published numbers, and a published external comparison at the same base scale — a
different pretraining objective (permutation-based, not masked-LM) evaluated on the same downstream
suite. Neither of those is a candidate I'm choosing between; they're context for how large a gap between
my four configurations would actually be interesting. If the spread among my four configurations is
small relative to the gap between BERT-base and that external model, the input-format question is a
second-order effect on top of a much larger objective-level gap I'm not trying to close at this rung.

One methodological decision I want to commit to now, independent of what the numbers turn out to show:
whichever of full-sentences or doc-sentences performs best or ties for best among the four, I intend to
carry *full-sentences* forward as the format for the rest of this study, not doc-sentences — even if
doc-sentences comes out slightly ahead. The reason is architectural rather than empirical: doc-sentences
produces variable batch sizes by construction, since a sequence near the end of a short document can
come in under the 512-token budget and needs a larger batch to compensate. That variability is a genuine
complication for the very next thing on this ladder — a controlled batch-size ablation — where I need
batch size to be a clean, fixed, comparable knob across settings, not a quantity that's already being
dynamically adjusted by the input-packing scheme for unrelated reasons. So the choice between these two
NSP-free formats will not be "pick whichever number is highest"; it will weigh a possibly-small accuracy
difference against a real methodological cost to every subsequent rung, and full-sentences is very
likely to be the pragmatic choice for that reason even if it isn't the single best number in the table.

So: run all four configurations — segment-pair+NSP (already established), sentence-pair+NSP,
full-sentences (no NSP), doc-sentences (no NSP) — at the same scale as before (BERT-base architecture,
dynamic masking, batch matched for total tokens, same 1M-step-equivalent training budget), evaluate on
the same fixed protocol (SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy, RACE accuracy), and let
that four-way comparison settle whether it's segment length or the NSP loss itself doing the work in the
original recipe.
