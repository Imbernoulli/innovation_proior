**Problem.** The original recipe's NSP loss is reported as important by its own authors — removing it
was said to hurt QNLI, MNLI, and SQuAD 1.1 — but that ablation, as far as can be told, dropped the loss
while *keeping* the two-segment input structure, which confounds "the loss doesn't help" with "an
unsupervised second segment actively confuses the model." Separately, several concurrent efforts have
questioned NSP's necessity. Input structure (segment length) and the NSP loss need to be disentangled
as two separate axes, not tested as one bundled change.

**Proposal.** Compare four input/objective configurations at BERT-base scale, dynamic masking already
adopted, batch sizes matched for total tokens across configurations:
- `segment-pair+NSP` (the anchor — already running as rungs 1-2's format)
- `sentence-pair+NSP` — same NSP loss, but segments are single natural sentences (batch size raised to
  match token throughput) — isolates segment length while holding the loss fixed
- `full-sentences` (no NSP) — sentences packed contiguously up to 512 tokens, may cross document
  boundaries (extra separator token at the seam)
- `doc-sentences` (no NSP) — same packing, but confined to a single document; batch size dynamically
  raised when a sequence lands short of 512 tokens near a document's end

If NSP is genuinely load-bearing, `sentence-pair+NSP` should hold up despite short segments while both
NSP-free formats should lag `segment-pair+NSP`. If span length is the real driver, `sentence-pair+NSP`
should be the weak point despite keeping NSP, while the full-length NSP-free formats should hold up or
improve.

**Reference points (not candidates).** BERT-base published numbers, and a published base-scale model
under a different (non-masked-LM) pretraining objective, included only to calibrate how large a gap
between the four format candidates would actually be interesting relative to an objective-level gap.

**Format decision, independent of outcome.** Whichever of `full-sentences`/`doc-sentences` scores
higher, `full-sentences` is adopted going forward: `doc-sentences` produces variable batch sizes by
construction (short documents force a batch-size bump), which would contaminate the controlled
batch-size ablation planned for the very next rung. This is a methodological choice, not a claim about
which format wins.

**Configuration under test (rung 3, delta from rung 2):**
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

**Evaluation.** SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy, RACE accuracy, across all four
configurations.
