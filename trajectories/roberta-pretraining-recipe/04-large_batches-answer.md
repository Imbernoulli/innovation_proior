**Problem.** The four-way format comparison showed sentence-pair+NSP as the weakest configuration on
every task despite keeping the NSP loss, while both NSP-free formats matched or beat segment-pair+NSP —
the load-bearing factor was span length, not the loss. Full-sentences (no NSP) is adopted going
forward, as committed before running that comparison, over the marginally-ahead doc-sentences, to keep
batch size a clean fixed knob for this rung. With format and masking settled, batch size is the next
lever: NMT has an established result that very large mini-batches, with a correspondingly scaled
learning rate, can improve both optimization speed *and* final task quality, not just training speed;
BERT specifically has separately been shown to tolerate batch sizes up to 32K without breaking.

**Proposal.** Sweep batch size across three points, holding total training compute
(sequences × steps) fixed via gradient accumulation, so batch size is the one isolated variable rather
than also changing total compute: 256 sequences × 1,000,000 steps (the current anchor) ≈ 2,048
sequences × ~125,000 steps ≈ 8,192 sequences × ~31,000 steps. Learning rate is retuned separately at
each point — the large-batch argument is contingent on scaling LR to match, not a claim that batch size
alone helps at a fixed LR.

**Why three points, not a single before/after.** The batch-size/quality relationship isn't assumed to
be monotonic a priori; it could plateau or have an interior optimum. 256 is the anchor; 2K is a
moderate jump; 8K pushes toward (but stays well inside) the up-to-32K range already shown workable for
BERT elsewhere, and is the batch size that would be standardized on for the data/step-scaling rungs
later if it holds up here, since fewer total optimizer steps per unit of data matters directly once
corpus size grows.

**Metrics.** Held-out perplexity on the MLM objective (a purer read on optimization quality, decoupled
from finetuning noise) alongside MNLI-m and SST-2 dev accuracy — the same downstream metrics used in
the format comparison, for direct comparability. Perplexity and downstream accuracy moving together
across the three settings would support reading any batch-size effect as a genuine optimization-quality
effect; divergence would be informative in its own right about how tightly pretraining quality and
downstream quality actually track each other.

**Configuration under test (rung 4, delta from rung 3):**
```
batch size:    sweep {256, 2048, 8192} sequences, learning rate tuned per setting
steps:         {1,000,000, ~125,000, ~31,000} respectively — compute-matched
               (same total sequences observed across all three)
[unchanged]:   dynamic masking, full-sentences (no NSP), BERT-base architecture,
               BookCorpus+Wikipedia 16GB
```

**Evaluation.** Held-out perplexity, MNLI-m dev accuracy, SST-2 dev accuracy, at each of the three
batch sizes.
