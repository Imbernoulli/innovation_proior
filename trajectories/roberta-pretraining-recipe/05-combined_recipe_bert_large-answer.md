**Problem.** The batch-size sweep found an interior optimum at 2K (best perplexity 3.68, best MNLI-m
85.2, best SST-2 92.9), not a monotonic "bigger is better" curve — 8K trails 2K on every column but
still beats the 256 anchor on perplexity and MNLI-m. Optimizing this one rung in isolation would pick
2K; but every remaining rung inherits whichever batch size is chosen here, and at the much larger scale
those rungs will run at, distributed-parallelization throughput becomes the binding constraint, not a
few tenths of dev-set accuracy. 8K is adopted deliberately against this rung's own single best number.

**Proposal — the combined recipe.** Stack four ingredients: dynamic masking (locked in at rung 2),
full-sentences without NSP (locked in at rung 3), 8K batch size (learning rate retuned accordingly), and
byte-level BPE at 50K subword units in place of the original character-level 30K vocabulary. The BPE
switch has no dedicated ablation in this study — early informal comparisons showed only a slight
accuracy cost on some tasks — and is adopted for a robustness reason rather than an accuracy one: a
byte-level vocabulary can encode any input text without an unknown-token fallback, a property that
matters specifically once the data-scaling rungs ahead bring in text from sources (news crawls, web
text, story-style crawls) well outside the narrow BookCorpus+Wikipedia corpus the original 30K
vocabulary was learned on.

**Scale-up decisions.** Move from BERT-base to BERT-large (L=24, H=1024, A=16, ~355M params) for this
rung and everything after — the question of whether the corrected recipe beats the originally published
numbers is best answered at the scale those numbers were reported at. Train for only 100,000 steps
(a fraction of the original 1M-step-equivalent budget) over the same 16GB BookCorpus+Wikipedia data
BERT-large itself used — deliberately matched on data and held to a modest step count, before the more
expensive data-scaling and step-scaling rungs ahead. This isolates what the recipe change alone is
worth, rather than conflating it with the also-planned increases in data volume and training duration.

**What this rung will tell me.** How a combined-recipe, 100K-step, 16GB-data BERT-large configuration
compares to BERT-large's own originally published numbers (same architecture, full 1M-step original
recipe). Closing most or all of that gap on a tenth of the step budget and no extra data would say the
individual recipe corrections compound; falling meaningfully short would say step count and data volume
were doing more of the original's work than the per-change ablations credited them for.

**Configuration under test (rung 5, delta from rung 4):**
```
architecture:   BERT-large (L=24, H=1024, A=16, ~355M params)
masking:        dynamic
input format:   full-sentences, no NSP
batch size:     8K sequences, learning rate retuned (peak LR 4e-4, warmup 30K steps)
vocabulary:     byte-level BPE, 50K subword units
steps:          100,000
data:           BookCorpus + English Wikipedia, 16GB (same as original BERT-large)
```

**Evaluation.** SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy, against BERT-large's published
numbers under the original recipe.
