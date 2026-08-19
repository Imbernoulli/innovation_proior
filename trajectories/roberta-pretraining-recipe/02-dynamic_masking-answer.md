**Problem.** BERT's original masking is *static*: the 15%-selection / 80-10-10 corruption is computed
once during preprocessing, and the training corpus is duplicated tenfold so that across 40 epochs each
sequence is masked in ten different ways (any one pattern recurring four times). That caps mask
diversity per sequence at ten patterns for reasons of preprocessing convenience, not anything
principled — and repeatedly training against an already-seen corruption pattern should teach
diminishing returns relative to a genuinely novel one, a ceiling that matters most exactly when
training for more epochs, more steps, or over more data (all axes this study intends to explore later).

**Proposal.** Dynamic masking: generate the mask fresh every time a sequence is fed to the model
(at data-loading time), instead of once at preprocessing. The masking algorithm itself is unchanged —
still 15% of tokens selected, still 80% `[MASK]` / 10% unchanged / 10% random-token — only *when* the
selection happens changes. This also removes the need for tenfold data duplication, an efficiency win
independent of any accuracy effect.

**Why now, this cheaply.** This is the smallest possible change on the ladder — it touches neither the
objective, the architecture, nor the data, only the masking schedule — so it's the right first ablation
to run before touching anything more expensive. If it's comparable or better, it becomes the default
for every subsequent rung and removes a potential confound (a hard mask-diversity ceiling) from later,
much larger scaling experiments.

**Expectation, stated honestly.** Under the current 1M-step/16GB regime, each sequence is seen roughly
40 times either way, which isn't yet deep into where a ten-mask ceiling should bind hard — so this rung
is expected to land close to static masking, not necessarily a large jump. "Comparable, with a
meaningful preprocessing/storage efficiency win" is already sufficient to adopt it going forward; a
clear win would be a bonus, and a clear loss would be the surprising result worth investigating (would
suggest the training dynamics benefit from repeated exposure to a fixed corruption, not just from mask
freshness).

**Configuration under test (rung 2, delta from rung 1):**
```
masking:      dynamic (regenerated per exposure at data-loading time; 15% select,
              80% [MASK] / 10% unchanged / 10% random, same split as static)
              — no tenfold data duplication needed
[unchanged]:  segment-pair + NSP input format, batch 256, 1M steps,
              BookCorpus+Wikipedia 16GB, BERT-base architecture
```

**Evaluation.** Same fixed protocol as rung 1 — SQuAD 2.0 F1, MNLI-m accuracy, SST-2 accuracy —
compared directly against this reimplementation's own static-masking numbers.
