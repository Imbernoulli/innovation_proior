**Problem.** Adding tenfold more, more diverse data while holding steps fixed at 100K produced only
few-tenths improvements on every metric (SQuAD +0.4/+0.4, MNLI-m +0.3, SST-2 +0.3) — modest given the
scale of the data increase. That leans toward duration, not data volume, as the binding constraint: at a
fixed step count, a much larger corpus means each document is sampled less often, so the model may
simply not have had enough gradient steps to make full contact with the newly added 144GB, independent
of how much that text has to teach.

**Proposal.** Hold the 160GB combined corpus, BERT-large architecture, 8K batch size, and every locked
recipe element (dynamic masking, full-sentences/no-NSP, byte-level BPE) fixed exactly as rung 6 left
them, and extend pretraining from 100,000 to 300,000 steps — a threefold increase, still well short of
the original million-step BERT-large budget, but three times what this recipe has been evaluated at.

**Why 300K, not further yet.** An intermediate point, not a jump to something near the original budget,
for the same sequencing reason as every prior rung: a clean read on whether extending duration at fixed
data continues to help, plateaus, or reverses, before committing to an even longer and more expensive
run. Continued improvement with no sign of stalling would support pushing step count further; a plateau
or reversal would instead point at data volume, not duration, as the ceiling at 160GB.

**Overfitting risk, named explicitly.** More steps over a fixed corpus can eventually mean memorizing
rather than generalizing — pretraining loss falling while downstream performance stalls or reverses.
The data-scaling rung held steps fixed and varied data, so it gives no direct evidence either way about
what happens holding data fixed and varying steps. A larger, 160GB corpus should push whatever
overfitting threshold exists further out than at the original 16GB scale, but that's a qualitative
expectation, not something the prior rungs demonstrate.

**What the result would mean, without presuming it.** A jump of a point or more on several metrics —
closer in size to the rung-5 recipe-combination jump than to the modest data-only gain — would support
reading step count as the binding constraint at 100K. Another modest few-tenths move, similar in size to
the data-only result, would instead suggest the recipe is approaching a general plateau at this data
scale regardless of which lever is pulled.

**Configuration under test (rung 7, delta from rung 6):**
```
steps:          300,000 (up from 100,000)
[unchanged]:    combined recipe (dynamic masking, full-sentences/no-NSP, 8K batch,
                byte-level BPE), BERT-large architecture, 160GB combined data
```

**Evaluation.** SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy, against rung 6's numbers.
