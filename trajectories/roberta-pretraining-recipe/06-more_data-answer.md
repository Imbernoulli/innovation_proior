**Problem.** The combined recipe already beat BERT-large's published numbers at a tenth of the original
step count and no extra data — the recipe corrections compound, and the original training wasn't
extracting anywhere close to what this recipe pulls from the same 16GB corpus. That reframes what's left
to test: is the remaining headroom a data ceiling, a duration ceiling, or both? These need to be tested
as separate levers, for the same reason batch size was isolated from total compute earlier — pulling
both at once means not knowing which one moved the numbers, at a much higher compute cost than either
change alone.

**Proposal.** Hold everything from rung 5 fixed — combined recipe, BERT-large architecture, 8K batch,
100,000 steps — and change only the pretraining data: add three additional English-language corpora to
the original 16GB BookCorpus+Wikipedia set. A broad news-crawl corpus (tens of GB after filtering, a
journalistic/current-events register distinct from Wikipedia or BookCorpus). An open recreation of a
large, community-curated web-text corpus (a general-interest web register). A smaller corpus filtered to
match the narrative style of common-sense reasoning benchmarks, included for genre diversity rather than
raw size. Together with the original 16GB this totals a substantially larger, more varied corpus,
gathered independently since not every comparable dataset used elsewhere is publicly obtainable.

**Known limitation, stated up front.** This rung conflates data *size* and data *diversity* — the added
corpora are both larger and drawn from different domains than the original two sources, and there's no
practical way to vary one while holding the other fixed without discarding usable text or sourcing
same-domain data at prohibitive scale. Any downstream shift this rung shows will be attributable to
"more and more varied data together," not cleanly to one or the other.

**Prediction, without presuming magnitude.** Given the general external finding that more pretraining
data improves downstream performance, and given rung 5's efficiency at 16GB, performance should improve
further on SQuAD/MNLI-m/SST-2 relative to rung 5, holding steps fixed at 100K. Whether the improvement
is small (recipe corrections already captured most of the headroom) or large (BERT-large was
data-starved as well as duration-starved) is what this rung is designed to distinguish, and shapes how
much is left for the step-count rung after it.

**Configuration under test (rung 6, delta from rung 5):**
```
data:          BookCorpus + English Wikipedia (16GB) + news-crawl corpus (~76GB
               after filtering) + open web-text recreation (~38GB) +
               narrative-style crawl (~31GB) -- ~160GB total
[unchanged]:   combined recipe (dynamic masking, full-sentences/no-NSP, 8K batch,
               byte-level BPE), BERT-large architecture, 100,000 steps
```

**Evaluation.** SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy, against rung 5's own numbers.
