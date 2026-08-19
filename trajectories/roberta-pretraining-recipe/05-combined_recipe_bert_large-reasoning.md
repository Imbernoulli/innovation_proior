The batch-size sweep gives a clean three-point curve: 256 sequences at 3.99 held-out perplexity, 84.7
MNLI-m, 92.7 SST-2; 2,048 at 3.68 ppl, 85.2 MNLI-m, 92.9 SST-2 — the best number in all three columns;
8,192 at 3.77 ppl, 84.6 MNLI-m, 92.8 SST-2 — worse than 2,048 on every column but still ahead of 256 on
perplexity and MNLI-m, and within a tenth of 256 on SST-2. So the relationship isn't monotonic: there's
an interior optimum somewhere around 2K, not a straight line of "bigger is always better." That's worth
sitting with for a moment before deciding what to carry forward, because the numeric optimum and the
right practical choice for the rest of this study are not automatically the same thing.

If I were optimizing this single rung in isolation, 2K would be the number to pick — best perplexity,
best MNLI-m, best SST-2. But I'm not optimizing this rung in isolation; I'm choosing the batch size that
every remaining rung in this study — combining every improvement so far, then scaling data by 10x, then
extending training duration by 3-5x — is going to inherit. At that scale, wall-clock and hardware
considerations stop being a footnote and start being the actual constraint on what's runnable at all.
8K parallelizes across distributed workers more cleanly than 2K, independent of any accuracy question,
and the gap between 8K and 2K here — a few tenths of perplexity, half a point of MNLI-m, a tenth of
SST-2 — is small next to the gap either of them opens up over the 256 anchor. An 8K batch that trades a
small amount of measured quality at the current small-scale, short-duration setting for meaningfully
better throughput at the scale I'm about to run everything else at is the right trade to make once, now,
rather than revisiting the batch-size question again after every subsequent rung. So: 8K, not 2K, goes
into the combined recipe — a deliberate choice against this rung's own single best number, made for
reasons this rung's numbers can't by themselves capture.

That settles three of the four ingredients I want to combine: dynamic masking (locked in after rung 2),
full-sentences without NSP (locked in after rung 3), and now 8K batch size. The fourth is one I haven't
run a formal ablation on at all, and I want to be honest about why. The original vocabulary is a
character-level byte-pair encoding of 30K subword units, learned after heuristic tokenization of the
input. An alternative construction builds the BPE vocabulary directly over raw bytes rather than unicode
characters, which lets a comparatively modest vocabulary — 50K units — encode arbitrary input text
without ever needing an "unknown token" fallback, at the cost of roughly 15-20M additional embedding
parameters depending on model size. Some early, informal comparisons between the two encodings turned up
only slight differences, with the byte-level vocabulary coming in slightly worse on some end-task
numbers. That's a real cost, and if I only cared about squeezing out the last half-point of accuracy at
this exact scale, character-level BPE would be defensible. But I'm adopting byte-level anyway, and the
reasoning has nothing to do with the small accuracy gap: a universal encoding scheme that can represent
any input text without an unknown-token fallback is a property I want *especially* once the data-scaling
rungs ahead of this one bring in text from sources well outside BookCorpus and Wikipedia — news
crawls, web text scraped from arbitrary domains, story-style text. Those sources are exactly where a
30K character-level vocabulary learned on a comparatively narrow corpus is most likely to hit unknown-
token gaps, and I'd rather not discover a vocabulary-coverage problem partway through a much more
expensive data-scaling run. So byte-level BPE at 50K units goes into the combined recipe on robustness
grounds, accepting a small, already-observed accuracy cost as the price, rather than because I expect it
to help accuracy at this rung.

Putting all four together — dynamic masking, full-sentences without NSP, 8K batch, byte-level BPE — is
what I'll call the combined recipe from here on. Before scaling anything else, I want to test this
combination on its own, and I want to do it in a way that specifically disentangles "does the combined
*recipe* help" from "does more data and more compute help," because those are two separate questions
this study needs separate answers to. The cleanest way to do that is to hold data and total compute
roughly comparable to the original BERT-large training run and change only the recipe.

Two decisions follow from that goal. First, architecture: move from BERT-base to BERT-large (24 layers,
1024 hidden, 16 heads, roughly 355M parameters) for this rung and everything after it. Every prior rung
in this study used BERT-base specifically because it's cheaper to iterate on, but the recipe question I
actually care about — does the corrected pretraining procedure beat the *originally published* numbers —
is best answered at the scale those original numbers were reported at, which for the strongest published
BERT configuration is the large architecture. Second, and more subtly, step count: rather than running
the combined recipe at the same total-compute budget the batch-size sweep used (1M-sequence-equivalent,
matching the original), I want to first run it for only 100,000 steps — a fraction of that budget —
over the same Books+Wikipedia 16GB data BERT-large itself was trained on. This is deliberately an
under-resourced first data point relative to what a fully-scaled run would use, and the reason is
sequencing: I have two levers left to pull after this rung — more diverse data, and more training steps
— and if I pull them both at once alongside every recipe change already stacked in, and something goes
well or badly, I won't know which of the five things stacked together (recipe, architecture, data
volume, step count, all at once) is responsible. Running the combined recipe first at a *matched* data
budget and a deliberately modest step count gives me a clean read on what the recipe change itself is
worth, holding data and roughly the compute scale where I can still afford to iterate, before I start
adding the far more expensive dimensions of more data and much longer training on top of it.

What I'm watching for, without presuming the answer: how this combined-recipe, 100K-step, 16GB-data
BERT-large configuration compares to BERT-large's own originally published numbers under the same
downstream protocol (SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy) — trained under the original
recipe, at the same architecture, for a full 1,000,000 steps at batch 256. If the combined recipe, even
at a tenth of the step budget the original used and no additional data, can close most or all of the gap
to (or exceed) that fully-trained original, that would be a strong signal that the recipe changes
compound in the direction each individual ablation suggested, and that the 900,000 steps and 15GB of
additional-source text the original recipe implicitly "spent" were not doing efficient work relative to
what a corrected procedure can extract from the same amount of raw text. If it falls meaningfully short
despite the recipe corrections, that would say the earlier per-change ablations don't compound as
cleanly as hoped, or that step count and data volume were doing more of BERT-large's original heavy
lifting than the individual ablations gave them credit for — either way, a result I want in hand before
committing the much larger compute of the data- and step-scaling rungs still ahead.

So: combine dynamic masking, full-sentences without NSP, 8K batch (learning rate retuned accordingly),
and byte-level 50K BPE into a single recipe; scale up to the BERT-large architecture; train on the
original 16GB BookCorpus+Wikipedia data for 100,000 steps; evaluate on SQuAD 1.1/2.0 dev F1, MNLI-m
accuracy, and SST-2 accuracy, against BERT-large's own published numbers under the original recipe.
