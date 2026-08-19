The batch-size sweep gives a clean three-point curve, and it's not the curve I might have guessed
going in. 256 sequences: 3.99 held-out perplexity, 84.7 MNLI-m, 92.7 SST-2. 2,048: 3.68 ppl, 85.2
MNLI-m, 92.9 SST-2 — the best number in every column. 8,192: 3.77 ppl, 84.6 MNLI-m, 92.8 SST-2 — worse
than 2,048 across the board, but still ahead of 256 on perplexity and MNLI-m, and within a tenth of it
on SST-2. So the relationship has an interior optimum somewhere around 2K, not a straight "bigger is
better" line. That's worth sitting with before I decide what to lock in, because the number that wins
this one rung and the number that's right for the rest of this study aren't automatically the same
choice.

If I were only optimizing this rung, 2K wins outright. But I'm choosing the batch size every remaining
rung inherits — combining every improvement so far, then scaling data roughly tenfold, then extending
training three to five times longer. At that scale, wall-clock and hardware throughput stop being a
footnote and become the actual constraint on what's runnable. 8K parallelizes more cleanly across
distributed workers than 2K, independent of accuracy, and the gap between them here — a few tenths of
perplexity, half a point of MNLI-m, a tenth of SST-2 — is small next to the gap either of them opens up
over the 256 anchor. Trading a small amount of measured quality at this small-scale, short-duration
setting for meaningfully better throughput at the scale I'm about to run everything else at is the right
call to make once, now, rather than revisiting batch size after every rung that follows. So 8K, not 2K,
goes into the combined recipe — a deliberate choice against this rung's own best number, for reasons
this rung's numbers alone can't capture.

That settles three of the four ingredients: dynamic masking, locked in after rung 2; full-sentences
without NSP, locked in after rung 3; and now 8K batch size. The fourth I haven't run a formal ablation on
at all, and I want to be honest about that. The original vocabulary is a character-level byte-pair
encoding of 30K subword units. An alternative builds the BPE vocabulary directly over raw bytes instead
of unicode characters, which lets a comparatively modest 50K-unit vocabulary encode arbitrary input text
with no unknown-token fallback ever needed, at the cost of roughly 15-20M extra embedding parameters.
Early, informal comparisons between the two turned up only slight differences, with the byte-level
vocabulary slightly worse on some end-task numbers. That's a real, already-observed cost, and if
squeezing the last half-point of accuracy at this exact scale were the only goal, character-level BPE
would be defensible. I'm adopting byte-level anyway, for a reason that has nothing to do with that small
accuracy gap: a universal encoding that never needs an unknown-token fallback is a property I want
specifically once the data-scaling rungs ahead bring in text from sources well outside BookCorpus and
Wikipedia — news crawls, arbitrary web text, story-style crawls — exactly where a 30K vocabulary
learned on a narrower corpus is most likely to hit coverage gaps. Better to accept the small,
already-measured cost now than discover a vocabulary problem partway through a much more expensive
data-scaling run.

Putting all four together — dynamic masking, full-sentences without NSP, 8K batch, byte-level BPE — is
what I'll call the combined recipe from here on. Before scaling anything else, I want to test this
combination on its own, in a way that specifically separates "does the recipe help" from "does more data
and more compute help," because this study owes separate answers to those two questions. The cleanest
way is to hold data and roughly the compute scale constant relative to the original and change only the
recipe.

Two decisions follow. First, architecture: move from BERT-base, which every prior rung used because
it's cheap to iterate on, to BERT-large (24 layers, 1024 hidden, 16 heads, roughly 355M parameters) —
because the question I actually care about, whether the corrected recipe beats the originally published
numbers, is best answered at the scale those numbers were reported at, and the strongest published BERT
configuration is the large architecture. Second, and more subtly, step count: rather than running the
combined recipe at the full compute budget the batch-size sweep used, I want to first run it for only
100,000 steps over the same 16GB Books+Wikipedia data BERT-large itself trained on — deliberately a
fraction of what a fully-scaled run would use. The reason is sequencing. Two levers are still left after
this rung — more diverse data, and more training steps — and if I pull both of those at once alongside
every recipe change already stacked in, and the result moves, I won't be able to tell which of five
simultaneous changes (recipe, architecture, data volume, step count, all together) is responsible. A
matched-data, modest-step first data point gives me a clean read on what the recipe change alone is
worth, at a scale I can still afford to iterate on, before I commit the much larger compute that more
data and much longer training require.

What I'm watching for, without presuming which way it goes: how this combined-recipe, 100K-step,
16GB-data BERT-large configuration compares to BERT-large's own originally published numbers — the
original recipe, the same architecture, a full 1,000,000 steps at batch 256. Closing most or all of that
gap on a tenth of the step budget and no extra data would say the individual corrections compound in the
direction each ablation separately suggested, and that the 900,000 extra steps and any implicit data
advantages of the original recipe weren't doing efficient work relative to what a corrected procedure
can extract from the same raw text. Falling meaningfully short despite the corrections would say the
per-change ablations don't compound as cleanly as hoped, or that step count and data volume were doing
more of BERT-large's original heavy lifting than credited — either way, something I want to know before
committing the much larger compute the rungs after this one require.

```
architecture:   BERT-large (L=24, H=1024, A=16, ~355M params)
masking:        dynamic
input format:   full-sentences, no NSP
batch size:     8K sequences, learning rate retuned (peak LR 4e-4, warmup 30K steps)
vocabulary:     byte-level BPE, 50K subword units
steps:          100,000
data:           BookCorpus + English Wikipedia, 16GB (same as original BERT-large)
```

I'll evaluate SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, and SST-2 accuracy against BERT-large's own
published numbers under the original recipe.
