Adding the three extra corpora, holding steps fixed at 100,000, moved every downstream number in the
right direction — SQuAD from 93.6/87.3 to 94.0/87.7, MNLI-m from 89.0 to 89.3, SST-2 from 95.3 to 95.6
— but every one of those movements is a few tenths of a point, not the kind of jump I'd have hoped for
from tenfold more, substantially more diverse pretraining text. That's an honest and informative result
in its own right: it says the recipe corrections established through the earlier rungs were already
capturing most of the available headroom at a fixed 100K-step budget, and data volume alone, at this
step count, is a real but modest lever rather than the dominant one. It answers the question the
previous rung was built to isolate — data ceiling or duration ceiling — with a clear lean toward
duration. If tenfold more data barely moves the needle while everything else stays fixed at 100,000
steps, then either the model is already close to saturating what 100,000 steps of optimization can
extract regardless of corpus size, or it simply hasn't had enough gradient steps to make full use of the
much larger corpus now available to it — and both readings point at step count, not data volume, as
where the next real gain is most likely sitting.

There's a structural reason to expect step count specifically matters here, beyond the general "train
longer" intuition. Multiplying data tenfold while holding steps fixed at 100,000 also implicitly changes
how much of that expanded corpus the model actually gets to see: at a fixed batch size and step count, a
larger total corpus means each individual document is sampled less often over training, not more. So the
modest data-alone gain I just measured could partly reflect that the model hasn't had the gradient steps
to make full contact with the newly added 144GB, rather than reflecting that the newly added text has
little more to teach beyond what the original 16GB already provided. Extending step count at the now
160GB scale is the direct way to test that distinction: if the model was starved for steps relative to
the size of its training set, giving it more steps over the same larger corpus should produce a
materially bigger jump than the few-tenths data-alone produced, because more steps means more effective
coverage of data that's already sitting there — not just more raw compute spent.

I also want to weigh a real risk pulling in the other direction: overfitting. A model trained for
substantially more steps over a fixed corpus can, past some point, start memorizing rather than
generalizing, which would show up as pretraining loss continuing to fall while downstream performance
stalls or reverses. I can't rule this out from the data-scaling rung — that rung held steps fixed and
varied data, so it tells me nothing directly about what happens when I hold data fixed and vary steps.
The corpus is now large, 160GB, several times the size that originally trained BERT-base or BERT-large,
which should push whatever overfitting threshold exists further out than at the original 16GB scale —
more text generally buys more resistance to memorization at a fixed step count — but "should push it
further out" is a qualitative expectation, not a guarantee, and this rung is exactly where I'd expect
the first sign of it if the recipe's efficiency starts running into diminishing or reversing returns.

The concrete move: hold the 160GB combined corpus, BERT-large architecture, 8K batch size, and every
locked-in recipe element — dynamic masking, full-sentences without NSP, byte-level BPE — exactly as the
previous rung left them, and extend training from 100,000 to 300,000 steps. Threefold, still well short
of the million steps the original BERT-large recipe used, but three times what this recipe has been
evaluated at so far. I'm choosing 300K as an intermediate point rather than jumping straight toward
something close to the original million-step budget, for the same sequencing reason that's guided every
rung in this study: a clean read on whether extending duration at fixed data continues to help,
plateaus, or starts to hurt, before committing to an even longer and more expensive run. Continued
improvement with no sign of stalling is the signal to keep pushing step count further; a plateau or
reversal would instead say the recipe has found something close to its ceiling for this corpus size, and
further gains would need to come from data rather than duration after all.

What I'm watching for, as a real open question rather than a foregone conclusion: does SQuAD, MNLI-m,
and SST-2 continue moving upward from 94.0/87.7, 89.3, 95.6 by an amount that looks more like the
recipe-combination jump — a full point or more on several metrics — than the modest few-tenths the
data-only rung produced, which would support step count as the binding constraint at 100K? Or does it
move by another modest few-tenths, similar in size to the data-only result, which would instead suggest
the recipe is approaching a general plateau at this scale regardless of which lever is pulled? I don't
want to presume which of those the numbers will show before running the extended training.

```
steps:          300,000 (up from 100,000)
[unchanged]:    combined recipe (dynamic masking, full-sentences/no-NSP, 8K batch,
                byte-level BPE), BERT-large architecture, 160GB combined data
```

I'll evaluate the same fixed protocol as before — SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy
— against rung 6's numbers.
