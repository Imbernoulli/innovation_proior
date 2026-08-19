Adding the three extra corpora, holding steps fixed at 100,000, moved every downstream number in the
right direction — SQuAD from 93.6/87.3 to 94.0/87.7, MNLI-m from 89.0 to 89.3, SST-2 from 95.3 to 95.6
— but every one of those movements is a few tenths of a point, not the kind of jump I might have hoped
for from tenfold more and substantially more diverse pretraining text. That's an important, honest
result in its own right: it says the recipe corrections established through rungs 2 through 5 were
already capturing most of the available headroom at the fixed 100K-step budget, and data volume alone,
at this step count, is a real but modest additional lever rather than the dominant one. That answers the
question this rung's predecessor was built to isolate — data ceiling versus duration ceiling — with a
clear lean toward duration. If a tenfold increase in data barely moves the needle while everything else
is held fixed at 100,000 steps, then either the model is already close to saturating what 100,000 steps
of optimization can extract regardless of how much text is available to draw from, or the model simply
hasn't had enough gradient steps to make full use of the much larger corpus now sitting in front of it —
both readings point toward step count, not data volume, as where the next real gain is most likely
sitting.

There's a structural reason to expect step count specifically matters here, distinct from the general
"train longer" intuition. Multiplying data tenfold while holding steps fixed at 100,000 also changes,
implicitly, how much of that expanded corpus the model actually gets to see: at a fixed batch size and
step count, more total pretraining text means each individual document or passage is sampled less
often over the course of training, not more. So the modest data-alone gain measured in the last rung
could partly reflect that the model simply hasn't had the gradient steps needed to make full contact
with the newly added 144GB of text, rather than reflecting that the newly added text has little more to
teach beyond what the original 16GB already provided. Extending step count at the now-160GB data scale
is the direct way to test that: if the model was starved for steps relative to the size of its training
set, giving it more steps over the same larger corpus should produce a materially bigger jump than the
few-tenths gain data alone produced, precisely because more steps means more effective coverage of the
data that's already sitting there, not just more raw compute spent.

I also want to weigh this against a real risk in the other direction: overfitting. A model trained for
substantially more steps over a fixed corpus can, past some point, start memorizing rather than
generalizing, which would show up as pretraining loss continuing to fall while downstream task
performance stalls or reverses. This is a genuine possibility I have to keep in mind, not something I
can rule out from the data-scaling rung alone — that rung held steps fixed and varied data, so it tells
me nothing directly about what happens when I hold data fixed and vary steps. The corpus is now large
(160GB, several times the size that originally trained BERT-base or BERT-large), which should push
whatever overfitting threshold exists further out than it would be at the original 16GB scale — more
text generally buys more resistance to memorization for a fixed step count — but "should push it
further out" is a qualitative expectation, not a guarantee, and this rung is exactly where I'd expect to
see the first sign of it if the corrected recipe's efficiency (already evident in how far ahead of
BERT-large's original run it landed at only 100K steps) starts running into diminishing or reversing
returns.

The concrete move: hold the 160GB combined corpus, the BERT-large architecture, the 8K batch size, and
every recipe element (dynamic masking, full-sentences without NSP, byte-level BPE) exactly as the
previous rung left them, and extend training from 100,000 to 300,000 steps — a threefold increase, still
well short of the million steps the original BERT-large recipe used, but three times what this recipe
has been evaluated at so far. I'm choosing 300K specifically as an intermediate point rather than
jumping straight to something close to the original million-step budget, for the same sequencing reason
that's guided every rung in this study: I want a clean read on whether extending duration at fixed data
continues to help, plateaus, or starts to hurt, before committing to an even longer and more expensive
run. If 300K steps shows continued improvement over 100K with no sign of the downstream numbers
stalling or reversing, that's the signal to keep pushing step count further; if it shows a plateau or a
reversal, that would be the point to stop extending duration and consider that the recipe, at 160GB of
data, has found something close to its ceiling for this corpus size, and further gains would need to
come from data rather than duration after all.

What I'm watching for, stated as a real question rather than a foregone conclusion: does SQuAD, MNLI-m,
and SST-2 continue moving upward from rung 6's 94.0/87.7, 89.3, 95.6 by an amount that looks more like
the recipe-combination jump from rung 5 (a full point or more on several metrics) than the modest
few-tenths the data-only rung produced — which would support the reading that step count, not data
volume, was the binding constraint at 100K steps — or does it move by another modest few-tenths, similar
in size to the data-only result, which would instead suggest the recipe is simply approaching a general
plateau at this scale regardless of which lever is pulled. Either reading is informative, and I don't
want to presume which one the numbers will show before running the extended training.

So: extend pretraining from 100,000 to 300,000 steps, holding the 160GB combined data, BERT-large
architecture, 8K batch size, and every locked-in recipe element fixed, and evaluate on the same
protocol — SQuAD 1.1/2.0 dev F1, MNLI-m accuracy, SST-2 accuracy — against rung 6's numbers.
