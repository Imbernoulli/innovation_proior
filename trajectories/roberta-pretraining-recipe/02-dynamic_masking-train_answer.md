The reimplementation checks out well enough to build on. Against BERT-base's published dev numbers I
land at 90.4/78.7 SQuAD 1.1/2.0, 84.0 MNLI-m, 92.9 SST-2, 64.2 RACE — SQuAD a bit ahead of the
88.5/76.3 reference, everything else within a couple tenths of a point — and a second, independent
slice of the same static-masking configuration gives 78.3 SQuAD2.0 / 84.3 MNLI-m / 92.5 SST-2 against
a 76.3/84.3/92.8 reference. No task is systematically behind. That's the signature I was looking for:
the spread looks like ordinary noise and the minor per-implementation retuning I already flagged as
legitimate, not a bug. So I now have a trustworthy fixed point, and I can start actually asking design
questions instead of debugging.

I want to start with the cheapest, most mechanical thing on the list: not the objective, not the
architecture, not the data, just *how the existing masking objective's random corruption gets
generated*. The original recipe computes the mask once, at preprocessing time, and works around the
staleness of a single fixed mask by duplicating the training corpus tenfold — so across 40 epochs each
sequence ends up masked in ten different ways, and any one of those ten patterns recurs four times over
the course of training. Looked at closely this is a strange design. The entire point of random masking
is that the model can't memorize which positions get corrupted; it has to build a genuine distributional
model of language to recover arbitrary held-out tokens. But a *static* mask breaks that randomness at
exactly the granularity that should matter most over a long run: the fourth time the model sees a given
instance under the same mask, it isn't seeing a fresh corruption anymore — it's rehearsing one it has
already trained against three times. The tenfold duplication papers over this, but it also caps the
ceiling: ten patterns per sequence, a number set by preprocessing convenience (ten duplication passes),
not by anything principled about how many distinct corruptions a sentence needs to teach a general
recovery skill.

The fix is obvious once stated: generate the mask fresh every time a sequence is fed to the model,
instead of fixing it once and repeating it. Dynamic masking changes nothing about the corruption
algorithm itself — still 15% of tokens selected, still the 80/10/10 split among `[MASK]`, unchanged,
and random-token — only *when* the selection happens: at data-loading time, per exposure, rather than
once before training starts. And it removes the need for the tenfold duplication outright, since
there's no longer a fixed ten-mask pool to draw a repeat from — a strict storage and preprocessing win
on its own, independent of whatever it does to accuracy. If dynamic masking is even comparable to
static, it's already worth adopting on cost grounds; if it also helps accuracy, the case is
overdetermined.

Why expect any accuracy benefit at all, rather than just a cost wash? The argument is about scale. With
a fixed ten-mask pool, the marginal value of seeing the same instance again under the same precomputed
mask should shrink — the model has already extracted what that particular corruption of that particular
sentence has to teach, and a genuinely novel corruption should teach more. That effect should bite
hardest exactly when training runs longer, over more epochs, more steps, or more data — every one of
which is a direction this study is already committed to exploring in the rungs ahead, where batch size,
step count, and corpus size all get pushed well past the current regime. I'd rather remove a hard
mask-diversity ceiling now, at the cheapest possible rung, than discover much later that it was quietly
capping the returns on a far more expensive change like tripling the step count or adding ten times the
data.

I don't expect this to be dramatic yet, and I want to say that honestly rather than oversell the change.
In the current 1M-step/16GB regime, each sequence is seen on the order of 40 times regardless of
masking scheme, which isn't deep into the range where a ten-pattern ceiling should really bind. So my
actual expectation is comparable-to-mildly-better numbers here, with the real payoff more likely to
show up once the corpus and step count are pushed further — which is exactly what "comparable, plus a
free efficiency win, plus removing a confound from later scaling experiments" is sufficient grounds to
adopt for the rest of the ladder even without a dramatic gain right now. A clear loss, on the other
hand, would be the genuinely surprising result — it would suggest the training dynamics actually benefit
from rehearsing a fixed corruption, something like an implicit regularization effect from the objective
being made artificially harder by mask repetition, and that would be worth understanding before locking
dynamic masking in as a default.

```
masking:      dynamic (regenerated per exposure at data-loading time; 15% select,
              80% [MASK] / 10% unchanged / 10% random, same split as static)
              — no tenfold data duplication needed
[unchanged]:  segment-pair + NSP input format, batch 256, 1M steps,
              BookCorpus+Wikipedia 16GB, BERT-base architecture
```

I'll evaluate this exactly the way I evaluated rung 1 — SQuAD 2.0 F1, MNLI-m accuracy, SST-2 accuracy —
against my own reimplementation's static-masking numbers, not against the originally published
reference, since the whole point from here on is a controlled, one-variable-at-a-time comparison
against an instrument I've already validated.
