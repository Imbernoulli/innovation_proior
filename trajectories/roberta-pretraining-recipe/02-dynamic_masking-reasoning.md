The reimplementation checks out. Against BERT-base's published dev numbers (SQuAD 1.1/2.0 F1
88.5/76.3, MNLI-m 84.3, SST-2 92.8, RACE 64.3), my fairseq build lands at 90.4/78.7 SQuAD, 84.0 MNLI-m,
92.9 SST-2, 64.2 RACE under segment-pair+NSP with static masking — SQuAD actually a bit ahead, MNLI-m
and RACE within a couple tenths, SST-2 within a tenth. And under a second, independent slice of the
same static-masking configuration I get 78.3 SQuAD2.0 / 84.3 MNLI-m / 92.5 SST-2 against a 76.3/84.3/92.8
reference. No task is systematically behind; the spread looks like ordinary run-to-run noise plus the
minor per-implementation LR/warmup retuning I flagged as legitimate going in, not a bug. That means I
now have a trustworthy fixed point, and everything after this is a genuine one-variable-at-a-time
ablation rather than a debugging exercise. So: what's the first thing worth changing?

I want to start with the cheapest, most mechanical change on the list — one that doesn't touch the
objective, the architecture, or the data, only *how the existing MLM objective's random mask is
generated*. The original recipe computes the mask once, during preprocessing, and then works around
the staleness of that single mask by duplicating the training corpus tenfold, so that across 40 epochs
each sequence gets ten different mask patterns, and any one of those ten patterns recurs four times.
This is a genuinely strange design when I look at it closely. The masking pattern is supposed to be a
random corruption the model has to learn to undo in general — the whole point of masking 15% of tokens
at random is that the model can't memorize which positions get masked, it has to build a real
distributional model of language to recover arbitrary held-out tokens. But a *static* mask breaks that
randomness at exactly the granularity that matters most for a long training run: every time the model
sees a given training instance for the fourth time under the same mask, it isn't seeing a fresh random
corruption of that sentence anymore, it's rehearsing a corruption it has already been trained against
three times before. The tenfold data duplication is a workaround for this, not a fix — it multiplies
storage and adds sequences to the corpus, but it caps the effective mask diversity at exactly ten
patterns per sequence, a number chosen for practical reasons (ten passes of preprocessing) rather than
anything principled about how many distinct corruptions a sentence needs to be shown to teach a general
recovery skill.

The alternative is obvious once stated: generate the mask fresh every time a sequence is fed to the
model, rather than fixing it once and repeating it. Call this *dynamic* masking. It requires no change
to the masking algorithm itself — still 15% of tokens selected, still 80/10/10 split among `[MASK]` /
unchanged / random-token — only a change to *when* that selection happens: at data-loading time, per
epoch, per exposure, rather than once at preprocessing time before training starts. It removes the need
for the tenfold data duplication entirely, since there's no longer a fixed pool of ten pre-computed
masks to draw from; a sequence's mask is simply resampled every time it's presented. That is also a
meaningful *efficiency* win independent of any accuracy effect: no tenfold storage blow-up, no
preprocessing pass to generate the duplicated, pre-masked corpus. If dynamic masking performs even
comparably to static, it's a strict improvement on cost grounds alone; if it also improves accuracy,
the case for it is overdetermined.

Why would I expect it to help accuracy at all, rather than just being a wash with a storage benefit
attached? The argument is about what happens as training scales — either to more epochs over the same
data, or, as the later data-and-steps rungs of this study are already flagged to explore, to
substantially larger pretraining corpora and longer schedules. With a fixed ten-mask pool, the *marginal*
value of an additional pass over the same instance under the *same* pre-computed mask should shrink:
the model has already seen that particular corruption pattern for that particular sentence some number
of times, and repeating it teaches diminishing returns on the general masked-recovery skill compared to
what a genuinely novel corruption of the same sentence would teach. This becomes crucial precisely when
pretraining for more steps or over more data — the two axes this whole study is already aimed at
testing later. If I'm eventually going to scale batch size, training duration, and corpus size well
past the original 1M-step/16GB regime, I don't want an artifact of the masking scheme — a hard ten-mask
ceiling per sequence — quietly capping how much signal additional training steps can extract from a
sequence I've already exhausted the mask diversity of. Better to remove that ceiling now, at the
cheapest possible rung, than to discover much later in this study that it was silently limiting returns
on a much more expensive change like doubling the corpus or tripling the step count.

I don't expect this change to be dramatic in the current 1M-step/16GB regime specifically — the current
setup, static or dynamic, sees each sequence roughly 40 times over the course of training (once per
epoch under the effective mask cadence I inherited), which is not yet deep into the range where the
ten-mask ceiling should bind hard. So my honest expectation is that this rung should look roughly
comparable to, or only mildly better than, static masking under the current pretraining budget — the
real payoff of removing the ceiling might not show up cleanly until the corpus and step count are
substantially larger. But "roughly comparable, with a large storage/preprocessing efficiency win
attached" is already sufficient grounds to keep dynamic masking for the rest of this study even absent
a large accuracy gain right now, since every subsequent rung's ablations become cheaper to run and the
mask-diversity ceiling stops being a confound in the scaling experiments later on. If it turns out to be
meaningfully *worse* than static under the current regime, that would be the surprising result — it
would suggest something about the model's training dynamics genuinely benefits from the *repetition* of
a fixed corruption pattern (perhaps acting as an implicit regularizer against the objective being too
easy when every corruption is novel), which would be worth understanding before committing to it as a
default for the rest of the ladder.

So: replace static, preprocessing-time masking with dynamic, per-exposure masking, keeping the 15%
selection rate and 80/10/10 split unchanged, everything else in the rung-1 configuration held fixed
(segment-pair+NSP input format, batch 256, 1M steps, 16GB Books+Wiki data), and measure the same
protocol — SQuAD 2.0 F1, MNLI-m accuracy, SST-2 accuracy — against the static-masking numbers this
reimplementation already produced.
