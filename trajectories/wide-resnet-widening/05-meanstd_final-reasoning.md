The dropout result is genuinely mixed, and I don't want to paper over that with a tidy story before
I've actually understood it. CIFAR-100 moved the direction I hoped — 20.5% down to 20.0% — but CIFAR-10
moved the other way, 4.17% up to 4.39%, a small but real regression on the exact same architecture and
the exact same regularizer. If dropout inside the residual branch were simply "correct" the way ruling
it off the shortcut was correct, I'd expect it to help on both datasets, even if by different amounts —
CIFAR-100's ten-times-fewer-images-per-class regime giving it more room to help there specifically.
Instead one dataset improved and the other got slightly worse on the identical model and training run.
That's not the clean "regularization helps a heavily-parameterized model" story I was expecting; it's
either a real, dataset-dependent effect, or noise sitting close to the resolution of a median-of-5-runs
comparison, or a symptom of something in the *evaluation protocol* rather than the architecture itself.
I want to rule out that third possibility before I draw any conclusion about dropout's value, because
if the protocol is the confound, it would be sitting underneath every number in this entire design
process, not just the dropout comparison.

There is a real, already-known asymmetry in the protocol I've been running, and I haven't examined it
carefully until now. Every measurement since the block-type rung has used ZCA whitening as CIFAR
preprocessing, inherited as a default without much scrutiny. But the baselines I keep comparing
against — the thin pre-act-ResNet family, original ResNet, stochastic depth — all report their numbers
under simple mean/std normalization, not ZCA whitening. Every comparison I've drawn against those
baselines so far, including the one right above (WRN-28-10-dropout at 4.39/20.0 against pre-act-
ResNet-1001 at 4.92/22.71), has been an apples-to-a-different-fruit comparison: my numbers are ZCA,
theirs are mean/std, and I've been treating the gap between them as if it were purely architectural
when part of it could be a preprocessing artifact in either direction. I even flagged, back when I
first read the pre-act-ResNet-1001 baseline, that its headline 4.92% used the same batch size (128) I
use throughout, while a parenthetical 4.64% used batch size 64 and isn't directly comparable — I was
already being careful about training protocol matching for that one baseline. I should be at least as
careful about preprocessing, which touches every single comparison rather than one baseline's footnote.

So there are two separate reasons pointing at the same next move, not one. First, the direct
motivation: if I want any of my WRN-vs-thin-ResNet comparisons to be trustworthy rather than
confounded by two different preprocessing pipelines, I need my own numbers measured under the same
mean/std convention the baselines use. This isn't a response to the dropout result specifically — it's
a correctness requirement for every comparison I've been making since rung 3, and it's overdue
regardless of what dropout did. Second, and more speculative: switching preprocessing gives me a
natural way to check whether the dropout regression on CIFAR-10 was protocol-dependent. ZCA whitening
decorrelates and rescales input channels in a way mean/std normalization doesn't; if dropout's
interaction with batch normalization is even mildly sensitive to the input statistics batch norm's
first layer sees, a preprocessing change could plausibly shift where dropout's cost/benefit trade-off
lands, in either direction. I don't have a strong mechanistic argument for why it would flip a small
regression into a small gain specifically — this is a real unknown, not a hypothesis I'm confident in —
but re-measuring under the corrected, baseline-matched protocol lets me find out rather than continuing
to build on top of an unexamined confound.

Concretely: switch CIFAR preprocessing from ZCA whitening to mean/std normalization, and re-measure.
Two things need re-measuring, not one. First, the no-dropout comparison itself: I want CIFAR-10 and
CIFAR-100 numbers for the strongest configurations the grid found — 40-4, 16-8, and 28-10 — under
mean/std, so I have a clean, protocol-matched head-to-head against the thin pre-act-ResNet family
(110/164/1001 layers) and against stochastic depth and the original ResNet, all of which I can now
compare like-for-like rather than across preprocessing conventions. This also gives me a chance to
check something I couldn't check before: whether 40-4, at 8.9M parameters — genuinely close to the
1001-layer reference's 10.2M — actually matches or beats that reference now that the comparison isn't
muddied by different preprocessing. That's the clean depth-vs-width-at-matched-budget test I wanted
back at the grid rung and couldn't quite get, because 28-10 turned out to be the grid's accuracy winner
at a much larger 36.5M parameters, not a parameter-matched one.

Second, dropout itself needs re-running under the new protocol, not assumed unchanged from the ZCA
measurement — that's the whole point of suspecting the protocol as a confound in the first place. I'll
re-test dropout on 28-10 again, so it's directly comparable to its own ZCA-preprocessed predecessor,
and I'll broaden the dropout comparison to two more points I haven't tested dropout on at all yet: a
smaller model (16-4) to see whether dropout's effect depends on how much spare capacity there is to
regularize, and a much deeper-but-thin model (52-1, close to the "thin" end of this family) to see
whether the branch-only placement helps even where width isn't the source of extra capacity. And since
dropout's motivating case was always heavy-overfitting risk under light augmentation, I want the SVHN
setting in this rung too — no augmentation at all, and by far the setting where any regularization
gap should show up most clearly, if it's real. That's a genuinely different stress test than CIFAR's
flip-and-crop regime and one I haven't touched at all since the initial context flagged it as the
low-augmentation edge case.

So this rung is doing two things that are really one thing: fixing a protocol mismatch that's been
silently sitting under every comparison since rung 3, and using that fix as the occasion to properly
close out the dropout question with a wider, more honest test than the single mixed-result comparison
I have right now. I'm not assuming the mean/std switch flips CIFAR-10's dropout regression into a
win — I flagged above that I don't have a strong mechanistic reason to expect that specific outcome —
but I do expect the no-dropout baseline comparisons to become more trustworthy the moment they're
measured on the same footing as the numbers I'm comparing them against, and that alone is worth doing
independent of what happens to dropout.
