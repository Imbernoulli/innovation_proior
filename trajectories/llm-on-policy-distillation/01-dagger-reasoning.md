The loss is the whole experiment, but it bolts onto a fixed on-policy loop, and the floor I want to
start from is the crudest thing that respects that loop's one real lesson — train on the states the
student actually visits — while ignoring everything the queryable teacher makes available beyond its
single preferred token. That floor is hard-target imitation, and starting there is deliberate,
because it isolates the exact axis the rest of the ladder will move along: how much of the teacher's
distribution the loss is allowed to use. If I begin with the loss that throws away the *most*, I get
a clean reading of what the soft-distribution losses buy on top of it, and every later rung becomes a
measured increment over a known floor rather than a number floating in a vacuum.

So let me reason about what the trainer hands me and what the crudest defensible loss does with it.
The substrate is on-policy by construction: before each step, with probability `lmbda` the trainer
swaps the dataset batch for the student's own sampled completions, relabels them, and only then calls
my loss. That mixing is the imitation-learning fix already wired in, and it is worth pricing out how
much it buys, because the price is the reason the loop is built this way. Plain behavior cloning fits
the student to demonstrator states, but at inference the student conditions on its *own* prefixes; a
single early slip lands it in a context the expert never visited, its error rate there is higher, and
the mistakes compound. The standard bound on that cascade is a regret that grows like `ε·T²` in the
horizon `T`, against the `ε·T` you would get if the training states matched the test states. Put
numbers on it: at a per-token error `ε ≈ 0.05` over a `T ≈ 200`-token solution, the off-policy bound
is `ε·T² ≈ 0.05 · 40000 = 2000` and the on-policy bound is `ε·T ≈ 10` — two orders of magnitude,
the difference between "a few slips" and "the whole tail of the completion is off-support." I do not
have to earn that gap; the trainer's `lmbda` coin already pays it by injecting the student's own
prefixes into training roughly half the time. So the data axis is settled upstream. What I have to
decide is the *label*: given the states the student visited, what target do I push it toward at each
token?

It also matters that the budget is small, because that decides whether a brittle loss can be rescued
by simply training longer. The trainer runs `max_steps=2000` with per-device batch 4, gradient
accumulation 4, across 2 GPUs, so the effective batch is `4·4·2 = 32` sequences per optimizer step
and the run sees `32·2000 = 64000` sequence-updates total. Against a 10k-prompt subset that is only
`64000 / 10000 ≈ 6.4` passes over the data, and with `lmbda = 0.5` about half of those — some 32k
sequence-updates — are the student's own rollouts. This is a short run. Whatever loss I choose has to
be well-conditioned from the first steps; there is no long tail of epochs in which a pathological
gradient could quietly anneal itself out. That reinforces the plan to establish a floor first: I want
to know what the crudest, most robust label does with 6 passes before I spend the same budget on
anything cleverer.

The most conservative answer, and the one with the longest imitation-learning pedigree, is to push
the student toward the *expert's chosen action* — a hard target. In the control setting an expert is
queried at a visited state and returns the one action it would take; the learner is fit to reproduce
that action by ordinary supervised classification. Map that onto tokens: the "state" is the partial
sequence the student has generated, the "expert" is the frozen teacher, and the teacher's chosen
action at that state is its most likely next token — the argmax of its conditional distribution.
Train the student by cross-entropy to predict exactly that token. This is DAgger's loss in its most
literal autoregressive instantiation: collect the learner's states (the trainer does this), label
each with the expert's top-1 action (the teacher's argmax), minimize cross-entropy. The label source
is the teacher; the per-token weight is one; there is no KL term, no temperature, no advantage
weighting. It is the supervised-classification core of the imitation reduction, nothing more.

Let me be precise about why this is the *crudest* loss and not just *a* loss, because that framing is
the reason it belongs at the bottom of the ladder. The teacher does not just have a preferred token;
it has a full distribution over the vocabulary, and that distribution carries the relative
plausibility of every alternative — the "dark knowledge" that soft distillation was invented to
exploit. Hard-target cross-entropy collapses that entire distribution to a single one-hot vector at
its mode. Everything the teacher knows about the *second*-best continuation, about how confident it
is, about which wrong tokens are near-misses versus absurd, is discarded. So this loss uses strictly
less of the available signal than any divergence-based loss could. If a soft loss cannot beat it,
the dark-knowledge premise is wrong for this teacher–student pair; if it can, the gap measures
exactly what the soft information is worth. That is what I want from rung one, and it is why the floor
is not a throwaway: it is the denominator against which every later number will be read.

There is a competing candidate for "the floor" I should dispatch explicitly, because the scaffold
already ships one: the default body is forward KL against the full teacher softmax, and I could just
declare *that* the floor and start climbing from a soft loss. I reject it, and the reason is that the
scaffold default is not crude — it already spends the entire teacher distribution, so it cannot serve
as the reference that measures what spending the distribution is *worth*. What I want at the bottom is
the loss that uses the least, so the first climb has something to be an increment over. And there is a
clean sense in which the hard target sits on the *same axis* as the scaffold rather than off to one
side, which is what convinces me it is the honest floor and not a different experiment. Cross-entropy
against the argmax is exactly the forward KL `KL(q ‖ p_S)` where `q` is the teacher's distribution
sharpened to zero temperature: as the teacher softmax temperature `τ → 0`, `softmax(teacher_logits/τ)`
concentrates all mass on the argmax and becomes the one-hot vector `e_ã`, and `KL(e_ã ‖ p_S) =
Σ_v e_ã(v)·(log e_ã(v) − log p_S(v)) = −log p_S(ã)` because the one-hot's own entropy term is `0`.
That is precisely the per-token cross-entropy the DAgger body computes. So the hard target is not a
loss from some other family; it is the `τ → 0` endpoint of the scaffold's own forward KL — the
teacher cranked to maximal sharpness. The scaffold default sits at `τ = 1` on that same knob. Framing
the floor as an endpoint of the axis I am about to move along is exactly what I want: the first rung
will pull `τ` back off zero, restoring the mass the argmax threw away, and the accuracy change will be
a clean read of what that mass is worth.

Now the wall I have to be honest about, because it is what I expect to bite. A hard target is a *low
information* target, and at the same time it can be a *high noise* one. Consider where the teacher's
distribution is flat — two or three continuations almost equally likely, as happens constantly in
mathematical prose ("therefore" vs "hence" vs "so", or the order in which independent terms are
written). Let me actually make this concrete instead of asserting it, because the size of the effect
is the whole worry. Suppose at some position the teacher's post-crop logits put three connective
tokens at `6.00, 5.95, 5.90` and everything else at or below `2`. Exponentiating, `e^{6.00} ≈ 403.4`,
`e^{5.95} ≈ 383.8`, `e^{5.90} ≈ 365.0`, so those three carry mass `≈ 403.4/1152.2 = 0.350`,
`383.8/1152.2 = 0.333`, `365.0/1152.2 = 0.317` (the tail below logit 2 is negligible against a sum of
1152). The argmax is the first token, but it wins by a `0.05` logit margin — `1.7` percentage points
of probability — over a token the teacher considers essentially interchangeable. Cross-entropy now
commands the student to place *all* its mass on that first token and treat the other two, each with a
third of the teacher's belief, as outright errors. Push it further: imagine the student has already
learned to match the teacher perfectly here, `p_S = (0.35, 0.33, 0.32, …)`. The cross-entropy
gradient on the student's logit `z_j` is `p_S(j) − 1{j = ã}`, so at this teacher-matched student the
gradients are `0.35 − 1 = −0.65` on the winner (push up hard), `+0.33` on the second token (push
down), `+0.32` on the third (push down). A *correct* soft match still gets a large gradient tearing
it off the teacher's real distribution and toward one-hot on an essentially arbitrary winner. Compare
the forward-KL gradient at the same point, `p_S − p_T = 0` — it vanishes exactly where the student
matches the teacher. So the hard target's minimum is nowhere near where the teacher's distribution
actually lives; it manufactures a sharp, partly spurious gradient precisely at the flat positions.
That is the mechanism, and I did not have to guess its magnitude: it is a `0.65`-scale push on the
winner at a position where the teacher was nearly indifferent.

There is a second, subtler reason this should sit low, specific to the capacity gap. The student is a
0.5B base model; the teacher is a 7.6B math-tuned instruct model — roughly a `7.6/0.494 ≈ 15×`
parameter gap, and worse, the teacher is instruction- and math-tuned while the student is a raw base
model. A hard target says "match my argmax" with no temperature softening, which means the student is
asked to become as sharp as the teacher's mode at every position — but the student cannot represent
the teacher's reasoning, so forcing one-hot confidence on the teacher's top token, with no credit for
the surrounding mass, gives it nothing to interpolate toward when it is wrong. Soft losses degrade
gracefully (a near-miss still earns partial credit through the distribution); the hard target is
all-or-nothing at the token level. Under a large capacity gap, all-or-nothing token supervision is
exactly the brittle regime, and the flat-position example above shows the failure is not rare — it
fires on every interchangeable connective in a long chain.

There is a sharper version of this worry once I remember that half the batches are the student's own
rollouts. On the dataset half, the teacher scores curated solution tokens it would plausibly have
written itself, so its distribution there is relatively peaked and the argmax is a fairly reliable
label. On the on-policy half, the teacher is scoring a *student-generated* prefix — a sequence the
7.6B model did not write and may find off-distribution — and its next-token distribution over such a
prefix is exactly where I expect it to be flattest and most hedged, because the teacher is uncertain
how to continue a context it would not have produced. That is the failure mode I just quantified:
flat teacher distribution, arbitrary argmax, spurious one-hot demand. So the hard target is *least*
reliable precisely on the on-policy states the loop was constructed to emphasize, which is a small
irony worth naming — the crudest label degrades most on the very data that the expensive on-policy
machinery works to put in front of it. It does not sink the floor (grade-school chains stay peaked
even on student prefixes), but it tells me where the soft losses should win first: on the harder,
longer, more off-distribution completions where the teacher's hedging carries the information the
argmax throws away.

Let me settle the operational details against the actual edit surface, because the loss has to be the
literal body the trainer calls and nothing more — I do not get to touch the loop, add a frozen
network, or change `lmbda`. The teacher's chosen action at each position is `teacher_logits.argmax`
over the vocabulary, giving a `[B, T]` tensor of target token ids. I must preserve the trainer's
padding convention: positions where `labels == -100` are prompt or padding and must not contribute,
so I overwrite the argmax targets with `-100` wherever the labels say so, then let cross-entropy's
`ignore_index=-100` drop them. The cross-entropy is taken between the student's logits, flattened to
`[B*T, V]`, and the flattened target ids — token-level negative log-likelihood of the teacher's top
token, summed over the valid completion positions and divided by their count for the `batchmean`
reduction, with the denominator clamped to at least one so an all-masked batch cannot divide by
zero. One alignment check I should not skip: the trainer crops the teacher head from 152064 to the
student's 151936 before handing me the tensor, so `teacher_logits.argmax` returns an id in
`[0, 151935]`, which is a valid student-vocabulary target — the 128 teacher-only ids that the crop
removed can never appear as a label, and cross-entropy against the student's 151936-wide logits is
well-posed. If the crop were the other way I could be indexing a token the student cannot emit; it is
not, so the hard label is always representable.

Critically, there is **no temperature scaling on either side**, and I want to justify that rather
than inherit it. The imitation loss is plain `log π_θ(ã|s)` for the teacher's action `ã`, and the
target `ã = argmax_v teacher_logits_v` is *temperature-invariant*: dividing every logit by a positive
`T` preserves their order, so `argmax(teacher_logits / T) = argmax(teacher_logits)` for any `T > 0`.
The temperature could therefore only touch the student side, where dividing its logits by `0.9`
before the cross-entropy would rescale the gradient by a constant and shift nothing about which token
is being demanded — a pure, pointless reparameterization of the step size. So I leave both untouched.
This is one place the task's loss differs from a generic soft-distillation body: the temperature knob
the signature exposes is simply unused here, on purpose, because the hard target has no soft
distribution for temperature to soften.

The cost side confirms this is the right thing to sit at the bottom, too, which reassures me the
floor is cheap enough that later rungs are paying for signal and not merely for compute. The scaffold
default materializes the teacher's `log_softmax` over the full `[B, T, 152064]` tensor at every step;
at `B=4` and a completion length of a few hundred tokens that is on the order of a gigabyte of
float32 per side just to hold the soft target. The hard target needs none of it: an argmax collapses
the teacher tensor to a `[B, T]` integer id (kilobytes), and cross-entropy fuses the student's
`log_softmax` internally without ever storing a second full-width distribution. So the floor is not
only the least *informative* loss, it is close to the *cheapest*, which means when a soft loss later
beats it I will know I am buying accuracy with the teacher's distribution and not with a coincidental
change in numerical conditioning.

I should also note what this loss does *not* do relative to the full DAgger procedure, so I do not
over-claim. DAgger's aggregate-and-refit (Follow-The-Leader over a growing dataset) and its decaying
expert-mixing schedule `β_i` live in the *data* layer, and the data layer is the trainer's, not
mine. The trainer mixes student-generated and dataset batches by a *static* `lmbda`, not a decaying
`β_i`, and it replaces the batch per step rather than aggregating all rounds. So what I am landing is
DAgger's *loss* — hard cross-entropy on the expert's action at the learner's visited states — on top
of the trainer's static on-policy mixing, not DAgger's full no-regret schedule. The loss body has no
way to see whether a given batch came from the student or the dataset, so it cannot even condition on
the data source. That is fine for a floor: the point is the label, and the label is the teacher's
top-1 token regardless of where the state came from.

Before I commit, one small verification that the body is numerically safe in the corner cases the
trainer will actually produce. If a batch is all padding — every `labels == -100`, which can happen
when a rollout is short and the rest is pad — then every argmax target is overwritten to `-100`,
cross-entropy with `ignore_index=-100` returns a zero-length reduction, `valid_mask.sum()` is `0`,
and the `clamp_min(1)` makes the denominator `1` so the `batchmean` return is `0/1 = 0` rather than a
`nan` from `0/0`. Good. And on a normal batch the return is exactly the mean per-token NLL of the
teacher's top token over the completion positions, which is the quantity I meant to minimize — no
hidden temperature factor, no KL cross-term, nothing but supervised classification against the
argmax. The body does what the derivation says.

One patch I could reach for and should refuse, because refusing it is what keeps the floor honest, is
label smoothing: soften the one-hot target to `(1−ε)·e_ã + ε·uniform`, which would blunt the
over-confident one-hot demand at flat positions. It is cheap and it would nudge the calibration, but
it fixes the wrong thing. Smoothing spreads the leftover `ε` mass *uniformly* across all 151936
tokens, asserting that every non-argmax token is equally slightly-correct — which is exactly false,
because the teacher's second-best at a flat position is a *specific* token (the "hence" against the
"therefore"), not the undifferentiated tail. So label smoothing would trade one wrong target for
another: it removes some over-confidence but replaces the teacher's real relative structure with a
flat prior that has nothing to do with the teacher. The correct softening is not toward uniform but
toward the teacher's *own* distribution, and once I am willing to carry the teacher's distribution I
am no longer at the floor — I am doing soft distillation, which is the next rung. So I keep the hard
target pure precisely so that the thing the next rung adds is unambiguously "the teacher's mass," not
"some smoothing constant I smuggled in at the bottom."

Putting it together, the step-1 edit is the hard-target fill: argmax the teacher to get per-token
labels, mask them with the trainer's `-100`, take masked token-level cross-entropy of the student
against those labels, reduce by `batchmean`. No KL, no temperature, no soft distribution. The full
scaffold body is in the answer.

Now reason about what this floor must do, because that is the entire reason to run it. The three
benchmarks span difficulty and, more importantly, span statistical reliability, and I want to fix
that reliability now so I read the eventual table honestly. GSM8K has 1319 problems; a result near
one-half has a standard error of about `sqrt(0.5·0.5/1319) ≈ 0.0138`, so a difference between two
losses of a percentage point is roughly one standard error — real signals on GSM8K have to clear
`±1.4%` to mean anything. MATH-500 has 500 problems; near `0.3` its standard error is about
`sqrt(0.3·0.7/500) ≈ 0.020`, `±2%`, so it is noisier per point but still readable. AMC23 is 40
problems on an avg@8 protocol: a single problem is `1/40 = 2.5%` of the score, so two or three solved
problems swing the number by more than any plausible effect I am chasing, and I will treat AMC as
noise and read the story off GSM8K and MATH-500. I expect hard-target imitation to *function* — the
student will learn to imitate the teacher's preferred tokens and will solve a meaningful fraction of
GSM8K, because grade-school arithmetic chains are short and the teacher's argmax is usually
unambiguous there, so the flat-position pathology fires less often. But on MATH-500, where chains are
longer and the teacher's per-token distribution is more often multi-modal, the discarded soft
structure and the spurious sharpening of flat regions should cost it: I expect it to land at or near
the bottom of the ladder on both reliable metrics, beating nothing decisively and plausibly trailing
every soft-distribution loss. The diagnosis that would point to the next rung is already written into
the construction: if the hard target underperforms, the fix is to stop throwing away the teacher's
distribution — to give the student *soft* per-token targets that preserve the relative mass on the
alternatives, starting with the cheapest faithful way to summarize that distribution rather than
collapsing it to its mode.
