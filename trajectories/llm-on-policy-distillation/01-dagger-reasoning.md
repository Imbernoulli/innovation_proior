The loss is the whole experiment, but it bolts onto a fixed on-policy loop, and the floor I want to
start from is the crudest thing that respects that loop's one real lesson — train on the states the
student actually visits — while ignoring everything the queryable teacher makes available beyond its
single preferred token. That floor is hard-target imitation, and starting there isolates the exact
axis the rest of this work will move along: how much of the teacher's distribution the loss is
allowed to use. If I begin with the loss that throws away the most, every later choice becomes a
measured increment over a known floor rather than a number floating in a vacuum.

The substrate is on-policy by construction: before each step, with probability `lmbda` the trainer
swaps the dataset batch for the student's own sampled completions, relabels them, and only then calls
my loss. That mixing is the imitation-learning fix already wired in. Plain behavior cloning fits the
student to demonstrator states, but at inference the student conditions on its *own* prefixes; a
single early slip lands it in a context the expert never visited, and the mistakes compound — the
standard bound on that cascade grows like `ε·T²` in the horizon against the `ε·T` you get when
training states match test states. At a per-token error `ε ≈ 0.05` over a `T ≈ 200`-token solution
that is `2000` versus `10`, two orders of magnitude. The trainer's `lmbda` coin already pays for that
by injecting the student's own prefixes into training roughly half the time, so the data axis is
settled upstream. What I have to decide is the *label*: given the states the student visited, what
target do I push it toward at each token?

The budget is small, which matters because it decides whether a brittle loss can be rescued by
training longer. `max_steps=2000`, per-device batch 4, gradient accumulation 4, 2 GPUs — effective
batch `4·4·2 = 32`, so `64000` sequence-updates total, only `~6.4` passes over the 10k-prompt subset,
about half of them the student's own rollouts. There is no long tail of epochs in which a
pathological gradient could quietly anneal itself out, so whatever loss I choose has to be
well-conditioned from the first steps. That reinforces establishing a floor first.

The most conservative label, with the longest imitation-learning pedigree, is the expert's chosen
action — a hard target. Map the control setting onto tokens: the state is the partial sequence the
student generated, the expert is the frozen teacher, its chosen action is its most likely next token
— the argmax of its conditional distribution. Train the student by cross-entropy to predict exactly
that token. This is DAgger's loss in its most literal autoregressive form: collect the learner's
states (the trainer does this), label each with the teacher's argmax, minimize cross-entropy. No KL,
no temperature, no advantage weighting, unit token weight.

Why this is the *crudest* loss and not just *a* loss: the teacher has a full distribution over the
vocabulary carrying the relative plausibility of every alternative — the dark knowledge soft
distillation exists to exploit — and hard-target cross-entropy collapses all of it to a one-hot at
the mode. So it uses strictly less of the available signal than any divergence-based loss could. If a
soft loss cannot beat it, the dark-knowledge premise is wrong for this teacher–student pair; if it
can, the gap measures what the soft information is worth. That is why the floor is the denominator
against which every later number is read.

There is a competing candidate for the floor that the scaffold already ships: the default body is
forward KL against the full teacher softmax, and I could just start climbing from that soft loss. I
reject it as the floor because it already spends the entire teacher distribution, so it cannot
measure what spending the distribution is *worth*. And the hard target sits on the *same* axis as
that default rather than off to one side: cross-entropy against the argmax is exactly the forward KL
`KL(q ‖ p_S)` with `q` the teacher sharpened to zero temperature. As `τ → 0`,
`softmax(teacher_logits/τ)` concentrates all mass on the argmax and becomes the one-hot `e_ã`, and
`KL(e_ã ‖ p_S) = −log p_S(ã)` (the one-hot's own entropy term is zero) — precisely the per-token
cross-entropy the DAgger body computes. So the hard target is the `τ → 0` endpoint of the scaffold's
own forward KL, and the scaffold default is `τ = 1` on the same knob. The first climb will pull `τ`
back off zero, restoring the mass the argmax threw away, and the accuracy change reads directly as
what that mass is worth.

Now the wall I expect to bite. A hard target is a *low information* target and can be a *high noise*
one. Where the teacher's distribution is flat — two or three continuations almost equally likely, as
happens constantly in math prose ("therefore" vs "hence" vs "so", or the order in which independent
terms are written) — the argmax picks an arbitrary winner and cross-entropy demands one-hot
confidence on it. Make it concrete: three connective tokens at logits `6.00, 5.95, 5.90`, everything
else at or below `2`, carry mass `≈ 0.35, 0.33, 0.32` (the tail is negligible); the argmax wins by a
`0.05`-logit margin over tokens the teacher considers essentially interchangeable. Suppose the
student has already matched the teacher here, `p_S = (0.35, 0.33, 0.32, …)`. The cross-entropy
gradient on logit `z_j` is `p_S(j) − 1{j = ã}`, so at this teacher-matched student the gradients are
`0.35 − 1 = −0.65` on the winner, `+0.33` on the second, `+0.32` on the third — a `0.65`-scale force
tearing a *correct* soft match off the teacher's real distribution toward one-hot on an arbitrary
winner. The forward-KL gradient at the same point is `p_S − p_T = 0`, vanishing exactly where the
student matches. So the hard target's minimum is nowhere near where the teacher's distribution
actually lives; it manufactures a sharp, partly spurious gradient precisely at the flat positions,
and it fires on every interchangeable connective in a long chain.

There is a second reason this should sit low, specific to the capacity gap. The student is a 0.5B
base model, the teacher a 7.6B math-tuned instruct model — a `~15×` parameter gap plus an
instruction-tuning gap. A hard target asks the student to become as sharp as the teacher's mode at
every position, with no credit for the surrounding mass, so when it is wrong it has nothing to
interpolate toward. Soft losses degrade gracefully; the hard target is all-or-nothing at the token
level, which under a large capacity gap is the brittle regime.

And it is worst exactly on the on-policy half. On the dataset half the teacher scores curated
solution tokens it would plausibly have written, so its distribution there is relatively peaked and
the argmax is a fairly reliable label. On the on-policy half it scores a *student-generated* prefix
it did not write and may find off-distribution — precisely where its next-token distribution is
flattest and most hedged. So the crudest label degrades most on the very data the expensive on-policy
machinery works to put in front of it. That does not sink the floor (grade-school chains stay peaked
even on student prefixes), but it tells me where the soft losses should win first: on the harder,
longer, more off-distribution completions where the teacher's hedging carries the information the
argmax throws away.

Operationally, against the actual edit surface — I only fill the loss body, no touching the loop,
adding a frozen network, or changing `lmbda`. The teacher's chosen action is `teacher_logits.argmax`
over the vocabulary, a `[B, T]` tensor of target ids. I preserve the trainer's padding convention by
overwriting the argmax targets with `-100` wherever `labels == -100`, then let cross-entropy's
`ignore_index=-100` drop them; the student logits are flattened to `[B*T, V]`, the CE summed over
valid positions and divided by their count for `batchmean`, denominator clamped to at least one so an
all-masked batch cannot divide by zero. One alignment point: the trainer crops the teacher head from
152064 to the student's 151936 before handing me the tensor, so `teacher_logits.argmax` returns an id
in `[0, 151935]` — always a valid student-vocabulary target, never one of the 128 teacher-only ids
the crop removed.

There is deliberately no temperature scaling on either side. The target `ã = argmax_v
teacher_logits_v` is temperature-invariant, since dividing every logit by a positive `T` preserves
their order; and temperature on the student side would only rescale the gradient by a constant, a
pointless reparameterization of the step size. So the temperature knob the signature exposes is
simply unused here — the hard target has no soft distribution for it to soften.

The cost side confirms the floor belongs at the bottom. The scaffold default materializes the
teacher's `log_softmax` over the full `[B, T, 152064]` tensor every step, on the order of a gigabyte
per side; the hard target collapses the teacher to a `[B, T]` integer id and cross-entropy fuses the
student's `log_softmax` internally without storing a second full-width distribution. So the floor is
close to the *cheapest* loss as well as the least informative, which means when a soft loss later
beats it I will know I am buying accuracy with the teacher's distribution and not with a coincidental
change in numerical conditioning.

What this does not do relative to full DAgger, so I do not over-claim: the aggregate-and-refit
(Follow-The-Leader over a growing dataset) and the decaying expert-mix `β_i` live in the *data*
layer, which is the trainer's static `lmbda` mixing, not mine. The loss body cannot even see whether
a batch came from the student or the dataset. So I am landing DAgger's *loss* — hard cross-entropy on
the expert's action at the learner's visited states — on the trainer's static on-policy mixing, not
its full no-regret schedule. That is fine for a floor: the point is the label.

One patch I could reach for and should refuse is label smoothing — softening the one-hot to
`(1−ε)·e_ã + ε·uniform`. It would blunt the over-confident demand at flat positions, but it fixes the
wrong thing: it spreads the leftover `ε` mass *uniformly* across all 151936 tokens, asserting every
non-argmax token is equally slightly-correct, which is exactly false — the teacher's second-best at a
flat position is a *specific* token (the "hence" against the "therefore"), not the undifferentiated
tail. The correct softening is toward the teacher's *own* distribution, and once I carry that I am no
longer at the floor; I am doing soft distillation, which is the next step. So I keep the hard target
pure, so that what the next step adds is unambiguously "the teacher's mass."

So the edit is the hard-target fill: argmax the teacher, mask with the trainer's `-100`, take masked
token-level cross-entropy of the student against those labels, reduce by `batchmean`. The full body
is in the answer.

Now what this floor must do. The three benchmarks span statistical reliability, which I want to fix
now so I read the eventual table honestly. GSM8K has 1319 problems; near one-half its standard error
is `sqrt(0.5·0.5/1319) ≈ 0.014`, so a difference between two losses has to clear about `±1.4%` to
mean anything. MATH-500 has 500; near `0.3` its standard error is `~0.020`, noisier per point but
still readable. AMC23 is 40 problems on avg@8, where a single problem is `2.5%` of the score, so two
or three solved problems swing the number past any effect I am chasing — I will treat AMC as noise
and read the story off GSM8K and MATH-500. I expect hard-target imitation to *function* — grade-school
chains are short and the teacher's argmax is usually unambiguous there, so the flat-position pathology
fires less often — but to land at or near the bottom on both reliable metrics, worst on MATH-500 where
chains are longer and the teacher's per-token distribution is more multi-modal. The diagnosis that
would point to the next step is written into the construction: if the hard target underperforms, stop
throwing away the teacher's distribution and give the student *soft* per-token targets that preserve
the relative mass on the alternatives, starting with the cheapest faithful way to summarize that
distribution rather than collapsing it to its mode.
