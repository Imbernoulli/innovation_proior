The loss is the whole experiment, but it bolts onto a fixed on-policy loop, and the floor I want to
start from is the crudest thing that respects that loop's one real lesson — train on the states the
student actually visits — while ignoring everything the queryable teacher makes available beyond its
single preferred token. That floor is hard-target imitation, and starting there is deliberate,
because it isolates the exact axis the rest of the ladder will move along: how much of the teacher's
distribution the loss is allowed to use. If I begin with the loss that throws away the *most*, I get
a clean reading of what the soft-distribution losses buy on top of it.

So let me reason about what the trainer hands me and what the crudest defensible loss does with it.
The substrate is on-policy by construction: before each step, with probability `lmbda` the trainer
swaps the dataset batch for the student's own sampled completions, relabels them, and only then calls
my loss. That mixing is the imitation-learning fix already wired in — the states the student
conditions on at inference (its own prefixes) appear in training, so the cascade that dooms plain
behavior cloning, where one early slip lands the model in a context the expert never visited and the
error compounds to a cost that grows like the square of the sequence length, is structurally
addressed upstream. I do not have to re-derive on-policy data collection; the trainer's `lmbda` coin
*is* the data axis. What I have to decide is the *label*: given the states the student visited, what
target do I push it toward at each token?

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
exactly what the soft information is worth. That is what I want from rung one.

Now the wall I have to be honest about, because it is what I expect to bite. A hard target is a *low
information* target, and at the same time it can be a *high noise* one. Consider where the teacher's
distribution is flat — two or three continuations almost equally likely, as happens constantly in
mathematical prose ("therefore" vs "hence" vs "so", or the order in which independent terms are
written). The argmax picks one of them essentially at random, by a hair of logit, and cross-entropy
then commands the student to put all its mass on that arbitrary winner and treat the equally-good
alternatives as errors. A soft KL would have told the student "these three are interchangeable"; the
hard target tells it "only this one is correct," manufacturing a sharp, partly spurious gradient.
So I expect hard-target imitation to be simultaneously starved of signal (it ignores the soft
structure) and over-confident where the teacher itself is uncertain (it sharpens flat regions into
one-hot demands). On reasoning chains, where the teacher's per-token distribution is often genuinely
multi-modal but the *answer* depends on getting the rare load-bearing tokens right, that combination
should leave accuracy mediocre.

There is a second, subtler reason this should sit low, specific to the capacity gap. The student is a
0.5B base model; the teacher is a 7.6B math-tuned instruct model. A hard target says "match my
argmax" with no temperature softening, which means the student is asked to become as sharp as the
teacher's mode at every position — but the student cannot represent the teacher's reasoning, so
forcing one-hot confidence on the teacher's top token, with no credit for the surrounding mass,
gives it nothing to interpolate toward when it is wrong. Soft losses degrade gracefully (a near-miss
still earns partial credit through the distribution); the hard target is all-or-nothing at the token
level. Under a large capacity gap, all-or-nothing token supervision is exactly the brittle regime.

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
zero. Critically, there is **no temperature scaling on either side**: the imitation loss is plain
`log π_θ(ã|s)` for the teacher's action `ã`, and dividing logits by a temperature would change the
target only through the argmax (which is temperature-invariant anyway) while needlessly rescaling the
student's logits, so I leave both untouched. This is one place the task's loss differs from a generic
soft-distillation body — the temperature knob the signature exposes is simply unused here, on
purpose, because the hard target has no soft distribution for temperature to soften.

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

Putting it together, the step-1 edit is the hard-target fill: argmax the teacher to get per-token
labels, mask them with the trainer's `-100`, take masked token-level cross-entropy of the student
against those labels, reduce by `batchmean`. No KL, no temperature, no soft distribution. The full
scaffold body is in the answer.

Now reason about what this floor must do, because that is the entire reason to run it. The three
benchmarks span difficulty: GSM8K (grade-school, 1319 problems) is the most statistically reliable
read; MATH-500 (competition-style, 500 problems) is harder; AMC23 (40 problems, avg@8) is so small
that two or three solved problems swing the number, so I will treat it as noise and read the story
off GSM8K and MATH-500. I expect hard-target imitation to *function* — the student will learn to
imitate the teacher's preferred tokens and will solve a meaningful fraction of GSM8K, because grade-
school arithmetic chains are short and the teacher's argmax is usually unambiguous there. But on
MATH-500, where chains are longer and the teacher's per-token distribution is more often multi-modal,
the discarded soft structure and the spurious sharpening of flat regions should cost it: I expect it
to land at or near the bottom of the ladder on both reliable metrics, beating nothing decisively and
plausibly trailing every soft-distribution loss. The diagnosis that would point to the next rung is
already written into the construction: if the hard target underperforms, the fix is to stop throwing
away the teacher's distribution — to give the student *soft* per-token targets that preserve the
relative mass on the alternatives, starting with the cheapest faithful way to summarize that
distribution rather than collapsing it to its mode.
