PowerSign and AddSign both fell short of AdamW on every column, at both scales: 77.36/83.39/65.17
and 77.37/83.36/64.52 against AdamW's 78.89/84.61/66.73 on ViT-S/16, and the same pattern at
ViT-B/16. That settles the cheap question and opens the expensive one. My prediction going in was
that a fixed-operand, fixed-momentum search space caps what can be found, because it never lets
the search touch how the memory buffer itself is tracked — only how a gradient combines with a
buffer someone else already decided how to update. The result is consistent with that prediction,
though it can't distinguish it from other explanations (maybe the tree depth was too shallow,
maybe RL search is just weaker than other search strategies). I'm not going to try to settle that
attribution question by introspection; I'm going to build a search space that removes the
restriction I'm suspicious of, and let its own results speak.

So: build a program representation where an update rule is an arbitrary short sequence of
assignment statements over arrays — weight, gradient, a fixed number of extra state buffers, and a
learning rate — using primitive math functions, exactly the way I'd write pseudocode. This is
strictly more expressive than a fixed-operand tree: nothing stops a mutation from touching the
statement that updates a buffer, only the statement that consumes it. I fix the signature to
match AdamW's — two extra state variables, both init zero — so whatever the search finds has
memory no larger than Adam's, which matters if I ever want this adopted. I warm-start the
population with AdamW itself, written as a program, rather than random noise, so the search
explores outward from something already known-good rather than searching blind.

Two facts tell me this search space needs to be *directed*, not just *free*. First: even sampling
two million random programs on the cheap proxy task, the best one found is still meaningfully
worse than AdamW. That's not a close call — it means good rules are needle-in-haystack sparse in
this space, and undirected sampling at any budget I could afford is not going to find one.
Second: I run regularized evolution — keep a population, repeatedly tournament-select a few
candidates, mutate the best of them (insert, delete, or modify one statement), and compare it
against two baselines given *four times* the compute budget: hyperparameter tuning of AdamW
(mutating only constants) and the same random sampling as above. The evolutionary search
significantly outperforms both, even with a fourth of their compute. So directed mutation over
whole programs is finding something neither brute-force sampling nor constant-tuning-alone can
reach. That's the case for running this search at all; it doesn't yet tell me what the winning
program looks like.

Mutations touching one statement at a time create a bookkeeping problem worth naming, because it
shapes how I read the raw output later: a useful multi-statement structure sometimes needs several
individually-useless-looking statements as stepping stones, so I have to let the population carry
redundant statements rather than pruning them away during search. That tolerance costs program
bloat, which I offset with an abstract-execution pass — infer shapes before ever running a program
(reject malformed mutants), hash what a program actually computes so semantically-duplicate
candidates hit a cache instead of a fresh evaluation, and flag statements that don't affect the
output. Concretely, across five search runs of 300K programs each, about 70% of statements end up
flagged redundant by the end (removing them shrinks a program roughly threefold), and the cache
catches close to 90% of would-be evaluations, cutting search cost by an order of magnitude. Both
numbers matter for what I do next: the raw program that survives selection is going to be
*mostly redundant text*, and reading it as written — before stripping the dead weight — would be
reading noise, not signal.

Let me look at what the search actually surfaced, after redundant-statement removal (the
functionally-inert ~70% is gone, but I haven't touched anything that still affects the output).
It's still not clean: there's a `clip` and an `arcsin` applied to the incoming gradient, a chain
that computes `m*m`, then `sqrt` of that, then divides `m` by the result, a weight-decay term, an
lr scale, and two chained `interp` calls that track two buffers, which I'll call `m` and `v` for
now since that's how the search happened to name them. Read the arithmetic chain first, since it's
checkable without touching training at all: m² is nonnegative, √(m²) = |m|, so `m / sqrt(m*m)` is
exactly `m / |m|`, which is element-wise sign(m) everywhere `m` isn't exactly zero (and in
floating point, an accumulated momentum buffer essentially never lands exactly on zero). So
three of the raw statements collapse to one call: `sign(m)`. That's a real structural finding, not
noise — the search independently arrived at signing a momentum-derived quantity, the same family
signSGD-momentum lives in, without me telling it to.

That reading — "it signs an accumulated buffer" — is the natural, minimal way to interpret what's
left once the dead arithmetic is stripped, and it's the hypothesis I want to test directly rather
than assume. If signing an accumulated buffer is the whole story, then the two `interp` calls
should be collapsible to *one* momentum EMA at *some* rate β, updated and then signed, exactly the
signSGD-momentum shape: `m = interp(g, m, β); update = sign(m)`. The raw program's own two
constants — roughly 0.9 on one `interp` call and roughly 1.1 on the other — hand me two natural
candidate values to try for that single β: 0.9, matching the rate ordinary momentum uses
everywhere else I've seen it, and 0.99, closer to the second of the two raw constants and
implying a memory horizon (1/(1−β)) about ten times longer than the usual 0.9. I don't yet have a
principled reason to prefer one over the other, and I don't yet have a principled reason to
believe collapsing to one constant loses anything at all — the two-`interp` structure could just
be an artifact of how mutation happened to write it, with no functional consequence, in which case
the simplest single-buffer read is the right final answer and I can stop here with something even
lighter than AdamW.

So the test I'm committing to for this rung is exactly that: strip the dead cosh/arcsin/clip
(deferred to later scrutiny — they're candidates for removal but not what's being tested here),
take the sign-of-accumulated-momentum reading as literally as the raw program supports, and train
two variants that differ only in β — `Ablation_0.9` and `Ablation_0.99` — under the identical
protocol as rung 1: ViT-S/16 and ViT-B/16 from scratch, RandAugment + Mixup, each variant's own
lr/λ tuned on a log scale. One thing I already know has to change relative to AdamW's tuning: the
raw output of `sign(·)` is exactly ±1 per coordinate, which is a much larger step than AdamW's
m/√v ever produces, so both variants need a substantially smaller learning rate than AdamW's — I
carry that adjustment into the tuning sweep rather than treating it as a separate finding.

What decides the next move is not just whether these beat AdamW, but *how* they compare to rung 1.
If a single-constant sign-of-momentum rule already beats PowerSign and AddSign, that's independent
confirmation that freeing the search space over how momentum itself gets combined and signed —
not just widening the tree depth — is where the real gain sits, since I removed exactly the
restriction I suspected (m is no longer handed in fixed; the whole momentum-then-sign structure is
now searched). Whether it also clears AdamW is the open question this rung actually exists to
answer, and I'm not going to guess at it: if a single β is enough, I have my answer and a lighter,
one-buffer rule than AdamW; if it isn't, the raw program's insistence on *two different* constants,
which I've been treating as possibly cosmetic, becomes the thing to take seriously instead of
explaining away.
