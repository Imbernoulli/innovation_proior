PowerSign and AddSign both fell short of AdamW at every column, both scales — consistent with my
suspicion that a fixed-operand, fixed-momentum search space caps what can be found, because it
never lets the search touch how the memory buffer itself is tracked, only how a gradient combines
with a buffer someone else already decided how to update. Rather than argue the attribution by
introspection, I built a freer search space and let its own results speak: an update rule as an
arbitrary short sequence of assignment statements over the weight, gradient, a fixed pair of extra
state buffers (matching AdamW's own signature, so memory never exceeds Adam's), and the learning
rate, using primitive math functions — strictly more expressive than a fixed-operand tree, since
nothing stops a mutation from touching the statement that *updates* a buffer, not just the
statement that consumes it. I warm-started the population with AdamW itself rather than random
noise.

Two facts told me this space needs directed search, not just freedom. Even two million randomly
sampled programs on a cheap proxy task, the best one found is still meaningfully worse than
AdamW — good rules are needle-in-haystack sparse. And regularized evolution — tournament
selection, mutate the best of a random few (insert/delete/modify one statement) — significantly
outperforms both an AdamW-hyperparameter-tuning baseline and random sampling, even when those two
get four times the compute. Mutations touching one statement at a time forced me to tolerate
redundant statements during search (useful multi-statement structures sometimes need
individually-useless stepping stones), which I offset with an abstract-execution pass: infer
shapes before ever running a program, hash what a program actually computes to cache duplicate
evaluations, flag statements that don't affect the output. Across five 300K-program runs, about
70% of statements end up flagged redundant (removing them shrinks a program roughly threefold),
and the cache catches close to 90% of would-be evaluations. Both numbers matter for what comes
next: the raw surviving program is mostly dead text, and reading it before stripping that text
would be reading noise.

After redundant-statement removal, what's left still isn't clean — a `clip` and an `arcsin` on the
incoming gradient, a chain computing `m*m`, then `sqrt` of that, then dividing `m` by the result,
plus a weight-decay term, an lr scale, and two chained `interp` calls tracking two buffers. The
arithmetic chain is checkable without touching training at all: m² is nonnegative, so √(m²) = |m|,
so `m / sqrt(m*m)` is exactly `m / |m|` = sign(m) everywhere m isn't exactly zero (which an
accumulated float buffer essentially never is). Three statements collapse to one call: the search
independently landed on signing a momentum-derived quantity, the same family signSGD-momentum
lives in.

I propose testing that reading literally, as the simplest hypothesis consistent with what survived
pruning, before trusting the raw program's second, differently-tuned `interp` call as anything
more than an artifact: collapse the two `interp` calls to *one* momentum EMA at a single rate β,
then sign it — `m = interp(g, m, β); update = sign(m)`. The raw program's own two constants —
roughly 0.9 on one call, roughly 1.1 on the other — hand me two natural candidates to try for that
single β: 0.9, the rate ordinary momentum uses everywhere else, and 0.99, closer to the second raw
constant and implying about ten times the memory horizon of 0.9. I have no principled reason yet
to prefer one, or to believe collapsing to one constant loses anything — if it doesn't, I ship
something even lighter than AdamW and I'm done. What decides the next move is not just whether
these beat AdamW but how they compare to PowerSign/AddSign: beating rung 1 would confirm that
freeing the momentum-tracking rule itself — not just deepening the expression tree — is where the
real gain sits.

```python
def train(w, g, m, lr):           # ONE extra state buffer, not two
    m = interp(g, m, beta)        # (1-beta)*g + beta*m
    update = sign(m)
    update = update + w * weight_decay
    update = update * lr
    return update, m
```

`Ablation_0.9` (β=0.9) and `Ablation_0.99` (β=0.99), same protocol as rung 1, own lr/λ tuned, lr
scaled down from AdamW's since sign(·) is exactly ±1 per coordinate, a much larger step than
AdamW's m/√v.
