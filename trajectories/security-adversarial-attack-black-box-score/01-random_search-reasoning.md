The attack is the whole point, but it runs inside a fixed oracle, and the crudest thing I can do with
that oracle is the floor I have to start from — so the pain to begin with is just turning forward-only
score queries into a misclassification at all, with no cleverness about *where* or *how big* the moves
are. I am locked out of the gradient: I send the oracle an image and read back logits, nothing else.
There is exactly one lever — propose a perturbed image, query, and decide whether to keep it — and the
floor uses that lever in the most naive way available.

Let me write down the objective so I do not fool myself. For a correctly classified `(x, y)` I want
`argmax_k f_k(x_adv) != y` with `||x_adv - x||_inf <= eps` and `x_adv in [0,1]`. The cheapest scalar
that tracks progress toward that, using only the logits I already pay for, is the **correct-class
score** `f_y(x_adv)`: it starts high (the model is confident in `y`) and the prediction flips once some
other class overtakes it. I do not even need the full margin to hill-climb — I just need a number that
goes down as I get closer, and `f_y` is the simplest such number, one `gather` on the logits. So the
floor minimizes `f_y` by greedy accept-if-better: keep a candidate only when it lowers the correct-class
logit.

Now the only design questions left are the proposal and the budget bookkeeping, and the floor answers
both as bluntly as possible. The proposal: draw uniform noise in a small box, `uniform(-step, step)`
with `step = eps/2`, add it to the current best, then re-project into the feasible set —
`clamp(x + clamp(cand - x, -eps, eps), 0, 1)`. This is the textbook random-search move: a fixed-radius
isotropic perturbation of the current iterate, evaluated and kept only if it improves. There is no
structure in it at all — no notion that the model is convolutional, no notion that successful `L_inf`
perturbations sit at corners, no decay of the step size, no concentration of the change into a region.
It scatters small nudges across all `C*H*W` coordinates at once. The `step = eps/2` choice means a typical
proposal lands well inside the box rather than on its boundary, so the floor is not even spending its
full per-component budget — it is making timid interior moves.

The budget bookkeeping is where the floor quietly throws away most of its allowance, and this is the
detail that decides its number. The harness gives a per-sample budget `n_queries` (the runs use 1000),
and the oracle exhausts the *whole batch* to failure the instant the running count crosses
`batch_size * n_queries`. The floor does not walk anywhere near that line: it runs a fixed
`n_steps = max(1, min(n_queries, 64))` iterations, each costing one query per sample, and it queries
*only the candidate* each step — it does not re-query the current best, because it carries the best
score forward from the previous accept. So with `n_queries = 1000` it caps itself at **64** candidate
queries plus one initial query of the clean image, ~65 total. That is the `avg_queries = 65` I should
expect to read back: the floor deliberately leaves ~935 of every 1000 queries on the table. It is the
weakest configuration by construction — minimal moves, no structure, and a self-imposed ceiling far
below the real budget.

There is one more subtlety worth naming because it is the floor's only piece of vectorized care. The
accept rule operates per sample: I compute the candidate's correct-class score for every image in the
batch, form `improve = cand_score < best`, and update only the rows where it holds via a masked
`torch.where`. So a single batched query advances every still-improving sample at once, and an image
whose candidate got worse simply keeps its previous best — the budget is shared across the batch but the
decision is independent per image. That is the right structure; the floor just refuses to use it for
more than 64 steps or with anything smarter than uniform noise.

Now reason about what this floor must do, because running it is the entire point of step 1. Greedy
random search with isotropic interior noise in a ~3000-dimensional space (CIFAR is `3*32*32`) is the
classic high-dimensional failure: a random direction is almost orthogonal to whatever direction would
actually lower `f_y`, so most proposals are rejected, and the few accepted ones make tiny progress.
With only 64 such steps the perturbation barely moves off the clean image. On the *easier*
(model, dataset) pairs — where the decision boundary is close and even crude noise stumbles across it —
the floor should flip a fair fraction of images. On the harder pairs — a more robust architecture, or
CIFAR-100 where the correct class has 99 competitors and the margin is structured — the same 64 timid
moves should mostly fail, and `asr` should sag well below half. The split across scenarios will be wide
and the mean modest, precisely because nothing in the floor adapts to the model or spends the budget it
was given.

That diagnosis already points at step 2. The floor's two crippling choices are economic and structural:
it caps itself at 64 queries when 1000 are available, and it makes unstructured interior nudges when the
geometry says to sit on the boundary and the model says to concentrate the change. The first thing I
will do is stop leaving the budget unused and stop estimating nothing — but the cleaner, more principled
next move is to go back to the one black-box primitive that *does* reconstruct a descent direction from
forward queries alone: estimate the gradient from a simultaneous random perturbation and descend it.
That is the next rung. The distilled floor module — the literal scaffold edit — is in the answer.
