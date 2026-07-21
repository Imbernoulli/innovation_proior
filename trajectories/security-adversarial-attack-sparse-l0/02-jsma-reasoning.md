The Pixle floor came back almost exactly as I feared, and the numbers tell me *why*. `asr = 0.0067` on
`Rebuffi-R18-L2`, `0.020` on `Augustin-L2`, `0.0067` on `Engstrom-L2`, mean `~0.011`. Read as raw counts
these are integers over the sample size: `1/150 = 0.0067`, `3/150 = 0.020`, so Pixle flipped one image on
Rebuffi, three on Augustin, one on Engstrom — five successful attacks in roughly `450` attempts. This is
not noise around a real signal; it is a search that essentially never found anything, sitting one or two
counts off zero. The diagnosis is precise: Pixle spends *no* information deciding which pixels to touch —
with `pixel_mapping="random"` it copies a value onto a uniformly random destination, and fifteen blind
throws almost never land on a fragile pixel of an `L2`-trained model whose surface was flattened to make
those sensitivities scarce. Augustin being softest (three flips versus one) is consistent, but three-of-150
against one-of-150 is essentially-never against essentially-never; the ratio is real but it is a ratio of
tiny integers, not signal I can build on. The lesson is unambiguous: against robust models I cannot guess
locations. I have to *compute* which pixels matter.

So this rung introduces a per-pixel importance signal. The harness grants gradients, so I no longer have to
stay black-box — and I should choose deliberately among the ways a gradient becomes a *sparse* attack.
Reading a single loss gradient `grad_x J` and keeping its top-`24` coordinates is the dense-attack reflex,
and I can reject it on a shape argument: `J` collapses the whole `10`-dimensional output and the label into
one scalar, giving one direction over all `3072` features — it says how to move every pixel a little for a
dense step, but it carries no per-class structure, so its top coordinates are pixels sensitive *on average*,
not pixels that specifically trade true-class score for another class. That is the wrong object for
*selecting* a tiny support. Jumping straight to a pure evaluative search is the black-box regime Pixle just
came from, skipping the question of whether the gradient is worth anything. So the disciplined step is to
use the gradient through the object built for *selection*: the Jacobian `dF_j/dx_i`, the forward derivative
— per-feature *and* per-class, and crucially signed. With it I can ask, per pixel, the targeted question a
single loss gradient cannot express: does pushing this pixel raise some chosen class while dragging the
others down?

Each Jacobian row `j` is one backward pass seeded at logit `j`, so the whole `n_classes x num_features`
matrix costs ten backward passes on CIFAR-10 — cheap in absolute terms. But the greedy loop rebuilds it
every iteration because the network is non-linear and sensitivities shift after each move; with the budget
forcing twelve iterations (derived below), the attack spends `10*12 = 120` backward passes per image. That
is an order of magnitude more model work than Pixle's ~15 forward passes, the price of *directed* placement.
Whether the trade pays is exactly what this rung measures.

The saliency structure is the point. To flip toward a target class `t`, a useful feature must do two things
when increased: raise the target logit (`dF_t/dx_i > 0`) and lower the rest (`sum_{j!=t} dF_j/dx_i < 0`). A
feature failing either scores zero; among those passing both, the score is the *product*
`(dF_t/dx_i) * |sum_{j!=t} dF_j/dx_i|`. Product, not sum, and it matters: a feature with `dF_t = 5.0` but
others-sum `-0.1` shoves the target up hard while barely suppressing competitors — a wasted pixel on a
robust model — yet a *sum* rule scores it `4.9`, far above a balanced feature with `dF_t = 1.0`, others-sum
`-1.0` scored `0.0`. The product scores them `0.5` versus `1.0`, correctly preferring the balanced trade; it
demands *both* factors be substantial. One subtlety pins the choice: the two conditions carry independent
information only on the *logits*, not the softmax probabilities — probabilities sum to one, so
`dp_t/dx_i > 0` mechanically forces `sum_{j!=t} dp_j/dx_i < 0`, making the second gate vacuous, and the
softmax's saturated derivatives flatten the ranking on top. On the logits the outputs are unconstrained, so
requiring both signs genuinely selects the rare favorable features. And because a single feature is rarely
favorable on both axes — most pixels help the target but slightly help a competitor too, and get gated out —
the search modifies *two* features at a time, letting one pixel's strongly-negative others-sum compensate
the other's slightly-positive one, so a favorable *pair* forms where neither member could alone. Each
selected feature is saturated to its extreme in one shot, dropped from the domain, and the Jacobian
recomputed; stop when the prediction reaches the target or the budget is spent. This also tells me where the
method will still fail — when *no pair* clears the combined gate, the local-optimum stall I expect on the
hardest model.

Now the two ways the harness's JSMA configuration diverges from the textbook, both load-bearing. First,
`torchattacks.JSMA` is targeted-only — it needs a target class per sample. The textbook `(y+1) % n_classes`
is a weak target: an arbitrary neighbor may be a mid-pack class the attack must overtake everything above
*and* stay ahead of everything below to reach, needlessly constrained, and "harder target" means "fails
more" on a robust model. But my metric is untargeted — the image is fooled the instant *any* class
overtakes the true one — so I do not need to *reach* the target. Aiming at the *least-likely* class, the one
the model rates lowest, is the most aggressive way to *collapse the true class*, because the both-signs
saliency it induces suppresses the true logit hardest, and some easier class usually overtakes first and
triggers the untargeted flip before the least-likely target is ever reached. So
`set_mode_targeted_least_likely` turns a targeted-only primitive into the strongest untargeted proxy, for
any `n_classes`.

Second, the `L0` budget — the failure the harness explicitly guards against. JSMA counts in *feature* space:
it computes `num_features = C*H*W = 3072` and sets `max_iters = ceil(num_features * gamma / 2)`, modifying
two features per iteration, so it can touch `num_features * gamma` features total. A spatial pixel counts as
changed if *any* of its three channels moves, so touching that many features can cover up to that many
*distinct spatial pixels*. The fill sets `gamma = pixels / (C*H*W) = 24/3072 = 0.0078125`, giving
`max_iters = ceil(24/2) = 12` iterations and `24` features touched, upper-bounding distinct spatial pixels
by `24` — exactly the budget, possibly lower if two features share a pixel. The landmine: copy a constant
`gamma` from a `28x28`-grayscale reference where `10/1024 = 0.009766` reads naturally, plug it into this
`3072`-feature space, and `max_iters = ceil(15) = 15` iterations touch `30` features — six over budget. The
harness validates the `L0` count channel-wise *after* the attack and rejects any over-budget sample as a
failure, collapsing the measured ASR regardless of how well the saliency worked. Getting `gamma` right is
the difference between a valid attack and one thrown out wholesale. So the edit is
`JSMA(model, theta=1.0, gamma=pixels/num_features)`, then `set_mode_targeted_least_likely(quiet=True)`.

One more choice: `theta = +1`, saturating each chosen feature *up* rather than down. A channel lives in
`[0,1]` and the two directions are not symmetric — driving it to `1.0` injects a bright, high-contrast
impulse a conv filter reads as strong positive activation, while driving it to `0.0` mostly *removes*
signal, easily inpainted by the surrounding context adversarial training taught the model to rely on. Adding
intensity produces a more confident misclassification than removing it, so `+1` extracts the most per-pixel
leverage under a budget that charges nothing for magnitude, and it keeps the saliency's sign convention (the
gate was written for *increasing* a feature) consistent rather than fighting it.

I could instead just retune Pixle — crank restarts and patch sizes to cover more of the grid — but that
scales queries linearly for a search that stays *blind*, testing locations with no reason to believe they
are fragile; to inspect even half the `1024` locations needs hundreds of probes per image. JSMA reads the
sensitivity of all `3072` features in ten backward passes and spends its `24` pixels on the highest-scoring
ones. On a surface where fragile locations are a fraction of a percent, directed ranking is the only route
that could plausibly beat luck.

So my expectation against the Pixle floor: JSMA *uses* a per-pixel importance signal instead of guessing, so
it should beat `0.011`. But I am not expecting a large number, and the reason is structural. JSMA's saliency
is a *first-order, local* quantity read at the clean image, and the greedy walk commits to the best pair,
saturates, recomputes, never reconsidering — and the first-order signal is exactly what `L2` training
hardened against, so the both-signs gate finds few features that clear it and the walk stalls in local
optima where a *different* support would have worked. On the hardest model the least-likely target may
simply be unreachable within 12 iterations. So I expect JSMA to clear the floor but land low, in the
few-percent range — single-digit-to-low-double-digit counts per model rather than Pixle's one-to-three —
with the per-model ordering no longer guaranteed to track Pixle's, since a first-order signal is suppressed
differently by each model's training than blind luck is. If it comes back only marginally above Pixle, the
next rung's diagnosis is written: greedy first-order saliency is too brittle on robust surfaces, and I need
either a richer gradient move that does not commit irrevocably, or a search that can reconsider.
