I have to fill the `run_attack` body, and the harness gives me full model access — gradients are on the
table — but I want to start from the cheapest, most conservative thing that is still a real sparse
attack, so I know what the floor looks like before I spend any complexity. The cheapest realistic threat
model never differentiates the network at all; it only feeds in an image and reads back the probability
vector. So the first rung is the *black-box, query-only* attack that needs nothing but `model(x)`. Among
the sparse attacks that live in that regime, the one that is structurally the leanest — no population, no
gradient estimation, one forward pass per probe — is the pixel-rearrangement attack, and that is where I
start.

The idea is worth stating precisely because it is what makes this rung so cheap. A natural image already
contains, somewhere inside it, a huge palette of pixel values — bright pixels, dark pixels, every hue the
scene happens to hold. So to drop a high-contrast, out-of-place pixel onto a sensitive location, I do not
need to *synthesize* a new color; I can *copy* an existing pixel's value onto the target. That single
observation collapses the whole search. The other black-box sparse attack I know, the differential-
evolution one-pixel attack, has to search a continuous `[0,1]^c` color cube *per modified pixel* on top
of choosing the location — three real dimensions of color for every pixel — and it evaluates an entire
population every generation, hundreds of forward passes before it even takes a step. Rearranging deletes
the color search entirely: I am no longer asking "what new color do I paint this pixel," only "which
existing pixel's value do I copy here." The continuous per-pixel color dimension vanishes, the search
becomes positions-times-positions, and I get two things for free that I would otherwise have to enforce
by hand — every value I write is by construction already a legal value in `[0,1]`, so I never clip or go
out of range, and the `L0` count is clean because the only pixels that change are the destinations I
overwrite.

There is a second reason this is a good *floor* and not just a cheap one: it borrows the lesson from the
query-efficient dense black-box attacks that *localized* updates work far better than scattered ones,
because convolutional networks are unusually sensitive to a few *neighboring* pixels changed together with
high contrast. So the rearrangement does not move random scattered pixels; it copies a small *contiguous*
source patch. That keeps the attack honest as a baseline — it is using the one structural prior that black-
box sparse attacks are known to exploit — while still being the leanest possible instantiation of it.

So the candidate I sample is: take a small contiguous *source patch* of the image (an origin plus tiny
side lengths), and for each source pixel pick a *destination* coordinate where its value gets written;
overwrite the destinations, leave everything else untouched. Because the patch is small only a few pixels
change per step, which keeps me sparse; because I copy existing values I stay in range. The destinations
I choose by the cheapest possible map — `pixel_mapping="random"`: each source pixel goes to a uniformly
random target location, no extra queries spent deciding where. The outer search is plain accept-if-
improves random search with restarts: keep a committed image, sample a candidate built off it, evaluate
the true-class probability, record a candidate only if it lowers that probability, and at the restart
boundary move the committed image to the best recorded candidate; stop the instant the prediction flips.
One evaluation per probe, no population — this is the leanest sparse attack there is, and that is exactly
why it belongs on the bottom rung.

Now I have to be honest about how this rung is configured *in this task*, because the scaffold's choices
are not the paper's defaults and they matter for what I should expect. The harness wraps the
`torchattacks` implementation directly, and the fill I land is the literal call:
`Pixle(x_dimensions=(1, 2), y_dimensions=(1, 2), pixel_mapping="random", restarts=3, max_iterations=5,
update_each_iteration=False)`. Read those numbers against the budget of 24 pixels and the picture is
stark. The patch side lengths are drawn from `(1, 2)` in *both* dimensions, so each sampled patch is one
to four pixels — a single pixel up to a `2x2` block. With `restarts=3` and `max_iterations=5` the attack
gets, at most, on the order of fifteen candidate evaluations per image before it gives up, and it commits
the running best only at the three restart boundaries (`update_each_iteration=False`, the restart-
iterative variant, not the greedy one). Fifteen random tiny patches, copied to random destinations, on a
*robust* model. That is an extremely thin search. The budget allows 24 changed pixels, but this
configuration will rarely even approach it — most candidates touch one to four pixels and the loop ends
long before twenty-four are usefully placed. This is the floor by design: the leanest attack, given the
fewest queries, against the hardest targets.

Let me reason about what this rung must do, because the entire point of running it is to set the bar the
next rung has to clear. Random search throws away everything it learns — each probe is independent, it
never uses the geometry of past probes to concentrate where the boundary turned out to be close. On an
*undefended* network that is fine: vulnerable pixels are dense, a random patch lands on one quickly, and
even a thin search saturates. But these are `L2`-adversarially-trained models, whose entire training
objective was to *flatten* the loss surface in a neighborhood of each input — to make exactly the local,
few-pixel sensitivities that a random-search attack relies on as scarce and as small as possible. So a
scattershot of fifteen tiny random patches is searching for rare needles in a haystack that was
deliberately built to hide them. I expect this rung to almost entirely fail: a near-zero success rate on
all three models, with whatever handful of successes come from the easiest samples where even a robust
model left a single fragile pixel. The random mapping is the weakest link — it spends no queries
deciding *where* to put the out-of-place pixel, so against a robust model it is essentially hoping to hit
a sensitive location by luck, fifteen times, and luck against a flattened surface is thin.

That diagnosis already points at the next rung. If random rearrangement fails because it never uses any
signal about *which* pixels matter, the obvious next move is an attack that *does* compute a per-pixel
importance — even a crude one — and spends its budget on the pixels that score highest, instead of
sampling locations blind. That is the direction the ladder climbs from here. But first I need the number,
because "almost entirely fails" has to become a measured floor before the next rung can be said to beat
it.

To make this concrete as the literal scaffold edit: where the default `run_attack` returned the clean
image untouched (ASR 0 by construction), this rung instantiates `torchattacks.Pixle` with the tiny-patch,
random-mapping, restart-iterative configuration above and returns `attack(images, labels)`. The
`pixels`, `device`, and `n_classes` arguments are unused — Pixle's own patch-size and budget knobs govern
sparsity, and the harness validates the `L0` count after the fact, rejecting any sample that somehow
exceeds 24. The full scaffold module is in the answer. What I am watching for: a success rate close to
zero across `Rebuffi-R18-L2`, `Augustin-L2`, and `Engstrom-L2`, confirming that a blind, query-starved
sparse search is the wrong tool against robust models — and giving me the floor the rest of the ladder is
measured against.

The causal chain in one breath: I want the cheapest real sparse attack as the floor → the leanest is
black-box, query-only, and the rearrangement trick (copy existing pixel values instead of searching a
color cube) makes it leaner still, killing the continuous color dimension and guaranteeing in-range,
budget-clean candidates → wrapped in accept-if-improves random search with restarts, configured here with
one-to-four-pixel patches, random destinations, and ~15 evaluations per image → against `L2`-robust
models whose surfaces were trained flat, a blind query-starved search should almost entirely fail,
setting a near-zero floor and pointing the next rung at *using* per-pixel importance instead of guessing
locations.
