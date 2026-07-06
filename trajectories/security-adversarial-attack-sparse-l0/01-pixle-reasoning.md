I have to fill the `run_attack` body, and the harness gives me full model access — gradients are on the
table — but I want to start from the cheapest, most conservative thing that is still a real sparse
attack, so I know what the floor looks like before I spend any complexity. The cheapest realistic threat
model never differentiates the network at all; it only feeds in an image and reads back the probability
vector. So the first rung is the *black-box, query-only* attack that needs nothing but `model(x)`. Among
the sparse attacks that live in that regime, the one that is structurally the leanest — no population, no
gradient estimation, one forward pass per probe — is the pixel-rearrangement attack, and that is where I
start.

Before I commit to that, let me actually enumerate what lives in the query-only sparse regime, because
"leanest" has to mean something measurable, not a gut call. Three candidates present themselves. One is a
differential-evolution one-pixel search: it holds a population of candidate perturbations and, every
generation, forms children by scaled coordinate differences and keeps the ones that lower the true-class
probability. Two is a coordinate-wise saliency estimator that probes each pixel by finite differences to
rank importance, then spends the budget on the top ranks. Three is the rearrangement attack, which copies
existing pixel values around and keeps changes that help. I can rank these by the one currency that
matters at the floor, forward passes spent before the search even takes a productive step. The
finite-difference ranker is immediately disqualified: on CIFAR-10 the image is `3*32*32`, so there are
`1024` spatial locations, and estimating a per-pixel importance by one two-sided probe each already costs
`~2048` forward passes *before* placing a single adversarial pixel — that is not a floor, that is a
mid-tier query budget. Differential evolution is cheaper per useful step but still evaluates a whole
population every generation, so it too spends tens to hundreds of forward passes before it commits. The
rearrangement attack spends exactly one forward pass per probe and commits or discards immediately. On the
"forward passes before first commit" axis the ordering is unambiguous — rearrangement `<` DE `<`
finite-difference ranking — so the rearrangement attack is the floor, and the other two are rungs I can
climb to later if and only if a cheaper thing has already been measured to fail.

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

Let me put a number on that collapse, because "the color dimension vanishes" is the whole reason this is
the floor and I want it concrete. If I were to encode a `d`-pixel perturbation the way the differential-
evolution attack does, each modified pixel is a five-tuple — two location coordinates plus three color
channels — so at the full budget `d = 24` the search variable is a `5*24 = 120`-dimensional continuous
vector, and three-fifths of those `120` dimensions, `72` of them, are color coordinates that a
continuous optimizer has to hunt through the `[0,1]^3` cube per pixel. Rearrangement erases all `72`: a
candidate is fully specified by a source-patch origin (two integers), the patch side lengths (two small
integers), and a destination map. With the map chosen by the cheapest possible rule — copy each source
pixel to a *uniformly random* target, spending zero queries to decide where — the effective free search
dimension is on the order of *four* small integers per probe, against `120` continuous reals for the DE
encoding. That is the mechanical sense in which this is the leanest instantiation: I have traded a
`120`-dimensional continuous optimization for what is essentially a four-integer lottery ticket per
forward pass, and the harness will never reject a ticket for going out of range or over budget because
both properties are structural, not enforced.

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

I should pause and ask whether a slightly richer *mapping* would be a better floor, because the random map
is the part I am most tempted to improve. The alternative in this family maps each source pixel to the
destination whose current value is *most different* from the source, a similarity-guided placement that
tries to maximize contrast. It is genuinely more effective per probe — but it is not free: scoring
destinations by similarity requires reading pixel values and, in the query-scoring variants, extra forward
evaluations to decide *where*, which is precisely the cost the floor is supposed to avoid paying. If I let
the floor spend queries deciding where to place pixels, then a later rung that *computes* placement from
the gradient has nothing to beat — the whole point of the bottom rung is to isolate the effect of using
*no* placement information at all. So I deliberately keep the random map. The same logic rejects bumping
`restarts` from `3` to, say, `30`, or the patch sizes from `(1,2)` to larger blocks: each of those buys
real success, but each also stops being the floor, converting the measurement from "what does a blind,
query-starved search get" into "what does a moderately-resourced blind search get," and I need the
former as the anchor. The leanest configuration is not an accident I am stuck with; it is the choice that
makes the floor *mean* something.

Now I have to be honest about how this rung is configured *in this task*, because the scaffold's choices
are not the leanest method's own defaults and they matter for what I should expect. The harness wraps the
`torchattacks` implementation directly, and the fill I land is the literal call:
`Pixle(x_dimensions=(1, 2), y_dimensions=(1, 2), pixel_mapping="random", restarts=3, max_iterations=5,
update_each_iteration=False)`. Read those numbers against the budget of 24 pixels and the picture is
stark. The patch side lengths are drawn from `(1, 2)` in *both* dimensions, so each sampled patch is one
to four pixels — a single pixel up to a `2x2` block. The area of a patch is the product of two side
lengths each uniform on `{1,2}`, so the four equally-likely shapes are `1x1`, `1x2`, `2x1`, `2x2` with
areas `1, 2, 2, 4`; the expected patch touches `(1+2+2+4)/4 = 2.25` pixels. With `restarts=3` and
`max_iterations=5` the attack gets, at most, on the order of fifteen candidate evaluations per image
before it gives up, and it commits the running best only at the three restart boundaries
(`update_each_iteration=False`, the restart-iterative variant, not the greedy one). Fifteen random tiny
patches, copied to random destinations, on a *robust* model. That is an extremely thin search. The budget
allows 24 changed pixels, but this configuration will rarely even approach it — most candidates touch one
to four pixels and the loop ends long before twenty-four are usefully placed. This is the floor by design:
the leanest attack, given the fewest queries, against the hardest targets.

Let me convert "thin search" into an actual coverage figure, because that is what tells me whether to
expect single-digit successes or essentially none. Fifteen probes, each landing an average of `2.25`
pixels at uniformly random destinations, place about `15 * 2.25 = 33.75` pixel-writes over the run. Those
writes collide, so the *distinct* locations touched is smaller: with `1024` locations and `~34`
independent uniform draws, the expected number of distinct positions probed is
`1024 * (1 - (1 - 1/1024)^34) ~= 1024 * (1 - 0.9668) ~= 34` before collisions, or a hair under that once
overlaps bite — call it roughly `33` distinct locations. That is about `33/1024 ~= 3.2%` of the image
surface examined across the entire attack. So the mechanism is transparent: the attack blindly inspects
about three percent of the pixel grid and hopes one of those locations is fragile enough that copying a
high-contrast value onto it flips a robustly-trained classifier. Even if a handful of genuinely fragile
pixels exist on an image, sampling three percent of locations at random will miss them the overwhelming
majority of the time.

It is worth tracing the accept-and-commit dynamics once by hand, because the `update_each_iteration=False`
setting changes what "fifteen probes" actually buys and I do not want to overstate the search. Within one
restart, the loop runs up to five iterations; each iteration samples a patch off the *committed* image,
evaluates the true-class probability, and *records* the candidate only if it lowered that probability — but
the committed image is not advanced mid-restart, so all five probes of a restart are sampled against the
*same* base image. Only at the restart boundary does the best recorded candidate become the new committed
image. So the three restarts give at most three genuine advances of the working image, each chosen as the
best of up to five independent tiny patches on the then-current base. That is materially weaker than a
greedy variant that would compound every accepted improvement immediately: here a good patch found in
iteration two of a restart does not seed iterations three through five: they still perturb the old base.
The practical consequence is that the effective depth of the search is three, not fifteen — fifteen is the
probe count, but the number of times the attack actually *builds on* a success is three. Against a
flattened surface where a single tiny patch almost never lowers the probability at all, most restarts
record nothing and the committed image never moves, so the realized attack frequently returns an image
that is barely perturbed. This is the restart-iterative variant behaving exactly as its name says, and it
sharpens rather than softens my expectation of a near-zero floor.

One more mechanism deserves a line, because it is the single structural prior that keeps this an honest
baseline rather than a strawman: the contiguity of the source patch. A CIFAR-scale convolutional network
stacks `3x3` convolutions, so a first-layer unit already integrates a `3x3 = 9`-pixel neighborhood, and
after two such layers the receptive field is `5x5`. A single changed pixel injects a high-contrast
impulse that one `3x3` filter sees but the surrounding units average away; a contiguous `2x2` block, by
contrast, falls entirely inside a single `3x3` receptive field and co-activates several neighboring
units with correlated high contrast, which is exactly the kind of localized structure the query-efficient
dense black-box attacks found most disruptive. So copying a `2x2` block rather than four scattered pixels
is not decoration — it concentrates the same four-pixel budget into one receptive field instead of
spreading it across four, giving each probe its best shot per query. Keeping that prior in the floor means
that when the floor fails, it fails for the *right* reason — blind placement — and not because I crippled
it with a placement prior known to be bad.

Let me reason about what this rung must do, because the entire point of running it is to set the bar the
next rung has to clear. Random search throws away everything it learns — each probe is independent, it
never uses the geometry of past probes to concentrate where the boundary turned out to be close. On an
*undefended* network that is fine: vulnerable pixels are dense, a random patch lands on one quickly, and
even a thin search saturates. Let me make that limit concrete as a sanity check on the whole picture: if
some large fraction `f` of locations were individually fragile — the undefended regime — then the chance
of *missing* all of them across `~33` distinct probed locations is about `(1-f)^33`, which for `f = 0.1`
is `(0.9)^33 ~= 0.03`, i.e. a `~97%` success rate even for this starved search. That is exactly why a thin
random rearrangement saturates on undefended models, and it confirms the attack is not broken — it is
correctly a strong attack in the regime where fragile pixels are dense. But these are
`L2`-adversarially-trained models, whose entire training objective was to *flatten* the loss surface in a
neighborhood of each input — to make exactly the local, few-pixel sensitivities that a random-search
attack relies on as scarce and as small as possible. Push `f` down to the robust regime, a fraction of a
percent of locations, and the same formula flips: `(1 - 0.003)^33 ~= 0.905`, a `~90%` *miss* rate per
image, so a `~10%` success at best and, once I account for the fact that a single fragile pixel rarely
suffices and a robust flip usually needs a coordinated few, far lower still. So a scattershot of fifteen
tiny random patches is searching for rare needles in a haystack that was deliberately built to hide them.
I expect this rung to almost entirely fail: a near-zero success rate on all three models, with whatever
handful of successes come from the easiest samples where even a robust model left a single fragile pixel.
The random mapping is the weakest link — it spends no queries deciding *where* to put the out-of-place
pixel, so against a robust model it is essentially hoping to hit a sensitive location by luck, fifteen
times, and luck against a flattened surface is thin.

Let me also confirm I have the interface mechanics right by tracing the shape and validity, because a
floor that silently returns invalid tensors would score zero for the wrong reason and corrupt the anchor.
`images` arrives as `(N, C, H, W) = (N, 3, 32, 32)` in `[0,1]` on `device`; the attack overwrites some
destination pixels of each image with values copied from source pixels of *that same image*, so every
returned value is a value that was already present in the input and therefore already in `[0,1]` to
machine precision — it passes the harness's finiteness and range checks by construction. The `L0` count
the harness computes is channel-collapsed: a spatial pixel counts as changed iff any of its three channels
moved. Rearrangement only writes to destination locations, and the number of distinct destinations across
the thin loop is far under `24` in this configuration, so the budget check passes with room to spare. The
returned tensor is `attack(images, labels)`, same shape `(N, 3, 32, 32)`. Nothing about the plumbing can
inflate or deflate the measured floor; whatever ASR comes back is a faithful reading of how often a blind
three-percent-coverage search flips a robust model. That is what I want the anchor to measure.

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
measured against. The falsifiable form of that expectation, stated in the only metric the harness reports:
per-model ASR in the low single digits of a percent, on the order of one to a few flips out of the up-to
`150` samples, and no model materially above the others — because a blind search has no mechanism to
exploit whatever makes one robust model's surface a hair less flat than another's. If instead this rung
came back with a double-digit ASR on any model, my whole premise that these surfaces are flattened against
few-pixel sensitivities would be wrong, and I would have to rethink the ladder from the bottom.

The causal chain in one breath: I want the cheapest real sparse attack as the floor → the leanest is
black-box, query-only, and the rearrangement trick (copy existing pixel values instead of searching a
color cube) makes it leaner still, killing the continuous color dimension and guaranteeing in-range,
budget-clean candidates → wrapped in accept-if-improves random search with restarts, configured here with
one-to-four-pixel patches, random destinations, and ~15 evaluations per image covering only about three
percent of the pixel grid → against `L2`-robust models whose surfaces were trained flat, a blind query-
starved search should almost entirely fail, setting a near-zero floor and pointing the next rung at *using*
per-pixel importance instead of guessing locations.
