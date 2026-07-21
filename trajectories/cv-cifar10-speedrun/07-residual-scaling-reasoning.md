airbench94 is done: 94.01% mean accuracy in 9.9 epochs / 3.83 seconds, 3.29 compiled, a 4.8× speedup
over the 18.3-second baseline built entirely from structural fixes. But the speedrun has a second, harder
bar I have ignored — 96% mean accuracy — and that is a genuinely different problem. At the 94% bar the
binding constraint was *time*: the network had ample capacity and I was racing to use it efficiently, so
every change removed wasted epochs. At 96% the constraint flips. Two points higher is not "the same race, a
little longer" — it asks the model to represent a more accurate function and to train without overfitting
while it does. So I should expect to spend *more* epochs and *more* FLOPs here, and the design goal is to
spend them well.

Two things stop airbench94 at ~94%. First, raw capacity: the network is small (blocks of roughly
64/256/256, two convs each) and trained for only ~10 epochs, tuned to *just* clear 94% fast, so it lacks
the representational room and the training time for 96%. Second, generalization: scale that capacity up
and train much longer and a small-data problem like CIFAR-10 (only 50,000 images) starts to overfit, and
the train/test gap becomes what holds me below 96%. So the recipe has to do three things at once — add
capacity, add depth, and add regularization — because capacity and depth alone overfit, and regularization
alone underfits.

A sanity check on how expensive 96% *should* be, so I know whether my recipe is in the right ballpark.
Going from 94% to 96% halves the error, 6% → 4% — a 33% reduction. Error-versus-compute on a well-tuned
image classifier tends to fall on a straight log-log line, error ∝ FLOPs^(−k) for some modest k. If k is
around 0.3–0.5, cutting error by 4/6 = 0.667 costs a FLOPs multiple of 0.667^(−1/k): about 2.3× at
k = 0.5, about 3.9× at k = 0.3. Fold in a longer schedule (tens of epochs) and a wider network and a
total cost on the order of ~10× airbench94's wall-clock is what the line predicts — costly, not
pathological. If my recipe came out at 100× I would suspect I was doing something badly; ~10× says I am on
the line, and I will hold the recipe to that envelope.

There is a lazy path and an extravagant one, both off the line. Pure width-and-epochs — keep the
airbench94 architecture, make it wide, train 40 epochs — stalls short of 96% or reaches it only at a
wildly inefficient FLOPs point, because shallow two-conv blocks cannot compose enough nonlinear stages no
matter how wide. Swapping in a full ResNet-50-scale network clears 96% but is enormously more compute than
the bar needs and throws away everything the lineage built. The efficient path is surgical: take the
network already sitting on the line at 94% and extend it *minimally* along the three axes the diagnosis
named. Crucially it *keeps* the entire conditioning stack — the frozen whitening front end, Dirac init on
every deep conv, BatchNorm scale frozen at 1 with the 64× bias rate — so I am not re-deriving a good
starting point, only adding capacity, depth, and regularization on top of a base that is already
well-conditioned. That is why the extension can be minimal: the hard optimization problems were solved at
94% and carry forward.

Capacity first. I widen the blocks — the third from 256 up to 512, earlier blocks to match (roughly
128/384/512) — and train for tens of epochs. A 3×3 conv's parameters scale as 9·C_in·C_out, so widening a
256→256 block (≈590k) to a 384→512 block (≈1.77M) roughly triples its parameters — real capacity — but
width buys breadth of features per stage, not the *number* of stages, and beyond a point more channels on
the same shallow blocks stop helping. So, depth: I add a *third* conv to each block, 7 conv layers → 10.
But I already know from my own ladder what naive deepening does — the deeper layers have to relearn to
pass signal through, which is the exact problem Dirac init fixed. I am still using Dirac, so the deep
stack does start near identity, but identity init only sets the *starting* point, and over a ~37-epoch
run the optimizer can walk a deep block far from init into a configuration where gradients no longer flow
cleanly back to the earlier convs — unlike the ~10-epoch 94% run, a 37-epoch run gives them plenty of
distance to wander. The standard fix for gradient flow that holds *throughout* training, not just at init,
is a residual connection: wrap a skip around a sub-stack so its output is x + F(x). So I add a residual
across the last two convs of each block — save the activation after the first conv, run the next two
convs+norms, add the saved activation back before the final activation (the ConvGroup is in the answer).

This looks like a device I already have, so the redundancy question is real: Dirac is also a
"start-near-identity" mechanism, so do I need an explicit skip on top of it? Yes, and the distinction is
precise. Dirac init is a property of the weights at *t = 0*: it makes F ≈ 0 at the start but says nothing
about later, and ∂F/∂x is free to grow as the weights move. A residual skip is a property of the
architecture at *all t*: with output = x + F(x) the Jacobian is I + ∂F/∂x, which always carries the
identity term, so gradient always has a direct path back to x regardless of what F has become — the "+I"
cannot be trained away because it is structural, not initialized. In a short 94% run the weights stay near
their Dirac init, ∂F/∂x stays small, and a skip would be nearly dead weight (which is exactly why I did
*not* add one for airbench94). In a long, deep 96% run the weights move far, ∂F/∂x can grow, and the
structural +I is what keeps the deeper stack trainable once the init-only guarantee has decayed. Same idea
at two timescales — identity at init, identity throughout — and the long run is where the second earns its
keep.

The skip goes over the *last two* convs, not all three, for a shape reason. conv1 does two structural jobs
at once: it changes the channel count and is followed by the `MaxPool2d(2)` that halves the spatial
resolution, so the block's input and conv1's output differ in *both* channels and spatial size — a skip
jumping across conv1 would have mismatched shapes and need a learned 1×1 projection. conv2 and conv3 are
both channels_out → channels_out at the same pooled resolution, so the activation saved after conv1
matches conv3's output exactly and `x + x0` is a clean, parameter-free identity skip. The placement is the
only place a projection-free skip fits — and it happens to be the deepest, most gradient-starved part of
each block. The skip also earns its place on the *generalization* axis, which matters since I am about to
add capacity that pushes toward overfitting: writing the block as x + F(x) makes its default the identity
and forces it to only *learn the correction* F, a lower-complexity hypothesis than representing the full
map from scratch. On a small dataset a lower-complexity default is a form of regularization — the extra
depth does not automatically become extra memorization capacity.

Now regularization, because capacity + depth + epochs *will* overfit 50,000 images. The augmentation I
have — flip plus 2-pixel translate — was appropriate for a ~10-epoch run that could not overfit much, but
is too weak for a 37-epoch run with time to memorize. The classic strong-but-cheap augmentation is Cutout:
mask a random square patch of each training image, forcing the network to classify from partial views
rather than lean on any single region. I add 12-pixel Cutout — a 12×12 square is 144 of 1024 pixels,
about 14% blanked, enough to bite without destroying the object — and raise the random translation from 2
to 4 pixels. Two things in the implementation matter (the code is in the answer). The random top-left
corner is drawn with `randint(0, h-size+1)`, whose upper bound guarantees the square fits entirely inside
the image. And `masked_fill(mask, 0)` fills the hole with *zero* on the *normalized* images — and zero in
per-channel-normalized space is, by construction, the per-channel *mean* in raw pixel space. So a Cutout
hole is not a black square (an out-of-distribution dark blob the network could learn to detect) but a
neutral, dataset-mean gray patch — the uninformative "nothing here" I want. The normalization makes zero
*be* the mean for free.

A few schedule details follow from the longer run: I let the learning rate decay all the way to zero at
the end (with tens of epochs there is time for a proper anneal), and shorten the warmup as a fraction of
the now-longer run. And I keep alternating flip — the prior evidence shows its effective speedup from
random→alternating actually *grows* with the epoch budget (27.1% at 20 epochs, 38.3% at 40), so if
anything it matters *more* in a 37-epoch run, the redundancy it removes compounding over a longer
schedule.

The cost estimate closes the loop: the schedule grows ~9.9 → ~37 epochs, a factor ~3.7 in passes over the
data; the per-step FLOPs grow ~2.5× from the architecture (the 384/512 blocks are ~3× the parameter cost
of the old 256 blocks, and a third conv per block adds another conv's compute). Product 3.7 × 2.5 ≈ 9×,
landing right in the ~10× envelope the log-log line predicted for a 6%→4% error halving. That agreement is
itself a check: had the recipe implied 3× or 30×, one of my two estimates would be wrong.

This is a different *kind* of prediction than the earlier changes — not a seconds-down move on a fixed bar
but a *capability* question, answering how fast I can reach 96% at all. The bet is that width + a third conv
per block + residual skips + Cutout break through the ~94% ceiling to 96% mean accuracy, at a FLOPs/error
point on the same favorable log-log line, so 96% costs the ~10× wall-clock the line predicts but not
pathologically more. The specific risk I am least sure about is whether the residual skip is redundant
with Dirac and adds nothing — but the Jacobian argument says the skip's +I is structural where Dirac's is
only initial, and the long training is exactly where init-only conditioning decays. The result is the
airbench96 training; the code is in the answer.
