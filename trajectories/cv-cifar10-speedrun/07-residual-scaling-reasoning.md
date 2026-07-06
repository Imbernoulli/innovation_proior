airbench94 is done: 94.01% mean accuracy in 9.9 epochs / 3.83 seconds, 3.29 compiled, a 4.8× wall-clock
speedup over the 18.3-second baseline built entirely from structural fixes. But the speedrun has a second,
harder bar I have been ignoring — 96% mean accuracy — and the question now is not "how do I shave seconds
off 94%" but "what is the fastest way to reach a *higher accuracy at all*." That is a genuinely different
problem, and I should name the difference before designing, because it changes which lever matters. At the
94% bar the binding constraint was *time*: the network had ample capacity to hit 94% and I was racing to
use that capacity efficiently, so every rung was about removing wasted epochs. At 96% the binding
constraint flips. Two percentage points higher is not "the same race, a little longer" — it is asking the
model to represent a more accurate function and to train without overfitting while it does. So I should
expect to spend *more* epochs and *more* FLOPs here, and the design goal is to spend them well.

Let me diagnose what actually stops airbench94 at ~94%, because that tells me what to add. Two things.
First, raw capacity: the network is small (blocks of roughly 64/256/256 channels, two convs per block)
and trained for only ~10 epochs — it is tuned to *just* clear 94% fast, so it simply lacks the
representational room and the training time to push to 96%. Second, generalization: if I scale that
capacity up and train much longer, a small-data problem like CIFAR-10 (only 50,000 images) will start to
overfit, and the train/test gap becomes the thing that holds me below 96%. So the 96% recipe has to do
three things at once — add capacity, add depth, and add regularization — and if I do only the first two I
will overfit, and if I do only the third I will underfit. All three together.

Before I build, a sanity check on how expensive 96% *should* be, so I know whether my recipe is in the
right ballpark or wildly off. Going from 94% to 96% means halving the error rate from 6% to 4% — a 33%
error reduction. Error-versus-compute on a well-optimized image classifier tends to fall on a straight
log-log line, error ∝ FLOPs^(−k) for some modest exponent k. If k is somewhere around 0.3–0.5, then
cutting error by the factor 4/6 = 0.667 costs a FLOPs multiple of 0.667^(−1/k): about 2.3× at k = 0.5,
about 3.9× at k = 0.3. So the *smooth* extrapolation says the error halving alone is worth a few times the
FLOPs, and once I fold in a longer schedule (tens of epochs instead of ten) and a wider network, a total
cost on the order of ~10× airbench94's wall-clock is exactly what the log-log line predicts — costly, but
not pathological. If my recipe came out at 100× I would suspect I was doing something badly; ~10× says I
am on the line. That is the target envelope, and I will hold the recipe to it.

It is worth pausing on the alternatives to this three-pronged recipe, because there is a lazier path and a
more extravagant one and I want to reject both explicitly. The lazy path is pure width-and-epochs: keep
the airbench94 architecture, make it wide, train it for 40 epochs. That is the plateau I just described —
shallow blocks cannot compose enough nonlinear stages no matter how wide, so this stalls short of 96% or
reaches it only at a wildly inefficient FLOPs point, off the log-log line. The extravagant path is to swap
in a much larger standard architecture — a full ResNet-50-scale network. That would clear 96% but it is
enormously more compute than the problem needs at this bar, and it throws away everything the speedrun
lineage has built (the whitening front end, the Dirac stack, the tuned optimizer), landing far off the
FLOPs/error line I want to stay on. The efficient path is the surgical one: take the network that already
sits on the line at 94% and extend it *minimally* along the three axes the diagnosis named — a little more
width, one more conv per block, a structural skip, and stronger augmentation — so the 96% point lands on
the same line rather than above it. That is the recipe. Crucially, the surgical path *keeps* the entire
conditioning stack the ladder already built — the frozen whitening front end, Dirac init on every deep
conv, BatchNorm scale frozen at 1 with the 64× bias learning rate — so I am not re-deriving a good starting
point, I am adding capacity, depth, and regularization on top of a base that is already well-conditioned.
That is why the extension can be minimal: the hard optimization problems were solved at the 94% bar and
carry forward, and the only new work is representational room and overfitting control.

Capacity first, the easy part. I widen the blocks — the third block from 256 up to 512, with the earlier
blocks widened to match (roughly 128 / 384 / 512) — and I train for tens of epochs instead of ten. Widen
is pure tuning and it gets me partway, but I know it plateaus: a 3×3 conv's parameter count scales as
9·C_in·C_out, so widening a 256→256 block (9·256·256 ≈ 590k params) to a 384→512 block (9·384·512 ≈
1.77M) roughly triples that block's parameters — real capacity — but beyond a point more channels on the
same *shallow* two-conv blocks stop helping, because the network cannot compose features over enough
nonlinear stages. Width buys breadth of features per stage; it does not buy the number of stages.

So, depth. I add a *third* convolution to each block, taking the network from 7 conv layers to 10. But I
already know from my own ladder what goes wrong when you naively deepen a conv stack: the deeper layers
have to relearn to pass signal through, and training conditioning degrades — that is the exact problem
Dirac init was invented to fix, and I am still using Dirac init, so the deep stack does start near
identity. The trouble is that identity init only sets the *starting* point. Over a long, ~37-epoch
training, the optimizer can walk a deep block far from its init into a configuration where gradients no
longer flow cleanly back to the earlier convs — and unlike the ~10-epoch 94% run, where the weights never
travel far, a 37-epoch run gives them plenty of distance to wander. The standard structural fix for
gradient flow that holds *throughout* training, not just at init, is a residual connection: wrap a skip
around a sub-stack so its output is x + F(x). So I add a residual connection across the last two convs of
each block — save the activation after the first conv, run the next two convs+norms, and add the saved
activation back before the final activation.

There is a tension I have to name and resolve with a real argument, because it looks like I am adding a
device I already have. I am *already* using Dirac initialization, which is itself a "start-near-identity"
mechanism — so do I even need an explicit residual skip on top of it, or are they redundant? They are
not, and the distinction is precise. Dirac init is a property of the weights at *t = 0*: it makes F ≈ 0 at
the start, but it says nothing about later, and ∂F/∂x is free to grow as the weights move. A residual skip
is a property of the architecture at *all t*: with output = x + F(x), the Jacobian is ∂(x + F(x))/∂x = I +
∂F/∂x, which *always* carries the identity term, so gradient always has a direct path back to x regardless
of what F has become — the "+I" cannot be trained away because it is structural, not initialized. Let me
make the redundancy question concrete: in a short 94% run the weights stay near their Dirac init, so
∂F/∂x stays small, so a skip would be nearly dead weight (I + small ≈ I either way) — which is exactly why
I did *not* add a skip for airbench94. In a long, deep 96% run the weights move far, ∂F/∂x can grow, and
the structural +I is what keeps the deeper stack trainable when the init-only guarantee has long since
decayed. So Dirac and the skip are the same idea at two different timescales — identity at init, identity
throughout — and the long run is precisely the regime where the second one earns its keep. I keep both.

```python
class ConvGroup(nn.Module):
    def forward(self, x):
        x = self.conv1(x); x = self.pool(x); x = self.norm1(x); x = self.activ(x)
        x0 = x
        x = self.conv2(x); x = self.norm2(x); x = self.activ(x)
        x = self.conv3(x); x = self.norm3(x)
        x = x + x0           # residual over the last two convs
        x = self.activ(x)
        return x
```

There is a reason the skip goes over the *last two* convs and not over all three, and it is a shape
argument I should check rather than wave at. In the block, conv1 does two structural jobs at once: it
changes the channel count (channels_in → channels_out) and it is followed by the `MaxPool2d(2)` that
halves the spatial resolution. So the block's input and conv1's output differ in *both* channel count and
spatial size — a skip that jumped across conv1 would have mismatched shapes and would need a learned
projection (a 1×1 conv, extra parameters, extra compute) to even add them. The last two convs, conv2 and
conv3, are both channels_out → channels_out at the same pooled spatial resolution, so the activation I
save after conv1 (`x0`, shape N × channels_out × H/2 × W/2) matches conv3's output shape exactly, and `x +
x0` is a clean, parameter-free identity skip with nothing to project. So the skip is placed precisely where
the shapes already line up — and that is also precisely the two-conv sub-stack that most needs the gradient
path, the deepest part of each block. The placement is not a free choice dressed up as a principled one;
it is the only place a projection-free skip fits.

The residual skip actually earns its place on the *generalization* axis too, not only the gradient-flow
one, which is worth noting since I am about to add capacity that pushes toward overfitting. Writing the
block as x + F(x) means its default behavior is the identity and it only has to *learn the correction* F —
a lower-complexity hypothesis than a block that must represent the full input-to-output map from scratch.
On a small dataset, a lower-complexity default is a form of regularization: the network reaches for the
simplest function (pass through) unless the data pushes it to add a correction, so the extra depth I am
adding does not automatically translate into extra memorization capacity. So the skip does double duty —
it keeps the deep stack trainable *and* it biases the added depth toward parsimony — which is exactly what
I want when the whole reason I am adding depth is that shallow-and-wide plateaued but naive-deep would
overfit.

Now regularization, because capacity + depth + epochs *will* overfit 50,000 images. The augmentation I
have — flip plus 2-pixel translate — is light, which was appropriate for a ~10-epoch run that could not
overfit much in the time available, but is too weak for a 37-epoch run that has time to memorize. The
classic strong-but-cheap augmentation for CIFAR is Cutout: mask out a random square patch of each training
image (set it to zero), forcing the network to classify from partial views and not lean on any single
region. I add 12-pixel Cutout — a 12×12 square is 144 of the 1024 pixels, about 14% of the image blanked
each time, enough to bite without destroying the object — and I strengthen the random translation from 2
to 4 pixels to match the heavier-augmentation regime. Cutout is a per-batch masked fill, so it is nearly
free at run time, and it attacks exactly the failure mode a bigger, longer-trained network suffers: overfit
to specific regions of specific images.

```python
def make_random_square_masks(inputs, size):
    n,c,h,w = inputs.shape
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)
    corner_y_dists = torch.arange(h, device=inputs.device).view(1,1,h,1) - corner_y.view(-1,1,1,1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1,1,1,w) - corner_x.view(-1,1,1,1)
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)
    return mask_y * mask_x

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)
```

Two things about this implementation are worth checking, because a subtly-wrong Cutout can hurt rather than
help. First, correctness of the mask: `make_random_square_masks` draws a random top-left corner per image
with `randint(0, h-size+1)`, whose upper bound `h-size+1` guarantees the square fits entirely inside the
image (the corner can be at most h−size, so corner+size ≤ h). It then builds the mask by broadcasting —
`corner_y_dists` is the signed distance of every row from the corner, and the mask keeps rows where that
distance is in [0, size) and likewise for columns, so `mask_y * mask_x` is exactly the axis-aligned
size×size square with that corner. It is fully vectorized (one random corner per image, no Python loop over
the batch), so it is cheap. Second, and easy to miss: `masked_fill(mask, 0)` fills the hole with *zero* on
the *normalized* images — and zero in the per-channel-normalized space is, by construction, the per-channel
*mean* in raw pixel space. So a Cutout hole is not a black square (which would be an out-of-distribution
dark blob the network could learn to detect); it is a neutral, dataset-mean gray patch — exactly the
uninformative "nothing here" I want, forcing the network to look elsewhere rather than handing it a
distinctive artifact. Filling with the mean is the right thing, and the normalization makes zero *be* the
mean for free.

A few schedule details follow from the longer run. I let the learning-rate schedule decay all the way to
zero at the end rather than to a small floor — with tens of epochs there is time for a proper anneal, and
a hard zero at the end squeezes out the last of the noise — and I shorten the warmup as a fraction of the
now-longer run. And I keep alternating flip: I have Table 2 evidence from the 94% work that its effective
speedup from switching random→alternating actually *grows* with the epoch budget (27.1% at 20 epochs, up
to 38.3% at 40 epochs), so if anything the derandomized flip matters *more* in a 37-epoch run than it did
in the ~10-epoch one — the redundancy it removes compounds over a longer schedule. So the derandomization
carries over rather than being a 94%-only trick.

Let me close the loop on the cost estimate by multiplying out where the ~10× actually comes from, so the
recipe and the envelope meet. The schedule grows from ~9.9 to ~37 epochs, a factor of about 3.7 in passes
over the data. The per-step FLOPs grow from the architecture changes: widening the blocks to roughly
128/384/512 raises the channel-count products (a 3×3 conv costs 9·C_in·C_out per spatial position, and the
384/512 blocks are ~3× the parameter cost of the old 256 blocks), and adding a third conv to each block
adds another conv's worth of compute per block — together on the order of ~2.5× the per-step FLOPs. Product:
3.7 × 2.5 ≈ 9×, landing right in the ~10× wall-clock envelope the log-log line predicted for a 6%→4% error
halving. So the recipe's cost is not arbitrary — it is the epoch stretch times the per-step growth, and it
comes out where the smooth error-versus-compute line says a 96% point should sit. That agreement is itself
a check: if the recipe had implied 3× or 30×, one of my two estimates would be wrong; ~9× from the
architecture matching ~10× from the line is the consistency I want before committing.

The prediction, and it is a different *kind* of prediction than the earlier rungs made. This is not a
seconds-down rung on a fixed bar; it is a *capability* rung — it answers a question 94% could not, namely
how fast I can reach 96% at all. The bet is that width + a third conv per block + residual skips + Cutout,
together, break through the ~94% ceiling to 96% mean accuracy, and that the resulting FLOPs/error point
falls on the same favorable log-log line that extrapolates from the airbench94 point I already have — so
96% costs more, roughly the ~10× wall-clock that line predicts, but not pathologically more. The cost is real and I accept it: tens
of epochs, several times the FLOPs and wall-clock of airbench94. The specific risk I am most unsure about
is whether the residual skip is redundant with Dirac init and adds nothing — but the Jacobian argument
says the skip's +I is structural where Dirac's is only initial, and the long training is exactly where
init-only conditioning decays, so I expect the skip to earn its place rather than sit idle. What the 96%
seconds-and-accuracy table will tell me is whether the recipe actually clears 96% (falsifying or
confirming that capacity+depth+regularization was the right diagnosis) and whether its cost lands on the
log-log line or off it. The result is the airbench96 training; the code is in the answer.
