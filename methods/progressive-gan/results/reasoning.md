OK, let me think this through from scratch. I want a generator that makes high-resolution images — say faces at 1024×1024 — that look real and are *varied*, and I want training to actually finish instead of collapsing. Right now I can train an adversarial generator at 64² or 128² and get something passable with a Wasserstein critic, but every time I push the resolution up the whole thing gets slower and more fragile, the samples get worse, and often it just falls apart. So the question isn't "scale up the network and the dataset," because that demonstrably doesn't work. The question is *why* high resolution breaks adversarial training, and whether there's a way to sidestep the cause rather than fight its symptoms.

Let me get precise about the failure. The generator only ever learns through the gradient the discriminator hands it. And there's a known, sharp reason that gradient goes bad: if the data distribution and the generated distribution live on manifolds that barely overlap, then a discriminator can tell them apart essentially perfectly, and once it can, the gradient it passes back to the generator either vanishes (it's saturated, confidently right everywhere) or points in nearly arbitrary directions (tiny perturbations of a perfect classifier). Two thin manifolds floating in a huge pixel space almost never overlap — that's the generic situation, not the exception. So a too-good discriminator is poison, and the discriminator's job is exactly to get too good.

Now overlay resolution on that. At 4×4 an image is 48 numbers; the space of plausible 4×4 face-blobs and the space of generated 4×4 blobs are fat, low-frequency, and they *do* overlap a lot — it's genuinely hard to be sure a 4×4 patch is real or fake, there just isn't enough information. At 1024² an image is three million numbers, and real photographs have a very particular fine-grained statistical signature — sensor noise, edge sharpness, the exact texture of skin and hair — that a generator will get subtly wrong everywhere. The discriminator has three million highly informative coordinates to find the tell in. So the higher the resolution, the *easier* the discrimination, the faster the discriminator becomes near-perfect, and the worse the gradient pathology. Resolution doesn't just make the target harder to hit; it actively sharpens the exact mechanism that kills the gradient. Generating straight at 1024² is the single worst case for it. And there's a second, dumber problem stacked on top: a 1024² minibatch eats so much memory that I'm forced to shrink the batch, which makes every gradient estimate noisier and the batch statistics flakier, degrading stability further right where I can least afford it.

So I have a loss-side fix already — use the Wasserstein distance with a gradient penalty, where the critic is constrained to be 1-Lipschitz and so never saturates into a useless step function; it keeps handing back a finite, meaningful gradient even when the supports are disjoint. Good, I'll keep that as my loss. But notice what it does *not* fix: it makes the gradient well-behaved, but it doesn't make the *problem* easy. Asking a single network, from random init, to learn the entire map from a latent vector all the way to a coherent three-million-pixel photograph — getting the global face layout *and* the eyelash texture right simultaneously — is just an enormously hard optimization problem, and the critic is at its most lethal exactly at that highest resolution where I'm asking the most. The loss fix and the difficulty are orthogonal. I need something on the difficulty side.

Let me stare at the asymmetry I just noticed: discrimination is *easy* for the discriminator and *hard* for the generator at high resolution, but at low resolution discrimination is genuinely *hard* (the distributions overlap) and generation is *easy* (few coarse numbers to get right). Low resolution is the regime where the game is balanced and the gradient is informative. What if I just... never start at high resolution? Train the whole thing at 4×4 first, where the game is fair, get a generator that nails the coarse structure — the rough head shape, the light/dark layout — and only *then* worry about detail.

But I can't train a 4×4 generator and a separate 1024² generator from scratch; the second one throws away everything the first learned. I want one network that *grows*. Start G and D as a tiny mirror-image pair operating at 4×4. Train them until the 4×4 distributions match well. Then append one more block to each — a block that upsamples and adds a couple of convolutions, taking the generator to 8×8, and a mirror block on the discriminator that takes 8×8 down to 4×4 — and continue training, now at 8×8. Keep all the old layers trainable; don't freeze anything. Then 16×16, 32, 64, ... 1024. At every stage the generator's job is only ever "take the representation I already have at resolution R and add the next octave of detail to reach 2R," which is a small, local refinement, not the whole map. And crucially, at every stage the discriminator is comparing distributions at the *current* resolution, where they still overlap enough that its job isn't trivial — I'm continually feeding it a fair fight instead of handing it the three-million-pixel turkey shoot on day one.

This is a curriculum, and it falls out of the structure rather than being imposed: early layers see only the coarse problem, converge on large-scale structure, and by the time the fine-resolution layers come online the coarse layers have largely settled, so each new stage only has to learn its own octave. And it's *cheap*: the vast majority of training iterations happen at low resolution where the networks are shallow and a forward/backward pass costs almost nothing, so I get to the same quality far faster than training the full deep network the whole time — most of the wall-clock at high res is just the last short stretch.

Now the obvious problem with "append a block and continue." A freshly added block has random weights. The instant I insert it, it maps the perfectly good 4×4 features into garbage at 8×8, the discriminator (also freshly grown) screams, and a huge gradient comes ripping back through into the lower layers that I spent all that effort training. I've shocked the trained part of the network with noise from the new part. That defeats the whole point of having converged at 4×4 first.

So I can't *snap* the new layer in; I have to ease it in. Think about what the network looked like the instant before I grew it: it produced a good R×R image, and to get a 2R×2R-shaped output I'd just upsample that R×R image with nearest-neighbor — a blocky but correct image. The new 2R×2R block, once trained, will produce a genuinely detailed image. So I have two candidate outputs at the new resolution: the trivially-upsampled old image, which is correct-but-blurry and available immediately, and the new block's output, which starts as noise but will become the real thing. I want to *cross-fade* from the first to the second.

So treat the new block like a residual branch with a blend weight α going from 0 to 1. Concretely, give the network a little "to-RGB" head — a 1×1 convolution that projects a feature map to actual RGB pixels — at each resolution. When I'm transitioning into resolution 2R, the generator's emitted image is

  image = (1 − α) · upsample(toRGB_old(features_at_R)) + α · toRGB_new(block(features_at_R)),

with α ramped linearly from 0 to 1 over, say, the next several hundred thousand images. At α = 0 the new block contributes literally nothing to the output — the network behaves exactly like the old one, just upsampled, so there's no shock at all. As α creeps up, the new block's contribution is mixed in gently, and the gradient flowing into it (and back through the old layers) starts small and grows smoothly; the old layers are never hit with a step change. At α = 1 the old upsampled path is gone and the new block fully owns the output, and then it's just a normal layer I keep training. The 1×1 convs are the right tool for the RGB heads precisely because they're per-pixel projections with no spatial mixing — they convert features to colors without inventing structure.

The discriminator has to mirror this exactly or the two will be inconsistent. Where the generator has a toRGB that projects features → RGB, the discriminator has a "from-RGB" 1×1 convolution that projects an input RGB image → features at that resolution, and then the rest of D processes them down. During the transition I feed the real (or fake) image into D at the new resolution through fromRGB_new, run it through the new downsampling block, and blend that with what I'd get by first downsampling the image to the old resolution and running fromRGB_old:

  D_input_features = (1 − α) · fromRGB_old(downsample(x)) + α · downblock(fromRGB_new(x)),

same α, ramped together. So the generator and discriminator grow in lockstep and the blend is symmetric. One more subtlety: while α is between 0 and 1 the generator is emitting a mixture of two resolutions, so the *real* images I show the discriminator should be the same kind of mixture — take the real image, make a blurred version by average-pooling to the lower resolution and upsampling back, and use (1 − α) times the blurred image plus α times the sharp image. Otherwise the discriminator could cheat during the fade by detecting "this one has crisp high-frequency content, so it's real," which is exactly the spurious tell I'm trying to avoid. With both sides faded by the same α the discriminator only ever sees a consistent resolution and can't shortcut.

I'll bookkeep all of this with a single scalar I'll call the level-of-detail, lod. As I show more images, lod decreases continuously from (say) 8 down to 0; floor(lod) picks the active resolution. During a transition from an old level k+1 to a new level k, the fractional part of lod starts near 1 and falls to 0, so it is the *old* path's weight. The new path's weight is therefore α = 1 − (lod − floor(lod)). That sign matters: just after a transition starts, α is tiny, so the random new block barely contributes; at the end α reaches 1 and the new block fully takes over. The same lod drives the generator's output blend, the discriminator's input blend, and the real-image blend.

Good — that's the core. Progressive growing plus smooth fade-in. Now the symptoms I haven't addressed: variation, and magnitude escalation. Let me take variation first.

Generators love to collapse onto a subset of the data — produce a few prototypes over and over. The reason the discriminator doesn't punish this is that it judges each image *in isolation*: a single generated face can look perfectly real even if the generator only ever makes that one face. The discriminator has no way to notice "this entire batch is suspiciously similar" because it never looks across the batch. The known fix is to let it: compute statistics across the minibatch and feed those to the discriminator, so that a batch of near-clones looks statistically unlike a batch of real images, which has real diversity. The original version of this learns a big projection tensor that maps each image's features to a set of per-sample cross-batch statistics and concatenates them in. That works, but it's heavy — it adds a learned tensor, it adds hyperparameters (how many statistics, what dimensions), and I have to decide where to put it. I want the variation pressure without paying for any of that, because every learnable knob is one more thing that can destabilize an already-touchy adversarial game.

What's the simplest cross-batch statistic that captures "this batch lacks variety"? Just the standard deviation of the features across the batch. If the generator is producing near-identical images, the per-feature standard deviation across the batch will be tiny; real batches have healthy spread. So: take the discriminator's activations [N, C, H, W], and for each feature channel at each spatial location, compute the standard deviation over the N images in the batch — s_{c,h,w} = sqrt( mean_n (x_{n,c,h,w} − mean_n x_{n,c,h,w})² + ε ). Now I have a [C, H, W] map of per-location variabilities. Average it down to a single scalar — the average standard deviation across all features and positions. Then replicate that scalar into a constant [N, 1, H, W] feature map and concatenate it as one extra channel onto the activations. Every image in the batch now carries, as a feature, a measurement of how varied the batch it came in was. The discriminator can read it and learn "real batches have this much spread; if I'm being shown a batch with too little, it's fake." That directly pressures the generator to keep its outputs diverse, on pain of being caught.

And look what this costs: nothing. No learnable parameters — it's a fixed reduction. No new hyperparameter to speak of. I just compute a std, average it, tile it, concatenate. (I can optionally compute it over small groups of the batch rather than the whole batch, which makes the statistic a little more local, but that's a detail.) Where to put it? It only needs to be somewhere the discriminator can use it, and putting it near the very end — at the 4×4 stage, right before the final scoring layers — is cleanest, since by then the spatial map is tiny and the single scalar of batch-variability is exactly the kind of global summary that belongs next to the final real/fake decision. So: one extra constant feature map, appended near the end of the discriminator, parameter-free. Much simpler than the learned-tensor version.

Now the magnitude problem. I've watched mode collapses happen, and they don't creep — they detonate, over maybe a dozen minibatches. The pattern is: the discriminator overshoots, its gradients get exaggerated, the generator responds, and the *magnitudes of the signals* in both networks start escalating, each net's growth feeding the other's, until it's a runaway. It's an unhealthy competition expressed as a magnitude blow-up. Most people damp this with BatchNorm in the generator (and sometimes the discriminator). But BatchNorm was invented to fix internal covariate shift — to keep the distribution of each layer's inputs stable as the layer below it changes — and I don't actually observe covariate shift being the problem in this adversarial setting. What I observe is *magnitudes spiraling*. So I don't want BatchNorm's mean/variance-tracking machinery and its learnable scale/shift (more parameters, more state, batch-size dependence — and I'm being forced to tiny batches at high res, where batch statistics are unreliable anyway). I want something whose only job is to refuse to let magnitudes escalate, with no learnable parameters to get dragged into the competition.

The escalation lives in the generator's feature magnitudes, so put the brake there. After each convolution in the generator, normalize each pixel's feature vector to a fixed length. Take the vector of feature values at pixel (x,y) across all N channels, and divide by its root-mean-square:

  b_{x,y} = a_{x,y} / sqrt( (1/N) Σ_{j=0}^{N−1} (a^j_{x,y})² + ε ),

with ε = 1e-8 to avoid dividing by zero. This is a per-pixel feature-vector normalization — a variant of the old local-response-normalization idea, where each activation is scaled by an aggregate over the feature channels at its own location. It pins the magnitude of every pixel's feature vector to unit RMS regardless of what the competition is doing, so there's simply no degree of freedom along which the magnitudes can run away. It has no learnable parameters, so it can't be co-opted into the escalation. I was worried this is too heavy-handed — forcing every feature vector to unit length sounds like it would throw away useful information — but the direction of each feature vector is preserved, only the scale is fixed, and the network has plenty of capacity to encode what it needs in directions and in the relative pattern across pixels. I'll only put it in the generator; the discriminator with the Wasserstein gradient penalty is already magnitude-controlled by the Lipschitz constraint, and I don't want to clamp its scores.

One more thing nags at me, and it's about *learning speed across layers*, which turns out to interact with the adaptive optimizer in a way I'd been ignoring. Adam (and RMSProp) don't take a raw gradient step; they divide each parameter's update by a running estimate of that parameter's gradient standard deviation. The consequence is that the update step is essentially *scale-invariant in the parameter*: whether a weight is intrinsically big or small, Adam moves it by roughly the same characteristic step size, because it's normalizing out the gradient magnitude. Now, careful modern initialization — He init — deliberately sets different layers to different weight scales (variance 2/fan_in) so that activation variance stays constant through depth. So I have layers whose weights live at genuinely different dynamic ranges by design. Feed those through Adam's scale-invariant update and the layers with a larger dynamic range take *relatively longer* to traverse their range — the same absolute step is a smaller fraction of where they need to go — so they effectively learn slower than the small-scale layers. A single global learning rate is then simultaneously too large for one layer and too small for another, and there's no single value that's right. That's a quiet, pervasive imbalance, and in a sensitive adversarial game I'd rather not have layers silently learning at different rates.

The fix is almost perverse. Instead of putting the scale into the initialization, I rip it out of the initialization and put it back at runtime. Initialize *every* weight from a plain N(0,1) — all layers now start at the same unit dynamic range. Then, at runtime, scale each layer's weights by the He constant before using them: ŵ_i = w_i · c, where c = gain / sqrt(fan_in) is exactly the per-layer normalization constant He's scheme would have used (gain = √2 for leaky ReLU). The forward pass is identical to a He-initialized network — same effective weights, same activation variances — so I lose nothing on the signal-propagation side. But the *learned* parameters w_i, the things Adam actually updates, now all live at the same unit scale, so Adam's scale-invariant step moves every layer at the same effective rate. The dynamic range, and therefore the learning speed, is equalized across all weights. It's a one-line change to how I fetch weights — multiply by a constant — and it removes the cross-layer learning-rate imbalance for free.

Let me also pin down the loss bookkeeping, since I'm committing to the Wasserstein-with-gradient-penalty critic. The discriminator minimizes E[D(fake)] − E[D(real)] plus the penalty λ·E[(‖∇_{x̂} D(x̂)‖₂ − 1)²] on interpolates x̂ between real and fake (λ = 10), and the generator minimizes −E[D(fake)]. One small addition: with nothing pinning it, the critic's output can slowly drift to large positive or negative values over a long run, which is just numerically unhealthy. So I add a tiny term ε_drift · E[D(real)²], ε_drift = 1e-3, that gently pulls the real scores toward zero without affecting the relative ordering that the Wasserstein objective cares about. And because the critic now stays informative throughout, I don't need many critic steps per generator step — I can just alternate one-and-one per minibatch.

Now let me lay the architecture out concretely. Latent vector of 512 dimensions; I'll draw it from a normal and normalize it onto the hypersphere (and I can pixel-norm it on the way in too). The generator's 4×4 block is a dense layer reshaped to 512×4×4 followed by a 3×3 conv; every subsequent block is upsample (2× nearest-neighbor replication) then two 3×3 convs, with the feature-map count halving as resolution doubles past 32² (512, 512, 512, 512, 256, 128, 64, 32, 16 for 4²→1024²). Each conv is followed by leaky-ReLU (slope 0.2) and the pixelwise feature normalization. A 1×1 toRGB at the top. The discriminator is the mirror: a 1×1 fromRGB, then blocks of two 3×3 convs followed by 2× average-pool downsampling, feature maps doubling as resolution halves, and at the 4×4 end I splice in the minibatch-standard-deviation feature map, then a 3×3 conv, a dense layer, and a final linear score. Both use leaky-ReLU everywhere except the linear output, equalized learning rate on every weight, no batch/layer/weight norm anywhere. Train with Adam, α = 1e-3, β1 = 0, β2 = 0.99. Keep an exponential moving average of the generator's weights (decay 0.999) to sample from at evaluation time, since the averaged weights give smoother output. Schedule: train 4×4 for some hundreds of thousands of images, then for each new resolution spend one phase fading the new block in (α: 0→1) and one phase stabilizing at α = 1, shrinking the minibatch as resolution grows to fit memory (16 down to 3 at 1024²).

Let me write the load-bearing pieces as code so the bookkeeping is unambiguous.

```python
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def leaky_relu(x, a=0.2): return torch.maximum(x * a, x)

def upscale2d(x, f=2):                       # nearest-neighbor 2x (element replication)
    N, C, H, W = x.shape
    return x.view(N, C, H, 1, W, 1).expand(N, C, H, f, W, f).reshape(N, C, H*f, W*f)

def downscale2d(x, f=2):                      # average pool
    return F.avg_pool2d(x, f)

# --- equalized learning rate ---------------------------------------------
# weights are N(0,1); the He constant c = gain/sqrt(fan_in) is applied at RUNTIME,
# so every weight has unit dynamic range and Adam's scale-invariant step moves
# all layers at the same effective learning rate.
class EqConv2d(nn.Module):
    def __init__(self, cin, cout, k, gain=np.sqrt(2)):
        super().__init__()
        self.w = nn.Parameter(torch.randn(cout, cin, k, k))     # N(0,1) init
        self.b = nn.Parameter(torch.zeros(cout))
        self.c = gain / np.sqrt(cin * k * k)                    # He constant, runtime scale
        self.pad = k // 2
    def forward(self, x):
        return F.conv2d(x, self.w * self.c, self.b, padding=self.pad)

class EqLinear(nn.Module):
    def __init__(self, cin, cout, gain=np.sqrt(2)):
        super().__init__()
        self.w = nn.Parameter(torch.randn(cout, cin))
        self.b = nn.Parameter(torch.zeros(cout))
        self.c = gain / np.sqrt(cin)
    def forward(self, x):
        return F.linear(x, self.w * self.c, self.b)

# --- pixelwise feature normalization (generator only) --------------------
# each pixel's feature vector -> unit RMS; no learnable params; caps magnitudes.
def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)

# --- minibatch standard deviation (appended near end of D) ---------------
# one constant extra feature map carrying the batch's average per-location stddev,
# so D can detect a batch that lacks variation; parameter-free.
def minibatch_stddev(x, group_size=4):
    N, C, H, W = x.shape
    G = min(group_size, N)
    y = x.view(G, -1, C, H, W)                  # split batch into groups of G
    y = y - y.mean(dim=0, keepdim=True)         # subtract per-group mean
    y = y.pow(2).mean(dim=0)                     # variance over the group
    y = (y + 1e-8).sqrt()                        # stddev over the group  [M,C,H,W]
    y = y.mean(dim=[1, 2, 3], keepdim=True)      # average over features+pixels -> [M,1,1,1]
    y = y.view(-1, 1, 1, 1).expand(N, 1, H, W)   # replicate to one feature map
    return torch.cat([x, y], dim=1)

# --- generator / discriminator blocks ------------------------------------
class GBlock(nn.Module):                         # upsample then two 3x3 convs
    def __init__(self, cin, cout, first=False):
        super().__init__()
        self.first = first
        if first:                                # 4x4: dense(z)->4x4 feature map, then conv
            self.dense = EqLinear(cin, cout * 16, gain=np.sqrt(2) / 4)
            self.conv  = EqConv2d(cout, cout, 3)
            self.cout  = cout
        else:
            self.conv0 = EqConv2d(cin, cout, 3)
            self.conv1 = EqConv2d(cout, cout, 3)
    def forward(self, x):
        if self.first:
            x = pixel_norm(x)                                  # normalize latent
            x = self.dense(x).view(-1, self.cout, 4, 4)
            x = pixel_norm(leaky_relu(x))
            x = pixel_norm(leaky_relu(self.conv(x)))
        else:
            x = upscale2d(x)
            x = pixel_norm(leaky_relu(self.conv0(x)))
            x = pixel_norm(leaky_relu(self.conv1(x)))
        return x

class DBlock(nn.Module):                         # two 3x3 convs then downsample (or the 4x4 head)
    def __init__(self, cin, cout, last=False, label_size=0):
        super().__init__()
        self.last = last
        if last:                                 # 4x4 head: +stddev map, conv, dense, score
            self.conv  = EqConv2d(cin + 1, cin, 3)
            self.dense0 = EqLinear(cin * 16, cout)
            self.dense1 = EqLinear(cout, 1 + label_size, gain=1)
        else:
            self.conv0 = EqConv2d(cin, cin, 3)
            self.conv1 = EqConv2d(cin, cout, 3)
    def forward(self, x):
        if self.last:
            x = minibatch_stddev(x)
            x = leaky_relu(self.conv(x))
            x = leaky_relu(self.dense0(x.flatten(1)))
            return self.dense1(x)
        x = leaky_relu(self.conv0(x))
        x = leaky_relu(self.conv1(x))
        return downscale2d(x)

# toRGB / fromRGB are 1x1 convs (per-pixel color projection / its inverse)
def toRGB(cin):   return EqConv2d(cin, 3, 1, gain=1)
def fromRGB(cout): return EqConv2d(3, cout, 1)

# --- progressive generator with fade-in ----------------------------------
# lod fractional part = the cross-fade alpha for the stage being introduced.
class Generator(nn.Module):
    def __init__(self, nf):                      # nf: feature maps per resolution stage
        super().__init__()
        self.blocks = nn.ModuleList([GBlock(512, nf[0], first=True)] +
                                    [GBlock(nf[i-1], nf[i]) for i in range(1, len(nf))])
        self.torgb = nn.ModuleList([toRGB(c) for c in nf])
    def forward(self, z, lod):
        top = len(self.blocks) - 1 - int(np.floor(lod))     # index of the current top block
        alpha = lod - np.floor(lod)                          # fade weight in [0,1)
        x = self.blocks[0](z); prev = x
        for i in range(1, top + 1):
            prev = x; x = self.blocks[i](x)                  # prev = features before the top block
        img = self.torgb[top](x)
        if alpha > 0 and top >= 1:                           # blend in the new (top) stage
            old = upscale2d(self.torgb[top - 1](prev))       # old toRGB of pre-top features
            img = (1 - alpha) * old + alpha * img
        return img

# --- training step --------------------------------------------------------
def process_reals(x, lod):
    # blend real images between the two resolutions by the same alpha,
    # then upscale to the network's working size.
    alpha = lod - np.floor(lod)
    if alpha > 0:
        blur = upscale2d(downscale2d(x))                     # one-octave-lower version
        x = (1 - alpha) * blur + alpha * x
    f = int(2 ** np.floor(lod))                              # upscale to the active working size
    return upscale2d(x, f) if f > 1 else x

def train_step(G, D, G_opt, D_opt, reals, lod, lam=10.0, drift=1e-3):
    z = torch.randn(reals.size(0), 512, device=reals.device)
    z = z / z.norm(dim=1, keepdim=True)                      # latent on the hypersphere
    reals = process_reals(reals, lod)

    # --- discriminator step (WGAN-GP + drift) ---
    fake = G(z, lod).detach()
    d_real, d_fake = D(reals, lod), D(fake, lod)
    eps = torch.rand(reals.size(0), 1, 1, 1, device=reals.device)
    xhat = (eps * reals + (1 - eps) * fake).requires_grad_(True)
    g = torch.autograd.grad(D(xhat, lod).sum(), xhat, create_graph=True)[0]
    gp = (g.flatten(1).norm(dim=1) - 1).pow(2).mean()
    d_loss = d_fake.mean() - d_real.mean() + lam * gp + drift * d_real.pow(2).mean()
    D_opt.zero_grad(); d_loss.backward(); D_opt.step()

    # --- generator step ---
    g_loss = -D(G(z, lod), lod).mean()
    G_opt.zero_grad(); g_loss.backward(); G_opt.step()
    # (maintain an EMA copy of G's weights, decay 0.999, for evaluation sampling.)
```

So the causal chain: high resolution makes the discriminator near-perfect almost immediately because three million pixels expose every flaw, and a near-perfect discriminator on barely-overlapping distributions returns a useless gradient — so instead of learning the latent→megapixel map at once, grow the networks from 4×4 upward, keeping the game fair at the current resolution and turning the problem into a sequence of easy octave-by-octave refinements that is also far cheaper; bring each new resolution in by cross-fading it as a residual branch (toRGB/fromRGB blended by α, with the real images blended to match) so the trained lower layers are never shocked; append a parameter-free minibatch-standard-deviation feature map to the discriminator so it can detect and punish a generator that drops variation; replace BatchNorm with parameter-free pixelwise feature normalization in the generator so the magnitudes can't escalate in the two-network competition; and initialize all weights at unit scale while applying the He constant at runtime so that Adam's scale-invariant updates move every layer at the same effective learning rate — all on top of a Wasserstein-gradient-penalty critic with a tiny drift term keeping its output honest.
