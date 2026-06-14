Let me start from what actually goes wrong when I train an image-producing network. I have a network that emits an image `yhat` and a target image `y`, and I optimize squared pixel error `||yhat - y||_2^2` because that is the obvious differentiable distance. The results come out soft — edges blurred, fine texture washed away — and they look worse than the number suggests. So before I reach for any clever loss, I want to understand *why* this is structural, not a tuning failure. Two pixels at the same coordinate are compared in isolation; the loss has no idea that they sit inside an edge or a texture, no idea that a faint global blur destroys structure a person sees instantly while barely moving the pixel values. That is the asymmetry that bites: blur is a tiny `L2` change and a huge perceptual change. And there is a second, sharper reason the *training* blurs. Suppose for a given input several outputs are all plausible — say two equally good fine textures `v1` and `v2`. The predictor that minimizes expected squared error to a randomly drawn target is the conditional mean, and the mean of the two sharp textures is `(v1+v2)/2`, which is neither texture: it is their blurry average. So whenever the problem is genuinely ambiguous, squared error *rewards* hedging toward the mean, and the mean of sharp things is smooth. The metric I optimize is not the thing I care about, and the gap is exactly the thing I keep seeing — softness.

So what I really want is a *perceptual* distance: a function that is small exactly when a human would say the two images look the same, and large when they look different. If I had that and it were differentiable, I would just train under it and be done — the measurement problem and the supervision problem are the same coin. The trouble is that "looks the same to a person" is notoriously hard to write down: human similarity depends on high-order structure, it is context-dependent, it may not even obey the triangle inequality. The instinct is to collect human judgments and fit a function straight to them — show people images, regress a similarity score. But the judgment that people can reliably give is *relative and pairwise* — "is A or B more similar to this reference?" — not an absolute number, and fitting a free-form function to pairwise, context-laden judgments is the kind of thing that overfits the seen distortions and fails to generalize to new ones. Let me hold that worry; I will come back to whether a directly-fit metric generalizes, because I suspect it does not, and that suspicion will end up shaping the design.

What is already on the table to do better than `L2`? The hand-designed perceptual metrics — SSIM and its multiscale and feature-similarity descendants. SSIM compares two images through local luminance, contrast, and structure statistics over a sliding window, multiplies them, averages spatially. It genuinely beats `L2` on a lot of distortions, and that is real progress. But stare at its assumptions. Every term it computes is a shallow, fixed function of pixel statistics in a *registered* window: it lines up location `(i,j)` in one image with `(i,j)` in the other and compares the local stats. The moment the distortion is geometric — a one-pixel shift, a small warp, a resample — the window in image A no longer contains the same structure as the window in image B, and the comparison falls apart even though a person barely notices the shift. SSIM was simply not built for spatial ambiguity. And because it is hand-designed with no learning, there is no knob to *teach* it which differences humans care about. It is a clever guess at perception, frozen.

Now the other thread, the one that actually smells like the answer. The synthesis community noticed something I should take seriously: the *internal activations* of a deep network trained for ImageNet classification form a space where Euclidean distance behaves much more like perception than raw pixels. Gatys showed image "content" is captured by squared error between high-level VGG feature maps, and "style/texture" by the Gram matrices of those features. Johnson made it a training loss — the feature reconstruction loss `(1/(C_j H_j W_j))||phi_j(yhat) - phi_j(y)||_2^2` over a fixed VGG layer `phi_j` — and networks trained under it come out sharper and more semantically faithful than pixel-loss networks. People started calling any such feature-space distance a "perceptual loss." Why on earth would features trained to *classify* objects be good at measuring perceptual *similarity*? Here is the way I can make myself believe it: to classify well, a representation has to be invariant to exactly the nuisances humans ignore (small shifts, lighting, noise) and sensitive to exactly the structure humans attend to (edges, parts, textures, shapes). A representation tuned to be predictive of semantically important structure in the world is, almost by construction, a representation in which Euclidean distance tracks the kind of similarity people perceive. Perception need not be a special-purpose function; it could be a *consequence* of having learned a good representation. If that hypothesis is right, it has a strong, testable edge: the effect should not be unique to VGG, or even to ImageNet supervision — any representation that learned to predict useful structure should work, and a network that learned *nothing* (random weights) should not.

But "perceptual loss" is a name that outran its evidence, and that is the wound to probe. Johnson and Gatys use a hand-picked layer and weight every channel and every spatial location *equally*, with zero grounding in human judgments. Nobody has checked how perceptual these distances actually are, which channels carry the signal, whether the architecture or the supervision matters, or whether training is needed at all. So step one is not to invent; it is to *measure*. I need human judgments over a much wider distortion space than the existing IQA datasets cover — crucially including the artifacts that deep networks themselves produce (autoencoding, denoising, colorization, super-resolution outputs), because those are what a perceptual metric will face in the wild — and I need the judgments in the form humans give reliably: a two-alternative forced choice, "which of these two is closer to the reference?" I will collect that, on patches rather than whole images, because patches keep the comparison at the low-level perceptual scale and dodge the high-level "different senses of similarity" problem that makes whole-image judgments context-dependent. With that data in hand I can finally ask the questions empirically.

Now, how do I turn a network's activations into a single distance, carefully, so that the answer is not an accident of feature magnitudes? I extract activations from `L` chosen layers of the backbone for both images. The first thing I have to confront is that channels within a layer have wildly different magnitudes; if I take a raw Euclidean distance, whichever channels happen to be large will dominate, regardless of whether they are perceptually meaningful. That contaminates everything. So I unit-normalize the activations in the channel dimension first — at each spatial location, divide the channel vector by its own length. Call the normalized features `yhat^l` for layer `l`. Now every channel sits on the same scale, and what decides a channel's influence is no longer its raw magnitude but a weight I get to choose. Then I take the squared difference per channel, weight it, sum over channels, average over the spatial positions of the layer, and sum across layers:

```
d(x, x0) = sum_l (1 / (H_l W_l)) * sum_{h,w} || w_l ⊙ (yhat^l_{hw} - yhat0^l_{hw}) ||_2^2 ,
```

with `w_l` a non-negative per-channel weight vector for layer `l`. Look at the special case `w_l = 1` for all `l`: the weighted squared distance on unit-normalized vectors is just `||yhat - yhat0||^2 = 2 - 2 (yhat · yhat0)`, which is (twice) one-minus-cosine-similarity — so with uniform weights this *is* cosine distance in feature space averaged over layers, the off-the-shelf perceptual loss. That is the right baseline to sit inside my construction, because it means the only new degree of freedom is `w_l`, and setting it to one recovers exactly what Gatys and Johnson were doing. Good — the new thing is small and interpretable.

Why a *learned, per-channel* weight and not just cosine distance? Because not every feature channel is equally perceptual. Some channels in a classification network respond to things humans weight heavily in similarity, others to things we barely notice; uniform weighting throws that information away. So I want to *calibrate*: learn `w_l` from the human judgments. And why must `w_l` be non-negative? Because the weight scales a squared feature difference into the distance; if I allowed a negative weight, increasing the difference in that channel would *decrease* the total distance, which is nonsense — moving two patches apart in some feature should never make the metric call them closer. So I constrain `w_l >= 0`, enforced the blunt way: after each gradient step, clip any negative weight to zero (project onto the non-negative orthant). This is a calibration of an existing feature space — a tiny number of parameters on top of a frozen network (for VGG's five conv layers it is on the order of a thousand-odd weights), not a retraining. And the cleanest way to *implement* the weighted channel-sum on the squared differences is a `1x1` convolution with a single output channel and no bias: a `1x1` conv over `C` input channels into one output *is* exactly `sum_c w_c * (·)_c`, so I square the normalized feature differences first and let the conv carry the non-negative `w_l`.

Now I need a loss to train `w_l` against the 2AFC data, and this is where my earlier worry about directly-fit metrics pays off. Each training item is a triplet `(x, x0, x1, h)`: a reference and two distortions, with `h in {0,1}` the human's choice of which is closer. From my metric I get two distances `d0 = d(x,x0)` and `d1 = d(x,x1)`. The naive thing is a ranking loss: force a fixed margin so the chosen patch's distance is smaller by some constant. But a fixed margin is wrong — how much closer the preferred patch *should* be depends on the case, and humans on a hard triplet split their votes. So instead of hard-coding the map from `(d0,d1)` to a preference, I learn it: a small network `G` takes the two distances and outputs a probability `hhat in (0,1)` that patch 0 is the chosen one, trained with binary cross-entropy against `h`:

```
L(x,x0,x1,h) = - h log G(d0,d1) - (1-h) log(1 - G(d0,d1)) .
```

Conceptually `G` is just a couple of small fully-connected layers into a sigmoid — enough to learn a soft, case-dependent comparison of the two distances rather than a rigid margin. When two judges on a training pair disagree, I set the target to `0.5`, which is honest about the ambiguity. One refinement I want once I write the code: feeding `G` the bare pair `(d0,d1)` makes it relearn obvious relational features from scratch, so I hand it the relational quantities directly — the difference `d0-d1` and the two ratios `d0/(d1+eps)`, `d1/(d0+eps)` alongside `d0,d1` — five inputs that make "which is smaller, and by how much relatively" trivially available; small `1x1` convs with LeakyReLU then map those five to the logit. This BCE-through-a-learned-comparator handles the noisy, pairwise nature of the judgments far better than a fixed-margin rank loss — which I confirm beats the rank-loss alternative when I try both. And critically, because the *only* free parameters being fit to human data are the small `w_l` (and `G`) sitting on top of a frozen, generally-trained representation, I am not free-form-fitting a similarity function to the seen distortions — the heavy lifting is done by the pretrained features, and calibration only re-weights them. That is the structural reason this should generalize where a from-scratch fit to judgments would overfit: most of the metric's competence is borrowed from a representation that learned about the world, not memorized from the distortion set.

Let me make sure I actually answer the questions I set out to, in my head from what this measurement will show. Across architectures — a tiny SqueezeNet, AlexNet, VGG — the feature distances should land at similar, high agreement with humans, well above SSIM/FSIM/`L2`, which would say the effect is not about one special network. If I swap supervised ImageNet training for self-supervised objectives — solving jigsaw puzzles, cross-channel prediction, generative modeling — and they perform on par, that says it is not about classification labels but about *having learned useful structure at all*. And a randomly-initialized network with the same architecture should be much worse, pinning the effect on the *training signal*, not the architecture. That is the shape of the emergent-property claim, and it is falsifiable in exactly these comparisons. The uniform-weight cosine version already beats the hand-designed metrics; the learned `w_l` calibration then squeezes out a further, safe improvement, especially on the real-algorithm tasks I most care about. So my landing point for the *measurement* problem is: a learned, channel-calibrated, feature-space distance — call it LPIPS, the learned perceptual image patch similarity — with the distance formula above and the BCE-trained weights.

But measurement was never the whole point; I wanted a *training loss*. And here LPIPS slots in beautifully, because the metric is differentiable in `yhat`: feed the produced image and the target through the frozen backbone, normalize, take weighted squared feature differences, sum — every operation is differentiable, the only frozen thing is the backbone and the calibrated `w_l`, and gradients flow back into the generator. So I can train an image network to minimize `LPIPS(yhat, y)` directly, and now my optimization target is finally aligned with perception instead of with pixel-aligned squared error. Concretely, in the velocity-based generator I do not want to disturb, the network does not emit an image directly, it emits a velocity, but the linear path is `z_t = (1-t)x + t*eps` with conditional velocity `v = eps - x`, and rearranging the *same* equation gives an image estimate: `x_hat = z_t - t*v_pred`. That is the `yhat` I apply the perceptual loss to, and applying image-space supervision there does not change what the network's main target means — the velocity MSE stays the anchor, the image losses ride along as auxiliaries.

Before I go further I should understand how this auxiliary actually couples back to the velocity across the noise level `t`, because the time-dependence will decide a schedule. The clean image satisfies `x_t - t*v_target = (1-t)x + t*eps - t(eps - x) = (1-t)x + t*x = x` exactly, so the *true* velocity reconstructs `x` with no residual. Then `x_hat - x = (x_t - t*v_pred) - (x_t - t*v_target) = t*(v_target - v_pred)`: the error in the implied denoised image is exactly `t` times the velocity error. So at small `t` the implied image is *not* ill-conditioned or exploding — it is the opposite, the image error vanishes linearly in `t`, and the gradient of any image-space loss back to `v_pred` is scaled by `dx_hat/dv_pred = -t`, i.e. *weakened* by `t`. The honest reading: small `t` is a weak-leverage regime, where `z_t` is already nearly clean and the image loss has little to add over the fact that the sample is barely noised. At the other end, `t` near one, `x_hat` is reconstructed by subtracting a large `t` times a velocity from a nearly-pure-noise `x_t`, so the implied image is too unreliable to be a meaningful perceptual target. So the schedule writes itself from the two ends: concentrate the image-space terms on *low-noise but non-infinitesimal* samples. A factor `(1-t)^2` rises toward the clean endpoint where `x_hat` is a faithful image, and a hard gate `1[t > 0.1]` drops the tiny-`t` region where the auxiliary's velocity leverage has shrunk to nearly nothing; together, `perceptual_w = (1-t)^2 * 1[t > 0.1]`. I will also clamp `x_hat` to `[-1,1]` before any perceptual term, because early in training the implied image can leave the valid range and the feature backbone (and the FFT amplitude comparison against a `[-1,1]` clean image) expects inputs in that range.

LPIPS, for all that it is the right *core*, is not by itself a complete training loss, and I can predict its blind spots from how it is built. It is a feature-space distance: high-order, semantic, and somewhat spatially tolerant — which is exactly why it is perceptual, but it also means it does not pin down every fine local detail. A network can satisfy LPIPS while leaving edges a touch soft, because the metric forgives small spatial slop and aggregates over space. So I want a complementary term that puts *direct* pressure on local detail. The blur problem again — and Mathieu already diagnosed it and built the antidote: penalize the difference of *image gradients*, not just of pixels. His Gradient Difference Loss compares the finite-difference neighbor-differences of prediction and target,

```
L_gdl = sum_{i,j} | |Y_{i,j}-Y_{i-1,j}| - |Yhat_{i,j}-Yhat_{i-1,j}| |^alpha
                 + | |Y_{i,j-1}-Y_{i,j}| - |Yhat_{i,j-1}-Yhat_{i,j}| |^alpha ,
```

and with `alpha=1` this is an `L1` on the horizontal and vertical neighbor differences — a finite-difference edge loss. Why does this help where LPIPS does not? Because edges *are* large local gradients, and matching the gradient field forces the prediction's edges to land where the target's edges are, with the right sharpness; an `L2` or even a feature loss can average a sharp edge into a ramp and pay little, but a gradient loss sees the ramp's reduced slope immediately. So a small finite-difference gradient `L1` term, on `x_hat` vs the clean image. Why keep it *small* relative to LPIPS, though? Because it is a low-level, single-pixel-neighborhood term with no semantics — useful as a sharpener, dangerous as a primary objective (chasing gradients alone invites high-frequency noise that looks like detail but is not structure). LPIPS leads; the gradient term refines.

The next gap I can foresee is *scale*. Both LPIPS (computed at the backbone's resolutions) and a one-pixel gradient term are biased toward fairly fine spatial frequencies; coarse, large-scale structure — overall layout, low-frequency shading — can drift while the fine terms are satisfied. The standard fix is multi-scale supervision: downsample prediction and target by successive factors and compare at each resolution, so coarse structure gets its own error signal. So a multi-scale downsampled `L1` term, again small, again on `x_hat` vs clean. This is the Mathieu/LapSRN pyramid idea reused as a loss term rather than an architecture. It plugs the coarse-structure hole the fine terms leave.

I could stop here — feature-space LPIPS for perceptual structure, a gradient term for edges, a multiscale term for coarse layout — and that is a solid spatial-domain perceptual stack. But there is one more blind spot, and it is the one all three share, because they are all *spatial-domain* and mostly *local*. Let me think about where their gradients vanish. A generator trained under spatial losses systematically loses high-frequency content — the fine, repeating texture that lives at the top of the spectrum. Why? Because each spatial loss compares values (or local differences) at locations, and a slightly-too-smooth output is, location by location, only a little wrong; the *aggregate* deficit of high-frequency energy across the whole image never gets concentrated into a strong, coherent gradient. The error is diffuse in space, so spatial losses see it weakly. But the same deficit is *concentrated* in the frequency domain: it is precisely a shortfall of amplitude in the high-frequency coefficients. If I transform both images by the FFT and compare their spectra, the missing detail becomes a large, focused error exactly where it lives. That is the case for a frequency-domain term — not as a replacement, but because it sees what the spatial terms are structurally blind to.

So let me derive the frequency term carefully rather than bolt on an FFT. Take the 2D discrete Fourier transform of an image channel, `F{x}_{u,v} = (1/sqrt(HW)) sum_{h,w} x_{h,w} exp(-i 2*pi (u h / H + v w / W))`. Each complex coefficient splits into amplitude `|F{x}_{u,v}| = sqrt(R^2 + I^2)` and phase. What do I want to penalize — the complex coefficients, the amplitude, the phase? The failure mode I am fixing is *missing energy at a frequency*, and energy is amplitude; phase encodes *where* structure sits, which the spatial terms already handle and which is sensitive to the small shifts I deliberately want to tolerate. So I should compare *magnitudes*: `| |F{yhat}_{u,v}| - |F{y}_{u,v}| |`. Penalizing amplitude difference puts pressure on the spectrum's energy distribution — fill in the high frequencies — while staying tolerant to the positional shifts that an amplitude is invariant to anyway. An `L1` (not `L2`) on the amplitude difference, to match the robust, sparse-error flavor of the other terms and avoid letting a few huge coefficients dominate.

One more thing falls out for free. The image is real, so its Fourier transform is Hermitian-symmetric: `F{x}_{u,v} = conj(F{x}_{-u,-v})`. That means half the spectrum is redundant — the negative-frequency half is just the conjugate of the positive-frequency half, so its magnitudes are identical. I do not need to compute or penalize the whole complex grid; the real-input FFT (`rfft`) returns exactly the non-redundant half, and averaging the amplitude `L1` over that half loses no information (it just counts each independent frequency once). The careful Fourier-loss formulations normalize this with a `2/(UV)` factor over the half-spectrum, but for an auxiliary term a plain mean over the returned coefficients carries the same gradient direction with a fixed scale I will fold into the term's weight anyway. So the spectral term is: take `rfft2` of `x_hat` and of the clean image per channel, take the absolute value (amplitude), and average the `L1` of their difference,

```
L_spec = mean( | rfft2(x_hat).abs() - rfft2(x).abs() | ) .
```

This is global guidance — every coefficient summarizes the whole image, unlike the pixel-local spatial terms — aimed straight at the high-frequency detail the spatial stack underproduces. Spatial and frequency are complementary: one localizes and carries semantics, the other globally enforces the energy spectrum.

Now I have to weight the five terms, and I want each coefficient justified, not tuned blind. The velocity MSE stays unscaled, at weight one — it is the *correctness anchor*: the perceptual terms make the image look right, but the network's actual job is to predict the right velocity, and I must never let the auxiliary terms pull the prediction off the velocity target. The four image-space terms ride on top, and their relative sizes follow their roles: LPIPS is the *primary* perceptual signal, so it gets the largest coefficient, `0.5`; the gradient term is the next strongest sharpener at `0.3`; the multiscale and spectral terms are complementary refinements at `0.2` each. The ordering `0.5 > 0.3 > 0.2 = 0.2` is the ordering of how central each is to perceptual quality — feature-space first, then edges, then the two structural/spectral helpers — and keeping them all well below the MSE's effective weight respects the anchor. These four coefficients are not handed down by any of the prior work; they are the stack's own knobs, and the defensible thing is the *ordering*, set by centrality, not the exact values.

Let me assemble the whole objective, per sample:

```
loss = ||v_pred - v_target||^2                          # unscaled velocity anchor
     + perceptual_w * ( 0.5 * LPIPS(x_hat, x)           # primary perceptual signal
                      + 0.3 * L_grad(x_hat, x)           # finite-difference edge L1
                      + 0.2 * L_multi(x_hat, x)          # multi-scale downsampled L1
                      + 0.2 * L_spec(x_hat, x) )         # FFT-amplitude L1 (spatial+freq)
perceptual_w = (1 - t)^2 * 1[t > 0.1] ,   x_hat = x_t - t * v_pred .
```

Before I write it, let me re-examine LPIPS itself in code, because the metric is the load-bearing piece and I want its forward pass exact. Push both images through a frozen pretrained backbone (VGG is the de-facto choice for synthesis; the lighter AlexNet/SqueezeNet also work, which is the point of the architecture-invariance finding). The released metric standardizes its input first — optionally mapping `[0,1]` to `[-1,1]`, then a fixed per-channel affine shift/scale — before the backbone, so I keep that. At each chosen layer, unit-normalize the activation in the channel dimension — divide by the per-location channel-norm, with a tiny epsilon so a zero activation does not divide by zero. Take the squared difference of the two normalized feature maps. Apply the learned per-channel weights as the `1x1` conv with non-negative weights. Spatially average over `H,W`. Sum across layers. That is `d(x,x0)` from the formula, term for term. For training, the small comparator `G` and the BCE loss fit `w_l`; for *use as a loss* I just call the calibrated metric and backprop.

```python
import torch
import torch.nn as nn


def normalize_tensor(x, eps=1e-10):
    # unit-normalize in the channel dimension at each spatial location:
    # puts every channel on the same scale so w_l, not raw magnitude, decides weight
    norm = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))
    return x / (norm + eps)


def spatial_average(x):
    return x.mean([2, 3], keepdim=True)             # average over H, W of a layer


class NetLinLayer(nn.Module):
    """The learned per-channel weight w_l: a 1x1 conv (no bias) = weighted channel sum,
    applied to the squared normalized feature differences. Weights projected to >= 0."""
    def __init__(self, chn_in):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False))


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity: channel-normalized, learned-weighted,
    feature-space L2, averaged over space and summed over layers."""
    def __init__(self, backbone, chns):
        super().__init__()
        self.net = backbone                          # frozen pretrained feature extractor
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.L = len(chns)
        self.lins = nn.ModuleList([NetLinLayer(c) for c in chns])   # one w_l per layer

    def forward(self, x, x0):                         # inputs in [-1, 1]
        outs0, outs1 = self.net(x), self.net(x0)     # per-layer activations
        val = 0
        for kk in range(self.L):
            f0 = normalize_tensor(outs0[kk])         # channel unit-normalize
            f1 = normalize_tensor(outs1[kk])
            diff = (f0 - f1) ** 2                     # squared feature difference
            # 1x1 conv applies non-negative w_l and sums channels; avg over space
            val = val + spatial_average(self.lins[kk].model(diff))
        return val                                    # d(x, x0); sum over layers
```

And the training calibration that fits `w_l` against the human 2AFC judgments — frozen backbone, learn only `w_l` (and the comparator), project `w_l` to non-negative after each step. The comparator takes the five relational distance features so "which is smaller, relatively" is handed to it directly:

```python
class Dist2Logit(nn.Module):
    """G: maps a pair of distances to P(patch 0 is the human's choice), via the five
    relational features (d0, d1, d0-d1, d0/(d1+eps), d1/(d0+eps))."""
    def __init__(self, chn_mid=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1), nn.Sigmoid(),
        )

    def forward(self, d0, d1, eps=0.1):
        return self.model(torch.cat(
            (d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


def calibrate(metric, comparator, triplet_loader, opt):
    bce = nn.BCELoss()
    for x, x0, x1, h in triplet_loader:              # reference, two distortions, human choice in {0,1}
        d0, d1 = metric(x, x0), metric(x, x1)        # distances from the metric
        hhat = comparator(d0, d1)                     # learned soft comparison, not a fixed margin
        loss = bce(hhat, h)                           # -h log G - (1-h) log(1-G)
        opt.zero_grad(); loss.backward(); opt.step()
        for lin in metric.lins:                       # project w_l onto w >= 0:
            for p in lin.parameters():                # a larger feature diff must never reduce distance
                p.data.clamp_(min=0)
```

Now the full training loss, filling the one empty slot in the velocity-based generator harness, with the perceptual terms riding on the implied denoised image. I keep the shape handling explicit: reshape `t` for image broadcasting, clamp only the images passed to the perceptual helpers, allocate per-sample zero losses so masked-out samples contribute only the velocity MSE, and read the device from `v_pred` so nothing lands on the wrong device:

```python
import torch


def lpips_spectral_auxiliary(v_pred, v_target, x, x_t, t,
                             lpips_fn, compute_gradient_loss, compute_multiscale_loss):
    """LPIPS + finite-difference-gradient + multiscale + FFT-amplitude on the implied
    denoised image x_hat = x_t - t*v_pred, scheduled by (1-t)^2 and masked at small t.
    Returns the per-sample auxiliary loss; the velocity MSE is added outside."""
    B = v_pred.shape[0]
    t_img = t.reshape(B, *([1] * (v_pred.ndim - 1)))               # broadcast t over image dims
    t_flat = t_img.flatten()

    x_hat = x_t - t_img * v_pred                                   # implied denoised image
    mask = t_flat > 0.1                                            # drop weak-leverage small-t
    weight = ((1.0 - t_flat) ** 2) * mask.float()                 # (1-t)^2 schedule

    zeros = torch.zeros(B, device=v_pred.device, dtype=v_pred.dtype)
    loss_lpips = zeros.clone()
    loss_grad  = zeros.clone()
    loss_multi = zeros.clone()
    loss_spec  = zeros.clone()
    if mask.any():
        xh = x_hat[mask].clamp(-1, 1).float()                     # backbone/FFT expect [-1,1]
        xc = x[mask].clamp(-1, 1).float()
        loss_lpips[mask] = lpips_fn(xh, xc).view(-1).float()      # primary perceptual signal
        loss_grad[mask]  = compute_gradient_loss(xh, xc).view(-1).float()   # edge L1
        loss_multi[mask] = compute_multiscale_loss(xh, xc).view(-1).float() # coarse-structure L1
        # FFT amplitude L1: real-input rfft2 (half spectrum, Hermitian symmetry), |.|, L1 of diff
        fh = torch.fft.rfft2(xh, dim=(-2, -1)).abs()
        fc = torch.fft.rfft2(xc, dim=(-2, -1)).abs()
        loss_spec[mask] = (fh - fc).abs().mean(dim=(1, 2, 3)).float()

    return weight * (0.5 * loss_lpips      # feature-space perceptual: largest
                     + 0.3 * loss_grad     # edges: next
                     + 0.2 * loss_multi    # coarse structure: complementary
                     + 0.2 * loss_spec)    # frequency spectrum: complementary
```

The full objective is then the velocity MSE plus this auxiliary, averaged over the batch. Let me trace the whole causal chain back to make sure it holds together. I started stuck because squared pixel error is not perceptual: it treats pixels as independent and positions as fixed, so it is blind to structure and to spatial ambiguity, and as a training loss it drives the predictor to the blurry mean of plausible outputs. The hand-designed perceptual metrics (SSIM and kin) improve on `L2` but are shallow, fixed, and assume registration, so they break on geometric distortion and cannot be taught. The synthesis community's feature-space "perceptual loss" works far better, but it is uncalibrated — equal weights, hand-picked layers, no grounding in human judgments — so it was unknown which part is perceptual. The resolution was to *measure*: collect a wide 2AFC patch-judgment dataset including CNN artifacts, define a feature-space distance that unit-normalizes channels (so weights, not magnitudes, decide), weights each channel by a learned non-negative `w_l` via a `1x1` conv (recovering cosine distance at `w=1`), averages over space and layers, and calibrate `w_l` with a BCE-trained soft comparator fed the five relational distance features rather than a fixed-margin rank loss — fitting only a thin layer on top of a frozen, generally-trained representation, which is why it generalizes where a from-scratch fit overfits. The measurement confirms the distance tracks humans across architectures and even self-supervised training (an emergent property of having learned structure, not of classification labels), beating the shallow metrics. Because that distance is differentiable, it becomes the training loss I wanted, applied — in the velocity-based generator — to the denoised image implied by the predicted velocity, `x_hat = x_t - t*v_pred`, whose error is exactly `t*(v_target - v_pred)`, so the image-space gradient back to the velocity is scaled by `t`. LPIPS alone leaves fine and coarse and high-frequency detail under-supervised, so I added complementary terms each derived from where LPIPS is blind: a finite-difference gradient `L1` for edges, a multi-scale `L1` for coarse structure, and an FFT-amplitude `L1` for the high-frequency energy that all spatial terms diffusely miss, using `rfft2` because Hermitian symmetry makes half the spectrum redundant and penalizing amplitude (not phase) because the failure is missing energy and amplitude is shift-tolerant. The five terms are weighted by their centrality — unscaled MSE anchor, then `0.5/0.3/0.2/0.2` — scheduled by `(1-t)^2` so they concentrate where `x_hat` is a faithful image and the velocity leverage is real, and gated below `t=0.1` where that leverage has shrunk to nearly nothing, with a `[-1,1]` clamp so the backbone and the spectral comparison see valid inputs.
