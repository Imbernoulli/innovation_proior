Let me start from what actually bothers me. I can train a progressive-growing GAN today and it spits out gorgeous 1024×1024 faces. The FID is great. And yet I cannot *do* anything with it. I sample a `z`, I get a face. I want the same face with different hair — I have no handle. I want the same pose but a different person — no handle. I perturb one coordinate of `z` and the pose shifts and the identity shifts and the lighting shifts, all at once, all tangled. The generator is a black box that happens to produce nice pictures. The single thing I keep coming back to is: every visible property of the image is being controlled by `z` through the input layer, jointly, and I have no idea which part does what.

And there's a second thing nagging me. Look at the fine detail in a generated face: individual strands of hair, freckles, the speckle in an iris, pores. Real faces have a huge amount of this micro-randomness, and it should be *random* — if I imagine the "same" face twice it can have the hairs arranged differently without becoming a different person. But my generator's only input is a single deterministic `z`. So wherever that micro-randomness comes from, the network must be *manufacturing* spatially-varying pseudo-randomness internally, out of `z`, through the conv stack. That's a strange thing to ask a network to do. It costs capacity, and it's hard to hide the periodicity of a manufactured signal — which is exactly why I keep seeing those faint repeating textures in generated images. The network is bad at faking noise, and it shouldn't have to.

So the two pains are: no scale-specific control, and no clean source of stochastic detail. Let me hold both and look for tools.

Where would control over "what an image looks like at a given scale" come from? I keep thinking about style transfer, because that whole field is precisely about separating *what* an image depicts from *how* it looks. And the technical heart of fast style transfer is normalization. Let me reconstruct that carefully, because it's the closest existing machinery I know for the "how it looks" half of my problem.

Instance normalization: take a feature map, and for each sample `n` and each channel `c`, normalize over the spatial extent — `μ_nc = (1/HW)Σ_{h,w} x_nchw`, `σ_nc = sqrt((1/HW)Σ(x_nchw − μ_nc)² + ε)`, and then `IN(x) = γ·(x − μ(x))/σ(x) + β`. For style transfer this beats batch norm by a mile, and the reason people figured out is the interesting part. Someone trained matched IN and BN models on (a) original images, (b) contrast-normalized images, and (c) *style*-normalized images. IN's edge over BN survives contrast normalization, but it mostly *vanishes* once the images are already style-normalized. The only way to read that: instance normalization is itself doing style normalization. It's stripping the per-instance style and leaving content. And what is "style" in feature space? Gatys established it's the statistics of the feature maps — the Gram matrix — and it was later shown that matching just the per-channel mean and variance is essentially equivalent to matching the Gram matrix for this purpose. So the per-channel mean and variance of a feature map *are* the style. Instance norm wipes them; that's why it removes style.

Now there's a move people made on top of this. If instance norm wipes the style by killing `(μ, σ)` per channel, then maybe I can *write a new style in* by choosing the affine `(γ, β)` I put back. Conditional instance norm did exactly that — one network, many styles, by learning a separate `(γ^s, β^s)` per style `s`: `CIN(x; s) = γ^s·(x − μ(x))/σ(x) + β^s`. Same convolutions, you just swap the affine pair, and out comes a completely different style. Sit with that for a second: the convolution weights don't change at all, and yet the output style changes totally, purely from the per-channel scale and bias. So the per-channel affine of a normalization layer is enough to dictate style on its own — a remarkably low-dimensional handle.

CIN's limit is that the styles are a fixed learned set — `2FS` parameters, one pair per style, no arbitrary new styles. AdaIN fixed that: don't learn the affine, *compute* it from a style image's own statistics. `AdaIN(x, y) = σ(y)·(x − μ(x))/σ(x) + μ(y)`. Normalize content `x` to zero-mean unit-variance per channel, then re-scale and re-bias to the *style's* per-channel `(σ(y), μ(y))`. No learnable parameters, works for any style, costs almost nothing.

Here's where something clicks against my first pain. In style transfer the `(σ(y), μ(y))` come from an example *image*. But the operation doesn't care where those two numbers per channel come from — they're just a scale and a bias. What if I don't have a style image at all? What if the scale-and-bias *is* how I inject my latent into the generator? Instead of "style of this painting," it's "style of this `z`." I run my latent through something that emits a scale and a bias per channel, and I apply AdaIN at a convolution layer. The latent becomes a *style*.

And now the control I wanted starts to look reachable, because of *where* I can apply this. A progressive-growing generator already has a stack of conv layers, low resolution to high. Coarse layers shape big things (pose, face shape); fine layers shape small things (skin, hair texture). If I inject a fresh style — a fresh per-channel scale and bias — at *every* layer, then the latent is controlling the image *scale by scale*. The coarse-layer styles steer coarse attributes; the fine-layer styles steer fine attributes. That's exactly the scale-specific control I couldn't get when `z` only entered at the bottom.

But will the styles actually *stay* localized to their layer, or will a style applied at layer 5 just leak forward and contaminate everything downstream? Let me check the mechanics by tracing the per-channel statistics through two layers by hand. Take one channel with an incoming feature map `x = [1, 3, 2, 0]` over four positions. Style `k` is "scale 5, bias 10," applied as AdaIN: normalize `x` to `(x−μ)/σ`, then `5·x̂ + 10`. After normalization `x̂` has mean 0 and std 1, so `5·x̂ + 10` has mean exactly 10 and std exactly 5 — the channel's statistics are now precisely `(b_k, s_k) = (10, 5)`, the numbers style `k` wrote. Good: a style fully determines its layer's output per-channel mean and variance.

Now push that to layer `k+1`. Suppose the intervening convolution were the identity (I just want to know what happens to the *statistics*). Style `k+1` is "scale 2, bias −3," and AdaIN normalizes *first*: it subtracts the incoming mean (10) and divides by the incoming std (5), which lands back at mean 0, std 1 — the `(10, 5)` that style `k` wrote is gone — and then writes `2·x̂ + (−3)`, giving mean −3, std 2, i.e. `(b_{k+1}, s_{k+1})`. So as far as the *channel statistics* go, style `k` has been completely overwritten by the normalization step of layer `k+1`; it does not accumulate. To see that the normalization is load-bearing here, drop it: adding style `k+1` directly on top of style `k`'s output, `2·(5x̂+10) + (−3)`, gives mean 17, std 10 — style `k`'s `(10, 5)` is still riding along inside those numbers. So the order really matters: normalize, *then* style. With it, the direct statistical handle is renewed every layer; without it, layers smear together through their leftover channel statistics.

This isn't an absolute causal cutoff — the convolution at layer `k+1` reads the *spatial* pattern style `k` helped produce, so downstream content can still carry consequences of an earlier style. But the one thing a style most directly controls, the per-channel mean and variance, is reset and rewritten at each layer rather than compounding.

There's a nice consequence I didn't plan for. A style is *spatially invariant* — the same scale and bias hit the whole feature map. So a style can only express things that are global to the image at that scale: pose, identity, lighting, overall color. It physically cannot encode "this hair goes here and that freckle there," because that would require spatially varying values and the style is one number per channel. That's a clean division of labor falling out of the geometry of the operation. Which loops me straight back to my second pain — the stochastic detail.

If styles are inherently global, then I need a *separate*, spatially-varying input for the per-pixel randomness. The network was previously forced to fake this from `z`; let me just hand it real noise. After each convolution, add a single-channel image of uncorrelated Gaussian noise, broadcast across channels, before the nonlinearity. A fresh noise image per layer. And I want each channel to decide how much noise it wants, so broadcast it with a *learned per-channel scaling factor* `B` — `x ← x + B ⊙ noise`. Initialize those scales to zero so noise fades in rather than disrupting early training.

Will the network actually use this noise for the *right* thing — stochastic detail and not, say, pose? I think yes, and the discriminator is what enforces it. If the network tried to encode pose through the per-pixel noise, the pose would come out spatially inconsistent and incoherent across the image — and the discriminator would punish that, because real faces don't have pose that flickers per pixel. Whereas the style channel *can* express coherent global structure. So the network is pushed to route global content through the styles and stochastic content through the noise, with no explicit supervision telling it to. And will the noise effect stay localized to its layer's scale, or will the net pull randomness from earlier layers' activations the way it used to? It won't bother: a fresh, free source of noise is available at *every* layer, so there's no incentive to spend capacity synthesizing randomness from earlier activations when it can just grab the noise that's right there. Coarse-layer noise gives big stochastic effects — large hair curls, background blobs; fine-layer noise gives fine ones — individual strands, pores. And critically, the capacity the network used to waste manufacturing pseudo-noise (and producing those repeating artifacts) is freed.

So the synthesis side is taking shape: a stack of conv layers, and at each one, add noise, then AdaIN with a style derived from the latent. But two questions force themselves on me. First — what derives the styles from the latent, and is feeding raw `z` even the right thing? Second — if styles are injected at *every* layer, do I still need to feed `z` into the *bottom* input layer at all?

Take the second first, because it's almost philosophical. The bottom of the network traditionally receives `z` and turns it into a 4×4 feature block. But if the styles already inject the latent at every layer, the bottom input is at least suspect — the latent is everywhere. Let me try the aggressive version: start from a **learned constant** — a fixed 4×4×512 tensor, the same for every image, that gets trained like any other weight. The constant can serve as a learned coordinate frame and feature reservoir; the only thing that varies between images is then the styles and the noise. If this loses necessary variation, the latent-fed input can come back, but the cleanest hypothesis is that style at every layer is a sufficient interface to the generator.

Now the first question, the one I think is the deepest, because it ties to disentanglement. What maps the latent to the styles? The lazy answer is: a learned affine transform directly from `z` to `(scale, bias)` at each layer. But that means `z` itself is the latent the user manipulates, and I already know `z` is entangled. Let me reconstruct *why* it's structurally entangled, because the fix should come from understanding the cause.

The input `z` is sampled from a fixed prior — a round Gaussian. The generator has to reproduce the data density: the probability of any combination of factors of variation in latent space has to equal the frequency of that combination in the training data. Now suppose the data has a *hole* — some combination of attributes that simply never appears (say, in a face dataset, some pairing of attributes that never co-occur). The set of valid images is a manifold with a hole punched in it. But my `z` is a round Gaussian with *no* hole. To map a hole-free round blob onto a manifold-with-a-hole while preserving densities, the mapping `z → features` has no choice but to *curve* — it has to bend around so that the forbidden region in feature space receives no preimage from the sampled `z`. A curved mapping is, by definition, an entangled one: moving along a single `z` axis drags several factors along together, non-linearly. And this isn't a flaw I can train away — it's forced by the requirement that a *fixed* prior match a *non-uniform* data density. Any input distribution pinned to the data density inherits this curvature.

So the cause is: the latent the user touches is the *same* space that's forced to match the data density. That points at a move. What if the user-facing latent and the density-matching latent didn't have to be the same space? Interpose an **intermediate** latent space whose density I *don't* constrain. Let the input `z` still be Gaussian, but immediately pass it through a learned mapping `f: z → w`, and have `w` — not `z` — be what drives the styles. The space `w` lives in doesn't have to be Gaussian and doesn't have to match the data density; its density is whatever `f` induces. So `f` *can* absorb the curvature — take the round `z` and warp it into a `w`-space where the factors lie along flatter, more linear directions, leaving the synthesis network a disentangled handle.

But "can" isn't "will." Is there actual pressure for `f` to do this rather than just pass the entanglement through unchanged? I don't have a proof, only an argument: it should be *easier* for the synthesis network to produce realistic images from a disentangled representation than from a tangled one, so gradient descent has an incentive to push whatever unwarping is cheap into `f`, where it's now allowed to live. That's a hope about the optimization, not a guarantee — I can't constrain `w` to be flat directly. What I *can* say firmly is the negative half: I've removed the constraint that was provably *forcing* `z` to curve. Whether `f` uses the freedom is exactly the kind of thing I'll need a number for later, and it's why I'll want a way to actually measure how flat `w` ended up being.

How much mapping do I need? `f` is an MLP. Unwarping a curved manifold is not a one-layer affine job, so I want real depth, but not a side network so large that it dominates the generator. Eight fully-connected layers is a reasonable budget, with `z` and `w` both 512-dimensional and leaky-ReLU (α = 0.2) activations. One practical wrinkle shows up: a deep mapping network is easy to make too aggressive at the normal learning rate. The mapping is a different kind of beast from the conv stack — it's a long MLP whose output is then *re-scaled* per channel before injection, so its gradient scale is off. The clean fix is to slow it down: use a learning-rate multiplier of `0.01` for the mapping layers (`λ' = 0.01·λ`). I should also keep `z` on a consistent scale going in — normalize it to unit length first (pixel-norm on the input vector). That also makes interpolation in `z` well-defined as spherical, which I'll need later.

Let me also pin down the affine that turns `w` into a style at each layer. AdaIN needs a per-channel scale and bias, so for a layer with `C` channels the affine emits `2C` numbers. If I use the scale directly, the formula is `y_{s,i}·(x_i − μ(x_i))/σ(x_i) + y_{b,i}`, channel `i` normalized separately, then scaled by `y_{s,i}` and biased by `y_{b,i}`. But with small initialized weights and zero bias, direct scale would start near `0` and kill the signal. Better to have the affine emit a scale *deviation* `d_{s,i}` and use `(d_{s,i} + 1)·(x_i − μ(x_i))/σ(x_i) + y_{b,i}`, so that `d_s ≈ 0` initializes the actual scale near `1`. That means the scale-side bias stays at `0` in this parameterization; biasing it to `1` would initialize the actual scale to `2`. The equivalent direct-scale parameterization would drop the `+1` and initialize the scale bias to `1`. Biases and the noise scales start at 0; the learned constant input starts at 1.

That's the generator. Let me make sure the per-layer epilogue order is right, because I argued the ordering carries the localization. After a convolution at a layer: add the noise (with its learned per-channel scale), add the layer bias, apply leaky-ReLU, then instance-normalize, then apply the style (the scale-and-bias from `w`). The instance-norm-then-style is the AdaIN — normalize away the incoming statistics, write the new style. Noise goes in before the normalization so that the stochastic perturbation is itself subject to the same statistical reset and styling, and lands as genuine spatial variation on the styled features.

Now, I claimed the styles are localized — each controlling one scale — but training doesn't automatically give me clean *separation* between adjacent layers' styles. The network could still learn to assume that the style at layer `k` and the style at layer `k+1` are correlated (they both come from the same `w`, after all), and lean on that correlation, which would blur the per-scale control I'm after. I want to actively *break* that assumption. So during training, some fraction of the time, generate an image using **two** latents instead of one. Run `z_1` and `z_2` through the mapping to get `w_1, w_2`, pick a random crossover layer, and use `w_1` for the styles *before* the crossover and `w_2` for the styles *after*. The network never knows where the switch will happen, so it can't assume adjacent styles come from the same `w` — it's forced to make each layer's style stand on its own. This is mixing regularization, and I'll do it with high probability (~0.9). The payoff is twofold: stronger localization during training, and — for free — a controllable test-time operation. At test time I can take the coarse styles of face A and the fine styles of face B and synthesize A's pose-and-shape with B's coloring-and-microstructure, by choosing which layers each `w` drives. Coarse crossover swaps high-level attributes; fine crossover swaps small details.

One more generation-time nicety, borrowed from the observation that low-density regions of any latent prior give poor samples. I can trade variation for quality by shrinking `w` toward the mean. Compute the center of mass `w̄ = E_{z}[f(z)]`, and for a sample use `w' = w̄ + ψ·(w − w̄)` with `ψ < 1`. Because I'm doing this in `w` rather than `z`, it works reliably without touching the loss. And I can apply it *selectively* — only to the coarse (low-resolution) layers via a layer cutoff — so I clean up global structure while leaving fine detail at full variation. I'll track `w̄` as a moving average of the batch mean of `w` during training so it's ready at inference.

So the architecture is settled: `z` → pixel-norm → 8-layer MLP `f` → `w`; then a synthesis network `g` starting from a learned 4×4×512 constant, 18 styled layers for 1024² (constant epilogue and 4×4 convolution, then two convolutions at every doubled resolution through 1024²), each convolutional layer = conv, add noise (learned per-channel scale), bias, leaky-ReLU, instance-norm, AdaIN with a per-layer affine of `w`; mixing regularization during training; truncation in `w` at inference; output via a 1×1 toRGB conv. The discriminator and the loss I don't touch at all — non-saturating logistic loss with R1 (γ = 10) on faces, WGAN-GP elsewhere, Adam, progressive growing, equalized learning rate everywhere. I should note that since R1 lets FID keep falling for far longer than WGAN-GP, I'll train substantially longer (on the order of 25M images rather than 12M).

Now the part I can't skip: I've *claimed* `w` is more disentangled than `z`, but I have no number for it. The existing disentanglement metrics all need an encoder from images back to latents, or known ground-truth factors. My GAN has no encoder, and I refuse to bolt one on just to measure something — it's not part of the solution. I need metrics that work for *any* generator, with no encoder and no known factors. Let me invent two, from first principles about what disentanglement should *mean*.

First idea: if the latent space is flat and disentangled, then walking along an interpolation path should change the image *smoothly and at a steady rate*. If it's curved and entangled, interpolation produces lurches — features popping in and out mid-path, surprising non-linear jumps. So let me *measure the curviness* of the latent space by how drastically the image changes as I interpolate. I need a perceptual ruler for "how different are two images," and LPIPS is exactly that — a learned weighted L2 between VGG-16 embeddings, calibrated to human similarity. Subdivide an interpolation path into tiny segments and sum the perceptual distance over the segments; in the limit of infinitely fine subdivision this is the perceptual *arc length* of the path. A flat space gives short paths (the image moves at a constant, minimal perceptual rate); a curved space gives long paths (detours, lurches). I'll approximate the limit with a small step `ε`.

Let me get the normalization right, because there's a subtlety I want to nail down with an actual computation rather than hand-wave. Take two endpoints, interpolate to parameter `t` and to `t + ε`, generate both images, measure their perceptual distance `d`. LPIPS is built from a squared L2 in a VGG embedding, so for a small step it should behave like `d ≈ (local speed)²·ε²`. If that's right, the raw `d` collapses toward zero as I refine, and I'd need to divide by `ε²` to recover a stable density. Let me confirm the exponent on a concrete curved path rather than trust the heuristic.

Model the embedding of the generated image along the path as a smooth curve `φ(t)` and let the LPIPS-like distance be `d(t,ε) = ‖φ(t+ε) − φ(t)‖²`. Take `φ(t) = (sin 2t, t², cos t)` and evaluate at `t₀ = 0.3`. Stepping `ε` down by decades:

```
eps=1e-1   d=2.94e-2   d/eps=0.2940   d/eps^2=2.9396
eps=1e-2   d=3.15e-4   d/eps=0.0315   d/eps^2=3.1495
eps=1e-3   d=3.17e-6   d/eps=0.0032   d/eps^2=3.1698
eps=1e-4   d=3.17e-8   d/eps=0.0003   d/eps^2=3.1718
```

`d/ε` marches to 0 and `d` itself marches to 0, so neither the raw distance nor a divide-by-`ε` would give a finite limit — the path metric would just report "the finer I subdivide, the shorter the path," which is meaningless. But `d/ε²` settles to a constant, `≈ 3.172`. And that constant is exactly the true squared local speed: `φ'(0.3) = (2cos 0.6, 0.6, −sin 0.3)`, so `‖φ'(0.3)‖² = 3.1720…`, which matches the `ε=10⁻⁴` row. So dividing by `ε²` is the correct normalization — it turns the segment cost into the squared local speed `‖φ'(t)‖²`, whose expectation over the path is the length functional I'm after — and the quadratic behavior is real, not assumed. So define, for the input space (where `z` lives on a sphere because I unit-normalized it, hence *spherical* interpolation `slerp`):

`l_Z = E[ (1/ε²)·d( G(slerp(z_1, z_2; t)), G(slerp(z_1, z_2; t+ε)) ) ]`, with `z_1, z_2 ∼ P(z)`, `t ∼ U(0,1)`, `ε = 10^{-4}`, `G = g∘f`.

And for the intermediate space (where `w` is *not* normalized, so plain *linear* interpolation `lerp`, and I drive only `g`):

`l_W = E[ (1/ε²)·d( g(lerp(f(z_1), f(z_2); t)), g(lerp(f(z_1), f(z_2); t+ε)) ) ]`.

Average over many samples (100k). Shorter path length = flatter, more linear, less entangled space. I expect `l_W < l_Z` if the mapping really did flatten things — though "expect" is the honest word; whether `f` actually unwarped anything is precisely what this number is supposed to tell me, so I shouldn't presume the sign of the result, only that the metric is well-posed enough to read it off.

There's one bias in the comparison I can reason about ahead of time, even though I can't put a number on it without a trained generator. If `w`-space is genuinely a flattened unwarping of `z`, then the straight segment between two valid `w` endpoints may pass through regions that aren't on the input manifold — combinations `f` never produces from a real `z` — which the generator reconstructs poorly, inflating `d` along the *interior* of `w`-paths. The input space `z` has no such interior regions: every point on a `z`-geodesic is itself a valid Gaussian sample. So whatever the headline numbers say, the full-path measure is structurally biased *toward* `z` (against `w`). A way to probe this once I have a model: restrict the measure to the path *endpoints* only, `t ∈ {0, 1}`, which drops the interior off-manifold penalty. If the bias story is right, that should shrink `l_W` noticeably while leaving `l_Z` roughly unchanged — but that's a prediction to test, not something I can assert here, and if instead both shrink equally my off-manifold explanation is wrong.

Second idea, attacking disentanglement from a different angle: if a space is disentangled, then each individual factor of variation should correspond to a *linear direction*, which means a single binary attribute should be *linearly separable* in the latent space by a hyperplane. So let me measure linear separability per attribute. I don't have labels on latents directly, so I'll manufacture them: train auxiliary attribute classifiers (one per CelebA attribute; same architecture as the discriminator, minus the minibatch-stddev layer). Generate a large batch of images (200k) from `z ∼ P(z)`, classify each by the auxiliary network, sort by classifier confidence, and *throw away the least-confident half* — keeping 100k latents whose attribute label I trust. Why discard the uncertain half: the ambiguous cases carry noisy labels that would drown out the very linear-separability signal I'm trying to read. Now for each attribute, fit a linear SVM to predict the (trusted) label from the latent point — `z` for the traditional generator, `w` for this one — and measure how cleanly the SVM's hyperplane reproduces the labels. A clean way to score "how cleanly" is the conditional entropy `H(Y | X)`, where `Y` is the attribute label from the auxiliary classifier and `X` is which side of the SVM hyperplane the point falls on. `H(Y|X)` is how many extra bits I need to pin down the true label once I know the side of the plane — so a *low* value should mean the plane already determines the attribute, i.e. the attribute *is* a linear direction in this space.

Before I commit to this, let me sanity-check that `H(Y|X)` actually behaves at the two extremes the way I'm claiming, because it's easy to get a conditional-entropy formula subtly wrong. Construct two synthetic cases with 10k points. Case A: the hyperplane perfectly predicts the label, `side = Y`. Case B: the label is independent of the side, both fair coins. Computing `H(Y|X) = −Σ_s p(s) Σ_y p(y|s) log p(y|s)`:

```
perfectly separable:  H = 0.0000   exp(H) = 1.0000
independent:          H = 0.6931   exp(H) = 2.0000   (ln 2 = 0.6931)
```

So a perfectly linear attribute contributes 0 to the entropy and 1 (the multiplicative identity) after exponentiation, while a maximally entangled one contributes `ln 2` and a factor of 2 — the extremes land exactly where they should. Summing over the 40 attributes and reporting `exp(Σ_i H(Y_i | X_i))`: 40 perfectly-separable attributes give `exp(0) = 1`, the best possible score, and 40 fully-entangled ones give `exp(40·ln 2) = 2⁴⁰ ≈ 1.1×10¹²` — I confirmed `2**40` numerically to be sure the exponent isn't off. So the score ranges over `[1, 2⁴⁰]` with lower = more separable. The exponentiation just maps from the logarithmic (entropy) domain to a linear one so the scores are easy to compare, in the same spirit as the inception score. Lower = more separable = more disentangled.

If my unwarping argument is right, I'd predict `w` comes out consistently more separable than `z`, and deeper mapping networks improve it further. There's also a sharper, more falsifiable prediction I can wring out of the theory: stick a mapping network in front of a *traditional* generator (one where the latent still feeds the input layer). Then `z` still has to match the data density, so the mapping should curve `z` even *more* — hurting `z`'s separability — while the intermediate space it produces should be more separable than `z`. If I see separability of `z` get worse but the intermediate space get better, that's evidence the synthesis network prefers a disentangled input and that `f` is where the unwarping wants to live. If instead the mapping helps `z` too, my account is wrong somewhere. Either way it's a real test rather than a restatement, which is the point of bothering to build these metrics.

Let me write the core of it down, grounded in the actual module structure.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Equalized learning rate: store N(0,1) weights, apply He scaling at runtime so an
# adaptive optimizer advances every parameter at the same effective rate.
class EqLinear(nn.Module):
    def __init__(self, fin, fout, gain=2**0.5, lrmul=1.0, bias_init=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fout, fin) / lrmul)
        self.bias   = nn.Parameter(torch.full((fout,), float(bias_init) / lrmul))
        self.w_coef = gain / np.sqrt(fin) * lrmul   # runtime weight scale
        self.b_coef = lrmul                          # runtime bias scale
    def forward(self, x):
        return F.linear(x, self.weight * self.w_coef, self.bias * self.b_coef)

class EqConv2d(nn.Module):
    def __init__(self, fin, fout, k, gain=2**0.5, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fout, fin, k, k))
        self.bias   = nn.Parameter(torch.zeros(fout)) if bias else None
        self.w_coef = gain / np.sqrt(fin * k * k)
        self.pad = k // 2
    def forward(self, x):
        return F.conv2d(x, self.weight * self.w_coef, self.bias, padding=self.pad)

def pixel_norm(x, eps=1e-8):                          # keep z on the unit sphere
    return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + eps)

def blur2d(x, f=(1, 2, 1)):
    k = torch.tensor(f, dtype=x.dtype, device=x.device)
    k = (k[:, None] * k[None, :]); k = k / k.sum()
    k = k.expand(x.size(1), 1, 3, 3)
    return F.conv2d(x, k, padding=1, groups=x.size(1))

# z -> w : the learned mapping that is ALLOWED to be non-Gaussian, so it can absorb
# the curvature that a fixed prior matching the data density would otherwise force.
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, depth=8, lrmul=0.01):
        super().__init__()
        layers, fin = [], z_dim
        for i in range(depth):
            layers += [EqLinear(fin, w_dim, lrmul=lrmul)]
            fin = w_dim
        self.layers = nn.ModuleList(layers)
    def forward(self, z):
        w = pixel_norm(z)                            # normalize input latent first
        for fc in self.layers:
            w = F.leaky_relu(fc(w), 0.2)             # slowed down 100x via lrmul
        return w                                     # NOT normalized

# Per-pixel stochastic input: a fresh single-channel Gaussian image per layer,
# broadcast with a learned per-channel scale (init 0 so it fades in). This is the
# dedicated cheap source of stochastic detail the styles cannot express.
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3),
                                device=x.device, dtype=x.dtype)
        return x + self.scale.view(1, -1, 1, 1) * noise

# AdaIN: normalize incoming per-channel statistics, then write the style carried by
# w. The affine emits a deviation from 1 for the scale, so zero bias is identity-ish.
class StyleMod(nn.Module):
    def __init__(self, w_dim, channels):
        super().__init__()
        self.affine = EqLinear(w_dim, channels * 2, gain=1.0)
    def forward(self, x, w):
        x = F.instance_norm(x, eps=1e-8)              # zero-mean, unit-var per channel
        y = self.affine(w).unsqueeze(2).unsqueeze(3)  # [N, 2C, 1, 1]
        ds, yb = y.chunk(2, dim=1)
        return x * (ds + 1) + yb                      # deviation-from-1 scale

# One synthesis layer's epilogue: conv -> noise -> bias -> activation -> AdaIN.
class StyleLayer(nn.Module):
    def __init__(self, fin, fout, w_dim, upsample, blur):
        super().__init__()
        self.upsample = upsample
        self.blur = blur or blur2d
        self.conv  = EqConv2d(fin, fout, 3, bias=False)
        self.noise = NoiseInjection(fout)
        self.bias  = nn.Parameter(torch.zeros(fout))
        self.style = StyleMod(w_dim, fout)
    def forward(self, x, w):
        if self.upsample:
            x = self.blur(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.conv(x)
        x = self.noise(x)
        x = x + self.bias.view(1, -1, 1, 1)
        x = F.leaky_relu(x, 0.2)
        x = self.style(x, w)
        return x

class ConstStyleLayer(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.noise = NoiseInjection(channels)
        self.bias  = nn.Parameter(torch.zeros(channels))
        self.style = StyleMod(w_dim, channels)
    def forward(self, x, w):
        x = self.noise(x)
        x = x + self.bias.view(1, -1, 1, 1)
        x = F.leaky_relu(x, 0.2)
        return self.style(x, w)

# Synthesis network: starts from a learned constant (no latent through an input
# layer); the only per-image signal arrives as styles (and noise).
class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim=512, resolution=1024, channels=None, blur=None):
        super().__init__()
        log2res = int(np.log2(resolution))
        self.num_layers = log2res * 2 - 2            # 18 for 1024^2
        ch = channels or (lambda s: min(8192 // (2 ** s), 512))
        self.const = nn.Parameter(torch.ones(1, ch(1), 4, 4))
        self.input = ConstStyleLayer(ch(1), w_dim)
        self.layer0 = StyleLayer(ch(1), ch(1), w_dim, upsample=False, blur=blur)
        self.blocks = nn.ModuleList()
        for res in range(3, log2res + 1):            # 8^2 .. 1024^2, two layers each
            self.blocks.append(StyleLayer(ch(res-2), ch(res-1), w_dim, True,  blur))
            self.blocks.append(StyleLayer(ch(res-1), ch(res-1), w_dim, False, blur))
        self.to_rgb = EqConv2d(ch(log2res - 1), 3, 1, gain=1.0)
    def forward(self, ws):                            # ws: per-layer w, [N, num_layers, w_dim]
        x = self.const.expand(ws.size(0), -1, -1, -1)
        x = self.input(x, ws[:, 0])
        x = self.layer0(x, ws[:, 1])
        for i, blk in enumerate(self.blocks, start=2):
            x = blk(x, ws[:, i])
        return self.to_rgb(x)

class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, resolution=1024,
                 mixing_prob=0.9, w_avg_beta=0.995, blur=None):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.synthesis = SynthesisNetwork(w_dim, resolution, blur=blur)
        self.num_layers = self.synthesis.num_layers
        self.mixing_prob = mixing_prob
        self.w_avg_beta = w_avg_beta
        self.register_buffer('w_avg', torch.zeros(w_dim))

    def _broadcast(self, w):                          # same w to every layer
        return w.unsqueeze(1).repeat(1, self.num_layers, 1)

    def forward(self, z, truncation_psi=1.0, truncation_cutoff=None):
        w = self.mapping(z)
        if self.training:
            self.w_avg.lerp_(w.detach().mean(0), 1 - self.w_avg_beta)
        ws = self._broadcast(w)

        # Mixing regularization: a second latent drives the styles past a random
        # crossover layer, so no layer may assume its style shares w with its neighbor.
        if self.training and self.mixing_prob > 0 and torch.rand((), device=z.device).item() < self.mixing_prob:
            w2 = self.mapping(torch.randn_like(z))
            ws2 = self._broadcast(w2)
            cutoff = int(torch.randint(1, self.num_layers, (), device=z.device).item())
            idx = torch.arange(self.num_layers, device=z.device).view(1, -1, 1)
            ws = torch.where(idx < cutoff, ws, ws2)

        # Truncation in w (inference): shrink toward the mean, optionally coarse-only.
        if truncation_psi != 1.0:
            coefs = torch.ones(self.num_layers, device=z.device)
            if truncation_cutoff is not None:
                coefs[:truncation_cutoff] = truncation_psi
            else:
                coefs[:] = truncation_psi
            ws = self.w_avg.view(1, 1, -1) + (ws - self.w_avg.view(1, 1, -1)) * coefs.view(1, -1, 1)

        return self.synthesis(ws)
```

And the two metrics, which are how I'd actually check the disentanglement claim:

```python
def slerp(a, b, t):                                   # z lives on the sphere
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    omega = torch.acos((a * b).sum(-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    out = (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b
    return torch.where(so.abs() > 1e-7, out, lerp(a, b, t))

def lerp(a, b, t):                                    # w is not normalized
    return a + (b - a) * t

def perceptual_path_length(G, lpips, space='w', n=100000, eps=1e-4, batch=16, z_dim=512):
    device = next(G.parameters()).device
    total = 0.0
    seen = 0
    G.eval()
    with torch.no_grad():
        while seen < n:
            bsz = min(batch, n - seen)
            z1 = torch.randn(bsz, z_dim, device=device)
            z2 = torch.randn(bsz, z_dim, device=device)
            t = torch.rand(bsz, 1, device=device)
            if space == 'z':
                za = slerp(z1, z2, t); zb = slerp(z1, z2, t + eps)
                img_a, img_b = G(za), G(zb)
            else:
                w1, w2 = G.mapping(z1), G.mapping(z2)
                wa = lerp(w1, w2, t); wb = lerp(w1, w2, t + eps)
                img_a = G.synthesis(G._broadcast(wa))
                img_b = G.synthesis(G._broadcast(wb))
            # LPIPS is quadratic for small steps, so divide by eps^2 to get squared local speed.
            dist = lpips(img_a, img_b).reshape(bsz, -1).mean(1)
            total += dist.sum().item() / (eps ** 2)
            seen += bsz
    return total / n

def linear_separability(G, classifiers, space='w', n=200000, batch=32, z_dim=512):
    from sklearn.svm import LinearSVC
    import numpy as np
    device = next(G.parameters()).device
    G.eval()
    score = 0.0
    for attr, clf in enumerate(classifiers):
        clf.eval()
        xs, ys, confs = [], [], []
        with torch.no_grad():
            for start in range(0, n, batch):
                bsz = min(batch, n - start)
                z = torch.randn(bsz, z_dim, device=device)
                w = G.mapping(z)
                imgs = G.synthesis(G._broadcast(w))
                logits = clf(imgs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits[:, attr] if logits.ndim == 2 and logits.size(1) > 1 else logits.reshape(-1)
                xs.append((w if space == 'w' else z).detach().cpu())
                ys.append((logits > 0).long().cpu())
                confs.append(logits.abs().cpu())
        X_all = torch.cat(xs, 0).numpy()
        Y_all = torch.cat(ys, 0).numpy()
        conf_all = torch.cat(confs, 0)
        keep = conf_all.argsort(descending=True)[: n // 2].numpy()
        X = X_all[keep]
        Y = Y_all[keep]
        if np.unique(Y).size < 2:
            continue
        svm = LinearSVC(max_iter=10000).fit(X, Y)
        pred = svm.predict(X)
        # Conditional entropy H(Y | side-of-hyperplane): low => attribute is a linear direction.
        h = 0.0
        for s in (0, 1):
            m = pred == s
            if m.sum() == 0: continue
            p = Y[m].mean()
            for q in (p, 1 - p):
                if q > 0: h -= (m.mean()) * q * np.log(q)
        score += h
    return float(np.exp(score))                       # log -> linear domain, like inception score
```

Stepping back to the causal chain: the two pains were no scale-specific control and no clean stochastic source. Style transfer's normalization story told me that per-channel mean and variance *are* style and that an instance-norm layer's affine is a complete style controller — so I inject the latent as a per-layer AdaIN style, and the normalize-then-style order renews the direct channel-statistics control at each layer. Spatially-invariant styles can only carry global content, which both explains the scale-specific control and exposes the gap for stochastic detail — filled by a dedicated per-layer per-pixel noise input with learned per-channel scale, kept honest and localized by the discriminator and by the free availability of fresh noise at every layer. Because styles now inject the latent everywhere, I can start from a learned constant instead of feeding the latent through an input layer. The entanglement of `z` traces to a fixed prior being forced to match a non-uniform data density, so I interpose a learned 8-layer mapping to an unconstrained intermediate space `w` (slowed 100×) that can unwarp the curvature; mixing regularization decorrelates adjacent styles and doubles as style mixing; truncation in `w` trades variation for quality at chosen scales. Finally, since no existing encoder-free disentanglement metric exists, perceptual path length measures latent-space curvature via LPIPS-normalized interpolation speed (divided by `ε²` because LPIPS is quadratic), and linear separability measures whether attributes are linear directions via conditional entropy of a per-attribute SVM.
