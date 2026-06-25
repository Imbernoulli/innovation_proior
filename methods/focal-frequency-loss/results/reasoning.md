Let me start from the thing that keeps biting me: I train image generators — a VAE, a pix2pix, a SPADE
— under spatial losses (per-pixel `L1`/`L2`, a VGG perceptual term, sometimes an adversarial term), and
the outputs look broadly right but they keep losing *fine detail*. Edges soften, textures smear, the
crisp high-frequency structure that makes an image look real is the first thing to go. I want to
understand *why* a spatial loss does this, because the explanation should tell me where to intervene.

Per-pixel error treats the image as a bag of independent pixels and scores the energy of the residual.
Under reconstruction uncertainty — many plausible high-frequency completions of a given coarse content —
squared error is minimized by their average, which is a blur. So the spatial loss is not neutral about
frequencies; it has a *bias*. It pays cheaply for getting the low-frequency, smooth content right
(that's most of the pixel energy) and barely penalizes getting the high-frequency detail wrong (that's a
small fraction of the pixel energy but most of the perceptual content). The perceptual and adversarial
terms patch this indirectly, but none of them has an *explicit* notion of "you are missing this band of
the spectrum." The error I care about is structured in frequency space, and my loss is measuring it in
pixel space, where that structure is invisible. The obvious thing to try, then, is to measure the error
directly on the spectrum and see whether that exposes a usable handle.

The 2D discrete Fourier transform gives me the spectrum. For an image channel `f(x,y)` of size `H×W`,
`F(u,v) = Σ_x Σ_y f(x,y) exp(-i2π(ux/H + vy/W))` is a complex value per spatial frequency `(u,v)`,
`F(u,v) = a(u,v) + i b(u,v)`, carrying that frequency's amplitude and phase. Two images are identical
iff their full complex spectra are identical, so a distance between spectra is a perfectly legitimate
reconstruction objective. The most natural distance is the squared Euclidean distance between the
complex components: for a real spectrum `F_r` and a fake spectrum `F_f`, sum over all `(u,v)` of
`|F_r(u,v) - F_f(u,v)|² = (a_r - a_f)² + (b_r - b_f)²`. Averaged over the `M·N` components, that is a
plain frequency-distance loss.

Before I trust this as a complement to a spatial loss, I want to know how its *scale* compares — if the
spectral distance is, say, a hundred times larger than the `L2` it rides next to, then a fixed mixing
weight will be impossible to set. So I should pin down the relationship between the spatial residual and
the spectral residual. Parseval/Plancherel says a *unitary* transform preserves the squared norm, and
`torch.fft.fft2(..., norm='ortho')` is exactly the unitary normalization. Let me actually check it
rather than trust the name. Two random `4×4` channels `f`, `g`: the spatial sum of squared differences
comes out `26.4307`, and summing `|F_r - F_f|²` over the orthonormal spectrum gives `26.4307` — equal to
five decimal places. If instead I use the default (`norm='backward'`, unnormalized forward transform),
the spectral sum is `422.89`, exactly `16 = H·W` times larger. So the orthonormal FFT is not cosmetic:
it makes `Σ|F_r - F_f|²` literally equal to `Σ(f - g)²`, so a spectral term and a spatial term live on
the same scale and a single mixing weight `λ` means the same thing for both. Good — I'll keep `norm='ortho'`
and treat the spectral distance as a same-units sibling of the spatial one. Now back to the design,
because the plain version is going to be crude and I should find out where.

The crudeness is exactly the spectral-bias problem in a new guise. A plain frequency loss weights every
component equally. But the generator, trained under spatial losses, *already* matches most of the
low-frequency components — they're easy, they got fit early. The components that are still wrong are a
small minority of hard, mostly high-frequency ones. If I sum the spectral distance with equal weight,
the gradient is dominated by the sheer *number* of easy components (each contributing a small but
nonzero residual) and the few hard components — the ones that actually account for the perceptual gap —
get almost no attention. So an unweighted frequency loss would behave like the spatial loss it's meant
to fix: it would polish the already-good frequencies and ignore the missing ones. Equal weighting is the
wall. I need the loss to *adaptively concentrate on the hard frequencies* and down-weight the easy ones.

What does "hard" mean here, operationally? A frequency component is hard exactly when the model is
currently getting it wrong — when `|F_r(u,v) - F_f(u,v)|` is large. That's a quantity I already have, for
free, at every training step. So the reweighting suggests itself: weight each component's distance by a
matrix `w(u,v)` that is *large where the current frequency error is large* and small where it's small.
The simplest such matrix is the current frequency distance itself, raised to a tunable power:
`w(u,v) = |F_r(u,v) - F_f(u,v)|^α`.

I don't want to just narrate what I think this does — let me run it on a case small enough to read every
number. Take a `2×2` channel whose target is flat (`all 1`s), so its orthonormal spectrum is pure DC: a
single nonzero component at `(0,0)` and zeros elsewhere. Let the prediction be `[[1.1, 0.9],[0.85, 1.05]]` —
close in the bulk but perturbed. Computing `|F_r - F_f|` per component gives the four magnitudes
`[[0.050, 6e-8],[0.050, 0.200]]`: one component is essentially matched (`6e-8`), two are mildly off
(`0.050`), and one is the clear culprit (`0.200`). With `α = 1` and per-image normalization by the max,
the weight matrix is `[[0.25, 3e-7],[0.25, 1.0]]`. That is the behavior I was hoping for, now witnessed
rather than asserted: the already-matched component is gated to `~0` (no gradient wasted polishing it),
the hardest component is pinned to `1.0`, and the in-between ones sit at a quarter. Sweeping `α` confirms
it's a focusing knob: `α = 0` gives the all-ones matrix `[[1,1],[1,1]]` — exactly the unweighted loss —
while `α = 2` sharpens the contrast, dropping the `0.050` components to `0.0625` while the culprit stays
at `1.0`. And the weighted average tracks this: the unweighted loss on this example is `0.011250`, and
`α = 1` lowers the *reported* loss to `0.010312` precisely by discounting the easy components, so the
number the optimizer sees is dominated by the hard one. It's also *dynamic*: because `w` is recomputed
from the current error every step, once a hard frequency gets matched its magnitude falls and its weight
decays on its own, so the loss moves to the next-hardest one — a curriculum the loss generates itself,
frequency by frequency, with no hand-set band selection. `α = 1` is the natural default that makes the
weight scale linearly with the error magnitude.

But this gating only behaves like a "spotlight" if the weight is treated as a *constant* with respect to
the parameters — detached from the gradient. If I instead let gradients flow through
`w(u,v) = distance^α`, the loss term becomes `distance^α · distance² = distance^(α+2)`, and its gradient
is no longer "the ordinary spectral gradient, reweighted." Let me make this concrete on a single residual
`r` (a 1D proxy for one component) and actually differentiate both versions. With the weight detached,
`L_A = sg(|r|^α)·r²` has gradient `|r|^α·2r`. With the weight live, `L_B = |r|^α·r² = |r|^(α+2)` has
gradient `(α+2)|r|^(α+1)·sign(r)`. At `α = 1` the ratio `grad_B / grad_A = (α+2)/2 = 1.5`, and computing
it with autograd at `r = 0.1, 0.5, 2.0` gives `1.500` every time — a *constant* inflation factor.
So flowing gradients through the weight doesn't change *which* components are emphasized; it silently
multiplies the whole spectral gradient by `(α+2)/2` and, worse, ties that multiplier to `α`. Then the
exponent `α` is doing two jobs at once — setting the focusing contrast (which I want) *and* rescaling the
effective learning rate (which I don't) — and the two get tangled. The clean thing is to break that
coupling: at each step compute the weight matrix from the current error, *freeze it* (stop-gradient /
detach), and use it only as a fixed per-frequency importance on the distance. Then the gradient is
exactly `w(u,v)·∇(distance²)` — the ordinary frequency-distance gradient, reweighted by a constant that
says "spend your effort here." The weight is a spotlight, not a variable. So `w` is detached, always.

Two normalizations make the spotlight well-behaved. First, the raw `distance^α` can span orders of
magnitude across components and across images, so I normalize the weight matrix into `[0,1]` — divide by
its maximum (per image, or optionally per batch) so the largest current error gets weight 1 and
everything else is a fraction of that. (This is the same per-image-max division I already used in the
`2×2` check, which is what put the hardest component at exactly `1.0`.) This keeps the loss scale stable
as training progresses and the absolute error magnitudes shrink; without it the loss would silently
rescale itself and fight the learning rate. Second, when some errors are enormous relative to others, even
after the `[0,1]` normalization the distribution can be extremely peaked; an optional `log(1 + ·)` on the
weight matrix before normalizing compresses that range so the spotlight isn't pinned to a single outlier
frequency. Both are knobs (`batch_matrix`, `log_matrix`); the defaults are per-image normalization and no
log, which is the robust starting point. After normalizing I also clamp to `[0,1]` and zero out any NaNs
(a component with exactly zero error gives `0^α` issues at fractional `α`), so the weight is a clean
importance map in `[0,1]`.

Now assemble. Run the FFT on both images to get `F_r` and `F_f` (orthonormal normalization, so by the
Parseval check above the spectral distance has the same scale as a spatial one — this matters for
combining with spatial losses). Form the per-component squared distance `(a_r-a_f)² + (b_r-b_f)²`. Form
the detached, normalized, clamped weight matrix `w = clamp(norm(|F_r-F_f|^α), 0, 1)`. The focal frequency
loss is their weighted average over all components (and channels and the batch):
`FFL = (1/MN) Σ_u Σ_v w(u,v) · |F_r(u,v) - F_f(u,v)|²`. That's the whole object: a frequency-distance
loss whose every component is gated by how hard that component currently is.

One refinement for larger images: instead of one global FFT, I can crop the image into a grid of
`patch_factor × patch_factor` patches and take the FFT of each, then apply the same weighted distance
per patch. This gives a *patch-wise* spectrum that captures locally-varying frequency content (a textured
region and a smooth region have very different spectra) rather than averaging everything into one global
spectrum. For small images `patch_factor = 1` (a single global FFT) is the default and is fine.

This has to be a *complement*, not a replacement, and I should be careful about why. The frequency loss
says nothing about the absolute pixel values being right — it's a distance on the spectrum, and you can
in principle match the spectrum while drifting the spatial content, so it must ride alongside a spatial
reconstruction term that anchors the pixels (and, if present, the perceptual and adversarial terms that
handle global realism). The intended use is `total = spatial_loss + λ · FFL`, where `λ` weights the
frequency term; FFL's job is purely to drag the *frequency* statistics of the output toward the target's,
concentrating on the bands the spatial losses keep missing. I expect this stacking to improve a VAE,
pix2pix, and SPADE in both perceptual quality and quantitative metrics, but that's a hypothesis, not
something I can settle on this page — the validation I'd run is to add FFL on top of an existing recipe,
hold everything else fixed, and check that the frequency mismatch (and the downstream perceptual metric)
drops without the spatial metrics regressing.

Let me write the core of it as code, tying each block back to the reasoning — the FFT, the squared
complex distance, the detached normalized weight matrix, and the weighted average.

```python
import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """Focal frequency loss: a spectral-distance loss whose every frequency
    component is gated by how hard (how mismatched) that component currently is.
    Use as a complement to a spatial loss: total = spatial + loss_weight * FFL."""

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super().__init__()
        self.loss_weight = loss_weight      # lambda on the FFL term
        self.alpha = alpha                  # focusing strength; 0 -> unweighted, 1 -> linear in error
        self.patch_factor = patch_factor    # 1 = single global FFT; >1 = patch-wise spectra
        self.ave_spectrum = ave_spectrum    # average the spectrum over the minibatch first
        self.log_matrix = log_matrix        # log-compress the weight range before normalizing
        self.batch_matrix = batch_matrix    # normalize the weight by batch-max vs per-image-max

    def tensor2freq(self, x):
        pf = self.patch_factor
        _, _, h, w = x.shape
        assert h % pf == 0 and w % pf == 0, "patch_factor must divide H and W"
        ph, pw = h // pf, w // pf
        patches = [x[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                   for i in range(pf) for j in range(pf)]
        y = torch.stack(patches, 1)
        # orthonormal 2D FFT -> unitary, so spectral scale matches spatial scale
        freq = torch.fft.fft2(y, norm='ortho')
        return torch.stack([freq.real, freq.imag], -1)        # (..., 2): real & imag

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            # weight = |F_r - F_f|^alpha : large where the current frequency error is large
            tmp = (recon_freq - real_freq) ** 2
            tmp = torch.sqrt(tmp[..., 0] + tmp[..., 1]) ** self.alpha
            if self.log_matrix:
                tmp = torch.log(tmp + 1.0)                    # optional range compression
            if self.batch_matrix:
                tmp = tmp / tmp.max()                         # batch-max normalization
            else:
                tmp = tmp / tmp.max(-1).values.max(-1).values[:, :, :, None, None]  # per-image
            tmp[torch.isnan(tmp)] = 0.0
            tmp = torch.clamp(tmp, min=0.0, max=1.0)          # importance map in [0,1]
            weight_matrix = tmp.clone().detach()              # SPOTLIGHT: detached, a constant gate

        # squared Euclidean distance between complex components
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        # weighted average: gradient is w * d(distance^2), concentrated on the hard frequencies
        return torch.mean(weight_matrix * freq_distance)

    def forward(self, pred, target, matrix=None):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
```

So the causal chain, start to finish. Spatial losses (per-pixel, perceptual, adversarial) keep losing
fine detail because pixel error has a frequency bias — it pays for the easy low-frequency content and
ignores the hard high-frequency content — and none of them measures the error where it's structured, on
the spectrum. So measure it on the spectrum via the 2D FFT, as the squared Euclidean distance between
complex components, in the orthonormal normalization that (per the Parseval check) keeps it on the same
scale as the spatial loss it rides with. But a plain spectral distance inherits the same disease: equal
weighting lets the many already-matched easy frequencies dominate the gradient. So gate each component by
a weight matrix `|F_r - F_f|^α` that is large exactly where the current frequency error is large — the
`2×2` check showed it driving matched components to `~0` and the hardest to `1.0`, a self-generated
curriculum that decays as frequencies are matched — and *detach* it, because letting gradients through
the weight (as the differentiation check showed) just multiplies the whole spectral gradient by `(α+2)/2`
and entangles `α` with the learning rate, whereas as a constant gate the gradient is exactly
`w · ∇(distance²)`, the clean reweighted spectral gradient. Normalize it into `[0,1]` for scale
stability. The result, `FFL = mean(w · |F_r - F_f|²)` with `α = 1` the default, is a drop-in complement
that rides alongside the existing spatial loss and drags the output's frequency statistics toward the
target's, closing the high-frequency gap the spatial terms leave behind.
