The adversarial rung came in cleanly below perceptual, exactly as I bet, and the per-scale shape confirms
the whole ladder's logic. Small went `19.64 → 14.75` — the patch discriminator and feature matching found
the residual softness LPIPS tolerated, and they found the most of it where there was the most to find.
Medium went `10.69 → 7.13` and large `3.84 → 3.38`, the now-familiar diminishing returns up the capacity
ladder. The gmean task score fell from `9.31` to `7.08`. So three rungs of attacking *spatial* quality —
pixel anchor, then deep-feature perceptual distance, then patch-adversarial realism plus feature matching
— have each helped, and I should read *how much* each helped, because the trend is the whole argument for
what to do next. The gmean went `16.47 → 9.31 → 7.08`: a first cut of `43%`, then a second of only `24%`.
In absolute terms the gmean gains were `−7.16` then `−2.23` — each rung buying roughly a third of the
previous rung's improvement. And the per-scale fractional cuts this rung were `14.75/19.64 = 0.75` (small,
`25%`), `7.13/10.69 = 0.67` (medium, `33%`), `3.38/3.84 = 0.88` (large, only `12%`). Large has almost
stopped moving; it is close enough to the floor of what these three terms can do that the critic finds
little left to sharpen. The spatial story is near its floor: the discriminator and LPIPS together are
squeezing the last bits of generic softness out, and the curve is flattening. If I extrapolate the trend,
a *fourth* spatial term would buy maybe a third again of `2.23` — under a point of gmean — for a lot of
added machinery. To go meaningfully below `7.08` I need to attack a *different kind* of residual error,
not pile a fourth spatial term on top of three.

So let me diagnose what kind of error survives the recipe. All three of my objectives live in the spatial
domain: per-pixel L1 scores pixel energy, LPIPS scores deep-feature agreement, the discriminator and
feature matching score local patch realism. None of them has an *explicit* notion of which spatial
*frequencies* the reconstruction is getting wrong. And that matters, because a continuous VAE through an
`f=4` bottleneck has a structural frequency bias I can reason about directly. The `32×32` image is
compressed to an `8×8` latent grid, a `4×` spatial downsampling, so the latent's own spatial sampling can
only natively represent frequencies up to its Nyquist limit; everything above that band the decoder must
*synthesize* on the way back up from `8×8` to `32×32` rather than read out of the code. The encoder-decoder
fits low-frequency, smooth content fast — it is most of the pixel energy and cheap under L1 — and it
systematically struggles to reproduce the high-frequency detail (fine edges, texture grain) that is a
small fraction of the pixel energy but a large fraction of what rFID's deep feature extractor and a human
eye actually register. The spatial terms patch this only indirectly: L1 barely penalizes the lost
high-frequency residual, LPIPS pressures it through learned features, the discriminator pressures it
through realism — but none says "you are missing *this band* of the spectrum." The residual that survives
three spatial rungs is, I suspect, a *frequency-localized* error: a systematic mismatch in particular
spectral bands that no spatial loss is shaped to target directly. That is a genuinely different axis from
the softness the discriminator attacks, and it is the one the task description itself gestures at —
natural images decompose into frequency bands, and explicitly handling them during training is the open
lever I have not pulled.

The move, then, is to measure reconstruction error directly on the spectrum. The 2D discrete Fourier
transform gives me, per image channel, a complex value `F(u,v)` for every spatial frequency, carrying its
amplitude and phase. Two images match iff their full complex spectra match, so a distance between spectra
is a legitimate reconstruction objective — and unlike the spatial terms it is *natively* organized by
frequency, so it can target the bands the spatial losses keep missing. The natural distance is the squared
Euclidean distance between complex components, `|F_r(u,v) - F_f(u,v)|²`, averaged over the spectrum. But
before I commit to it I have to check it against the obvious failure, because a naive frequency loss would
repeat the spatial loss's sin in a new guise — and here I can actually settle it with a theorem rather
than a hunch.

Take the plain, equally-weighted spectral distance and ask what it equals. If I use an *orthonormal*
(unitary) 2D FFT, Parseval's theorem holds exactly: `Σ_{u,v} |F(u,v)|² = Σ_{x,y} |f(x,y)|²` for any
signal, so applying it to the difference signal `f_r - f_f` gives `Σ_{u,v} |F_r - F_f|² = Σ_{x,y} |x_r -
x_f|²`. In words: the unweighted squared spectral distance *is* the squared pixel distance, exactly the
same number. I checked it numerically on a random `32×32` pair to be sure I had the normalization right,
and the two sums agreed to seven digits — the orthonormal `norm='ortho'` flag is doing exactly the
unitary bookkeeping that makes the identity hold, and had I used the default unnormalized FFT the two
sides would have differed by a factor of the transform length and I might have mistaken a scaling
artifact for a real signal. So an equally-weighted frequency loss is not a new objective at
all — it is pixel MSE wearing a Fourier costume. It would spend its gradient exactly the way L2 does,
which is exactly the blur-hedging behavior I spent three rungs escaping. That kills the naive version
outright, and it also tells me precisely where any real value must come from: *not* from moving to the
frequency domain per se, but from *weighting the frequencies unequally*. The domain change buys nothing;
the weighting is the whole idea.

So what weighting? The pathology of the unweighted loss, restated, is that after three spatial rungs the
low-frequency components are mostly correct and dominate the sum, while the components still wrong are a
small minority of hard, mostly high-frequency ones. An unweighted sum spends its gradient polishing the
already-good frequencies and barely touches the missing ones — the spatial loss's behavior, relabeled. I
need the loss to *adaptively concentrate on the hard frequencies*. What is "hard"? Operationally, a
component is hard exactly when the current frequency error `|F_r - F_f|` is large — a quantity I have for
free every step. So gate each component's distance by a weight matrix that is large where the current
error is large: `w(u,v) = |F_r(u,v) - F_f(u,v)|^α`, normalized into `[0,1]`. Where the model already
matches a frequency, `w ≈ 0` and it is ignored; where it is far off, `w` is large and dominates; and it is
*dynamic* — as a once-hard frequency gets matched its weight decays and the loss moves to the
next-hardest. A self-generated curriculum over frequencies, no hand-set band selection.

Let me verify the weighting actually does what I claim rather than trust the story, by writing the
per-component contribution. With `w = (e / e_max)^α` for an error `e = |F_r - F_f|` normalized by the
per-image max `e_max`, the loss contribution of that component is `w · e² = e^{α+2} / e_max^α`. At `α = 0`
this is `e²` — the unweighted loss, which Parseval just showed is redundant. At `α = 1` it is `e³ / e_max`
— cubic in the error, so already-small errors contribute vanishingly while the largest error contributes
the full `e_max²`. I sanity-checked the shape on five errors evenly spaced up to the max: at `α = 0` the
mid-sized errors still carry real weight (a component at half the max error contributes about a quarter of
the max's contribution, `0.255`), whereas at `α = 1` that same mid component collapses to about an eighth
(`0.129`), and the tiny errors go to essentially zero. So raising `α` from `0` to `1` visibly rebalances
the gradient off the easy frequencies and onto the hard ones — exactly the curriculum I wanted, and
exactly what the redundant `α = 0` case cannot do. The default focusing strength is `α = 1` — the weight
scales linearly with the error magnitude — which is the robust starting point; `α = 0` recovers the loss
I just argued against, and much larger `α` would over-focus on a single frequency and starve the rest.

The one subtlety I must get right is that this weight has to be a *constant* with respect to the
parameters — detached, stop-gradient. If gradients flowed through `w = distance^α`, the loss would be
`e^{α+2}` and the optimizer would see the full derivative, including a term pushing on the *importance
map* itself — a perverse second-order incentive to make some frequencies look artificially hard or the
normalization max look artificially large, rather than simply reducing the distance. Detached, the
gradient is exactly `w(u,v) · ∇(distance²)`: the ordinary spectral-distance gradient, reweighted by a
fixed per-frequency spotlight that says "spend your effort here," and nothing else. Two normalizations
keep the spotlight stable: divide the weight matrix by its per-image max so the loss scale does not
silently drift as absolute errors shrink over training (without it, `w` would shrink with the errors and
the whole term would fade out just as it should be sharpening its focus), clamp to `[0,1]`, and zero any
NaN from the `0^α` corner where a perfectly-matched frequency gives `0/0`. For `32×32` CIFAR a single
global FFT (`patch_factor = 1`) is right; patch-wise spectra are a device for larger images where a global
FFT would blur spatial locality across the whole frame, which is not a concern at this size. Let me trace
the shapes so the normalization operates where I intend. With `patch_factor = 1` the whole image is one
patch, so `tensor2freq` produces a spectrum of shape `[B, 1, C, 32, 32]` and stacks real and imaginary
parts into `[B, 1, C, 32, 32, 2]`; the error magnitude `|F_r - F_f|^α` collapses that last axis back to
`[B, 1, C, 32, 32]`. The per-image max I divide by is taken over the two spectral axes (the `32×32`
frequency grid), so each image-and-channel's spotlight is normalized against *its own* hardest frequency —
which is what makes the `[0,1]` spotlight per-sample and keeps one image's easy spectrum from setting the
scale for another's. And the FFT uses orthonormal normalization — the same choice that made Parseval exact
above — so the transform is unitary and the spectral distance lands on the same scale as a spatial one,
which matters because I am adding it to spatial terms and want `loss_weight` to mean something comparable.

The `FocalFrequencyLoss` constructor also exposes `log_matrix`, `batch_matrix`, and `ave_spectrum`, and I
leave all three off, each for a reason rather than by default. `log_matrix` would replace the weight with
`log(1 + w)`, compressing its dynamic range — useful when frequency errors span many orders of magnitude
and a raw `w` would let one monster frequency dominate everything; here, after three spatial rungs the
errors are already fairly even and the linear `α = 1` weight is the cleaner curriculum, so I do not want to
flatten it. `batch_matrix` would normalize the spotlight by the max over the *entire batch* instead of
per-image, coupling every image's weighting to whichever image in the batch happens to carry the worst
frequency — that cross-image coupling adds variance I have no reason to invite, and per-image
normalization is the more stable choice. `ave_spectrum` would average the spectra across the batch before
comparing, which throws away per-image frequency structure and turns the loss into a distribution-level
statistic; but I want each reconstruction pulled toward *its own* target's spectrum, not toward the batch
mean, so it stays off. The defaults I keep — per-image, linear, per-sample spectrum — are the ones
consistent with a per-image reconstruction objective.

Now the composition, and this is the crucial design point: the focal frequency loss is a *complement*, not
a replacement. It constrains the spectrum but says nothing about absolute pixel values being right — you
can in principle match the spectrum's error-weighted structure while drifting the spatial content, since
the weighting throws away the easy low-frequency components that carry most of the coarse layout — so it
must ride on top of a spatial reconstruction term that anchors the pixels. The natural place to add it is
*on the strongest spatial recipe I have*, the vqgan rung: keep L1 + LPIPS + KL + feature matching + the
loop's adversarial machinery exactly as they are, and add `loss_weight · FFL(recon, target)` on top. FFL's
job is narrow and orthogonal — drag the output's frequency statistics toward the target's, concentrating
on the bands the three spatial rungs keep missing — so it should *stack*, recovering frequency fidelity
that the spatial terms structurally cannot, without disturbing what they already do well. I add the FFL
term always, and the contrast with the gated feature-matching term is instructive. Feature matching had to
wait for `disc_start = 5000` because matching a reconstruction's features to an *untrained* discriminator's
is matching noise — the signal does not exist until `D` has learned to see. The frequency mismatch is the
opposite: it is maximal at step *one*. An untrained decoder's low-pass bias is a property of the network
from initialization, so the spectrum is most wrong early and the FFL term has the most to correct exactly
when the discriminator is still silent. Gating FFL on the warm-up would throw away the quarter of training
where it is most useful. So FFL turns on immediately, weighted at `loss_weight = 1.0`, and I report its
value in the metrics so I can watch the spectral error fall from the first step and confirm it keeps
falling once the adversarial terms join in — if FFL and the discriminator were fighting rather than
stacking, the frequency error would stall or rise when `D` switches on at `5000`.

Let me size that `loss_weight = 1.0` against the other terms rather than accept it blindly, because "1.0"
sounds like it might dominate. By the Parseval identity, the *unweighted* spectral term equals the pixel
MSE, which for a decent reconstruction with per-pixel errors around `0.05–0.1` is order `0.0025–0.01`. The
focal weight `w ∈ [0,1]` then multiplies that down further — most frequencies are matched and get `w ≈ 0`,
so the mean weighted term is well below the raw MSE, landing around `0.001–0.005`. Against `L1 ~ 0.07` and
`0.5·LPIPS ~ 0.05–0.1`, an FFL term of a few thousandths at `loss_weight = 1.0` is roughly an order of
magnitude smaller than the anchors — so despite the unit weight, FFL is a *gentle nudge on the hard
frequencies*, not a term that can hijack the objective. That is the right posture for a complement: strong
enough to redirect gradient onto the missing bands, too small to override the spatial terms that are doing
the heavy lifting. Everything else — the frozen VGG LPIPS, the spectral-norm
`NLayerDiscriminatorWithFeatures`, the `disc_start = 5000` warm-up, stashing `_perceptual_loss` for the
loop's adaptive weight, `perceptual_weight = 0.5`, `kl_weight = 1e-6`, `feat_match_weight = 1.0` — is
carried over verbatim from the strongest baseline. The only new line in the objective is the frequency
term. (The distilled module, including the discriminator and the `FocalFrequencyLoss` class, is in the
answer.)

Let me be explicit about what stays out, because the loss-only edit surface forbids most of it anyway. I
am not touching the architecture, the optimizer, the schedule, the EMA, or the evaluation; the
`AutoencoderKL` stays continuous (no codebook, no quantizer) and the adversarial logic stays in the fixed
loop. The finale is a *pure loss-design* addition: one frequency-domain term, complementary to the three
spatial terms, gated by a detached hard-frequency spotlight.

Now the bar this has to clear and what I would validate, against the strongest baseline's real numbers.
The vqgan rung sits at small `14.75`, medium `7.13`, large `3.38`, gmean `7.08`. For the finale to justify
itself, the gmean must come in *below* `7.08`, and the per-scale story I would expect — by the same logic
that has held for three rungs — is that the frequency term helps most where the high-frequency residual is
largest. That should again be the small scale: the narrowest model through the tightest bottleneck (the
`12:1` compression I traced earlier) loses the most high-frequency detail, so an explicit spectral term
has the most band-mismatch to recover there, and I would expect `14.75` to fall the hardest. Medium and
large should tick down too but by less, large least of all, since at `3.38` it is already crisp enough
that little spectral residual remains — and large was already the scale that barely moved this rung, so a
frequency term with little to recover there is consistent, not surprising. PSNR and SSIM I would expect to
hold or improve slightly, and this is a real point of difference from the perceptual rung: matching the
spectrum more faithfully restores high-frequency content that, unlike the perceptual-versus-pixel trade,
is *not* in tension with pixel accuracy — a sharper, spectrally-correct reconstruction is also a
pixel-closer one, so if anything a faithful spectrum nudges PSNR up rather than trading it away.

The risks I would watch are two, and both are falsifiable. First, if `loss_weight = 1.0` somehow
over-weights the frequency term — if my magnitude estimate is off and FFL turns out comparable to the
anchors — the optimizer could chase spectral fidelity at the expense of the adversarial/perceptual balance
and *raise* rFID by introducing high-frequency artifacts; the signature would be the finale coming in
worse than vqgan, and the fix would be to lower `loss_weight`. Second, and more interesting, if the
surviving residual after three spatial rungs is *not* actually frequency-localized — if the discriminator,
by pressuring high-frequency realism, already captured the missing bands as a side effect — then FFL would
be redundant and the gmean would barely move below `7.08`. That null result would falsify the
"frequency-localized residual" diagnosis, and it is a real possibility precisely because the patch
discriminator does pressure high frequencies, just not by name. But the structural frequency bias of a
bottlenecked autoencoder, and the fact that none of the three spatial terms targets the spectrum
*explicitly* while FFL targets it by construction and concentrates on exactly the hard bands, make me
expect a real, scale-dependent gain — largest on the small scale — that pushes the gmean below the
strongest baseline.

The causal chain in one breath: three spatial rungs (pixel, perceptual, adversarial) have flattened the
curve to gmean `7.08` with shrinking `43% → 24%` cuts, because they all measure error in the spatial
domain and none targets the frequency bands a bottlenecked autoencoder structurally loses → so measure
reconstruction error directly on the 2D FFT spectrum as the squared complex distance → but Parseval shows
the unweighted orthonormal spectral distance equals pixel MSE *exactly* (checked to seven digits), so the
domain change alone is redundant and the weighting must carry all the novelty → gate each component by a
detached, `[0,1]`-normalized hard-frequency spotlight `|F_r - F_f|^α` (`α = 1`, whose `e³` contribution I
verified concentrates gradient off the easy frequencies onto the hard ones, where `α = 0` collapses back
to the redundant MSE) that self-generates a curriculum over the hardest bands → add this `loss_weight·FFL`
term as a *complement* on top of the full vqgan recipe (it constrains the spectrum, the spatial terms
anchor the pixels), sized at a gentle few-thousandths so it nudges rather than hijacks, changing one line
of the objective and nothing else → expecting the small scale's `14.75` to fall the hardest and the gmean
to clear `7.08`, with PSNR/SSIM held or up, while watching the frequency-weight and the
redundancy-with-the-discriminator risk that would falsify the diagnosis.
