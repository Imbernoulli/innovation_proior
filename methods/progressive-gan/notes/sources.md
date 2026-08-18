# Sources — progressive-gan, track B_selfaccount, key_step: real-image fade-in

Audit target: the decisive step "fade in each new block with `frac(lod)` AND fade the reals
identically so D cannot use high-frequency sharpness as a shortcut," landed in `reasoning.md`
around the passage containing "at the instant a new block appears — when it is still random and
producing only blocky output — the real images D sees are equally blocky."

## Verification pass

- type: primary
  path: `refs/primary/paper.tex`, line 107
  quote: "When new layers are added to the networks, we fade them in smoothly, as illustrated in
  Figure~\ref{fig:fadein}. This avoids sudden shocks to the already well-trained,
  smaller-resolution layers."
  supplies: the paper's own stated rationale for fading in new blocks at all (avoid destroying an
  already-trained lower-resolution solution). This is the general principle the real-image fade
  is an instance of.

- type: primary
  path: `refs/primary/figures.tex`, lines 20-22 (caption of `fig:fadein`)
  quote: "When training the discriminator, we feed in real images that are downscaled to match the
  current resolution of the network. During a resolution transition, we interpolate between two
  resolutions of the real images, similarly to how the generator output combines two resolutions."
  supplies: the explicit primary-source statement that real images are blended between the old and
  new resolution during a transition, by the same mechanism as the generator's output blend. This
  is the direct textual basis for "fade the reals identically." The paper states the fact and ties
  it to the generator blend by analogy; it does not spell out the discriminator-shortcut failure
  mode in words — that causal "why" is the reasoning trace's own derivation, and it is checked
  against the actual mechanism below rather than asserted as a paraphrase of the paper.

- type: official-code
  path: `code/progressive_growing_of_gans/train.py`, function `process_reals`, `FadeLOD` block
  quote (mechanism, not prose): `y = tile(avg_pool_2x2(x))` (one-octave blur via average-pool then
  tile back to original size); `x = tfutil.lerp(x, y, lod - tf.floor(lod))`, i.e.
  `x_faded = (1 - frac(lod)) * x + frac(lod) * blurred(x)`.
  supplies: the exact, runnable mechanism for the real-image fade — confirms the blend weight is
  `frac(lod)` on the blurred path, the same fractional schedule variable that drives the
  generator/discriminator block blend (`networks.py`, `lerp`/`lerp_clip` calls keyed off
  `lod_in - lod`). This is what lets the "checked at the boundaries" numeric verification in
  `reasoning.md` be checked against a real implementation rather than invented.

## Assessment

The decisive step was already grounded correctly: `reasoning.md`'s derivation states the concrete
failure mode (an ungated fade lets D use real-vs-fake sharpness as a trivial, uninformative tell)
and immediately verifies it numerically (residual-vs-blur measurement at old_weight = 0, 0.5, 1.0).
The blend formula and direction (`old_weight = frac(lod)`, real fade
`x_faded = (1-old_weight)*x + old_weight*upscale(downscale(x))`) match `process_reals` in
`train.py` exactly, and the general "avoid sudden shocks" rationale plus the explicit "interpolate
reals similarly to the generator" statement in the primary source support treating this as a real,
checkable mechanism rather than a stated-as-given trick. No factual error found in the fade
formula, sign, or schedule constants in `reasoning.md`, `answer.md`, or `train_answer.md`; all
three use the same `old_weight = frac(lod)` convention and the same real-fade formula, consistent
with the code. No edit made to `reasoning.md`.

## Self-account search (this pass)

Re-ran the search the earlier pass (`refs/self_accounts/search_log.md`, 2026-06-18) had done, plus
new queries aimed at more recent material:

- `Tero Karras "progressive growing" GAN interview discriminator shortcut fade in why`
- `Tero Karras StyleGAN oral history podcast "progressive growing" origin story GTC talk`

Found only: academic citations/derivative blog posts (Paperspace, MachineLearningMastery, Medium —
all third-party explainers, not author self-accounts), the official GitHub repo (already saved),
seminar-listing pages (FCAI, HIIT "Machine Learning Coffee" calendar entries for a Karras talk —
event announcements with no transcript or recording linked), and NVIDIA GTC talk listings for
Karras on unrelated later topics (limited-data GAN training). No long-form author interview,
retrospective, or transcript with additional technical content on the fade-in mechanism turned up.
This confirms the finding already on record in `search_log.md`. Relying on the primary paper +
official code combination for this step, as recorded above.
