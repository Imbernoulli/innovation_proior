Measured results — Improved DDPM (cosine schedule + learned reverse variance, hybrid loss).
CIFAR-10 FID, lower is better.

## CIFAR-10 FID
| configuration | FID |
|---|---|
| best reported ($L_{\text{simple}}$, 4000-step chain) | 2.90 |
| improved recipe ($L_{\text{hybrid}}$ + cosine schedule, learned variance) | 3.19 |
| likelihood-optimized ($L_{\text{vlb}}$ + cosine) | 11.47 |

**Sources.** All three are from Nichol & Dhariwal (2021), *Improved Denoising Diffusion Probabilistic
Models*, CIFAR-10 ablation table (500K iterations). The headline number for the *method* — the cosine
schedule with learned variances trained by the hybrid loss — is **3.19** (with the best balanced
likelihood, 3.17 bits/dim). The best CIFAR-10 FID they report overall is **2.90**, obtained with the
plain $L_{\text{simple}}$ on a longer 4000-step chain (not the hybrid+cosine config), which is the
"~2.90" anchor for this rung; the $L_{\text{vlb}}$ likelihood-optimized config gives the best NLL
(2.94 bits/dim) but a much worse FID of **11.47**, confirming that pure variational training, while best
for likelihood, is poor for sample quality — exactly why the hybrid loss keeps the flat
$\boldsymbol{\epsilon}$-MSE in charge of the mean.

The step below the floor is real but modest, and it lands where predicted: the cosine schedule carries
the FID gain, while the learned variance and hybrid loss buy likelihood and training stability rather
than a large FID move — and the deterministic sampler under test does not consume the learned variance
at all. The recipe is now a stack of separate fixes layered onto the same $\boldsymbol{\epsilon}$
parameterization and chain-derived weighting; the across-$\sigma$ emphasis and the network's
input/output scaling are still not chosen for the regression.
