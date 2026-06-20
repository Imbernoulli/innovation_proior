Measured results — ADM (improved denoiser U-Net architecture). FID, lower is better.

## ImageNet-64 FID (class-conditional)
| model | FID |
|---|---|
| Improved DDPM backbone | 2.92 |
| ADM | 2.61 |
| ADM (dropout) | 2.07 |

**Sources.** All from Dhariwal & Nichol (2021), *Diffusion Models Beat GANs on Image Synthesis*,
Table 5 (ImageNet $64\times64$, class-conditional, 250 sampling steps). The headline architecture
result is **ADM (dropout) = 2.07 FID** — the plain ADM U-Net with the multi-resolution attention,
64-channel heads, AdaGN conditioning, and BigGAN resampling, sampled **without** classifier guidance
(classifier guidance is a separate contribution applied at higher resolutions, not to this $64\times64$
number). The same table shows ADM stepping down from the iDDPM backbone (2.92 → 2.61 → 2.07), isolating
the architecture as the source of the gain.

This is a context rung: it does not move the CIFAR-10 training-design leaderboard, it resets the
*substrate*. The backbone is now strong enough — state-of-the-art at $64\times64$ on architecture alone
— that it can be carried into a training-design study "with no changes." The open problem is sharpened:
with a near-best backbone fixed, any remaining FID headroom must come from redesigning how the denoiser
is *trained* (preconditioning, loss weighting, $\sigma$-distribution, augmentation), since the
architecture is no longer the limiting factor.
