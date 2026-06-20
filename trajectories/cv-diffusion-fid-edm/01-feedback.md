Measured results — DDPM (VP preconditioning + simple $\boldsymbol{\epsilon}$-MSE loss). CIFAR-10
FID over 50,000 samples, lower is better.

## CIFAR-10 FID
| setting | FID |
|---|---|
| unconditional, native DDPM recipe (full ancestral sampler) | 3.17 |
| conditional, VP recipe under the fixed deterministic sampler (NFE = 35) | 2.48 |
| unconditional, VP recipe under the fixed deterministic sampler (NFE = 35) | 3.01 |

**Sources.** The native unconditional **3.17** is the DDPM "Ours ($L_{\text{simple}}$)" row
(Ho, Jain & Abbeel 2020, Table 1; Inception Score 9.46), the flat unweighted $\boldsymbol{\epsilon}$-MSE
on the linear-$\beta$ VP chain with full ancestral sampling. The two NFE = 35 figures are the same VP
training recipe re-evaluated under this trajectory's fixed deterministic sampler — the **config A
(Baseline)** row of the EDM additive-ablation table (Karras, Aittala, Aila & Laine 2022, Table 2),
CIFAR-10 conditional VP **2.48** and unconditional VP **3.01**. These NFE = 35 numbers are the
apples-to-apples baseline for the rungs that follow, since every later training-design change is
measured under the identical sampler.

The floor behaves as expected: a strong, GAN-free FID in both settings, with headroom left exactly
where the recipe inherited its choices from the chain rather than fitting them — the across-$\sigma$
loss emphasis and the unfitted reverse variance.
