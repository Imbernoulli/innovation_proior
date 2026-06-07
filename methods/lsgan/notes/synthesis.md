# LSGAN synthesis (Phase 1.5)

Verified: arXiv 1611.04076, "Least Squares Generative Adversarial Networks", Mao, Li, Xie, Lau, Wang, Smolley (ICCV 2017).
Canonical impl: based on carpedm20/DCGAN-tensorflow with LS loss; clean PyTorch = eriklindernoren/PyTorch-GAN/implementations/lsgan (saved in code/). D has NO sigmoid (least squares on raw linear output); adversarial_loss = MSELoss; D labels real=1 fake=0; G labels fake-as-1; 0.5 factor. = the 0-1 coding scheme.

## Pain point / research question
- Regular GAN's discriminator is a classifier with SIGMOID CROSS-ENTROPY loss.
- Problem (the diagnostic): when updating G, fake samples that are on the CORRECT side of D's decision boundary (classified as real) but still FAR from the real data cause almost NO loss / gradient under sigmoid cross-entropy — because the sigmoid saturates. So G gets no signal to pull those samples closer to the real data manifold. This is the vanishing-gradient problem.
- Want: a discriminator loss that still penalizes fakes that are correctly-classified-as-real but far from the boundary, so they get pulled toward the real-data manifold. And: more stable training.

## Central insight (insight-before-method)
- Sigmoid cross-entropy: flat (saturated) for large correct-side x → no gradient → no pull.
- Least-squares (L2) loss is flat at ONLY one point → it penalizes any sample whose D-output deviates from the target, including correctly-classified samples far on the correct side → generates gradient → pulls fakes toward the decision boundary.
- The decision boundary, for successful GAN learning, must cross the real-data manifold. So pulling fakes toward the boundary = pulling them toward the real data.

## The LSGAN objective (THE method) — a-b-c coding scheme
a = label for fake data, b = label for real data (for D); c = the value G wants D to believe for fakes.
  min_D V_LSGAN(D) = (1/2) E_{x~pdata}[(D(x) - b)^2] + (1/2) E_{z~pz}[(D(G(z)) - a)^2].
  min_G V_LSGAN(G) = (1/2) E_{z~pz}[(D(G(z)) - c)^2].
NOTE: D has no sigmoid; D outputs a real scalar, least squares regresses it to the labels.

## Relation to f-divergence: Pearson chi-square (the appendix-grade derivation)
Recall regular GAN minimizes JS: C(G) = KL(pd ‖ (pd+pg)/2) + KL(pg ‖ (pd+pg)/2) - log(4).

Extend the G objective to add a pd term (doesn't change the optimum since it has no G params):
  min_G V(G) = (1/2)E_{x~pd}[(D(x)-c)^2] + (1/2)E_{z~pz}[(D(G(z))-c)^2].

Optimal D for fixed G: minimize pointwise pd(D-b)^2 + pg(D-a)^2 over D.
  d/dD = 2pd(D-b) + 2pg(D-a) = 0 → D(pd+pg) = b·pd + a·pg → 
  D*(x) = (b·pd(x) + a·pg(x)) / (pd(x) + pg(x)).   ✓ (eq optimal_d)

Plug into 2C(G) = E_{x~pd}[(D*-c)^2] + E_{x~pg}[(D*-c)^2]:
  D* - c = (b·pd + a·pg)/(pd+pg) - c = ((b-c)pd + (a-c)pg)/(pd+pg).
  2C(G) = ∫ pd · [N/(pd+pg)]^2 dx + ∫ pg · [N/(pd+pg)]^2 dx   where N = (b-c)pd + (a-c)pg
        = ∫ (pd+pg)·N^2/(pd+pg)^2 dx = ∫ N^2/(pd+pg) dx
        = ∫ ((b-c)pd + (a-c)pg)^2 / (pd+pg) dx.
Rewrite N = (b-c)(pd+pg) - (b-a)pg.  [check: (b-c)pd+(b-c)pg-(b-a)pg = (b-c)pd+(a-c)pg ✓]
Set b-c=1 and b-a=2:
  N = (pd+pg) - 2pg = pd - pg = -(2pg - (pd+pg)).
  2C(G) = ∫ (2pg - (pd+pg))^2 / (pd+pg) dx = Pearson chi^2(pd+pg ‖ 2pg).
So minimizing the extended objective = minimizing Pearson chi^2 divergence between (pd+pg) and (2pg), provided b-c=1 and b-a=2.
[Self-check note: the integrand is (pd-pg)^2/(pd+pg); the paper labels this Pearson chi^2(pd+pg ‖ 2pg). With Pearson chi^2(P‖Q)=∫(P-Q)^2/Q this is the χ² with P=2pg shifted, denominator = pd+pg = P+Q form; the paper's labeling follows its own argument-order convention. Present the algebra explicitly; it is correct as derived.]

## Parameter selection (two valid schemes)
Scheme 1 (Pearson, satisfies b-c=1, b-a=2): a=-1, b=1, c=0:
  min_D: (1/2)E_pd[(D(x)-1)^2] + (1/2)E_pz[(D(G(z))+1)^2]
  min_G: (1/2)E_pz[(D(G(z)))^2].
Scheme 2 (0-1 binary coding, set c=b so G makes fakes "as real as possible"): a=0, b=1, c=1:
  min_D: (1/2)E_pd[(D(x)-1)^2] + (1/2)E_pz[(D(G(z)))^2]
  min_G: (1/2)E_pz[(D(G(z))-1)^2].
In practice both give similar performance; the paper USES scheme 2 (0-1) to train. Canonical code = scheme 2.

## Benefits (both derivable)
1. Higher quality: LS penalizes correctly-classified-but-far fakes → pulls toward boundary → toward real-data manifold (boundary crosses manifold).
2. Stability: more gradients (LS flat only at one point, sigmoid CE saturates for large x) → relieves vanishing gradient → more stable; LSGANs converge to a good state even WITHOUT batch normalization (following Arjovsky 2017's BN-exclusion stability test).

## Architectures (design)
- Model 1 (112x112 scenes): motivated by VGG. Vs DCGAN, ADD two stride-1 deconv layers after the top two deconv layers (in G). Discriminator identical to DCGAN except LS loss. ReLU in G, LeakyReLU in D (DCGAN). Generator: project z, deconv stack; discriminator: conv stack, NO sigmoid, output regressed by LS loss.
- Model 2 (many-classes, e.g. 3740 Chinese chars): conditional LSGAN. One-hot label vectors with thousands of classes too costly → use a LINEAR MAPPING layer Φ(y) to map large label vectors to small vectors, then concatenate to model layers. Need deterministic input→output relation (Hornik 1989): conditioning on label creates it.
  min_D = (1/2)E_pd[(D(x|Φ(y))-1)^2] + (1/2)E_pz[(D(G(z)|Φ(y)))^2]
  min_G = (1/2)E_pz[(D(G(z)|Φ(y))-1)^2].

## Training details (grounded line 331)
- Based on carpedm20/DCGAN-tensorflow, TensorFlow.
- lr: scenes 0.001, Chinese chars 0.0002. Adam, β1=0.5 (following DCGAN).
- Datasets: LSUN (5 scenes), HWDB1.0 (Chinese chars, 3740 classes).

## Baselines (prior art to elaborate)
- GAN (Goodfellow 2014): sigmoid cross-entropy D; minimizes JS divergence. Gap: vanishing gradients for far-but-correct fakes.
- DCGAN (Radford 2015): stable conv arch; the architecture LSGAN modifies.
- LAPGAN (Denton 2015), feature matching (Salimans 2016): quality improvements.
- f-GAN (Nowozin 2016): GAN as f-divergence minimization (JS is special case); generalizes to arbitrary f-divergences. LSGAN is another point in this space (Pearson χ²).
- WGAN (Arjovsky 2017): Wasserstein distance instead of JS; analyzed divergence properties; the BN-exclusion stability test. (LSGAN claims faster than WGAN since WGAN needs multiple D updates.)
- LS-GAN / Loss-Sensitive GAN (Qi 2016): loss assumes real < fake losses; non-vanishing gradient.
- EBGAN (Zhao 2016): energy/autoencoder D.
- Conditional GAN (Mirza 2014): conditioning on labels.

## Design-decision → why table
- Replace sigmoid-CE with least-squares D loss → CE saturates for far-correct fakes (no gradient); LS penalizes them → pulls toward boundary/manifold + more gradient → quality + stability.
- a-b-c coding (general labels) → flexibility; lets you hit the Pearson χ² conditions OR the "as-real-as-possible" condition.
- b-c=1, b-a=2 → makes the objective = Pearson χ² divergence (theoretical grounding).
- a=-1,b=1,c=0 OR a=0,b=1,c=1 → two concrete instantiations; use 0-1.
- D without sigmoid → least squares regresses the raw output to labels; no saturating squashing.
- Two extra stride-1 deconv layers in G (model 1) → VGG-motivated, deeper G for 112x112.
- Linear mapping Φ(y) for labels (model 2) → one-hot over thousands of classes infeasible; map to small dense vector, concat.
- Adam β1=0.5, DCGAN backbone → inherit stable GAN training recipe.
```
