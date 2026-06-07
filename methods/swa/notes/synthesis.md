# SWA synthesis (grounded)

## Verified
- arXiv 1803.05407, "Averaging Weights Leads to Wider Optima and Better Generalization", Izmailov, Podoprikhin, Garipov, Vetrov, Wilson (UAI 2018).
- Canonical impl: timgaripov/swa (PyTorch); upstreamed as torch.optim.swa_utils (AveragedModel, SWALR, update_bn).

## Pain point / research question
- Conventional SGD training converges to a single point that minimizes train loss, but train-loss and test-error surfaces are SHIFTED relative to each other, so the train-loss minimizer is off-center for test error -> suboptimal generalization.
- Want better generalization with ~no extra cost, as a drop-in replacement for SGD. Conjecture (Keskar 2017, Chaudhari 2016): WIDTH/flatness of the optimum correlates with generalization, because broad optima stay near-optimal under the train->test shift.

## Background / lineage
- Polyak-Ruppert averaging (Ruppert 1988, Polyak & Juditsky 1992): in convex optimization, averaging the iterates of SGD (with decaying LR) accelerates convergence. Not typically used for NN training. Practitioners sometimes use EMA (exponentially-decaying running avg) with decaying LR -> just smooths SGD, performs comparably (no real gain).
- Mandt 2017: SGD with constant LR ~ sampling from a Gaussian centered at the loss min, covariance set by LR. So constant-LR SGD iterates lie ~on the surface of a sphere (high-dim Gaussian). Averaging goes INSIDE the sphere to a higher-density central point.
- Garipov 2018 (FGE, Fast Geometric Ensembling): cyclical LR generates a sequence of weight-space points that are close but produce DIVERSE predictions; ENSEMBLE their predictions -> strong ensemble in the time of training one model. Also showed mode connectivity (optima connected by low-loss curves). FGE proposals are on the PERIPHERY of the good-weights set.
- Keskar 2017: sharp minima (large-batch) generalize worse; flat-in-most-directions but steep in some.

## Key observations (motivating, pre-method — about SGD trajectories)
- Run SGD with cyclical or constant LR from a pretrained point; the iterates explore the PERIPHERY of the set of high-performing networks.
- Cyclical-LR proposals are individually more accurate (each cycle fine-tunes with decreasing LR); constant-LR explores more but individual proposals worse.
- Train-loss and test-error surfaces are similar but shifted; averaging several proposals gives a more ROBUST CENTRAL point, centered on the shifted test mode -> higher test performance than any individual proposal.

## The SWA method (Sec 3.2, Alg 1)
- Start from pretrained w_hat (trained conventionally for full or e.g. 0.75B budget).
- Continue training with cyclical OR constant LR.
- Capture models w_i: at minima of the cyclical LR (once per cycle), or once per epoch for constant LR.
- Average their weights: w_SWA = (1/n) sum_i w_i.
- Running-average update (per captured model), n_models = current count:
  w_SWA <- (w_SWA * n_models + w) / (n_models + 1)
  [equivalent form used in PyTorch: avg <- avg + (w - avg)/(n_models+1); algebraically identical]
- Cyclical LR schedule: alpha(i) = (1 - t(i)) alpha_1 + t(i) alpha_2, t(i) = (1/c)(mod(i-1,c)+1), alpha_1>=alpha_2, cycle length c. DISCONTINUOUS jump from min back to max (unlike Smith/Garipov triangular) — exploration > individual-proposal accuracy. Constant LR: alpha(i)=alpha_1, c=1.
- BATCH NORM fix: BN running mean/var were never collected for w_SWA (it's an average of weights, never used in a forward pass during training). So after training, do ONE extra forward pass over the data in train mode with w_SWA to compute the BN statistics.

## Why it works (the derivations)
### Width of optima (Sec 3.4)
- Compare w_SWA vs w_SGD along random rays w(t,d)=w + t·d and along the segment w(t)=t·w_SGD+(1-t)·w_SWA.
- w_SGD has slightly LOWER train loss but higher test error; SWA's loss/error curves are WIDER (must step much further from w_SWA to raise error). SWA is in the SAME basin as SGD but a flatter, more central region.
- The loss is asymmetric: w_SGD sits near the boundary of a wide flat region, steep on one side; SWA moves to the flat interior. The train/test shift + this asymmetry explains SWA's better generalization.

### Connection to ensembling / FGE (Sec 3.5) — the Taylor argument
- f(w) = network's (scalar) prediction, twice differentiable. FGE points w_i close together, average w_SWA = (1/n) sum w_i, Delta_i = w_i - w_SWA, so sum_i Delta_i = 0.
- Averaging PREDICTIONS: f_bar = (1/n) sum f(w_i). Linearize each at w_SWA:
  f(w_j) = f(w_SWA) + <grad f(w_SWA), Delta_j> + O(||Delta_j||^2).
- f_bar - f(w_SWA) = (1/n) sum [<grad f, Delta_i> + O(||Delta_i||^2)] = <grad f, (1/n) sum Delta_i> + O(Delta^2) = O(Delta^2),
  because (1/n) sum Delta_i = 0 (first-order term VANISHES).
- Meanwhile predictions of different proposals differ by f(w_i)-f(w_j) = <grad f, Delta_i - Delta_j> + O(Delta^2) = FIRST order.
- So: ensembling-vs-weight-averaging differ at SECOND order, while the diversity between proposals is FIRST order. Hence averaging weights (one model) ≈ ensembling predictions (n models) for nearby proposals. SWA approximates an FGE ensemble with a single model -> ensemble-like generalization at single-model test cost.

### Convex-minimization view (Sec 3.6)
- Constant-LR SGD ~ Gaussian samples on a sphere (Mandt); averaging moves to the interior (higher density / more central), a flatter point.

## Complexity (Sec 3.3)
- Memory: keep one extra copy of weights (running average). But NN memory is dominated by activations not weights, so overhead ~10% during training; after training only store the single averaged model -> same as standard.
- Time: only the weighted-sum update once per epoch (or per cycle); negligible. SWA ≈ same cost as SGD.

## Design-decision -> why
- Average WEIGHTS not predictions (vs FGE): Taylor argument shows they agree to O(Delta^2); weight-avg gives a SINGLE model -> cheap test-time, no need to store/run n models.
- Cyclical or constant (high) LR (not decaying): need EXPLORATION of the good-weights region to get diverse, peripheral points to average; a decaying LR collapses to one point (that's why EMA+decaying-LR doesn't help).
- Discontinuous cycle (jump min->max): exploration matters more than per-proposal accuracy.
- Capture at LR minima (cyclical) / per epoch (constant): points are then individually decent and spread out.
- Start from pretrained w_hat: SWA refines within a good basin; you don't average random early junk.
- BN forward pass after: w_SWA never produced activations during training, so its BN running stats must be recomputed.
- Running incremental average (not store all w_i): O(1) extra memory.

## Eval settings (pre-method)
- CIFAR-10, CIFAR-100, ImageNet. Architectures: Preactivation ResNet-164, VGG-16, Wide ResNet-28-10, PyramidNet, DenseNet, Shake-Shake. Metric: test accuracy/error. Baselines: conventional SGD, FGE.

## Scaffold correspondence
- Pre-method scaffold: standard SGD training loop (model, optimizer, LR scheduler, epochs). Slots: the LR schedule in the tail, and "what final weights to use" (default = last SGD iterate).
- Final code: after pretraining budget, switch to constant/cyclical SWA LR; maintain AveragedModel running average updated each epoch/cycle; at the end recompute BN stats over the loader. (torch.optim.swa_utils: AveragedModel, SWALR, update_bn.)
