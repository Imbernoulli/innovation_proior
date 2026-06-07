# SimSiam synthesis

## Pain point / research question
Label-free visual representation learning via siamese nets "two views agree". The dominant
recipes all carry expensive machinery whose *only* job is collapse prevention:
- contrastive (SimCLR/MoCo): negative pairs (large batch / queue + momentum encoder)
- clustering (SwAV/DeepCluster): online clustering + balanced/Sinkhorn constraint
- BYOL: no negatives, but keeps a momentum target encoder + predictor; hypothesized that the
  momentum encoder is what avoids collapse (removing it -> 0.3% in BYOL Table 5).
Question: strip the siamese net to bare minimum (same encoder, shared weights, no momentum, no
negatives, normal batch) — what is *actually* doing the anti-collapse work?

## The method (SimSiam)
- two augmented views x1,x2 of one image x.
- SAME encoder f (backbone + projection MLP), shared weights, NO momentum.
- predictor MLP h on one branch.
- p1 = h(f(x1)), z2 = f(x2). D(p,z) = -(p/||p||)·(z/||z||) negative cosine. Equiv to MSE of
  L2-normalized vectors up to scale 2.
- symmetrized loss: L = 1/2 D(p1, stopgrad(z2)) + 1/2 D(p2, stopgrad(z1)). min = -1.
- stop-gradient on the target branch is ESSENTIAL: w/ stop-grad 67.7%, w/o -> collapse to
  constant, loss hits -1 immediately, std of L2-normed output -> 0, acc 0.1%.

## Key ablations (motivating/diagnostic — these are about SimSiam's own variants but they are the
## empirical core, central finding). For reasoning, lived inline.
- stop-grad: w/ 67.7 / w/o 0.1 (collapse). std curve: w/stopgrad ~ 1/sqrt(d), w/o -> 0.
- predictor h: remove h -> 0.1 (collapse). For symmetric loss, removing h makes loss
  1/2 D(z1,sg z2)+1/2 D(z2,sg z1); its gradient direction = gradient of D(z1,z2) scaled 1/2, so
  stop-grad becomes vacuous -> collapse expected. h fixed at random init -> 1.5 (not collapse,
  just doesn't converge; loss stays high). h with constant lr (no decay) -> 68.1, slightly better
  -> h should chase the latest representations, don't force it to converge early. => fix_pred_lr.
- batch size 64..4096: SGD, works 64-2048 (66.1..68.1), 4096 drops to 64.0 (SGD large-batch
  issue, not collapse). No LARS needed. SimCLR/SwAV need 4096.
- BN on MLP heads: none->34.6 (no collapse, just opt hard); hidden-only 67.4; default
  (proj output BN too) 68.1; +pred output BN -> unstable (oscillates, not collapse). BN helps
  optimization, NOT collapse prevention (same BN config collapses if stop-grad removed).
- similarity fn: cosine 68.1 vs cross-entropy (softmax target . log softmax pred) 63.2 — both
  avoid collapse -> collapse prevention not about cosine.
- symmetrization: sym 68.1, asym 64.8, asym2x 67.3 — helps accuracy, not collapse prevention.

## EM hypothesis (the derivation to lay out fully)
Loss with extra variable set eta (per-image representation, size ~ #images, NOT a network
output — an optimization argument):
  L(theta, eta) = E_{x,T} || F_theta(T(x)) - eta_x ||^2.
Solve min_{theta,eta}. Analogous to k-means: theta ~ cluster centers (encoder params), eta_x ~
assignment vector of x (one-hot in kmeans -> here the representation of x). Solve by alternation:
  theta^t   <- argmin_theta L(theta, eta^{t-1})    (M-step-like)
  eta^t      <- argmin_eta   L(theta^t, eta)         (E-step-like)
- Solving theta: SGD; eta^{t-1} is constant -> STOP-GRADIENT is the NATURAL CONSEQUENCE (no grad
  flows to eta).
- Solving eta: per-image, minimize E_T||F(T(x)) - eta_x||^2 -> eta^t_x = E_T[F_theta^t(T(x))]
  (mean over augmentations). (cosine: L2-normalize.)
- One-step alternation -> SimSiam: approximate eta by sampling ONE augmentation T':
  eta^t_x <- F_theta^t(T'(x)); plug into theta subproblem:
  theta^{t+1} <- argmin E_{x,T} || F_theta(T(x)) - F_theta^t(T'(x)) ||^2.  theta^t constant
  (the other branch, stop-grad'd); T' = the other view -> SIAMESE shape falls out. Reduce by ONE
  SGD step => SimSiam.
- Predictor: by definition h minimizes E_z||h(z1)-z2||^2 -> optimal h(z1)=E_z[z2]=E_T[f(T(x))],
  i.e. the expectation over augmentations that the one-sample approximation (eq eta) dropped. So
  h fills the gap of approximating E_T[.]; can't compute E_T explicitly, but a net can learn to
  predict it, sampling of T distributed across epochs.
- Symmetrization: denser sampling of T (an extra (T2,T1) pair); helps accuracy, not needed.
- Proof of concept: multi-step alternation (pre-compute eta, k inner SGD steps) all work
  (1/10/100-step 68.1/68.7/68.9, 1-epoch 67.0) -> alternating opt is valid, SimSiam = 1-step
  special case. Expectation-over-aug variant: maintain moving-average eta (m=0.8, like memory
  bank) -> 55.0% WITHOUT predictor h (vs 0.1 collapse without h and without moving avg) ->
  supports h ~ approximating E_T.
- Discussion: hypothesis says WHAT is optimized, not WHY no collapse. Why-no-collapse remains
  empirical: alternating gives a trajectory depending on init; init eta = output of random net is
  not constant; method doesn't take grad wrt eta jointly over all x, so hard to drift to constant.

## Canonical impl (facebookresearch/simsiam)
- SimSiam(base_encoder, dim=2048, pred_dim=512). encoder = resnet w/ fc replaced by 3-layer
  projection MLP: [Lin(prev,prev,bias=F),BN,ReLU] x2 then [fc(prev,dim,bias=F? -> bias removed),
  BN(dim, affine=False)] output BN no affine. predictor = 2-layer: Lin(dim,pred_dim,bias=F),
  BN(pred_dim),ReLU, Lin(pred_dim,dim). forward returns p1,p2,z1.detach(),z2.detach().
- criterion = nn.CosineSimilarity(dim=1); loss = -(crit(p1,z2).mean()+crit(p2,z1).mean())*0.5.
  (z1,z2 already detached in forward.)
- SGD, init_lr = 0.05*bs/256, momentum 0.9, wd 1e-4, cosine decay. fix_pred_lr: predictor uses
  constant lr (not decayed).
- output BN no affine; zero_init_residual; pred output fc has bias, no BN, no ReLU.
- predictor bottleneck: hidden = dim/4 (512 for 2048).
