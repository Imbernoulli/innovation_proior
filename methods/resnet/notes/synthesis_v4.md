# Synthesis V4 — Deep Residual Learning (notes-first; compose results FROM this)

This is the V4 working notes file. It carries (a) the design-decision → why table with rejected
alternatives + failure modes, (b) the load-bearing-ancestor write-ups, (c) the pre-method
Code-framework scaffold spec (CRITICAL V4 change), and (d) the discovery-order spine for
reasoning.md. results_v4/{context,reasoning,answer}.md are composed FROM this, not from memory.

V4 deltas vs V3 (per the re-read skill, which changed materially):
1. NOTES-FIRST: this file exists before the results; results transcribe it.
2. CODE-FRAMEWORK = MINIMAL PRE-METHOD SCAFFOLD. At context time we do NOT yet know about
   residual connections / identity shortcuts / bottlenecks / F(x)+x / Eltwise-SUM. The scaffold
   is a bare deep-CNN image-classification harness: generic conv-bn-relu block builder, a
   `class Net: # TODO: the architecture we'll design` that stacks blocks, SGD-with-momentum,
   standard augmentation, cross-entropy, train/eval loop. NO "reference implementation"/"official
   repo" wording. NO method names (residual/shortcut/skip/bottleneck/F(x)+x/Eltwise). Final code
   in reasoning.md/answer.md FILLS IN and corresponds piece-for-piece to this scaffold.
   Derivation method: write final paper/author-code-grounded code first, then hollow it out.
3. INSIGHT-BEFORE-METHOD enforced locally at every step; keep V3 fixes (no "provably absurd";
   no "here's the argument that turns X into Y"; surprise not paradox).
4. Standing rules: in-frame; context five sections; reasoning continuous first-person, ZERO real
   markdown headers, all derivations inline, 2.4 revision pass; answer opens with the method.

================================================================================
A. DESIGN-DECISION → WHY TABLE (with rejected alternatives + their failure modes)
================================================================================

D1. Reparameterize a block to learn F(x)=H(x)-x, recover H=F+x.
  WHY: Degradation is an OPTIMIZATION/conditioning failure, not representation (identity-by-
  construction proves a >=-as-good solution exists). Reparameterizing moves the cheap-to-reach
  region of weight space (small weights => F~0) on top of a sensible function (identity) instead
  of the useless zero map. Precedent that "reformulate around the residual eases optimization":
  VLAD/Fisher (encode residual to dict centers), residual VQ, Multigrid / hierarchical-basis
  preconditioning (solve residual between scales -> far faster convergence).
  REJECTED: learn H directly (status quo) -> identity must be built from scratch through
  conv-bn-relu, a fussy non-zero point in weight space SGD has no pull toward -> degradation.

D2. Shortcut is a PARAMETER-FREE IDENTITY (y = F(x)+x), not a learned/gated path.
  WHY: identity adds 0 params, 0 meaningful FLOPs -> plain vs. modified twin is a perfectly
  controlled comparison (no extra-capacity escape hatch); the identity path is NEVER closed, all
  info always flows.
  REJECTED A: Highway-style gate y = H(x)*T(x) + x*C(x), T/C data-dependent learned gates. Strictly
  more general (contains identity as T=const) BUT: gate has params (breaks clean comparison) and can
  CLOSE (carry->0), reverting that layer to the original hard non-residual problem exactly where
  depth needs the path most. More-general != better; the constraint is the point. Highway also had
  not shown gains past ~100 layers.

D3. Final ReLU AFTER the addition: sigma(F+x), not before.
  WHY: a ReLU on the residual branch output clamps F>=0, killing negative corrections (F must be
  able to represent a negative nudge). Rectify only the sum so the next block sees a clean signal.

D4. Dimension matching at stage transitions: OPTION B (1x1 projection only when channels/stride
  change; identity everywhere else).
  WHY: identity is free and correct wherever dims already match (the vast majority of skips);
  project only at the 2-3 places per net where channels double / spatial halves.
  OPTIONS:
   A = identity + zero-pad extra channels (still param-free). Failure: padded channels carry no
       learned signal across the transition -> a hair worse at transitions.
   B = project (1x1 conv + BN, with stride) ONLY at transitions; identity elsewhere. DEFAULT.
   C = project on EVERY skip (square W_s on matched skips too). Failure: adds params on dozens of
       skips, muddies the controlled comparison (gain could be from extra linear capacity), and for
       bottleneck DOUBLES the cost of the high-dim blocks. Marginal at best.
  Honest reading: A,B,C all >> plain; only slightly differ -> the LESSON is "projections are not
  essential; the identity skip is what fixes degradation."

D5. Falsifiable preconditioning prediction: learned residual responses should be SMALL on average;
  deeper nets -> even smaller per-layer responses (work spread over more blocks). Measure per-layer
  response std later.

D6. BOTTLENECK block 1x1 reduce -> 3x3 -> 1x1 restore (expansion 4) for deep nets (50/101/152).
  WHY: 3x3 cost is quadratic in channels it sees. Running it at reduced width (e.g. 256->64->...->256)
  makes one 3-layer bottleneck cost ~ one 2x(3x3) basic block -> same FLOP budget buys ~3 weight
  layers instead of 2 -> far more usable depth.
  REJECTED: stack two 3x3 at 512 channels deep -> blows FLOP budget long before 152 layers.

D7. Keep the bottleneck skip an IDENTITY (do NOT project it).
  WHY: the skip spans the two HIGH-dim ends (256->256). A 256->256 1x1 projection is ~as big as the
  rest of the block -> projecting roughly DOUBLES block cost+params. Identity makes it free. This is
  the strongest reason for B over C: C would double the cost of exactly the blocks we most stack.
  PAYOFF: 152-layer net ~11.3 GFLOPs < VGG-19 19.6 GFLOPs despite 8x depth.

D8. Stride-2 downsampling for the original bottleneck lives on the entry 1x1 reduce and the matching
  projection shortcut.
  WHY: this is the paper/authors' Caffe convention: the stage-transition branch2a 1x1 has stride 2,
  branch2b 3x3 has stride 1, and the projection shortcut uses stride 2. TorchVision's later
  ResNet-v1.5 variant moves the stride to the 3x3; do not use that variant for the paper-faithful
  artifact.

D9. Architecture: VGG-thin. Stem 7x7/64 stride2 -> 3x3 maxpool stride2; four stages 64->128->256->512,
  halve map / double channels per transition (VGG rule); downsample via stride-2 conv inside stages;
  global average pool (NIN) -> single 1000-fc -> softmax (NO fat FC head).
  WHY: keep VGG's 3x3-and-doubling philosophy (proven to scale) but drop its 19.6-GFLOP FC-heavy head.
  34-layer ~3.6 GFLOPs (~18% of VGG-19) yet deeper.
  Family: 18=[2,2,2,2] basic; 34=[3,4,6,3] basic; 50=[3,4,6,3] bneck; 101=[3,4,23,3]; 152=[3,8,36,3].
  Depth goes mostly into stage-3 (14x14) where there's spatial room and channels aren't maximal.

D10. Training recipe inherited UNCHANGED (only the reformulation is the changed variable):
  BN after each conv, before activation; He(2015) init (kaiming fan_out, ReLU-aware); train from
  scratch; SGD batch256, momentum 0.9, weight decay 1e-4, lr 0.1 /10 on plateau (up to 60e4 iters);
  VGG/AlexNet aug (scale jitter shorter-side [256,480], 224 random crop+flip, per-pixel mean subtract,
  color aug); no dropout (BN regularizes); test 10-crop or fully-conv multi-scale.
  CIFAR caveat: 110+ layers, lr 0.1 too hot -> brief 0.01 warmup until err<~80% then 0.1.

D11. BN's DOUBLE ROLE (why it's load-bearing for the argument, not just the recipe):
  (i) it's WHY the plain deep net converges at all; (ii) it's the EVIDENCE that degradation is NOT a
  vanishing-gradient problem — with BN, forward variance and backward gradient norms are healthy, so
  the deep plain net's failure can't be blamed on signal magnitude. This RULES OUT the easy
  explanation and leaves "optimization conditioning" as what remains.

D12. The identity-term gradient view — SELF-CORRECTED, kept as bonus not the explanation.
  Tempting: for the pre-activation sum s=F(x)+x, the exact Jacobian is ds/dx=J_F(x)+I; across L such
  sums the product of (I+J_F,l) terms has an additive identity term in its expansion. Exact v1 blocks
  also have the post-add ReLU, whose diagonal activity mask left-multiplies the block Jacobian. GENUINE
  local path when active, BUT must NOT become THE explanation: we already argued (via D11) that
  vanishing gradients aren't the cause here. So the load-bearing claim stays CONDITIONING; the clean
  gradient path is a welcome bonus. (Narrator states the tempting version, then catches himself and
  demotes it.)

================================================================================
B. LOAD-BEARING ANCESTORS (verified vs src/residual_v1_arxiv_release.tex)
================================================================================

- VGG (Simonyan & Zisserman, ICLR'15): uniform 3x3 stacks (two 3x3 = 5x5 RF + extra nonlinearity,
  three = 7x7); width rules (same map -> same #filters; halve map -> double filters keeps per-layer
  time ~const); big FC head + softmax; VGG-19 = 16 conv+3 FC, ~19.6 GFLOPs, most params in FC.
  GAP: proves uniform 3x3 stacking scales but stops ~19 layers, heavy FC head; naively deeper VGG
  hits degradation. REUSE: 3x3-and-doubling philosophy, minus FC head (-> global avg pool), slimmer.

- GoogLeNet/Inception (Szegedy et al., CVPR'15): Inception modules (parallel 1x1/3x3/5x5/pool,
  concatenated), 1x1 dimension-reduction "bottlenecks" keep 22 layers affordable; auxiliary
  classifiers at intermediate depths to inject gradient. GAP: hand-engineered heterogeneous modules;
  aux classifiers are a band-aid for depth's optimization difficulty, not a fix. REUSE: 1x1 reduce ->
  3x3 -> 1x1 restore bottleneck shape (do expensive spatial conv at reduced channels).

- Batch Norm (Ioffe & Szegedy, ICML'15): normalize pre-activation over mini-batch to 0-mean/unit-var,
  learnable gamma/beta: yhat = gamma*(x-mu_B)/sqrt(var_B+eps)+beta. Stabilizes layer-to-layer dist,
  permits larger lr, lets ~30-layer nets train in far fewer steps, mild regularizer (often replaces
  dropout). ROLE (not competitor): makes the puzzle SHARP — with BN, can't blame vanishing/exploding.

- He init / PReLU (He et al., ICCV'15): variance-preserving init for ReLU nets (var ~ 2/fan; Xavier
  under-scales by the factor reflecting ReLU's halved variance). ROLE: init for every plain/deep net,
  part of why they converge from scratch.

- Highway Networks (Srivastava, Greff & Schmidhuber, 2015) — CONCURRENT: y = H(x,W_H)*T(x,W_T) +
  x*C(x,W_C), data-dependent transform gate T and carry gate C (often C=1-T), each with own learned
  params. GAP (direct contrast): gates have params + data-dependent, so a closed gate (T->1,carry->0)
  reverts to ordinary non-residual transform -> skip not guaranteed open; no demonstrated gains past
  ~100 layers. The seam a parameter-free, never-closing skip pushes on.

- Shortcut lineage (older): linear input->output skip in MLPs (Bishop'95, Ripley'96, Venables&Ripley'99);
  centering responses/gradients via shortcuts (Schraudolph'98, Raiko'12, Vatanen'13); aux-classifier
  wiring. GAP: treat skip as gradient-injection/centering, not as changing a block's LEARNING TARGET.

- Smaller pieces: ReLU (Nair&Hinton'10); global avg pool + 1x1 conv (NIN, Lin'13); AlexNet aug /
  10-crop (Krizhevsky'12); dropout (Hinton'12, omittable once BN regularizes).

- Residual-representation / preconditioning analogies (MOTIVATION, not mechanism): VLAD (Jegou'12) /
  Fisher Vector (Perronnin&Dance'07) encode residual to dictionary centers; residual VQ (Jegou'11);
  Multigrid (Briggs'00) + hierarchical-basis preconditioning (Szeliski'90,'06) solve residual between
  scales -> dramatically faster convergence. Thread: reformulate/precondition around the residual ->
  easier optimization.

- Degradation observation: He&Sun "Convolutional neural networks at constrained time cost" (CVPR'15);
  Highway. Deeper plain nets get HIGHER TRAINING error (not overfitting). The pain point.

================================================================================
C. CODE-FRAMEWORK SCAFFOLD SPEC (PRE-METHOD — the V4 critical rewrite)
================================================================================

Vocabulary allowed: deep CNN, image classification, conv, BN, ReLU, block, stage, stride-2
downsample, global average pool, FC head, softmax/cross-entropy, SGD+momentum, step LR decay, weight
decay, He init, augmentation (scale jitter/crop/flip/mean-subtract), train/eval loop.

FORBIDDEN here: residual, shortcut, skip, bottleneck, identity-add, Eltwise/SUM, F(x)+x, "out +=
identity", any 2-conv-block-WITH-a-skip diagram, "reference implementation", "official repo", the
method name.

The scaffold is the bare harness that exists BEFORE the method. It has exactly ONE big empty slot —
the architecture (`class Net`) and the block it stacks — which the reasoning will fill. The block
builder is a GENERIC conv-bn-relu sub-stack with NO second branch and NO addition (we don't yet know
to add one). Pieces (each maps to a stub):

  - conv3x3 / conv1x1 helpers (bias=False; BN supplies the shift)              [KNOWN -> kept]
  - def conv_bn_relu_block(...):  # TODO: the layer block we'll design        [STUB - generic]
        a plain stack of conv->BN->ReLU; how many convs, how they connect = TODO
  - class Net(nn.Module):  # TODO: the architecture we'll design               [STUB]
        stem (7x7/64 s2 + 3x3 maxpool s2), four stages stacking the block with
        halve-map/double-channels, global avg pool -> FC. _make_stage stacks `block`.
  - He-init pass over convs; BN gamma=1/beta=0                                 [KNOWN -> kept]
  - optimizer = SGD(momentum=0.9, weight_decay=1e-4); StepLR /10                [KNOWN -> kept]
  - criterion = CrossEntropyLoss                                               [KNOWN -> kept]
  - train_transform = scale-jitter + 224 random crop + flip + mean-subtract     [KNOWN -> kept]
  - train/eval loop                                                            [KNOWN -> kept]

CORRESPONDENCE to final code: the generic `conv_bn_relu_block` stub becomes BasicBlock/Bottleneck
(now with the `out += identity` second branch); `class Net` becomes `class ResNet`; `_make_stage`
becomes `_make_layer` (which decides identity vs 1x1-projection downsample); the rest (helpers, init,
SGD/StepLR, criterion, transforms, loop) is unchanged. So the final code FILLS IN exactly these stubs.

================================================================================
D. DISCOVERY-ORDER SPINE for reasoning.md (insight-before-method at every step)
================================================================================

1. The blunt fact: depth wins (AlexNet 8 -> VGG 19 -> GoogLeNet 22), so is a better net just more
   layers? If yes someone would have a winning 100-layer net; nobody does. What stops us?
2. First suspect: vanishing/exploding gradients (Bengio'94, Glorot'10). But that wall is down:
   variance-preserving init (Xavier/Saxe/He) + BN -> 30-layer nets converge. With BN, forward var
   healthy and (check, don't assume) backward grad norms healthy. Easy explanation GONE -> progress.
3. Cleanest mental experiment: plain VGG-thin 20 vs 56 layer, both BN+He. Deeper has higher TRAINING
   error all the way. Sit with it: higher TEST error = overfitting (train err goes DOWN); here train
   err goes UP -> NOT generalization, it's OPTIMIZATION. General (constrained-time-cost, Highway,
   even MNIST). Name it: degradation.
4. Identity-by-construction: deeper net can copy shallower + set extra layers = identity -> same
   function -> same training error -> should NEVER be worse. Tried 3x iters; still degrades. A
   >=-as-good solution provably EXISTS in the space; optimizer can't reach it. Not a contradiction —
   a SURPRISE with sharp shape; that precision is the clue.
5. Stare at WHAT the solver fails to find: layers acting like identity. Why is that hard? ReLU wall
   at 0; small-random init centered at 0 makes a fresh conv-bn-relu stack compute a small RANDOM
   transform, not x; nothing pulls it toward identity (a fussy non-zero point). So "shallow + harmless
   extra layers" is hard -> exactly the measured degradation.
6. Reframe: not a representation problem (good solution exists) but a CONDITIONING problem -> fix is
   a REPARAMETERIZATION, not a new layer or solver.
7. Precedent for "reformulate around residual eases optimization": VLAD/Fisher, residual VQ,
   Multigrid/hierarchical-basis preconditioning. -> the move.
8. Apply: let block output F(x)=H(x)-x, recover H=F+x. Same expressive power (just changed coords).
   Reference point shifts: optimal=identity => F must be ZERO => trivial to reach (weights->0, where
   weight decay+small init already push). The pathologically-hard case becomes the EASIEST. The bet.
9. Not just exact identity: near-identity refine = small residual referenced to x, easier than whole
   function from scratch. Falsifiable: learned responses should be small; deeper -> smaller.
10. Realize H=F+x in a feedforward net: add x straight across with element-wise add. Skips aren't
    new (MLP linear skip; centering tricks) but those are gradient-injection/centering; I want the
    dumbest thing — a PURE IDENTITY, no weights/gate. y = F(x,{W_i})+x. Detail: final ReLU AFTER add
    (D3). Costs 0 params/0 compute -> HONEST controlled comparison.
11. Contrast Highway (D2): gate more general, contains identity, but can CLOSE and costs params ->
    constraint is the point.
12. Wall: dims don't match at transitions (halve map/double channels). Options A/B/C (D4) -> reason
    cost vs benefit -> B default; honest reading A,B,C all >> plain, slightly differ -> lesson.
13. Lay down architecture (D9), VGG-thin, 34-layer 3.6 GFLOPs; residual net = same graph + skips.
14. Wall 2: deep 3x3 at 512 is brutal -> bottleneck (D6); identity pays off AGAIN on bottleneck
    high-dim ends (D7) -> 152 lighter than VGG. Two choices reinforce.
15. Bottleneck stride placement (D8).
16. Training recipe inherited unchanged (D10); BN double role (D11).
17. Close the gradient story honestly: 1+dF/dx tempting but DEMOTE to bonus; load-bearing = conditioning
    (D12).
18. Cleanest test to validate (18/34 plain vs residual, option A so perfectly controlled; predict
    degradation reversed + small responses). No numbers.
19. Land on code: fills the scaffold; whole idea = one line `out += identity`. Causal-chain recap.
