# Ladder mining notes — wide-resnet-widening

Source of truth: `methods/wide-resnet/refs/primary/arxiv_source/wide_networks.tex` (final v4 arXiv
source, matches `refs/primary/arxiv-1605.07146.txt` header "arXiv:1605.07146v4 [cs.CV] 14 Jun 2017"),
plus `methods/wide-resnet/refs/official_code/git-history-szagoruyko-wide-residual-networks.txt` for
the superseded v1 README numbers, plus `refs/official_code/github-issues-all-threads.txt` issue #61
independently confirming the v1-vs-v2 renumbering.

All line numbers below are into `wide_networks.tex` unless noted.

## Rung 1 — block-type search B(M) (lines 160–176, 200–219, 242–244)
- Deepening factor l fixed at 2 for this study (l fixed at 2 = two convs/block); widening factor k=2
  fixed; only the internal kernel-size pattern M of the block varies (block structure B(M), M a list
  of kernel sizes: B(3,3), B(3,1,3), B(1,3,1), B(1,3), B(3,1), B(3,1,1)).
- Table \ref{table:blocks} (lines 204–218), caption line 217: "Test error (%, median over 5 runs) on
  CIFAR-10 of residual networks with k=2 and different block types. Time column measures one training
  epoch."
  block type | depth | #params | time,s | CIFAR-10
  B(1,3,1) | 40 | 1.4M | 85.8 | 6.06
  B(3,1)   | 40 | 1.2M | 67.5 | 5.78
  B(1,3)   | 40 | 1.3M | 72.2 | 6.42
  B(3,1,1) | 40 | 1.3M | 82.2 | 5.86
  B(3,3)   | 28 | 1.5M | 67.5 | 5.73
  B(3,1,3) | 22 | 1.1M | 59.9 | 5.78
- Prose conclusion (line 242): "Block B(3,3) turned out to be the best by a little margin, and B(3,1)
  with B(3,1,3) are very close to B(3,3) in accuracy having less parameters and less layers."

## Rung 2 — deepening factor l (lines 178–181, 200–239, 247–249)
- Table \ref{table:blocks_l} (lines 221–236), caption line 234: "Test error (%, median over 5 runs)
  on CIFAR-10 of WRN-40-2 (2.2M) with various l." Same param count (2.2e6) and same total conv-layer
  count held fixed across l by varying block count d.
  l | CIFAR-10
  1 | 6.69
  2 | 5.43
  3 | 5.65
  4 | 5.93
- Prose (line 249): "B(3,3) turned out to be optimal in terms of number of convolutions per block."

## Rung 3 — width-vs-depth grid, ZCA preprocessing (lines 252–276)
- Table \ref{table:width} (lines 258–274), caption line 274: "Test error (%) of various wide networks
  on CIFAR-10 and CIFAR-100 (ZCA preprocessing)."
  depth | k | #params | CIFAR-10 | CIFAR-100
  40 | 1  | 0.6M  | 6.85 | 30.89
  40 | 2  | 2.2M  | 5.33 | 26.04
  40 | 4  | 8.9M  | 4.97 | 22.89
  40 | 8  | 35.7M | 4.66 | -
  28 | 10 | 36.5M | 4.17 | 20.50
  28 | 12 | 52.5M | 4.33 | 20.43
  22 | 8  | 17.2M | 4.38 | 21.22
  22 | 10 | 26.8M | 4.44 | 20.75
  16 | 8  | 11.0M | 4.81 | 22.07
  16 | 10 | 17.1M | 4.56 | 21.59
- These are the same ZCA numbers github issue #61 (line ~2688–2699 of the issues file) cites as
  "the first version" (arxiv v1): "4.00% vs 4.17%, 4.27% vs 4.81%, 4.53% vs 4.97%" — i.e. issue #61's
  "first version" numbers 4.17 / 4.81 / 4.97 are exactly this table's 28-10 / 16-8 / 40-4 rows.

## Rung 4 — dropout arm under ZCA preprocessing (superseded, v1)
- Source: `refs/official_code/git-history-szagoruyko-wide-residual-networks.txt` lines ~940–958, a
  diff of README.md at commit 5e7ed6c ("v2", 2016-11-29). The `-` (removed) lines are the *old* (v1,
  ZCA) README table, which the v2 commit replaced:
  Method | CIFAR-10 | CIFAR-100
  pre-ResNet-164   | 5.46 | 24.33
  pre-ResNet-1001  | 4.92 | 22.71
  WRN-28-10        | 4.17 | 20.5
  WRN-28-10-dropout| 4.39 | 20.0
  (line 957: "-WRN-28-10-dropout| 4.39 | **20.0**")
- Dropout probability and placement per the main text (lines 190, 336): inserted between the block's
  two convolutions, after the second BN/ReLU; cross-validated to p=0.3 on CIFAR.
- This table predates the meanstd switch — WRN-28-10 baseline here is 4.17 (matches rung-3 table:width
  28-10 row exactly), so this dropout arm was measured on the ZCA-preprocessed rung-3 winner.

## Rung 5 — meanstd renormalization + final dropout numbers (published, v2/v4) (lines 196–197,
256–364, 420–435)
- Motivation for the preprocessing switch, stated as protocol rationale in the paper itself (lines
  196–197): "for some CIFAR experiments we instead use simple mean/std normalization such that we can
  directly compare with [pre-act-ResNet] and other ResNet related works that make use of this type of
  preprocessing." Independently confirmed as the actual reason in the git-history README diff (line
  958): "on CIFAR meanstd preprocessing (as in fb.resnet.torch) gives better results than ZCA
  whitening."
- Table \ref{table:final} (lines 284–313), caption: "Test error of different methods on CIFAR-10 and
  CIFAR-100 with moderate data augmentation (flip/translation) and mean/std normalization. We don't
  use dropout for these results... median over 5 runs."
  model | depth-k | #params | CIFAR-10 | CIFAR-100
  original-ResNet | 110  | 1.7M  | 6.43 | 25.16
  original-ResNet | 1202 | 10.2M | 7.93 | 27.82
  stoc-depth      | 110  | 1.7M  | 5.23 | 24.58
  stoc-depth      | 1202 | 10.2M | 4.91 | -
  pre-act-ResNet  | 110  | 1.7M  | 6.37 | -
  pre-act-ResNet  | 164  | 1.7M  | 5.46 | 24.33
  pre-act-ResNet  | 1001 | 10.2M | 4.92 (4.64 at batch64) | 22.71
  WRN (ours)      | 40-4 | 8.9M  | 4.53 | 21.18
  WRN (ours)      | 16-8 | 11.0M | 4.27 | 20.43
  WRN (ours)      | 28-10| 36.5M | 4.00 | 19.25
  Note line 278: pre-act-ResNet-1001's parenthetical 4.64% used batch size 64; all other numbers in
  this table (including WRN's) use batch 128, so 4.92% is the apples-to-apples baseline figure.
- Table \ref{table:dropout} (lines 349–364), caption: "Effect of dropout in residual block.
  (mean/std preprocessing, CIFAR numbers are based on median of 5 runs)"
  depth | k | dropout | CIFAR-10 | CIFAR-100 | SVHN
  16 | 4  | no  | 5.02 | 24.03 | 1.85
  16 | 4  | yes | 5.24 | 23.91 | 1.64
  28 | 10 | no  | 4.00 | 19.25 | -
  28 | 10 | yes | 3.89 | 18.85 | -
  52 | 1  | no  | 6.43 | 29.89 | 2.08
  52 | 1  | yes | 6.28 | 29.78 | 1.70
- Prose (line 338): "Dropout decreases test error on CIFAR-10 and CIFAR-100 by 0.11% and 0.4%
  correspondingly ... with WRN-28-10 ... There is only a slight drop in accuracy with WRN-16-4 on
  CIFAR-10."
- Table \ref{table:overall} (lines 420–435), best single-run headline numbers: CIFAR-10 WRN-40-10
  (dropout) 3.8%; CIFAR-100 WRN-40-10 (dropout) 18.3%; SVHN WRN-16-8 (dropout) 1.54%.
- Endpoint method embodied in code: `models/wide-resnet.lua` in
  `methods/wide-resnet/code/wide-residual-networks/` — final published `wide_basic` block builder
  (BN-ReLU-conv3x3-BN-ReLU-dropout-conv3x3, identity/1x1-projection shortcut, depth=6n+4,
  widths 16/16k/32k/64k) and `scripts/train_cifar.sh` / `scripts/train_svhn.sh` for the fixed
  hyperparameters (SGD Nesterov, lr 0.1, momentum 0.9, wd 5e-4, batch 128, CIFAR epoch_step
  60/120/160 over 200 epochs; SVHN lr 0.01, epoch_step 80/120 over 160 epochs, dropout 0.4, no aug).

## Pre-dating baselines (for 00-initial-context.md, not a rung)
- pre-act-ResNet (basic block, identity-mapping order) 110/1.7M: 6.37 CIFAR-10 (table:final);
  164/1.7M: 5.46/24.33; 1001/10.2M: 4.92(4.64)/22.71 — the strongest thin reference point.
- original ResNet 110/1.7M: 6.43/25.16; 1202/10.2M: 7.93/27.82 (degrades past ~1000 layers).
- stochastic depth 110/1.7M: 5.23/24.58; 1202/10.2M: 4.91/- (bypasses whole blocks at random,
  competitive at extreme depth).
- Zagoruyko's own prior (pre-WRN) self-account: VGG + BN + Dropout together on CIFAR-10 (Torch,
  2015-07) reaches 92.44% accuracy; removing either BN or Dropout drops it to 91.4% — grounds "BN
  does not fully substitute for dropout" as a pre-existing, already-measured fact.
  (`refs/self_accounts/torch-blog-2015-07-30-cifar-zagoruyko.txt`)
- identity-mappings ancestor: dropout at ratio 0.5 applied on the *shortcut* of a 110-layer residual
  network fails to converge to a good solution (>20% CIFAR-10 test error vs 6.61% baseline);
  generalized claim: multiplicative manipulations (scaling, gating, 1x1 conv, dropout) on the
  shortcut hamper information propagation. (`refs/ancestors/identity-mappings-1603.05027.txt`)
