# Ladder mining notes — pre-activation / identity mappings

Source file for all numbers: `methods/pre-activation/src/resnet_plus_arxiv.tex`
(the primary self-account, "Identity Mappings in Deep Residual Networks").
Baseline ResNet-110 = 6.61% is itself a pre-existing fact (He et al. 2016a,
`methods/pre-activation/refs/resnet_v1/residual_v1_arxiv_release.tex`), already
recorded as background in `methods/pre-activation/results/context.md`.

Protocol note (applies to every rung below): median of 5 runs per architecture
on CIFAR (resnet_plus_arxiv.tex:198, :368).

## Rung 1 — shortcut-path manipulations (grouped table, all variants fail baseline)

Table `tab:shortcuts`, resnet_plus_arxiv.tex:159-193. Motivation text: :212-232.
All rows tested on CIFAR-10, ResNet-110, f=ReLU (post-activation) unchanged.

| variant | on shortcut | on F | error % | line |
|---|---|---|---|---|
| original (baseline) | 1 | 1 | 6.61 | 167 |
| constant scaling | 0 | 1 | fail (plain net) | 172 |
| constant scaling | 0.5 | 1 | fail | 173 |
| constant scaling (frozen gating) | 0.5 | 0.5 | 12.35 | 173 |
| exclusive gating, b_g=0..-5 | 1-g(x) | g(x) | fail | 177 |
| exclusive gating, b_g=-6 | 1-g(x) | g(x) | 8.70 | 178 |
| exclusive gating, b_g=-7 | 1-g(x) | g(x) | 9.81 | 179 |
| shortcut-only gating, b_g=0 | 1-g(x) | 1 | 12.86 | 183 |
| shortcut-only gating, b_g=-6 | 1-g(x) | 1 | 6.91 | 184 |
| 1x1 conv shortcut | 1x1 conv | 1 | 12.22 | 187 |
| dropout shortcut (p=0.5) | dropout | 1 | fail | 190 |

Narrative lines used: 212 (constant scaling), 217-220 (exclusive gating,
init sensitivity), 222-224 (shortcut-only gating), 226 (1x1 conv, contrast
with He2016 option C on 34-layer/16-unit net), 228 (dropout), 232+234
(discussion: multiplicative ops hurt regardless of whether they contain
identity in their solution space -> optimization, not representational,
issue). lambda-product algebra: eq:grad1 derivation, lines ~140-148.

## Rung 2 — BN after addition (deliberate wrong-direction f test)

Table `tab:activations`, resnet_plus_arxiv.tex:246-259. Baselines for this
table: ResNet-110 6.61 (line 252), ResNet-164 5.93 (line 252, also stated in
prose at line 268 "baseline ResNet-164 has a competitive result of 5.93%").
BN-after-addition row: line 254, error 8.17 (ResNet-110) / 6.50 (ResNet-164).
Narrative: line 286 ("we go the opposite way... BN layer alters the signal
that passes through the shortcut... difficulties reducing training loss at
the beginning").

## Rung 3 — ReLU before addition

Table row: line 256, error 7.84 (ResNet-110) / 6.14 (ResNet-164). Narrative:
line 288 ("non-negative output from F... forward propagated signal is
monotonically increasing... may impact representational ability").

## Rung 4 — ReLU-only pre-activation

Table row: line 257, error 6.71 (ResNet-110) / 5.91 (ResNet-164). Narrative:
line 341 ("ReLU-only pre-activation performs very similar to the baseline...
this ReLU layer is not used in conjunction with a BN layer, and may not
enjoy the benefits of BN").

## Rung 5 — full pre-activation (BN + ReLU both moved before the weight layer)

Table row: line 258, error 6.37 (ResNet-110) / 5.46 (ResNet-164), bold/best
in the table. Also table `tab:preact` line 324: ResNet-110 baseline-unit 6.61
vs pre-activation-unit 6.37; line 328 (implicit, ResNet-164 row omitted from
tab:preact but present in tab:activations). Asymmetric-activation /
pre-activation equivalence derivation: lines 291-300 (eq:additive3).
Two-payoff narrative (ease of optimization; regularization): lines 335, 349,
and surrounding "Ease of optimization" / regularization discussion.

## Rung 6 — full pre-activation pushed to extreme depth (ResNet-1001)

Table `tab:preact`, resnet_plus_arxiv.tex:318-331.

| dataset | network | baseline unit | pre-activation unit | line |
|---|---|---|---|---|
| CIFAR-10 | ResNet-110 (1-layer skip) | 9.90 | 8.91 | 323 |
| CIFAR-10 | ResNet-110 | 6.61 | 6.37 | 324 |
| CIFAR-10 | ResNet-164 | 5.93 | 5.46 | 325 |
| CIFAR-10 | ResNet-1001 | 7.61 | 4.92 | 326 |
| CIFAR-100 | ResNet-164 | 25.16 | 24.33 | 329 |
| CIFAR-100 | ResNet-1001 | 27.82 | 22.71 | 330 |

1001-layer bottleneck architecture description: line 337 ("333 Residual
Units, 111 on each feature map size"). Ease-of-optimization narrative
(training loss reduced very slowly at the beginning for the baseline unit,
quickly for pre-activation): line 349. Params: ResNet-1001 10.2M (line 388,
also table `tab:cifar`). Median-of-5 protocol restated at line 368
(mean+-std for ResNet-1001 pre-act: 4.89+-0.14, line 388 — reported as a
footnote, headline number stays the rounded median 4.92 used everywhere
else in the table to avoid conflating median and mean).

Endpoint = full pre-activation residual unit (BN->ReLU->Conv, twice, clean
identity add, shared pre-activation feeding the 1x1 projection shortcut on
shape-changing units), demonstrated on ResNet-1001, the published method.

## Rung count and merge decisions

6 rungs (within the 3-8 target). Rung 1 merges 5 documented shortcut
variants (10 measured configurations across constant scaling / exclusive
gating / shortcut-only gating / 1x1 conv / dropout) into one rung because
the source presents them as a single grouped ablation (`tab:shortcuts`) that
answers one question ("can the shortcut carry more than the identity?") with
one uniform predicted-and-confirmed mechanism (the lambda-product argument).
Rungs 2-5 (activation placement) are kept as four separate rungs rather than
merged, because each is a distinct architectural hypothesis with its own
falsifiable prediction that the previous rung's result motivates (BN-after
calibrates the "any op on the merge path hurts" claim; ReLU-before tests the
naive identity-f fix and exposes the F>=0 problem; ReLU-only pre-activation
isolates the path fix from the normalization fix; full pre-activation adds
the normalization fix and is the first rung to beat baseline outright).
Rung 6 promotes the winning rung-5 design to the extreme-depth regime the
research question opened with (1202-layer generalizing worse than 110-layer;
1000-layer training loss falling "painfully slowly" at the start), closing
the loop with ResNet-1001.
