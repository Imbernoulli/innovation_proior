# Ladder mining notes — convnext-modernization

Source file for all numbers: `methods/convnext/refs/primary/source/main.tex`
(arXiv 2201.03545 source, "A ConvNet for the 2020s"). Line numbers below refer
to that file. Cross-checked against the rounded numbers stated in the roadmap
prose (Section "Modernizing a ConvNet: a Roadmap", lines 150-260) and the
precise per-step ablation table `\label{tab:modernizing-t}` (lines 807-830).
Final block/stem/downsampling code cross-checked against
`methods/convnext/code/ConvNeXt/models/convnext.py`.

| Rung | Variant | top1_acc | GFLOPs | Source line(s) |
|---|---|---|---|---|
| 0 (given) | ResNet-50, torchvision, original 90-ep recipe | 76.13 | 4.09 | main.tex:686, 809 |
| 1 | + modern (DeiT/Swin-style) training recipe, 300ep, AdamW, aug/reg bundle, LayerScale, no EMA | 78.82 ± 0.07 | 4.09 | main.tex:164 (prose "78.8"), 811, 557 (EMA disabled for modernizing-track), 292/570-589 (recipe hyperparameters table `tab:train_detail`) |
| 2a | + stage ratio (3,4,6,3) -> (3,3,9,3) | 79.36 ± 0.07 | 4.53 | main.tex:170 (prose "79.4"), 813 |
| 2b | + patchify stem (4x4 s4 conv replaces 7x7 s2 conv + maxpool) | 79.51 ± 0.18 | 4.42 | main.tex:176 (prose "79.5"), 813 |
| 3a | + depthwise 3x3 conv (groups=channels), width still 64 | 78.28 ± 0.08 | 2.35 | main.tex:184 (prose describes FLOPs/acc drop before width fix), 814 |
| 3b | + width 64 -> 96 (match Swin-T channels) | 80.50 ± 0.02 | 5.27 | main.tex:184 (prose "80.5%"), 815 |
| 4 | + inverted bottleneck (expand 4x -> depthwise -> project), still kernel 3x3 in old position | 80.64 ± 0.03 | 4.64 | main.tex:191-193 (prose "80.5% to 80.6%"), 816 |
| 5a | + move depthwise conv above the 1x1 expansion (still kernel 3x3) | 79.92 ± 0.08 | 4.07 | main.tex:212 (prose "79.9%"), 817 |
| 5b | kernel size sweep: 3 / 5 / 7 / 9 / 11 | 79.92 / 80.35 / 80.57 / 80.57 / 80.47 | 4.07 / 4.10 / 4.15 / 4.21 / 4.29 | main.tex:215 (prose: increases 79.9->80.6, saturates at 7x7), 817-821. Adopt k=7. |
| 6a | + ReLU -> GELU | 80.62 ± 0.14 | 4.15 | main.tex:224 (prose "stays unchanged (80.6%)"), 822 |
| 6b | + single activation per block (only between the two 1x1 layers) | 81.27 ± 0.06 | 4.15 | main.tex:228 (prose "+0.7% to 81.3%"), 823 |
| 6c | + single norm per block (drop to one BN before the 1x1 layers) | 81.41 ± 0.09 | 4.15 | main.tex:243 (prose "boosts... to 81.4%"), 824 |
| 6d | + BN -> LN | 81.47 ± 0.09 | 4.46 | main.tex:249 (prose "81.5%"), 825 |
| 6e | + separate 2x2 s2 downsampling layers + stabilizing LN (before each downsample, after stem, after final pooling) = final ConvNeXt-T | 81.97 ± 0.06 | 4.49 | main.tex:257 (prose "82.0%"), 827 |
| ref | Swin-T (comparison baseline, not a rung) | 81.30 | 4.50 | main.tex:829, 320 |

Rung grouping used for the trajectory (6 rungs; matches the documented
"macro design / ResNeXt-ify / inverted bottleneck / large kernel / micro
design" section structure of the source):
1. modern_recipe = row 1
2. macro_design = rows 2a+2b (bundled proposal, two measured sub-points)
3. resnextify = rows 3a+3b (bundled proposal, two measured sub-points)
4. inverted_bottleneck = row 4
5. large_kernel = rows 5a+5b (bundled proposal, six measured sub-points, adopt k=7)
6. micro_design = rows 6a-6e (bundled proposal, five sequential measured sub-points)

Training recipe fixed from rung 1 onward (main.tex:570-589, `tab:train_detail`,
applies to the modernizing-track runs per line 557 except EMA disabled):
AdamW, base lr 4e-3, weight decay 0.05, batch size 4096, 300 epochs, cosine
decay, 20-epoch linear warmup, RandAugment (9, 0.5), Mixup 0.8, CutMix 1.0,
Random Erasing 0.25, Label Smoothing 0.1, LayerScale init 1e-6, no EMA.

Gap: no author self-account beyond the paper/code was found
(methods/convnext/notes/source_matrix.md). Intermediate rung code (rungs
1-5, i.e. everything before the final architecture) is written to match the
documented block diagrams and FLOPs/accuracy deltas; only the final rung's
code is checked verbatim against the released implementation.
