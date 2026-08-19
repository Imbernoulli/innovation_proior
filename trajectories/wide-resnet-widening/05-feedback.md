Measured results — mean/std normalization (CIFAR), no preprocessing beyond `[0,1]` scaling (SVHN),
moderate augmentation (flip/translation, CIFAR only).

## No-dropout comparison (CIFAR-10 / CIFAR-100 test error, %, median of 5 runs)

| model | depth-k | # params | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| original-ResNet | 110 | 1.7M | 6.43 | 25.16 |
| original-ResNet | 1202 | 10.2M | 7.93 | 27.82 |
| stoc-depth | 110 | 1.7M | 5.23 | 24.58 |
| stoc-depth | 1202 | 10.2M | 4.91 | - |
| pre-act-ResNet | 110 | 1.7M | 6.37 | - |
| pre-act-ResNet | 164 | 1.7M | 5.46 | 24.33 |
| pre-act-ResNet | 1001 | 10.2M | 4.92 (4.64 at batch 64) | 22.71 |
| WRN (ours) | 40-4 | 8.9M | 4.53 | 21.18 |
| WRN (ours) | 16-8 | 11.0M | 4.27 | 20.43 |
| WRN (ours) | 28-10 | 36.5M | 4.00 | 19.25 |

## Dropout ablation (mean/std, CIFAR numbers median of 5 runs)

| depth | k | dropout | CIFAR-10 | CIFAR-100 | SVHN |
|---|---|---|---|---|---|
| 16 | 4 | no | 5.02 | 24.03 | 1.85 |
| 16 | 4 | yes | 5.24 | 23.91 | 1.64 |
| 28 | 10 | no | 4.00 | 19.25 | - |
| 28 | 10 | yes | 3.89 | 18.85 | - |
| 52 | 1 | no | 6.43 | 29.89 | 2.08 |
| 52 | 1 | yes | 6.28 | 29.78 | 1.70 |

Notes:
- The 40-4 / 16-8 / 28-10 no-dropout rows above are the same three configurations from the ZCA grid,
  now re-measured under mean/std: 40-4 moves 4.97->4.53 / 22.89->21.18; 16-8 moves 4.81->4.27 /
  22.07->20.43; 28-10 moves 4.17->4.00 / 20.50->19.25. All three improve under mean/std versus their
  ZCA counterparts on both datasets.
- 28-10 with dropout under mean/std moves to 3.89/18.85, versus 4.39/20.0 for the same architecture
  and dropout probability measured under ZCA.
- Best single-run headline numbers reached elsewhere in this design family (not part of the grid or
  dropout-ablation configurations above): CIFAR-10 WRN-40-10 with dropout, 3.8%; CIFAR-100 WRN-40-10
  with dropout, 18.3%; SVHN WRN-16-8 with dropout, 1.54%.
