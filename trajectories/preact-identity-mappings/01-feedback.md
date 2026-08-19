Measured results — shortcut-path sweep, CIFAR-10 test set, ResNet-110, median of 5 runs,
`f` = ReLU after addition unchanged in every variant.

## CIFAR-10 (ResNet-110)

| variant | on shortcut | on F | error (%) |
|---|---|---|---|
| original (identity shortcut) | 1 | 1 | 6.61 |
| constant scaling | 0 | 1 | fail (this is a plain net) |
| constant scaling | 0.5 | 1 | fail |
| constant scaling (frozen gating) | 0.5 | 0.5 | 12.35 |
| exclusive gating, b_g init 0 to -5 | 1-g(x) | g(x) | fail |
| exclusive gating, b_g init -6 | 1-g(x) | g(x) | 8.70 |
| exclusive gating, b_g init -7 | 1-g(x) | g(x) | 9.81 |
| shortcut-only gating, b_g init 0 | 1-g(x) | 1 | 12.86 |
| shortcut-only gating, b_g init -6 | 1-g(x) | 1 | 6.91 |
| 1x1 convolutional shortcut (every unit) | 1x1 conv | 1 | 12.22 |
| dropout shortcut (p=0.5) | dropout | 1 | fail |

Notes: "fail" = test error above 20%, i.e. the network did not converge to a useful solution.
For every gated variant the reported number is from a hyperparameter search over the gate-bias
initialization `b_g` in the range 0 to -10 (decrement -1), selected on the training set by
cross-validation; more negative `b_g` (closer to 1-g(x) ~ 1 at initialization) tracks closer to
the baseline in every gated case tested. Training-loss curves (not tabulated here) show higher
training error than the original ResNet-110 for the constant-scaling, exclusive-gating,
shortcut-only-gating (b_g=0), and 1x1-conv variants — the degradation is on the training set,
not only the test set. Both gated variants and the 1x1-conv shortcut can express the identity
shortcut inside their solution space (their parameters could in principle be optimized to
recover it).
