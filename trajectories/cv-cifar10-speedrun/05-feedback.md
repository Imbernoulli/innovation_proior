Measured result — multi-crop test-time augmentation (six weighted views), arXiv:2404.00498 §3.5.
Metric: A100-seconds to reach 94% mean accuracy, **lower is better**.

| configuration | epochs to 94% | A100-seconds |
|---|---|---|
| + lookahead (§3.4) | 12.0 | 4.6 |
| + multi-crop TTA (§3.5) | **10.8** | **4.2** |

"With this feature added, training reaches 94% in 10.8 epochs taking 4.2 A100-seconds." Multi-crop is the
one feature whose epochs-to-94% effect is *not* additive between the two measurement directions (Fig. 4):
adding it to the whitened baseline saves ~3 epochs, but removing it from the final airbench94 costs only
~1.3 epochs — the only non-additive interaction the paper reports. Without any TTA at all, the three
airbench methods attain 93.2%, 94.4%, and 95.6% mean accuracy respectively.
