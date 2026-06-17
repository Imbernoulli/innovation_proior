# x0 prediction (data/sample parameterization)

Train the denoising network to emit the clean image directly, and pass that raw output to the
sampler as the clean-image estimate.

For the local scaffold, with

```text
x_t = a_t * x_0 + b_t * eps
a_t = schedule["sqrt_alpha"][t]
b_t = schedule["sqrt_one_minus_alpha"][t],
```

the coupled target and recovery are:

```python
def compute_training_target(x_0, noise, timesteps, schedule):
    return x_0


def predict_x0(model_output, x_t, timesteps, schedule):
    return model_output
```

The implied DDIM noise direction, computed by the existing sampler, is then

```text
eps_hat = (x_t - a_t * x0_hat) / b_t.
```

This is algebraically consistent because the target is already the quantity `predict_x0` must
recover. There is no `1 / a_t` conversion from the raw model output into `x0_hat`, so the
clean-image estimate does not inherit epsilon prediction's low-SNR amplification:

```text
epsilon prediction: x0_hat = (x_t - b_t * eps_hat) / a_t
output error in eps -> clean-image error scaled by b_t / a_t.
```

Loss weighting is separate from this parameterization. In the Google Research reference code,
`mean_type == "x"` maps `model_output` directly to `model_x`, and the CIFAR configs pair that with
`mean_loss_weight_type == "snr_trunc"` (`max(SNR, 1)` weighting in x-space). If this scaffold has
only plain `F.mse_loss(output, target)`, the functions above implement the x-output
parameterization; a fully faithful CIFAR reproduction also adds the truncated-SNR loss weight
outside these two functions.
