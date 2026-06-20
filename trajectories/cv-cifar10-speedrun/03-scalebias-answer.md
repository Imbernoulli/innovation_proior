**Problem (from step 2).** After the initialization fixes (18 epochs / 6.8 s), one global learning rate is
applied uniformly to every parameter — implicitly assuming they all want to learn at the same rate. But the
trainable parameters are of two very different kinds: large redundant conv-weight tensors, and the
per-channel BatchNorm *biases* (the only trainable thing in each BatchNorm, since its scale is frozen at 1).
A bias rigidly shifts a whole channel relative to the GELU kink — a low-dimensional, high-leverage decision
whose gradient is a clean, concentrated signal that could support much larger steps. Under one shared lr the
biases are under-stepped and crawl to their operating points.

**Key idea.** Decouple the learning rate by parameter type: give the BatchNorm biases their own learning
rate, boosted by a fixed **64×** factor, while the conv weights stay on the base schedule. Scale only the
*learning rate*, not the weight decay (the decoupled-wd parametrization keeps the bias decay strength
unchanged), so faster steps aren't fought by harder decay.

**Why it works.** The BatchNorm biases are scalar, well-conditioned parameters that set each channel's
nonlinearity operating point; their gradient is a strong, low-variance signal that tolerates a large step,
so a 64× lr lets them snap to their right thresholds in a fraction of the epochs instead of crawling at the
conv weights' pace. The same 64× boost is the BN-bias scaling used by Page (2019) and hlb-CIFAR10. The risk
— a 64× lr usually diverges — is contained because it is applied only to these particular benign scalars.

**Change / code.** A two-group SGD optimizer: norm-bias group at `lr·64`, everything else at `lr`, with each
group's weight decay set to keep decay strength constant.

```python
# decoupled hyperparameters; bias_scaler = 64.0
lr_biases = lr * hyp['opt']['bias_scaler']

norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
param_configs = [dict(params=norm_biases,  lr=lr_biases, weight_decay=wd/lr_biases),
                 dict(params=other_params, lr=lr,        weight_decay=wd/lr)]
optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
```
