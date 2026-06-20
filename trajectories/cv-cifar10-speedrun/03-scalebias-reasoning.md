Whitening and Dirac init have together pulled me to 18 epochs and 6.8 seconds, and both were
*initialization* fixes — they conditioned the starting point. What's left to attack is the *trajectory*:
even with a clean start, the optimizer still spends 18 epochs walking the weights to their final values,
and I want to know whether all the parameters are moving at the rate they should be. The training uses a
single global learning rate schedule (the triangular ramp), applied uniformly to every parameter. That
implicitly assumes all parameters want to learn at roughly the same speed. Let me question that, because
the network has parameters of very different *kinds* mixed together.

Most of the trainable parameters are convolution weights — 3×3×C×C tensors, thousands to hundreds of
thousands of numbers each. But scattered among them are the **BatchNorm biases**: one scalar per
channel, and in this network the BatchNorm *weights* (scales) are frozen at 1, so the only thing each
BatchNorm contributes to learning is its per-channel bias — a single shift applied after normalization.
These biases are doing something quite different from the conv weights. A conv weight learns a spatial
feature; a BatchNorm bias learns *where to put the threshold* for a whole channel before the GELU — it
sets the operating point of the nonlinearity for that feature map. That is a low-dimensional, high-
leverage decision: moving one bias shifts an entire channel's activation distribution relative to the
GELU's kink.

Here is the asymmetry I think matters. A convolution weight's effect on the loss is diluted across its
many entries and across all spatial positions; the gradient for any single conv weight is a small piece
of a large, redundant ensemble, and the natural step size for it is modest. A BatchNorm bias, by
contrast, is a single number that rigidly translates an entire channel — its gradient is a clean,
concentrated signal about a global property of that channel, and it can tolerate (indeed wants) much
larger steps to reach its right value quickly. Under one shared learning rate, the biases are being
dragged along at the conv weights' pace: by the time the schedule has let the conv weights settle, the
biases — which could have snapped to their operating points in a fraction of the time — have been
crawling. The biases are *under*-stepped relative to what their clean gradient could support.

So the move is to decouple the learning rate by parameter type: give the BatchNorm biases their own,
much larger learning rate, while leaving the conv weights on the base schedule. The decoupled
hyperparameter parametrization already in place makes this clean — I can put the norm biases in their
own optimizer param-group and multiply their lr by a fixed scalar, without touching anything else. The
factor is large: empirically a **64×** boost on the bias learning rate. That sounds aggressive, but it
is reasonable precisely because these are scalar, well-conditioned parameters whose gradient is a
strong, low-variance signal — they are exactly the parameters that can absorb a big step without
diverging. And critically, I scale only the *learning rate*, not the weight decay: I want the biases to
move fast, but I do not want to also be decaying them 64× harder, which would just fight the larger
steps. The decoupled-wd parametrization keeps those two separate, so boosting the bias lr leaves the
bias weight-decay strength unchanged.

Concretely it is a per-group lr scaler. I split parameters into norm-biases and everything else, and the
norm-bias group gets `lr * bias_scaler` with `bias_scaler = 64`, while its weight decay is set to
`wd / lr_biases` so the *decay* matches the rest:

```python
lr_biases = lr * hyp['opt']['bias_scaler']          # bias_scaler = 64.0

norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
param_configs = [dict(params=norm_biases,  lr=lr_biases, weight_decay=wd/lr_biases),
                 dict(params=other_params, lr=lr,        weight_decay=wd/lr)]
optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
```

The prediction: if the BatchNorm biases really were the bottleneck — under-stepped scalars that set the
nonlinearity operating points — then letting them move 64× faster lets the network find its right
channel thresholds in far fewer epochs, and the whole training should compress again, with accuracy
held. I expect a solid step down from 18 epochs / 6.8 seconds. The size should be meaningful but not
another whitening-scale jump, because this is a refinement of the optimization rate on a small subset of
parameters, not a re-conditioning of the entire signal path. The risk is divergence: a 64× lr is the
kind of thing that, applied to the wrong parameters, blows up training — but applied to these
particular scalar, well-conditioned biases it should be stable, and the test is simply whether mean
accuracy stays at 94% while the epoch count drops. The change is the per-group bias lr scaler; code in
the answer.
