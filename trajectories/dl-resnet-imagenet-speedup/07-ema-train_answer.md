Squeeze-and-Excitation refilled the accuracy tank, but stacking these regularizers and augmentations has a side effect I have been ignoring that now matters: my validation metric is getting *noisy*. Between adjacent checkpoints the accuracy bounces around, and for a speed-chasing recipe that is a real problem, because I am trying to shorten the schedule to the minimum that still clears 76.6%. If the final-checkpoint accuracy has a built-in $\pm$ of half a point or more from step-to-step jitter, I cannot tell whether I have actually cleared the bar or just landed on a lucky step — I would have to leave a safety margin (train longer than necessary) to be confident, and that margin is wasted time. Where does the jitter come from? SGD with a finite learning rate does not converge to a point; it converges to a *distribution* bouncing around the minimum. Even late in training each step nudges the weights by a noisy gradient, so the iterate $\theta_t$ rattles around within a basin rather than sitting still — and the blur, resizing, and augmentations I have stacked only add to that stochasticity. Consecutive checkpoints are therefore different *samples* from the late-training distribution of weights, and their validation accuracies differ just because the weights differ by a step's worth of noise. The single final iterate is one draw, and I am betting my "did I clear the bar" decision on one draw.

The method I propose is **EMA**, an exponential moving average of the weights. The classic fix for "my estimator is a noisy single sample" is to average: if the late iterates are bouncing around a good region, the average of those iterates sits closer to the center of the basin than any single one, and it is lower-variance. Near a minimum the loss surface is roughly bowl-shaped, so averaging weights scattered around the bowl lands nearer the bottom and smooths out the per-step jitter. But a plain running mean over all of training is wrong — early weights are garbage and would poison the average — so I want a *recency-weighted* average that mostly reflects the recent, good iterates. The natural form is an exponential moving average: maintain a shadow copy $\theta_{\text{EMA}}$ and after each step pull it a small fraction toward the current $\theta$,

$$\theta_{\text{EMA}} \leftarrow s\,\theta_{\text{EMA}} + (1 - s)\,\theta,$$

with smoothing coefficient $s$ close to 1. Information from $k$ steps ago survives with weight $s^k$, so old (bad) weights fade exponentially and recent (good) ones dominate — exactly the recency-weighted average I want.

The one knob is the timescale. The interpretable form of it is the half-life — how many steps until a contribution decays to half — and it should be much longer than a single step but much shorter than all of training, so the average spans the recent good region without reaching back to the noisy early phase. A half-life on the order of a thousand batches is a sensible default (`half_life='1000ba'`), and $s$ is then derived from it. I do not strictly need to update every single step either: I can update every few steps (an `update_interval`) to save compute, and as long as the half-life is much larger than the interval, the generalization effect is essentially unchanged. So the cost is small — one weighted-add over the parameters, optionally only every several steps — plus extra device memory for the second copy of the weights and buffers, which for ResNet-50 is small relative to activations and optimizer state.

Two correctness points are load-bearing. First, *which* tensors to average: it is not just the learnable parameters. The batchnorm running statistics — the buffers — also evolve during training and matter at evaluation, so the EMA must track the buffers too, or the averaged model evaluates with mismatched normalization stats. The update therefore runs over both named parameters and named buffers. Second, *which* model I evaluate: the whole point is to evaluate the *averaged* weights, not the live training weights. At evaluation time I swap in $\theta_{\text{EMA}}$. This is a genuine trap — forget it, evaluate the training model, and I get none of the benefit and might conclude EMA hurt. I also honor one composition rule now: model-averaging methods do not stack, so running EMA *and* stochastic weight averaging at once would fight itself; I pick one, and EMA is it. What I expect is the reliable variance reduction I came for — smoother, less-noisy validation metrics that let me trust the final-checkpoint accuracy and shave the safety margin off the schedule — and sometimes a bonus genuine generalization improvement, because the averaged weights sit in a flatter, more central part of the basin than the last iterate. The core is the moving-average update, applied in place over both parameters and buffers.

```python
def compute_ema(model, ema_model, smoothing: float = 0.99) -> None:
    model_context_manager = get_model_context_manager(model)
    with model_context_manager:
        with torch.no_grad():
            if isinstance(ema_model, torch.nn.Module):
                ema_params = ema_model.state_dict()
                # iterate over BOTH parameters and buffers (buffers = batchnorm stats)
                for name, param in itertools.chain(model.named_parameters(),
                                                   model.named_buffers()):
                    if name in ema_params:
                        ema_params[name].copy_(
                            ema_params[name] * smoothing + param.data * (1. - smoothing))
            ...

# Usage: ema = EMA(half_life='1000ba'); evaluate with the averaged weights:
#   model = ema.ema_model
```
