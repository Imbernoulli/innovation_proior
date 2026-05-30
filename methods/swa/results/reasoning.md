Let me start from a discrepancy that nags at me every time I train a network: SGD drives the train loss down to a clean minimizer, but the test error at that exact point is not the best I could get. Why would the train-loss minimizer be the wrong place to stop? Because the train-loss surface and the test-error surface are not the same surface — they're qualitatively similar but *shifted* relative to each other. So the point that bottoms out the train loss is sitting off to one side of the test-error valley, not at its bottom. If that's the geometry, then driving harder toward the train minimum can actually hurt, and the thing I want isn't the deepest train point but a *central, robust* point that stays good under the shift.

That immediately tells me what kind of point to want: a *broad, flat* one. If I'm at the bottom of a wide flat basin, a small horizontal shift of the surface barely changes my loss; if I'm at a sharp narrow minimum, the same shift sends the error up steeply. Keskar et al. (2017) saw exactly this — small-batch SGD finds broader optima that generalize better than the sharp ones large-batch methods fall into, and the sharp ones are flat in most directions but brutally steep in a few. So width correlates with generalization, through this train-vs-test shift. The goal sharpens: end training at a *wide, central* point of the high-performing region, cheaply.

How do I find such a central point? Here's the key realization about what SGD is actually doing. With a non-vanishing step size, SGD doesn't converge to a point and sit there — it keeps wandering through the region of good networks. Mandt et al. (2017) make this precise: SGD with a *constant* learning rate behaves like drawing samples from a Gaussian centered at the loss minimum, with covariance set by the learning rate. And a curious fact about high-dimensional Gaussians is that their mass concentrates near the *surface* of a sphere, not the center — so every individual constant-LR iterate is out on the *periphery* of the good region, never at its dense center. The center, the robust point I want, is precisely where SGD does *not* go on any single step. But it's the *average* of those peripheral samples. Averaging the iterates would move me from the surface of the sphere to its interior — to the central, higher-density point.

So the move is: don't ship the last SGD iterate; ship the *average* of the iterates SGD visits while it's exploring. This isn't a new idea in the abstract — Ruppert (1988) and Polyak & Juditsky (1992) proved that averaging SGD iterates accelerates convergence in convex optimization. But that was with a *decaying* step size, and there's the rub: practitioners who tried a running average of weights for nets did it alongside a decaying learning rate, and it barely helped — it just smoothed the trajectory. Let me see why that fails, because the failure tells me what's essential. With a decaying learning rate, the iterates *collapse* toward a single point as the rate shrinks; they stop exploring. Averaging a cloud of points that are all piling onto the same spot gives you back that spot — no diversity, no benefit. The averaging only buys something if the points being averaged are *spread out* across the good region. Exploration is the precondition.

That means I must *keep the learning rate high* — constant, or cyclically returning to a high value — so SGD keeps taking large steps and visiting genuinely different points on the periphery, instead of settling. A constant learning rate `α(i) = α_1` gives maximal exploration. A cyclical schedule gives a middle ground: linearly decrease the rate from `α_1` to `α_2` within each cycle of length `c`, `α(i) = (1 - t(i)) α_1 + t(i) α_2` with `t(i) = (1/c)(mod(i-1, c) + 1)`, then jump straight back up to `α_1` to start the next cycle. I make the jump *discontinuous* — straight from the minimum back to the maximum — rather than ramping the rate up and down smoothly, because for my purpose exploration matters more than the accuracy of any single proposal; I want the schedule to fling SGD to a new place, then spend a few steps fine-tuning it with the decreasing rate before flinging it again. With the cyclical schedule I capture a model `w_i` at each cycle's *minimum* learning rate (the fine-tuned point); with a constant rate I just capture one per epoch. Either way I collect a set of spread-out, individually-decent points to average.

Now, *what* do I average — the weights, or the predictions? There's a strong existing answer for predictions: Fast Geometric Ensembling (Garipov et al. 2018) runs exactly this kind of cyclical-LR exploration to generate diverse nearby networks and then *ensembles their predictions*, getting ensemble-quality accuracy in single-model training time. But an ensemble means storing and running `n` networks at test time — the cost I'm trying to avoid. I want one model. So can I average the *weights* and get the same benefit as averaging the *predictions*? That seems too good — averaging parameters of different networks usually destroys them — but these networks aren't different; they're nearby points from the same exploration, close together by construction. Let me check whether weight-averaging and prediction-averaging actually agree when the points are close.

Let `f(w)` be the network's prediction (take it scalar, say one class probability, twice differentiable in `w`). I have points `w_i` from the exploration, with average `w_SWA = (1/n) Σ_i w_i`. Write each as an offset from the average, `Δ_i = w_i - w_SWA`; then by construction `Σ_i Δ_i = Σ_i (w_i - w_SWA) = n·w_SWA - n·w_SWA = 0`. Ensembling means averaging the predictions, `f̄ = (1/n) Σ_i f(w_i)`. Linearize each prediction around the average point:
`f(w_j) = f(w_SWA) + ⟨∇f(w_SWA), Δ_j⟩ + O(‖Δ_j‖²)`.
Average over `j` and subtract `f(w_SWA)`:
`f̄ − f(w_SWA) = (1/n) Σ_i [ ⟨∇f(w_SWA), Δ_i⟩ + O(‖Δ_i‖²) ] = ⟨∇f(w_SWA), (1/n) Σ_i Δ_i⟩ + O(Δ²)`.
But `(1/n) Σ_i Δ_i = 0`, so the entire first-order term *vanishes*, and
`f̄ − f(w_SWA) = O(Δ²)`.
The prediction of the *single averaged-weight network* `f(w_SWA)` and the *ensemble* prediction `f̄` differ only at *second* order in how far apart the points are. Compare that to how much the proposals themselves disagree: `f(w_i) − f(w_j) = ⟨∇f(w_SWA), Δ_i − Δ_j⟩ + O(Δ²)`, which is *first* order. So the diversity I'm trying to exploit is first-order, while the gap between averaging-weights and averaging-predictions is only second-order — negligible by comparison when the points are close (which they are, by design). Averaging the weights gives essentially the same thing as ensembling the predictions, but with one model. That's the whole payoff: ensemble-like generalization at single-model test-time cost. The `Σ Δ_i = 0` is doing all the work — it's exactly the property that kills the first-order error, and it holds *because* I average around the centroid.

So the algorithm is: pretrain a model `ŵ` with the conventional procedure (for the full budget or a fraction of it, say `0.75B`, stopping early without altering its schedule); then continue from `ŵ` with the constant or cyclical high learning rate; capture the spread-out points `w_i`; and maintain their running average. I don't want to store all the `w_i` and average at the end — that's `n` copies of the model. A running incremental average needs only one extra copy: when I capture the `(n_models+1)`-th model `w`,
`w_SWA ← (w_SWA · n_models + w) / (n_models + 1)`,
which folds the new weights into the average in `O(1)` memory and `O(1)` time per capture (equivalently, `w_SWA ← w_SWA + (w − w_SWA)/(n_models + 1)`).

One subtlety bites if the network uses batch normalization. Batch norm keeps running estimates of the per-layer activation mean and variance, collected *during forward passes* in training. But `w_SWA` is an *average of weights* — it was never itself used in a forward pass during training, so no activation statistics were ever collected for it. Its batch-norm running stats are stale (they belong to the last `w`, not to `w_SWA`). Using `w_SWA` as-is would normalize activations with the wrong statistics. The fix is a single extra pass: after training finishes, do one forward pass over the data with the `w_SWA` weights in training mode to recompute each layer's batch-norm running mean and variance. One epoch of forward-only work, negligible.

Let me also confirm the cost is genuinely negligible, since the pitch is "drop-in replacement for SGD." Memory: I keep one extra copy of the *weights* for the running average — but a network's memory footprint is dominated by *activations*, not weights, so this is a small bump during training (on the order of 10%), and after training I keep only the single averaged model, identical to standard. Time: per epoch (or per cycle) I do one weighted sum of two weight vectors to update the average — trivial next to a full epoch of gradients. So SWA costs about the same as SGD.

Here is the procedure, grounded in the standard implementation (an averaged-model wrapper that holds the running average and a forward-pass utility to refresh batch-norm statistics).

```python
import torch

def swa_train(model, loader, loss_fn, lr_init, swa_lr,
              pretrain_epochs, swa_epochs, cycle_len=1):
    opt = torch.optim.SGD(model.parameters(), lr=lr_init,
                          momentum=0.9, weight_decay=5e-4)

    # running average of weights (equivalent to w_SWA <- (w_SWA*n + w)/(n+1))
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    n_models = 0

    # --- phase 1: conventional pretraining to a good basin ---
    for epoch in range(pretrain_epochs):
        for x, y in loader:
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()

    # --- phase 2: explore with a high constant/cyclical LR, average captures ---
    for epoch in range(swa_epochs):
        for i, (x, y) in enumerate(loader):
            # cyclical: alpha = (1-t)alpha_1 + t*alpha_2, t = (mod(i-1,c)+1)/c;
            # constant LR is the special case c = 1 (alpha = swa_lr every step)
            t = ((i % cycle_len) + 1) / cycle_len
            alpha = (1 - t) * lr_init + t * swa_lr if cycle_len > 1 else swa_lr
            for g in opt.param_groups:
                g["lr"] = alpha
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
        # capture once per cycle (per epoch for constant LR) and fold into the average
        swa_model.update_parameters(model)   # w_SWA <- (w_SWA*n + w)/(n+1); n += 1
        n_models += 1

    # --- batch-norm fix: w_SWA never produced activations in training, so its
    #     BN running mean/var are stale.  Recompute with one forward pass. ---
    torch.optim.swa_utils.update_bn(loader, swa_model)
    return swa_model
```

The causal chain: the train-loss minimizer that SGD ships is off-center for test error because the two surfaces are shifted, and a broad, central solution is what stays good under that shift. Constant-/cyclical-LR SGD keeps exploring the periphery of the good region (peripheral because high-dimensional samples live on a sphere's surface), so the *average* of its iterates lands in the central, flat interior — which is exactly why a decaying-LR average fails: it collapses the exploration. Averaging the *weights* rather than the *predictions* is justified because, with `Σ Δ_i = 0`, the single averaged-weight model and the full ensemble agree to second order while the proposals differ at first order — ensemble-quality generalization from one model. Maintain the average incrementally for `O(1)` overhead, recompute batch-norm statistics once at the end, and the whole thing is a near-free drop-in for SGD that converges to wider optima.
