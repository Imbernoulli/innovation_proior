The one lever I am allowed to move is the per-epoch learning rate, and the scaffold hands it to me as a constant: $\eta = $ `base_lr` for all 200 epochs. A flat rate is structurally wrong at both ends, and the shape of that failure is what dictates the first rung. Model the loss near the weights as a quadratic with Hessian $H$; one gradient step along an eigendirection of curvature $\lambda$ multiplies the distance to the minimum by $(1 - \eta\lambda)$, which contracts only while $\eta < 2/\lambda$. The admissible step is therefore capped by $2/\lambda_{\max}(H)$, and $\lambda_{\max}$ is largest right after random init — a freshly initialized net sits in a sharp, high-curvature region, so $0.1$ can sit above the ceiling and the first steps overshoot. That is the early end. The late end is the mirror: near a minimum the *true* gradient has shrunk but the *stochastic* one has not, so a fixed $0.1$ keeps kicking the weights around the minibatch-noise floor instead of settling. The start wants a rate below the curvature ceiling and the end wants a tiny rate to beat the noise floor — different numbers — so the rate must be small-high-small.

I propose to start the ladder from a **constrained one-cycle** schedule: one smooth rise to a peak, then one smooth fall. The shape is read straight off the loss topology. Training makes steep, fast progress in the first handful of epochs, then enters a long nearly-flat valley where the slope is tiny, and only at the end has to thread a narrow trough to the minimum. Overlay the rate on that: small at the steep start so a big step does not overshoot, *large* across the flat valley so the step crosses it fast and punches through saddle plateaus rather than crawling over them, then small again to settle the trough without bouncing out. That is the cyclical-learning-rate insight (Smith, 2015) made into a single cycle — one rise, one fall — rather than a perpetual oscillation between two bounds, because the topology wants one fall that settles, not a band held forever.

What makes this rung load-bearing is being exact about *which* one-cycle I am running, because the name carries far more machinery than this harness exposes. The full one-cycle policy gets its power from three moves: a peak an order of magnitude above $0.1$, read off a pre-run LR range test, whose large in-valley rate is itself a strong regularizer (the super-convergence regime); a *reduction* of weight decay to rebalance that extra regularization; and momentum cycled *inversely* to the rate (high when the rate is low, dropped at the peak) because in the SGD-with-momentum update the displacement scales with both $\eta$ and $m$, so a rising rate stacked on high momentum blows past stability. None of these is available here. I edit only the body of `get_lr`, which returns one float per epoch; momentum is frozen at $0.9$ and the loop never re-sets it, so I cannot cycle it; weight decay is frozen at `5e-4`, so I cannot rebalance it; and `get_lr` is a pure function of `epoch`, so there is no range-test hook. Pushing the peak to ten times $0.1$ with momentum and weight decay frozen high would be exactly the configuration the full method warns is unstable, with none of the compensations. The faithful port is therefore the *tame* one: keep the rise-then-fall **shape**, but peak at `base_lr` itself, leave momentum and weight decay where the loop fixes them, and replace the range-test peak with the reference rate I am given.

With the peak pinned at `base_lr`, the rest is the shape of the two legs and the depth of the tail. I shape both legs with a cosine rather than the triangular policy's linear corners, because a cosine eases into and out of the extremes — it lingers a little near the top (sustained exploration at the peak) and near the bottom (a gentle landing) and, crucially, has zero slope at the peak, the most dangerous moment to jolt the dynamics. A cosine from a start value to an end value across a fraction `pct` running $0\to1$ is $\text{end} + \tfrac{\text{start}-\text{end}}{2}(\cos(\pi\,\text{pct}) + 1)$, equalling `start` at `pct=0` and `end` at `pct=1` with zero slope at both ends; I use it for both legs. The topology says the steep early region is short and the flat valley plus final descent is most of the run, so the climb takes the first 30% and the descent the last 70% (`pct_start = 0.3`). The climb does not start at the peak: I open it well below, at `base_lr/div_factor` with `div_factor = 25`, so the warmup genuinely begins small — $0.004$, comfortably under the curvature ceiling at init — and cosine-climbs to `base_lr` over the first 30%. For the tail I deliberately *do not* drive it orders of magnitude below the start. The full method's deep "annihilation" (a `final_div` of 1e4) earns its keep only when a large-rate phase first carried the iterate into a wide flat region to drop into; with the peak capped at the ordinary `base_lr`, there is no wide-region-then-annihilate story to honor, so I keep `final_div = 25` and end the run at `base_lr/25` as well.

The schedule is thus a cosine warmup from `base_lr/25` to `base_lr` over the first 30%, then a cosine anneal from `base_lr` back to `base_lr/25` over the remaining 70%, with progress measured over `total_epochs - 1` and no `arch`/`dataset` conditioning. I expect this to be the *weakest* rung — not because the shape is wrong but because the harness amputates what made the method strong, leaving two leaks. The up-leg spends the first 30% — sixty epochs — *below* `base_lr`, only touching the full rate at epoch 60, so a large slice of the most productive phase is spent under-stepping; and the down-leg stops at $0.004$, not zero, leaving residual jitter at the finish a true anneal-to-zero would have removed. Both should bite hardest on the deep ResNet-56 / CIFAR-100, which needs the productive high-rate middle and a clean low-rate finish most.

```python
def get_lr(epoch, total_epochs, base_lr, config):
    """OneCycleLR schedule (Smith & Topin, 2019).

    Phase 1 (0-30%): cosine warmup from base_lr/25 to base_lr.
    Phase 2 (30-100%): cosine anneal from base_lr to base_lr/25.
    """
    pct_start = 0.3
    div_factor = 25.0
    final_div = 25.0

    min_lr = base_lr / div_factor
    final_lr = base_lr / final_div

    progress = epoch / max(total_epochs - 1, 1)

    if progress <= pct_start:
        # Warmup phase: cosine from min_lr to base_lr
        t = progress / pct_start
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (1 - t)))
    else:
        # Anneal phase: cosine from base_lr to final_lr
        t = (progress - pct_start) / (1 - pct_start)
        return final_lr + (base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * t))
```
