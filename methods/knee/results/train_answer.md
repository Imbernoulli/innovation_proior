When I train a deep network with SGD-with-momentum, the lever that most decides final test accuracy — with the model, augmentation, batch size, weight decay, and optimizer all fixed by convention — is the schedule of the scalar learning rate $\eta_t$. And here two facts I believe sit in tension. The first is that generalization in these nets is governed not by how *fast* I reach a minimum but by *which* minimum I reach: wide, flat minima are robust to the shift between the empirical training surface and the true test surface and so generalize well, while sharp, narrow minima do not. The second is that every schedule I actually run is tuned for optimization *speed*: cosine annealing decays smoothly from the base rate to zero starting at step one, linear decay drops the rate to zero from step one as well, and step decay holds a single high plateau whose length is a hand-picked milestone before dividing by a factor. None of them was designed around the geometry. Cosine sits at its peak rate for a single instant ($\text{epoch}=0$) and cools monotonically thereafter; linear is the same disease in a straighter line and is the strongest of the simple monotone budgeted decays, which only makes it the bar to beat; step decay is closer to the right idea with its high plateau, but that plateau is a speed milestone, not a deliberate budget, and its drop is a discrete shock. So every baseline cools too early, and if there is accuracy being left on the table, this is where it is.

The reason the learning rate touches minimum width at all is the SGD noise scale. To first order the magnitude of the stochastic-gradient noise per step is $g \approx \epsilon N / (B(1-m))$, proportional to the rate $\epsilon$, the dataset size $N$, inversely to the batch $B$ and to $(1-m)$ for momentum $m$. A higher rate means more noise, and noise is exactly what lets SGD escape a narrow basin: a sharp minimum is a small target with steep walls that a noisy step bounces out of, whereas a wide minimum is a large flat region the same noise cannot escape. So a high learning rate is a wide-minimum-seeking device — it keeps the search hot enough to be ejected from narrow basins and to settle only where the basin is wide enough to hold it. The learning rate is the *temperature* of the search, and "decay the rate" really means "cool the search." Once it reads that way the question reorganizes from "what decay shape is fastest" into "when should I cool, and how long should I stay hot first." To answer that I need one more empirical fact, which I take as the working hypothesis: wide minima are *lower-density* than narrow ones — rare in a landscape full of narrow basins. If that holds, a search that cools quickly almost certainly falls into one of the abundant narrow basins long before it stumbles onto a rare wide one, and the only way to find a rare wide basin is to stay hot long enough to sample it and only *then* cool down to settle into it. This is consistent with what I already believe about large-batch training: large $B$ means *less* noise, a colder search, which by this argument settles into a sharper minimum sooner — and large batch is indeed known to generalize worse.

I propose the Knee schedule — an explore-then-exploit learning-rate curriculum in temperature: stay hot to find a rare wide basin, then cool to lock into it. It is three segments that tile the budget. First a brief linear *warmup* ramp from a small value up to the peak rate, which is pure init hygiene — on deep nets and especially on Transformers the rate at the sharp initial region can exceed the curvature ceiling, and the ramp protects it; warmup is complementary to the mechanism, not part of it. Then the *explore* phase, which is the new thing and carries the generalization load: I hold the rate *constant at the peak*, not a ramp and not a slow decay, because the entire point is to keep the temperature maximal and constant so the search keeps sampling basins for the full explore budget — any decay here is premature cooling, the exact failure of cosine and linear. The peak rate $\text{peak\_lr}$ is just the largest rate the net trains stably at, which is the same value the well-tuned baselines already use as their starting rate; I do not need to push it higher, I need to *hold* it longer. Finally the *exploit* phase: once explore has parked the iterate in or near a wide basin, I cool the search so the noise shrinks and SGD descends to the bottom of that basin and stays. For the cooling shape I deliberately do *not* reach for cosine or anything clever, because the explore phase is doing the generalization work and the exploit phase only has to bring the rate monotonically to zero over the remaining budget. A plain *linear* decay from $\text{peak\_lr}$ to $0$ is the strongest simple monotone budgeted decay, has one obvious slope, and reaches exactly $0$ at the budget boundary so the final steps are vanishingly small and the wide basin is fully settled. With $\text{decay\_steps}$ exploit steps the slope is $\text{slope} = -\text{peak\_lr}/\text{decay\_steps}$, the rate is $\text{peak\_lr} + \text{slope}\cdot(\text{steps into exploit})$, clamped at $0$ so it never goes negative — no milestones, no shape parameter.

Concretely, with $\text{peak\_lr}$ the reference rate, $\text{warmup\_steps}$ the ramp length, and $\text{explore\_steps}$ the plateau length, the three branches as a function of the global step are

$$\eta(t) = \begin{cases} \text{peak\_lr}\cdot t/\text{warmup\_steps}, & t \le \text{warmup\_steps} \\[2pt] \text{peak\_lr}, & \text{warmup\_steps} < t \le \text{warmup\_steps}+\text{explore\_steps} \\[2pt] \max\!\big(0,\ \text{peak\_lr} - \tfrac{\text{peak\_lr}}{\text{decay\_steps}}\,(t - \text{warmup\_steps} - \text{explore\_steps})\big), & t > \text{warmup\_steps}+\text{explore\_steps} \end{cases}$$

where the decay length is fixed by tiling the budget exactly, $\text{decay\_steps} = \text{total\_steps} - (\text{warmup\_steps} + \text{explore\_steps})$, and this must be non-negative. Checking the segments at the seams: warmup ends at $\text{peak\_lr}$ and explore opens at $\text{peak\_lr}$ — continuous; explore ends at $\text{peak\_lr}$ and exploit's first decay step has zero progress so it opens at $\text{peak\_lr}$ — continuous; exploit reaches $0$ at the budget boundary. The only non-smoothness is a *slope* change at each seam, never a value discontinuity, so there is no shock to the dynamics or to the momentum velocity.

The split between explore and exploit is the one real hyperparameter, and the density hypothesis tells me how to think about it. Too little explore and I cool before finding a wide basin, back to the cosine/linear failure; too much explore and too few epochs remain to settle, so I end *near* a wide basin but never descend into it and the training loss stays high. So there is an interior optimum, and because wide basins are rare it should be *large* — a substantial fraction of the budget spent hot. I would set it by sweeping a few explore fractions — say $0$, $30\%$, $60\%$, $100\%$ of the budget — and watching two things at once: the final accuracy, and as a direct test of the mechanism the *width* of the minimum reached, measured as the largest weight perturbation that keeps the loss inside a band. If the hypothesis is right, more explore epochs should monotonically widen the minimum and raise accuracy, up to the point where too little exploit budget remains to settle. I expect that to land around half the budget — enough exploration to find a rare wide basin, enough exploitation to settle it — so the default is explore $=50\%$ of the total budget, exploit the rest. That single fraction is the method's only knob. At the same budget this improves accuracy over the hand-tuned decays, and it reaches the original accuracy in fewer epochs; only the per-step learning rate changes, with the optimizer, loss, data pipeline, momentum, and weight decay untouched.

```python
import torch
from torch.optim.optimizer import Optimizer


class KneeLRScheduler:
    """Explore-exploit (Knee) learning-rate schedule.

    warmup : linear ramp 0 -> peak_lr over warmup_steps
    explore: hold peak_lr for explore_steps (defaults to ~50% of total budget)
    exploit: linear decay peak_lr -> 0 over the remaining decay_steps
    """

    def __init__(self, optimizer, peak_lr, warmup_steps=0, explore_steps=None, total_steps=0):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = int(0.5 * total_steps) if explore_steps is None else explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.current_step = 1

        assert self.decay_steps > 0

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.current_step)

    def get_lr(self, global_step):
        if self.warmup_steps > 0 and global_step <= self.warmup_steps:
            return self.peak_lr * global_step / self.warmup_steps
        elif global_step <= (self.explore_steps + self.warmup_steps):
            return self.peak_lr
        else:
            slope = -1 * self.peak_lr / self.decay_steps
            return max(0.0, self.peak_lr + slope * (global_step - (self.explore_steps + self.warmup_steps)))

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.current_step)
```
