Training deep convolutional networks with SGD and momentum is dominated by the learning-rate schedule. The standard approach is a piecewise-constant staircase: hold the rate fixed for long plateaus, then drop it by a hand-chosen factor at hand-chosen epochs. This schedule is discontinuous, couples the drop locations and drop factor to a pre-fixed total budget, leaves the rate too high for the region the iterate has entered between drops, and gives a well-tuned solution essentially only at the final epoch. What is needed is a schedule that implements the same coarse-to-fine transition smoothly, uses fewer hyperparameters, and provides usable checkpoints at many points during training.

The staircase is trying to do one simple thing: start with a large learning rate for broad exploration and gradually shrink it for fine convergence. The discontinuities are not doing any useful work; they are just an awkward way to approximate a smooth decay. The real question is what smooth decay shape preserves the exploratory start, transitions briskly through the middle, and lands gently at the end, and whether restarts can add anytime performance without the fragility of adaptive triggers or hard momentum resets.

The method I propose is cosine annealing with warm restarts. It replaces the staircase with a smooth half-cosine decay of the learning rate within each run, and periodically warm-restarts by raising the rate back to its maximum. A restart in a momentum method is functionally a reset of the acceleration phase: stop refining the current basin and explore broadly again. Rather than literally zeroing the momentum vector, which discards useful accumulated velocity, or relying on stochastic loss and gradient tests that are too noisy per minibatch, the schedule emulates a restart simply by raising the learning rate. Large steps from a high rate override stale momentum naturally, while preserving the velocity state.

Between restarts, the rate follows the half-cosine curve. Let s equal T_cur divided by T_i, the normalized progress through the current run. The within-cycle multiplier g(s) should be 1 at the start, 0 at the end, have zero derivative at both endpoints, and be monotone decreasing. The zero slopes at the endpoints give a sustained high-rate exploratory phase at the start and a gentle low-rate landing at the end, which also makes a clean restart point. The half-cosine g(s) = (1/2)(1 + cos(pi s)) satisfies all four conditions exactly. Linear ramps have constant slope and sharp corners, exponential decay is steep at the top and never reaches zero, and polynomial decays are flat only at the bottom. The cosine is the parameter-free member of the family of smooth S-curves with zero slope at both ends.

The full schedule grows the length of each run by a factor T_mult at every restart, so early cycles are short and produce a first usable solution quickly while later cycles are longer and refine more. Setting T_mult to 1 and running one cycle over the whole budget with eta_min equal to 0 and eta_max equal to base_lr collapses the schedule to eta_t = base_lr times (1/2)(1 + cos(pi times epoch divided by total_epochs)). This single smooth cosine decay needs only base_lr and total_epochs, with no milestones and no drop factor to tune, and is the closed-form equivalent of PyTorch's CosineAnnealingLR with eta_min set to 0. Because a restart temporarily raises the rate and can worsen the loss, the recommended checkpoint is the iterate at the end of the most recently completed cycle, where the rate has decayed to its minimum. This selection needs no validation set. When warm restarts are used, the sequence of trough iterates also provides several diverse candidate models from one training run.

```python
import math


def get_lr(epoch, total_epochs, base_lr, config=None):
    """Single-cycle cosine annealing from base_lr to 0."""
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))


class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts.

    Within run i, anneal each parameter group's base lr to eta_min along a
    half-cosine; after T_i epochs, wrap to the next run and grow T_i by T_mult.
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("T_0 must be a positive integer")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("T_mult must be an integer >= 1")

        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.T_cur = last_epoch
        self.last_epoch = last_epoch
        self.step(0 if last_epoch < 0 else last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * self.T_cur / self.T_i))
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur -= self.T_i
                self.T_i *= self.T_mult
        else:
            if epoch < 0:
                raise ValueError("epoch must be non-negative")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log(
                        epoch / self.T_0 * (self.T_mult - 1) + 1,
                        self.T_mult,
                    ))
                    self.T_cur = (
                        epoch
                        - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    )
                    self.T_i = self.T_0 * self.T_mult**n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        return lrs
```
