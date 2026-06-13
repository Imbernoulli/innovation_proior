# Context

## Research question

Stochastic gradient descent and its variants train essentially all successful deep networks, yet getting good performance out of them hinges on careful, expensive hyperparameter tuning — above all the learning rate. Set it too high and the iterates oscillate in high-curvature directions or diverge; set it too low and training crawls. The problem is sharpest in the stochastic setting, where each step uses a noisy minibatch gradient, so even at a "good" learning rate the iterates rattle around the optimum rather than settling. The two dominant families of improvements — adaptive learning rates (AdaGrad, Adam) and acceleration (heavy-ball, Nesterov) — both work by accumulating past gradient information, and both still demand tuning to behave. The question is whether there is a mechanism that is *orthogonal* to those two families: something that can wrap around whatever optimizer is already in use, lower the variance of its trajectory and make it robust to a wider range of learning rates, while costing almost nothing in compute or memory and changing as little as possible in an existing training pipeline.

## Background

The recent improvements to SGD fall into two camps. Adaptive methods — AdaGrad (Duchi et al. 2011) and Adam (Kingma & Ba 2014) — rescale each coordinate by an accumulated function of its past squared gradients, effectively giving a per-parameter learning rate. Accelerated methods — Polyak's heavy-ball (1964) and Nesterov momentum (1983) — add a velocity term that accumulates past gradients to damp oscillation and speed progress along consistent directions. Both rely on accumulated gradient history; both improve convergence but inherit a sensitivity to their step sizes.

A separate and older thread is *averaging the iterates themselves* rather than reshaping the gradient. Ruppert (1988) and Polyak & Juditsky (1992) showed that for stochastic approximation, the arithmetic average of the iterates converges with optimal asymptotic variance — better statistical efficiency than the last iterate — because averaging cancels the per-step noise. This is tail averaging: you run the iteration and average the late iterates. In deep learning this resurfaced as Stochastic Weight Averaging (Izmailov et al. 2018), which averages the weights visited by SGD under a cyclical or high constant learning rate and finds flatter, wider optima that generalize better. SWA has two notable features: it uses an *equal-weight arithmetic* average, and it requires choosing *when to start* averaging — begin too early and you fold in bad early models, too late and you average too few points. It also produces a final averaged model; it does not feed the average back into the optimization trajectory. Martens (2014) had observed, in the context of stochastic optimization, that an exponentially-decayed moving average of iterates "typically works much better in practice" than a plain Polyak average, because it emphasizes recent, better proposals.

The relevant theoretical lens for *why* averaging helps is the noisy quadratic model (Schaul et al. 2013; Wu et al. 2018), a tractable proxy for neural-network optimization. Take a diagonal quadratic with a noisy optimum, L̂(x) = ½(x−c)ᵀA(x−c) with c ~ N(x*, Σ), then translate coordinates so x* = 0 and every coordinate evolves independently. The expected loss decomposes into a bias term (squared mean iterate), a variance term, and an irreducible noise floor: E[L̂] = ½ Σᵢ aᵢ(E[θᵢ]² + V[θᵢ] + σᵢ²). Under SGD with learning rate γ the per-coordinate mean contracts as E[x⁽ᵗ⁺¹⁾] = (I−γA)E[x⁽ᵗ⁾] while the variance follows V[x⁽ᵗ⁺¹⁾] = (I−γA)²V[x⁽ᵗ⁾] + γ²A²Σ, so it relaxes to a nonzero steady-state variance V*_SGD = γ²A²Σ / (I−(I−γA)²). That steady-state variance is what keeps the loss above its floor; it grows with the learning rate throughout the stable scalar range 0 < γaᵢ < 2. Increasing γ speeds mean contraction in a coordinate only until γaᵢ reaches 1; beyond that, the mean sign-flips and oscillates even though the variance keeps rising. So there is a genuine tension: a large useful γ can reduce bias quickly but also inflates variance and can overshoot sharp directions, and short-horizon analyses (Wu et al. 2018) show SGD is biased toward exactly the large-γ regime when only a few steps are looked ahead. A mechanism that reduces the steady-state variance without forcing γ down would let one keep the fast bias contraction of a large learning rate.

The same tension is visible in training diagnostics before committing to any new mechanism: repeated high-learning-rate minibatch updates can make successive within-window iterates oscillate and overshoot in high-curvature directions. That behavior is exactly the high-variance trajectory the noisy quadratic model predicts.

## Baselines

**SGD with heavy-ball momentum (Polyak 1964).** Update v ← βv + g, θ ← θ − γv. The momentum term accumulates past gradients, accelerating along low-curvature directions and damping oscillation. Still needs the learning rate and momentum coefficient tuned together; in the under-damped regime it oscillates around the optimum, and in the stochastic setting it does not remove the steady-state noise.

**Adam (Kingma & Ba 2014).** Maintains exponential moving averages of the gradient (m) and squared gradient (v) and updates θ ← θ − α·m̂/(√v̂+ε). The √v̂ term is a diagonal approximation to curvature (the empirical Fisher), giving per-coordinate step sizes. Fast early convergence but sensitive to its learning rate and known to sometimes generalize worse than tuned SGD; like all the above it accumulates gradients but does not target iterate variance.

**Polyak-Ruppert / SWA iterate averaging (Ruppert 1988; Polyak & Juditsky 1992; Izmailov et al. 2018).** Average the visited weights to cut variance and find flatter optima. Equal-weight average; needs a start-of-averaging decision; yields a final model rather than influencing the ongoing trajectory.

**Outer/inner-loop relatives.** Reptile (Nichol & Schulman 2018) runs an inner optimizer then moves the initialization toward the inner result — but across *sampled tasks*, for meta-learning, not single-task convergence. Katyusha (Allen-Zhu 2017), an accelerated SVRG (Johnson & Zhang 2013), checkpoints parameters and pulls iterates back toward the checkpoint *every* inner step with an SVRG variance correction; it has convex guarantees but the SVRG correction works poorly for neural networks. Anderson acceleration (Anderson 1965) and nonlinear extrapolation (Scieur et al. 2018) keep *all* inner iterates and solve for the best linear combination extrapolating to the fixed point — powerful but with memory growing ~k× and a nontrivial sub-problem to find the combination.

## Evaluation settings

The natural yardsticks, all pre-existing: image classification on CIFAR-10 and CIFAR-100 (32×32 images, 50k train / 10k test, 10 and 100 classes) with ResNet-18 and the wider/preact ResNet variants; large-scale classification on ImageNet (≈1.28M train, 50k val, 1000 classes) with ResNet-50 and ResNet-152, top-1/top-5 single-crop accuracy; LSTM language modeling on Penn Treebank (perplexity), following the AWD-LSTM regularized setup (Merity et al. 2017); Transformer (Vaswani et al. 2017) neural machine translation on WMT-2014 English-to-German (BLEU on newstest), with the warmup-then-decay schedule and AdaFactor as a further optimizer comparison. Standard data augmentation for CIFAR (4-pixel pad, random 32×32 crop, horizontal flip, per-channel normalization). The metric of primary interest for an optimizer is training-loss convergence speed at matched data budget, with validation accuracy/perplexity as the generalization check.

## Code framework

The primitives that already exist: PyTorch's `torch.optim.Optimizer` base class with `param_groups`, per-parameter `state`, and a `step(closure)` contract; concrete inner optimizers like `torch.optim.SGD` and `torch.optim.Adam`; a standard training loop. An outer optimizer can wrap an existing optimizer by holding a reference to the inner one and forwarding the ordinary optimizer interface.

```python
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class OuterOptimizer(Optimizer):
    """Wraps an existing inner optimizer and leaves the outer update unspecified."""

    def __init__(self, optimizer, sync_steps=5):
        self.optimizer = optimizer
        self._outer_step = 0
        self._total_outer_steps = sync_steps
        self.state = defaultdict(dict)
        for group in optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                # TODO: initialize any per-parameter outer state.
                pass

    def __getstate__(self):
        # TODO: include the wrapper fields needed for serialization.
        pass

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_outer_step(self):
        return self._outer_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)   # let the inner optimizer take its step
        self._outer_step += 1
        if self._outer_step >= self._total_outer_steps:
            self._outer_step = 0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    # TODO: define the outer update here.
                    pass
        return loss

# Usage:
# base = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# opt  = OuterOptimizer(base, ...)
# for x, y in loader:
#     opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
```
