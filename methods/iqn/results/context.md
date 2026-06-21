## Research question

Value-based deep RL on Atari learns a single number per state-action — the expected return $Q(x,a)=\mathbb{E}[Z(x,a)]$ — and acts greedily on it. Distributional RL instead learns the law of the random return $Z(x,a)$, and prior distributional agents show that this richer target can help even when the policy still acts on the mean. The research question is how to design the distributional return representation and the policy that uses it.

## Background

A distributional RL algorithm is fixed by two choices: how the return distribution is parameterized, and which probability metric the learning minimizes. The distributional Bellman operator $\mathcal{T}^\pi Z(x,a)\overset{D}{=}R(x,a)+\gamma Z(x',a')$ is a $\gamma$-contraction in the maximal $p$-Wasserstein metric $\bar d_p$ — the $L_p$ distance between inverse CDFs (quantile functions). Wasserstein is the natural metric because the Bellman update scales and shifts mass off any fixed grid, and Wasserstein stays finite and meaningful even for distributions with disjoint support. The complication is a sampling result: the Wasserstein distance cannot be minimized by stochastic gradient descent from sample transitions — the empirical-Wasserstein gradient is biased — so the metric the operator contracts in is not directly trainable online.

Two families have been developed. The categorical family pins the *return locations* to a fixed comb $z_1<\cdots<z_N$ on a prescribed interval $[V_{\min},V_{\max}]$, learns the *probabilities* on those atoms, projects the shifted Bellman target back onto the comb, and minimizes a cross-entropy/KL after projection, with the projection tied to the Cramér ($L_2$-on-CDFs) geometry rather than Wasserstein. The quantile family transposes this: it fixes the *probabilities* to a uniform $1/N$ and learns the *locations*, which are then quantiles of the return; it uses quantile regression, whose minimizer is the desired quantile and whose sample gradient is unbiased, and the resulting projected Bellman operator contracts in the $\infty$-Wasserstein metric. Both share the Nature-DQN convolutional torso, a target network, a uniform replay buffer, $\epsilon$-greedy exploration, and a mean-greedy policy.

A separate strand of background concerns *risk*. Acting on the mean treats all spread in the return as equivalent. Classical decision theory offers two ways to be sensitive to the spread: expected utility theory (maximize $\mathbb{E}[U(Z)]$ for a concave/convex utility $U$), and Yaari's dual theory of choice, which replaces the independence axiom (violated by humans, as in the Allais paradox) with one over convex combinations of *outcomes*, yielding policies that maximize a *distorted expectation* — the expectation under a continuous monotone reweighting $h$ of the cumulative probabilities. Such an $h$ is a distortion risk measure; conditional value-at-risk, cumulative-prospect-theory probability weighting, and many others are special cases. A distorted expectation can be written as an integral of the quantile function $F_Z^{-1}$ against a non-uniform weighting over probability levels. Risk sensitivity, in this language, is a reweighting of which parts of the return distribution the policy cares about.

## Baselines

- **DQN (Mnih et al. 2015).** Convolutional $Q$-network trained by regressing $Q(x,a)$ onto $r+\gamma\max_{a'}Q(x',a';\theta^-)$ with a target network and experience replay. Learns the mean of the return.

- **C51 / categorical DQN (Bellemare, Dabney, Munos 2017).** Distribution over a fixed set of $N$ atoms on $[V_{\min},V_{\max}]$, learned probabilities, KL after a projection of the Bellman target. Demonstrates that learning the distribution helps even with a mean-greedy policy.

- **QR-DQN (Dabney et al. 2018).** Uniform $1/N$ mass on $N$ learned locations $\theta_i(x,a)$ — the locations are the midpoint quantiles $\hat\tau_i=(2i-1)/(2N)$ — trained by the quantile Huber regression loss on all pairs of predicted and bootstrapped target locations, acting greedily on the mean $\frac1N\sum_i\theta_i$. Removes the support bounds and the projection and gives unbiased gradients toward the $\infty$-Wasserstein projection. The network emits exactly $N$ values, one per predetermined level $\hat\tau_i$.

- **Dueling DQN (Wang et al. 2016).** Splits the head into a state-value stream and an advantage stream, $Q=V+A-\frac1{|\mathcal A|}\sum_{a'}A$. An architectural improvement to the mean estimator, orthogonal to whether the return is represented as a distribution.

## Evaluation settings

The Atari-57 ALE benchmark under the standard DQN protocol: $84\times84$ grayscale frames, 4-frame stack, frame-skip 4, reward clipping to $[-1,1]$, terminal-on-life-loss for training, a $10^6$-transition uniform replay buffer, $\epsilon$-greedy with $\epsilon$ annealed to a small floor, a periodically-copied target network, and a fixed interaction budget. Agents are compared by human-normalized mean and median episodic return across the suite, evaluated at the mean-greedy policy; the natural points of comparison are DQN, prioritized DQN, the distributional agents above, and the combined agent that stacks several orthogonal improvements at once.

## Code framework

The substrate is a standard distributional-DQN training loop. The convolutional torso $\psi$, the replay buffer, $\epsilon$-greedy acting, the target-network copy, and the optimizer all already exist. One module is open: the head that turns torso features into whatever return representation the agent learns, the loss that trains it, and the rule that turns that representation into an action.

```python
import torch
import torch.nn as nn

class ConvTorso(nn.Module):
    """Fixed Nature-DQN convolutional feature extractor: (B,4,84,84) -> (B, d)."""
    def __init__(self, d=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, d), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x / 255.0)


class ReturnHead(nn.Module):
    """Turns torso features into the agent's return representation."""
    def __init__(self, d, n_actions):
        super().__init__()
        self.torso = ConvTorso(d)
        # TODO: the return representation we will design

    def forward(self, x):
        psi = self.torso(x)
        raise NotImplementedError  # TODO

    def greedy_action(self, x):
        raise NotImplementedError  # TODO: act from the learned representation


def distributional_loss(online_head, target_head, batch, gamma):
    # TODO: the training objective for the learned representation
    raise NotImplementedError
```
