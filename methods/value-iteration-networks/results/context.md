# Context: learning policies that must plan, not just react

## Research question

We want a neural-network policy for sequential decision-making — navigation, control, trajectory prediction — that generalizes to new task instances it was never trained on. Train a policy on many random instances of a domain (e.g. grid-world navigation with randomly placed obstacles and a random goal), and test it on a new obstacle/goal layout drawn from the same distribution. Supervised one-step prediction generalizes well across instances; the question is whether a sequential decision-making policy can as well.

Good long-term behavior in these domains comes from planning — looking ahead through the consequences of actions to reach a goal. The policy representations that deep RL and imitation learning had standardized on are reactive: a perception stack maps the current observation directly to a distribution over actions. How can a differentiable neural-network policy, trainable by standard RL and imitation-learning algorithms, incorporate planning-like computation as part of its representation?

## Background

**Markov decision processes and value iteration.** The standard model for sequential decision-making (Bellman 1957; Bertsekas, *Dynamic Programming and Optimal Control* Vol. II, 4th ed. 2012) is an MDP with states $s\in\mathcal S$, actions $a\in\mathcal A$, reward $R(s,a)$, transition kernel $P(s'|s,a)$, and discount $\gamma\in(0,1)$. The value of a policy is $V^\pi(s)=\mathbb E^\pi[\sum_{t\ge0}\gamma^t r(s_t,a_t)\mid s_0=s]$; the optimal value $V^*(s)=\max_\pi V^\pi(s)$. Value iteration (VI) computes it by repeated Bellman backups:
$$Q_n(s,a)=R(s,a)+\gamma\sum_{s'}P(s'|s,a)V_n(s'),\qquad V_{n+1}(s)=\max_a Q_n(s,a).$$
$V_n\to V^*$ as $n\to\infty$, and an optimal policy is $\pi^*(s)=\arg\max_a Q_\infty(s,a)$. VI requires a known model and is run as a black-box iterative solver.

**Convolutional networks.** CNNs (Fukushima 1979; LeCun et al. 1998; Krizhevsky et al. 2012) stack convolution and max-pooling layers. A convolution layer maps an $l$-channel input $X$ to an $l'$-channel output $h_{l',i',j'}=\sigma\!\left(\sum_{l,i,j}W^{l'}_{l,i,j}X_{l,i'-i,j'-j}\right)$ via learned kernels; a max-pooling layer takes, per channel, the maximum over a local spatial neighborhood. Their power comes from two structural priors — **locality** (a kernel only touches a small neighborhood) and **weight sharing** (the same kernel sweeps the whole image) — which slash parameter count and match the statistics of images. They are trained by SGD with backpropagation.

**Reactive deep policies.** With CNNs working for perception, deep RL and imitation learning adopted them as policy networks: a CNN extracts features from the observation $\phi(s)$, fully connected layers map features to a distribution over actions, giving $\pi_\theta(a\mid\phi(s))$. DQN (Mnih et al., *Nature* 2015) learns such a network to predict action values on Atari; guided policy search (Levine & Abbeel 2014) and TRPO (Schulman et al. 2015) train such networks for robotic control; imitation learning (Ross et al. 2011, DAgger) trains them by supervised learning from expert actions. These algorithms are agnostic to the policy representation — they only need it differentiable. The representations themselves are essentially the supervised-learning architecture with actions in place of class labels: reactive, with the "deep computation" spent on perception.

**Model-based RL.** One classical route to generalization is model-based: identify a dynamics model from data, then solve it for a policy (Schmidhuber 1990; Deisenroth & Rasmussen 2011, PILCO). This generalizes by construction and is used in practice for domains where system identification is tractable.

**Approximating algorithms with neural networks.** A contemporary trend tried to make classical algorithms learnable as differentiable modules: Neural Turing Machines (Graves et al. 2014), Pointer Networks (Vinyals et al. 2015), and "learning to execute" recurrent nets (Zaremba & Sutskever 2015). Separately, Ilin, Kozma & Werbos (2007) showed a cellular *simultaneous recurrent* network — a recurrent net with a shared local cell update, run to a fixed point over a 2-D grid — could solve maze navigation, framed as approximate dynamic programming and trained (with an extended Kalman filter) much faster than a plain recurrent net. These are precedents for treating a planning/algorithmic computation as a recurrent, weight-shared, grid-structured network.

## Baselines

**Reactive CNN policy (inspired by DQN; Mnih et al. 2015).** A stack of convolution layers (in the grid-world study, five conv layers with $[50,50,100,100,100]$ $3\times3$ kernels and a couple of $2\times2$ max-pools) followed by a fully connected layer and a softmax over actions; the current position is supplied as an extra input channel. Core idea: map the observed map directly to action probabilities, leaning on the learning algorithm to instill good long-term behavior.

**Fully convolutional / dense-prediction policy (inspired by Long et al. 2015, FCN).** Treat "which action at each cell" like semantic segmentation: a fully convolutional net produces a per-cell action map. To let goal information reach every cell, the first convolution uses a filter spanning the whole image (e.g. $150$ filters of size $(2m{-}1)\times(2n{-}1)$), then $1\times1$ convs, then the action channels are read off at the current cell. Core idea: a global receptive field plus dense prediction.

**Inverse reinforcement learning / maximum-margin planning (Abbeel & Ng 2004; Ratliff et al. 2006; Ziebart et al. 2008; applied to overhead-image trajectory prediction by Kitani et al. 2012, Ratliff et al. 2009).** Learn a cost/reward from expert demonstrations, then run a planner on a *known* MDP to predict trajectories. Core idea: separate "learn the cost" from "solve the known model."

**Model-based RL (Schmidhuber 1990; PILCO, Deisenroth & Rasmussen 2011).** Identify the dynamics, then solve for a policy. Core idea: an explicit learned model that, once solved, transfers to new instances.

**Cellular simultaneous recurrent network for maze navigation (Ilin, Kozma & Werbos 2007).** A recurrent grid network with a shared local update, run to a fixed point, learns a navigation value-like field; framed as approximate dynamic programming. Core idea: a planning-like computation realized by a weight-shared recurrent net over a grid.

## Evaluation settings

- **Synthetic grid-world navigation.** Random $m\times n$ grids ($8\times8$, $16\times16$, $28\times28$) with randomly placed obstacles, a border, random start and goal. Observation: a 2-channel image (obstacle map; goal map) plus the agent's $(i,j)$ position. Action: one of 8 movement directions (or 4 in the RL variant). Training data are optimal shortest-path demonstrations (e.g. $5000$ instances $\times\,7$ trajectories) computed by an exact planner; imitation-learning by supervised classification of the expert action, or RL with reward $+1$ goal / $-1$ hole / $-0.01$ per step. Metrics: $0$–$1$ action-prediction loss, trajectory **success rate** (reaches goal without hitting an obstacle on roll-out), and trajectory-length difference from optimal. Evaluation is on a **held-out** set of maps disjoint from training.
- **Mars terrain navigation.** Overhead HiRISE grayscale terrain images; a $16\times16$ grid over a $128\times128$ patch; a cell is an obstacle if its terrain slope exceeds $10^\circ$ (computed from elevation data that is *not* given as input). Predict shortest-path trajectories directly from raw terrain images; metrics as above on held-out patches.
- **Continuous control.** A 2-D continuous point-mass / planning task trained with guided policy search (Levine & Abbeel 2014) under unknown dynamics (MuJoCo simulation), comparing policy representations.
- **Natural-language web navigation (WebNav; Nogueira & Cho 2016).** Pages/links of a website form a directed graph; given a query (sentences from a target page up to a few hops away), navigate from a start page to the target by clicking links. Bag-of-words features with pretrained embeddings; metrics: top-1 / top-4 link-prediction accuracy and average reward.

Standard optimizers and frameworks of the time: SGD with RMSProp (Tieleman & Hinton 2012), Theano (Theano Development Team 2016) for the GPU networks, RLLab (Duan et al. 2016) for TRPO, public GPS code (Finn et al. 2016) and MuJoCo (Todorov et al. 2012) for continuous control.

## Code framework

The pieces that already exist: a data pipeline that yields, per sample, an observation image and the agent's grid position with a supervised action label; convolution and channel-max-pooling primitives; an optimizer (RMSProp); a cross-entropy loss; and a standard supervised/RL training loop. What does not yet exist is the policy module itself — the part that turns the observation into an action while doing whatever computation makes the policy generalize across layouts. The scaffold lays out the known primitives and leaves that module as an empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- existing primitives ----
# Conv2d, max over a channel axis, Linear, Softmax, RMSprop, NLLLoss are all standard.

class Policy(nn.Module):
    """Maps an observation image + the agent's grid position to a distribution
    over actions. The internal computation is the open question."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # config: in_channels (obstacle+goal maps), grid size, num_actions, ...
        # TODO: the policy module we will design goes here.
        pass

    def forward(self, obs_image, state_x, state_y):
        # obs_image: (batch, in_channels, H, W); (state_x, state_y): current cell.
        # TODO: compute action logits from obs_image and the current position.
        raise NotImplementedError


def train(policy, data_loader, epochs, lr=0.01):
    opt = torch.optim.RMSprop(policy.parameters(), lr=lr)
    for _ in range(epochs):
        for obs_image, state_x, state_y, action in data_loader:
            logits, probs = policy(obs_image, state_x, state_y)
            loss = F.cross_entropy(logits, action)   # imitation; swap for an RL loss if needed
            opt.zero_grad(); loss.backward(); opt.step()
```
