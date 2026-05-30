# Research question

A reinforcement-learning agent interacting with an MDP $(\mathcal{X},\mathcal{A},R,P,\gamma)$
must, in addition to *improving* its policy from the data it has, *generate* the data in the
first place — it has to choose which states and actions to try. The dominant deep-RL agents of
the time explore with what is essentially noise bolted onto an otherwise deterministic decision:
$\epsilon$-greedy for value-based agents (with probability $\epsilon$ take a uniformly random
action, otherwise the greedy one) and an entropy bonus for policy-gradient agents (push the
action distribution toward uniform). Both inject perturbations that are *independent across time
steps* and, for $\epsilon$-greedy, *independent of the state*. The precise problem: such local
"dithering" of the policy is unlikely to produce the *coherent, multi-step, state-dependent*
deviations from the current behaviour that deep exploration requires; and the amount of
perturbation ($\epsilon$ schedule, entropy weight $\beta$) is a hand-tuned hyperparameter held
fixed (or annealed on a fixed schedule) across very different tasks. A solution would have to
(i) inject perturbations that are consistent over an episode and depend on the state, so the
agent can commit to an exploratory *strategy* rather than jitter; (ii) let the magnitude of
exploration be *learned* from the RL signal rather than set by hand; and (iii) drop into existing
agents (Q-learning, dueling, actor-critic) with negligible computational and conceptual overhead.

# Background

**MDPs and value functions.** An MDP is $(\mathcal{X},\mathcal{A},R,P,\gamma)$ with
$\gamma\in(0,1)$. A policy $\pi(\cdot\,|\,x)$ induces the action-value
$Q^\pi(x,a)=\mathbb{E}^\pi\big[\sum_{t\ge0}\gamma^tR(x_t,a_t)\big]$, and the optimal value is
$Q^\star(x,a)=\max_\pi Q^\pi(x,a)$. Value-based control learns $Q^\star$ and acts greedily; the
state-value is $V^\pi(x)=\mathbb{E}_{a\sim\pi}[Q^\pi(x,a)]$.

**Exploration heuristics in deep RL.** The standard mechanisms are *random perturbations of the
policy*: $\epsilon$-greedy (Sutton & Barto, 1998) injects a state-independent uniform action with
probability $\epsilon$; entropy regularisation (Williams & Peng, 1991; Mnih et al., 2016) adds a
bonus $\beta\,H[\pi(\cdot|x)]$ to keep the policy stochastic. Osband et al. (2016, 2017) argue
that these *dithering* schemes — decorrelated noise added at every step — cannot achieve the
"deep exploration" of committing to an informative multi-step trajectory, and that *randomised
value functions* (perturb the value estimate, then act greedily and consistently) provide a
provably efficient alternative in tabular/linear settings.

**Other exploration families and why they are awkward here.** *Optimism in the face of
uncertainty* (UCB-style; Kearns & Singh, 2002; Jaksch et al., 2010; Azar et al., 2017) has strong
guarantees but is tied to small or linear models. *Intrinsic-motivation* methods augment the
reward with a novelty/learning-progress/information-gain term (Schmidhuber; Houthooft et al.,
2016, VIME; Bellemare et al., 2016, pseudo-counts). Their drawback is structural: the metric for
intrinsic reward and, crucially, its *weight relative to the environment reward* must be chosen by
the experimenter; if mis-weighted the intrinsic term can shift or obscure the optimal policy, and
dithering is often still needed on top (Ostrovski et al., 2017). *Black-box / evolutionary*
policy-space search (Salimans et al., 2017) is generic but data-inefficient.

**Randomised value functions and bootstrapped nets.** Osband et al. (2016) extend randomised
value functions to deep nets via *bootstrapped DQN*: maintain $K$ separate value heads
(effectively duplicating sections of the network) trained on different bootstraps of the data;
acting under one head for a whole episode gives temporally consistent, state-dependent
exploration. The cost is the duplicated heads and the bootstrap machinery.

**Parameter-space noise (concurrent).** Plappert et al. (2017) independently add *constant*
Gaussian noise to the network parameters for exploration. This already gives state-dependent,
consistent perturbations, but the noise scale is fixed (or adapted by a separate heuristic),
not learned end-to-end by the RL loss, and the noise is restricted to Gaussian.

**Noise injection in optimisation.** Adding vanishing, non-trainable noise to weights is a known
optimisation aid (graduated optimisation, Hazan et al., 2016; neural diffusion, Mobahi, 2016) and
weight-uncertainty methods (Bayes-by-Backprop, Blundell et al., 2015; variational inference over
weights, Graves, 2011) maintain an explicit distribution over weights. These differ from what is
needed here: the former uses noise that vanishes on a fixed schedule and is not learned; the
latter is aimed at a posterior/compression objective, not at driving exploration with a noise
intensity tuned by the task reward.

**The reparameterisation trick.** If a weight is written $w=\mu+\sigma\,\varepsilon$ with
$\varepsilon$ a fixed-statistics noise, then any loss $L(w)$ becomes a function of
$(\mu,\sigma)$ and its expectation $\bar L(\mu,\sigma)=\mathbb{E}_\varepsilon[L(\mu+\sigma\varepsilon)]$
is differentiable in $\mu,\sigma$ with gradients estimable by a single Monte-Carlo sample
(Kingma & Welling, 2014; Blundell et al., 2015). This is the lever that lets a *noise scale*
be trained by ordinary backprop.

# Baselines

**DQN (Mnih et al., 2015).** A convolutional network $Q(x,\cdot;\theta)$ (conv 32×8×8/4 → conv
64×4×4/2 → conv 64×3×3/1 → FC 512 → FC $|\mathcal{A}|$) trained off-policy from a replay buffer
by minimising
$\mathbb{E}_{(x,a,r,y)\sim D}\big[(r+\gamma\max_bQ(y,b;\theta^-)-Q(x,a;\theta))^2\big]$ with a
periodically-copied target network $\theta^-$. Behaviour is $\epsilon$-greedy with $\epsilon$
annealed on a fixed schedule. **Gap:** exploration is state-independent and temporally
decorrelated, and the exploration rate is a hand-set schedule unrelated to the agent's own
uncertainty.

**Dueling DQN (Wang et al., 2016).** Splits the head into value and advantage streams sharing a
convolutional encoder $f$:
$Q(x,a;\theta)=V(f(x;\theta_{\text{conv}}),\theta_V)+A(f(x;\theta_{\text{conv}}),a;\theta_A)-\frac{1}{N_{\mathcal A}}\sum_bA(f(x;\theta_{\text{conv}}),b;\theta_A)$,
and optimised with the double-DQN target (action selected by the online net, evaluated by the
target net). **Gap:** same $\epsilon$-greedy exploration.

**A3C (Mnih et al., 2016).** A policy-gradient actor-critic. The network outputs a softmax policy
$\pi(\cdot|x;\theta_\pi)$ and a value $V(x;\theta_V)$; on a $k$-step rollout the policy gradient
is $\sum_i\nabla\log\pi(a_{t+i}|x_{t+i})\,A(x_{t+i},a_{t+i})$ with
$\hat Q_i=\sum_{j=i}^{k-1}\gamma^{j-i}r_{t+j}+\gamma^{k-i}V(x_{t+k})$ and
$A=\hat Q_i-V(x_{t+i})$, plus an entropy bonus $\beta\,H[\pi(\cdot|x)]$ to keep the policy from
collapsing too fast. **Gap:** exploration comes only from the entropy bonus — decorrelated
per-step stochasticity with a hand-set $\beta$.

**Bootstrapped DQN (Osband et al., 2016).** Achieves temporally consistent exploration via $K$
duplicated value heads and bootstrap masks; acting under one sampled head per episode. **Gap:**
the heads multiply parameters/compute and require the bootstrap apparatus; the exploration scale
is not itself a learned per-weight quantity.

**Constant parameter noise (Plappert et al., 2017).** Adds fixed-scale Gaussian noise to the
weights. **Gap:** the noise scale is not learned by the RL loss (adapted only by an external
distance heuristic) and is restricted to Gaussian.

# Evaluation settings

The natural yardstick is the Arcade Learning Environment (Bellemare et al., 2013): 57 Atari 2600
games with the standard DQN preprocessing (4-frame stacking of $84\times84$ grayscale frames,
frame-skip 4, max-over-2-frames, random no-op starts, reward clipping). The protocols are those
of the baselines: off-policy replay training with a periodically updated target network for the
value agents (DQN trained 200M frames), asynchronous $n$-step actor-critic for A3C (320M frames).
Evaluation suspends learning every 1M environment frames and runs the current agent for 500K
frames, with episodes truncated at 108K frames; scores averaged over 3 seeds. Performance is
summarised by the human-normalised score
$100\times(\text{Score}_{\text{agent}}-\text{Score}_{\text{Random}})/(\text{Score}_{\text{Human}}-\text{Score}_{\text{Random}})$,
aggregated as mean and median across the 57 games. A natural internal diagnostic is to track,
per noisy layer, the mean-absolute learned noise scale
$\bar\Sigma=\frac{1}{N}\sum_i|\sigma^w_i|$ over training, to see whether the agent drives its own
exploration up or down.

# Code framework

The deep-RL harness already provides the Atari wrappers, a replay buffer, the convolutional
torso, an optimiser, a target network, and the training loop. A standard linear layer
`nn.Linear` exists. The open slot is the linear-layer abstraction used in the heads — what, if
anything, replaces the plain affine map — and the consequent change to how actions are selected
(whether a separate exploration rule is still needed).

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExploringLinear(nn.Module):
    """Drop-in replacement for nn.Linear that is supposed to carry the
    agent's exploration. The internals are the open slot."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO: parameters of the layer

    def reset_parameters(self):
        pass  # TODO

    def resample(self):
        pass  # TODO: refresh whatever stochasticity the layer carries

    def forward(self, input):
        raise NotImplementedError  # TODO

class QNetwork(nn.Module):
    """Conv torso (frozen design) + heads built from the linear abstraction above."""
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = None  # TODO: built from ExploringLinear

    def forward(self, x):
        raise NotImplementedError  # TODO

def act(net, x):
    # how an action is chosen during data collection -- the exploration rule lives here
    raise NotImplementedError  # TODO

def loss_fn(online_net, target_net, batch, gamma):
    # TD loss; how/when stochasticity is (re)sampled for online vs target is the open slot
    raise NotImplementedError  # TODO
```
