# Research question

Reinforcement learning maximizes expected return, which works when rewards are dense enough that a
stochastic policy stumbles into them, but collapses on the hard-exploration games — Montezuma's
Revenge, Pitfall!, Private Eye — where rewards are hundreds of steps apart and a plain agent on the
extrinsic reward almost never sees one. The standard remedy adds an intrinsic bonus $i_t$ to the
extrinsic reward $e_t$, $r_t=e_t+\beta i_t$, with $i_t$ large in *novel* states. Every novelty
bonus in use shares one structural property: it is a *transient* signal that **vanishes** as the
agent gathers experience. That is exactly what you want for pushing the frontier outward once, and
exactly what you do *not* want for *maintaining* exploration: after the bonus on a region has
decayed, nothing pulls the agent back through it, even when traversing it again is the only way to
reach the next undiscovered area. The precise problem: design an exploration bonus that (i) rapidly
discourages revisiting a state *within* the current episode, so the agent keeps covering new ground
every episode; (ii) gives extra emphasis to states that are still novel *across* episodes without
letting lifelong familiarity erase the within-episode drive; (iii) ignores parts of the observation
the agent's actions cannot influence; and — because baking a non-vanishing exploratory drive into a
single value function would permanently distort the policy — (iv) admits a way to *separate* the
exploratory policy from a clean exploitative one, so dense-reward games are not sacrificed for the
sake of the hard ones.

# Background

**Exploration bonuses and the vanishing-novelty limitation.** Augmenting sparse extrinsic reward
$e_t$ with an intrinsic $i_t$ that is large in novel states is the dominant approach
(Bellemare et al. 2016; Pathak et al. 2017; Burda et al. 2018). The shared failure mode is that as
the agent becomes familiar with the environment the bonus disappears and learning is driven by the
extrinsic reward alone — sensible if the extrinsic reward is now reachable, but on the hardest games
the agent must keep *re-traversing* already-seen regions to extend its frontier, and a vanished
bonus gives it no reason to. Matching the decay rate of the bonus to the learning speed so that
exploration neither dies too early nor wastes effort requires careful, brittle calibration
(Ostrovski et al. 2017; Ecoffet et al. 2019).

**Count-based bonuses.** In a tabular MDP a principled bonus is a decreasing function of the
visitation count $n_t(s)$, e.g. $1/\sqrt{n_t(s)}$ (Strehl & Littman 2008), with regret guarantees.
In high-dimensional image spaces almost every state is seen at most once, so raw counts are useless
and must be generalized — *pseudo-counts* derive surrogate counts from a learned density model
(Bellemare et al. 2016), giving "counts" to unseen-but-similar states, at the cost of a heavy
density model that is awkward to scale.

**Prediction-error / random-distillation bonuses.** A cheaper family sets $i_t$ to the error of a
model on a self-supervised prediction problem about the agent's transitions — forward dynamics
(Pathak et al. 2017) or distilling a fixed random target network (Burda et al. 2018, RND). RND in
particular uses a frozen random network $g:\mathcal{O}\to\mathbb{R}^k$ and a trained predictor
$\hat g$ minimizing $\|\hat g(x)-g(x)\|^2$; the error is high on observations unlike the training
data, so it marks *global, lifelong* novelty, and it is cheap (two forward passes) and trivially
parallel. But because the predictor is optimized by gradient descent over *all* of training, its
error changes only slowly and *monotonically decreases* on a region once visited — it is a
lifelong, not a within-episode, signal, and it too vanishes.

**The controllable-state representation (inverse dynamics).** A forward-prediction bonus computed
on raw pixels is captured by unpredictable-but-irrelevant variation (TV static, moving leaves):
pixels mix what the agent controls, what affects it, and irrelevant noise. Pathak et al. (2017)
showed how to learn a feature map $f$ that keeps only *action-relevant* content by a self-supervised
*inverse-dynamics* task: from two consecutive observations $x_t,x_{t+1}$, recover the action $a_t$
via $p(a\mid x_t,x_{t+1})=h(f(x_t),f(x_{t+1}))$ with a softmax classifier, trained by maximum
likelihood. To predict the action, $f$ must encode whatever the action changed and has no incentive
to encode noise the action cannot affect; it also cannot collapse to a constant (that would destroy
the action information). The resulting $f(x)$ is a *controllable state*. This is a representation
component, separable from how the novelty signal is then computed.

**Slot-based episodic memory and kernel pseudo-counts.** Differentiable episodic memories store
embeddings in slots and read them by nearest-neighbor lookup with a smooth kernel (Blundell et al.
2016; Pritzel et al. 2017, Neural Episodic Control), using the inverse kernel
$K(x,y)=\epsilon/(d^2(x,y)/d_m^2+\epsilon)$ with a per-task running distance scale $d_m^2$. Such a
memory can hold the embeddings seen *so far in one episode* and answer "how close is the current
state to anything I've seen this episode?" — the natural substrate for a *within-episode* count.

**Universal value functions (UVFA).** Schaul et al. (2015) approximate a whole family of value
functions $V(s;g)$ with one network conditioned on a goal/parameter $g$, sharing representation
across the family. Conditioning a value function on the *intrinsic-reward weight* would let a single
network represent both a purely exploratory and a purely exploitative policy and switch between
them — and the exploratory members act as auxiliary tasks (Jaderberg et al. 2016) that keep the
shared representation improving even with no extrinsic reward.

**Distributed recurrent value-based RL (R2D2).** Kapturowski et al. (2019) combine an LSTM state,
prioritized experience replay, off-policy $n$-step / Retrace value learning (Munos et al. 2016), and
many parallel actors feeding a central replay — the regime where exploration must operate at scale.
A subtlety: adding a *non-stationary* intrinsic reward to the reward stream can turn the MDP into a
POMDP; feeding the intrinsic reward (and the conditioning weight) to the network *as input* keeps
the problem Markovian from the agent's view.

**The Random Disco Maze diagnostic.** A $21\times21$ gridworld where the wall colors are re-sampled
randomly every step, so the agent almost never sees the same pixel-state twice. The setting isolates
the failure mode a novelty signal has to avoid: raw-observation novelty is overwhelmed by irrelevant
color churn, so a pixel-keyed bonus can pay the agent for standing still while the world changes
around it. A workable representation has to collapse the colors and preserve the agent's
action-relevant position; a workable novelty signal has to care about distinct positions covered
within the current episode.

# Baselines

**PPO / value-based RL on extrinsic reward only (Schulman et al. 2017; Mnih et al. 2015).** Explore
via a stochastic policy and entropy. **Gap:** under very sparse reward the advantage is almost
always zero; on the hardest games the agent finds no positive reward at all.

**Forward-dynamics curiosity in a controllable-state space (Pathak et al. 2017, ICM).** Bonus = the
forward-model error $\frac{\eta}{2}\|\hat f(s_{t+1})-f(s_{t+1})\|^2$ in an inverse-dynamics feature
space. **Gap:** noise-robust, but the bonus still *vanishes* as the forward model masters a
transition, so it maintains exploration only as long as there is unmastered controllable dynamics
nearby, and reduces to undirected exploration once that runs out.

**Random Network Distillation (Burda et al. 2018, RND).** Bonus = distillation error against a
frozen random target; cheap, scalable, a strong lifelong/global novelty detector. **Gap:** it is a
slow, *lifelong* signal — its error on a region only ever decreases across training, it has *no
within-episode memory*, so once global novelty has decayed the agent has no drive to re-traverse a
region, and (because a single value function trained on $e_t+\beta i_t$ permanently mixes in the
exploratory bias) it cannot be cleanly switched off to exploit.

**Pseudo-count exploration (Bellemare et al. 2016).** A density-model count bonus; strong on
Montezuma's Revenge. **Gap:** the density model is heavy and harder to scale, and the count is
lifelong — same vanishing-novelty issue.

# Evaluation settings

The hard-exploration subset of Atari (Bellemare et al. 2016): Gravitar, Montezuma's Revenge,
Pitfall!, Private Eye, Solaris, Venture — sparse-reward games where agents typically find little
positive reward. Standard DQN preprocessing: grayscale $84\times84$, max-and-skip 4, extrinsic
reward transformed/clipped. Exploration is measured by mean episodic return on the extrinsic reward
(performance is always reported on $e_t$ alone, never the augmented reward), and, on a diagnostic
gridworld (Random Disco Maze) run with no extrinsic reward, by the fraction of distinct positions
visited — to isolate the inductive bias of the bonus from any task reward. The training regime is
large-scale and distributed (many parallel actors, a central prioritized replay, billions of
frames). A method is judged by improving across *many* games, not by helping a single one.

# Code framework

The substrate is a fixed policy-optimization loop (rollouts, advantage estimation, a clipped/value
update over minibatches) into which an *intrinsic-bonus module* plugs through a fixed interface. The
module owns whatever networks, memories, and normalization statistics the bonus needs; the loop
calls it to compute a per-transition bonus, to normalize the rollout's intrinsic rewards, and to
compute its training loss, and calls a mixing function to combine extrinsic and intrinsic
advantages. What goes inside that module — what signal is novelty, whether there is any per-episode
state, how the bonus is shaped and combined — is the open slot.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# provided by the fixed loop: layer_init, RunningMeanStd, RewardForwardFilter, last_frame, Args


class IntrinsicBonusModule(nn.Module):
    """Computes a per-transition intrinsic bonus and is trained on its own objective.
    What the novelty signal is, and whether any per-episode memory is kept, is the open slot."""

    def __init__(self, action_dim: int, device: torch.device, args: "Args"):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args
        # TODO: networks / memory / normalization the bonus needs

    def initialize(self, envs) -> None:
        ...  # TODO: any bootstrap (e.g. observation-normalization stats from a random rollout)

    def trainable_parameters(self):
        raise NotImplementedError  # TODO

    def update_batch_stats(self, batch_obs, batch_next_obs) -> None:
        ...  # TODO: running-stat updates

    def compute_bonus(self, obs, next_obs, actions) -> torch.Tensor:
        raise NotImplementedError  # TODO: the intrinsic reward i_t (per transition)

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        raise NotImplementedError  # TODO: scale the intrinsic stream

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:
        raise NotImplementedError  # TODO: how (if at all) the module is trained


def mix_advantages(ext_advantages, int_advantages, args: "Args") -> torch.Tensor:
    raise NotImplementedError  # TODO: how the two advantage streams combine
```
