# Research question

A reinforcement-learning agent improves its policy only when it receives a reward signal. In many
environments the extrinsic reward is *extremely sparse or absent* — a single $+1$ on reaching a
goal hundreds of steps away, and zero everywhere else — and no shaped reward can be hand-built.
Then the agent learns nothing until it *stumbles* into the goal, which random action noise will
essentially never do in any but the simplest environment. The question: can the agent generate
its own *intrinsic* reward that drives it to explore efficiently, learn useful skills, and
generalize — entirely from raw high-dimensional observations (images), with little or no extrinsic
reward? Such a signal would have to (i) scale to image state spaces, (ii) push the agent toward
states that genuinely teach it something about the consequences of its actions, and — the crux —
(iii) *not* be hijacked by parts of the environment that are unpredictable but irrelevant to the
agent, so that exploration does not stall on noise.

# Background

**Intrinsic motivation / curiosity.** When extrinsic reward is sparse, psychology's *intrinsic
motivation* (Ryan & Deci, 2000) — the drive of a child to explore a playground with no external
payoff — is the model. In RL, intrinsic-reward formulations fall into two broad families:
*novelty-seeking*, which rewards visiting rarely-seen states and requires a statistical/density
model of the state distribution (Bellemare et al., 2016, pseudo-counts; Lopes et al., 2012); and
*prediction-error / uncertainty-seeking*, which rewards actions whose consequences the agent
predicts poorly and therefore requires a *dynamics model* predicting $s_{t+1}$ from $s_t,a_t$
(Schmidhuber, 1991, 2010; Stadie et al., 2015; Houthooft et al., 2016, VIME). Both kinds of model
are hard to build in high-dimensional continuous (image) state spaces.

**The noisy-TV / artificial-curiosity trap.** The decisive diagnostic for prediction-error
curiosity: if the intrinsic reward is the *pixel-space* prediction error, the agent is permanently
rewarded by anything inherently unpredictable — white noise on a television, leaves moving in a
breeze, shadows, distractor objects, other agents whose motion is irrelevant to the agent's goals.
The pixel error there never decreases, so curiosity never moves on; the agent stalls, transfixed by
noise (Schmidhuber, 2010). Tabular novelty counts have the same pathology. A proposed remedy is to
reward only states that are *hard to predict but learnable* (Schmidhuber, 1991), but there is no
known computationally feasible way to estimate learnability (Lopes et al., 2012).

**Three sources of observation change.** Anything that changes the agent's observation is one of:
(1) things the agent *can control*; (2) things the agent *cannot* control but that *affect* it
(e.g. a vehicle driven by someone else); (3) things out of the agent's control *and* not affecting
it (moving leaves, TV static). A curiosity signal should respond to (1) and (2) and be blind to
(3) — the agent has no reason to care about variation that is inconsequential to it.

**Self-supervised feature learning from agent experience.** The agent's own
$(s_t,a_t,s_{t+1})$ tuples come labelled for free, which has been used as supervision to learn
visual features without external annotation — e.g. for recognition (Jayaraman & Grauman, 2015;
Agrawal et al., 2015) and for object pushing (Agrawal et al., 2016, where a joint inverse–forward
model used the forward model as a *regularizer* for features). Such self-supervised signals are
candidate ways to learn a representation of an image observation from interaction alone.

**Policy learning by actor-critic.** The asynchronous advantage actor-critic A3C (Mnih et al.,
2016) optimizes a policy $\pi(s;\theta_P)$ to maximize the expected return $\mathbb{E}[\sum_t r_t]$
via the policy gradient $\nabla\log\pi(a|s)\,A(s,a)$ with an advantage baseline and an entropy
bonus, trained on parallel asynchronous workers. It is a natural, scalable policy optimizer that
can consume any scalar reward, intrinsic or extrinsic.

# Baselines

**A3C with $\epsilon$-greedy / entropy exploration (Mnih et al., 2016).** The plain policy-gradient
agent: maximize $\mathbb{E}[\sum_t r_t^e]$ with the environment's extrinsic reward, exploring via
the stochastic policy and an entropy bonus. **Gap:** with sparse $r^e$ the gradient is almost
always zero; the entropy bonus produces only local dithering and cannot carry the agent across long
reward-free stretches.

**Pixel-prediction-error curiosity (Schmidhuber line; Stadie et al., 2015).** Augment the reward
with the error of a model predicting the next *observation* (pixels) from the current observation
and action. **Gap:** the noisy-TV trap — inherently unpredictable but irrelevant visual variation
keeps the error (and thus the bonus) permanently high, so the agent is rewarded for staring at
noise and stops making progress; and predicting pixels is itself hard and arguably the wrong
objective.

**Count/novelty-based intrinsic reward (Bellemare et al., 2016).** Reward visiting low-count
states via a density model over states. **Gap:** also responds to irrelevant stochastic variation
(every noisy frame looks novel), and needs a good state-density model in image space.

# Evaluation settings

Two visual environments. *VizDoom* 3-D navigation (`DoomMyWayHome-v0`): four discrete actions
(forward, left, right, no-op), a map of 9 rooms joined by corridors, a single sparse terminal
reward $+1$ for reaching the vest (zero otherwise), episodes capped at 2100 steps;
$\sim$350 steps for an optimal policy from the farthest room. Difficulty is varied by initial-goal
distance ("dense", "sparse", "very sparse" spawns), and a noise variant adds uncontrollable visual
noise to the input to probe the noisy-TV failure. *Super Mario Bros*: 14 composite joystick
actions, four levels, played with *no extrinsic reward at all* (curiosity only); long-range
dependencies (a long jump may require repeating an action up to 12 times). Generalization is tested
by pre-training on one level/map and measuring exploration speed on unseen levels/maps with new
textures. Inputs: RGB$\to$grayscale, $42\times42$, 4-frame stack; action repeat 4 (Doom) / 6
(Mario) at training, none at inference; A3C with 20 asynchronous workers, Adam (not shared across
workers).

# Code framework

An A3C agent already exists: a shared convolutional+recurrent torso producing a state feature, a
policy head and a value head, and the asynchronous policy-gradient training loop that maximizes a
scalar reward. The open slot is *the reward itself* when the environment's reward is sparse/zero:
what additional signal, if any, is added, and what machinery computes it from the agent's own
experience.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class A3CPolicy(nn.Module):
    """Shared conv (+ LSTM) torso; policy and value heads. Frozen design."""
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
        )
        self.lstm = nn.LSTMCell(288, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def forward(self, s, hidden):
        h = self.conv(s).flatten(1)
        h, c = self.lstm(h, hidden)
        return self.pi(h), self.v(h), (h, c)

class IntrinsicReward(nn.Module):
    """Produces an exploration bonus from the agent's own transitions.
    The internals -- what is predicted and in what space -- are the open slot."""
    def __init__(self, n_actions):
        super().__init__()
        # TODO

    def reward(self, s_t, a_t, s_tp1):
        raise NotImplementedError  # TODO: the intrinsic signal r^i

    def loss(self, s_t, a_t, s_tp1):
        raise NotImplementedError  # TODO: how the predictor(s) are trained

def total_reward(r_extrinsic, r_intrinsic):
    raise NotImplementedError  # TODO: how the two combine
```
