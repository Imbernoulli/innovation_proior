# Research question

Reinforcement learning maximizes expected return, which works when rewards are dense enough that
random action sequences stumble into them, but fails when rewards are sparse and far apart — the
canonical case is Montezuma's Revenge, where rewards can be hundreds of steps apart even under
optimal play and most RL agents never find a single positive reward. Such tasks need *directed*
exploration. At the same time, the strongest RL results come from *scale*: many copies of the
environment run in parallel, billions of frames of experience. So the exploration mechanism must
*scale cheaply* — many exploration methods (counts, pseudo-counts, information gain, prediction
gain) are awkward or expensive to run across thousands of parallel actors. The precise problem:
design an exploration bonus that is (i) trivial to implement, (ii) effective with high-dimensional
image observations, (iii) usable on top of any policy-optimization algorithm, (iv) cheap — ideally
a single forward pass of a network on a batch — and, crucially, (v) does *not* get trapped by the
parts of the environment that are unpredictable but irrelevant.

# Background

**Exploration bonuses.** A standard way to drive exploration under sparse environment reward
$e_t$ is to add an intrinsic bonus, replacing $e_t$ with $r_t=e_t+i_t$, where $i_t$ should be
large in *novel* states and small in frequently-visited ones.

**Count-based and pseudo-count bonuses.** In a tabular MDP, set $i_t$ to a decreasing function of
the visitation count $n_t(s)$ — e.g. $1/n_t(s)$ or $1/\sqrt{n_t(s)}$ (Strehl & Littman, 2008;
Bellemare et al., 2016). In high-dimensional spaces almost every state is visited at most once, so
counts must be generalized; *pseudo-counts* (Bellemare et al., 2016) derive surrogate counts from
changes in a learned state-density model, giving positive "counts" to unseen-but-similar states.
These density models are the part that is hard to scale.

**Prediction-error bonuses and the noisy-TV problem.** An alternative defines $i_t$ as the error
of a model solving a prediction problem about the agent's transitions — most commonly *forward
dynamics*, predicting the next observation (or its features) from the current observation and
action (Schmidhuber; Stadie et al., 2015; Pathak et al., 2017, ICM; Burda et al., 2018). Such
errors shrink as the agent gathers experience like the current transition, so high error marks
novelty. But there is a well-known failure: an agent maximizing forward-prediction error is
attracted to transitions whose answer is a *stochastic* function of the input — TV static, a coin
flip, randomly moving noise — because the error there is *irreducible* and never decays. The agent
gets stuck maximizing entropy it can never predict. This "noisy-TV" trap is the central obstacle
for prediction-error curiosity.

**Sources of prediction error.** A prediction error can come from (1) *too little training data*
near the input — epistemic uncertainty, which is the desirable novelty signal; (2) *stochasticity*
of the target function — aleatoric, the noisy-TV source; (3) *model misspecification* — the model
class cannot fit the target or lacks necessary inputs; (4) *learning dynamics* — the optimizer
fails to fit a target the model class contains. Only (1) is wanted; (2) and (3) are the harmful
ones that a curiosity bonus should not respond to.

**Prediction-improvement methods.** To dodge (2)/(3), prior work measures how much the predictor
*improves* upon seeing a new datapoint (learning progress) rather than the raw error (Schmidhuber,
1991; Oudeyer et al., 2007; Lopes et al., 2012). These are computationally expensive and hard to
scale to massively parallel training.

**Random features and randomized prior functions.** Features of randomly initialized neural
networks have long been studied (Rahimi & Recht, 2008; Saxe et al., 2011) and used for exploration
(Burda et al., 2018, who note that a forward model in a *random* feature space works about as well
as any other). Osband et al. (2018) propose *randomized prior functions* for uncertainty: train an
ensemble $g_\theta=f_\theta+f_{\theta^*}$ with $\theta^*$ from a prior and $\theta$ minimizing
$\mathbb{E}\|f_\theta(x_i)+f_{\theta^*}(x_i)-y_i\|^2+\mathcal{R}(\theta)$; the ensemble
approximates a posterior, and the spread of predictions estimates predictive uncertainty.

**Policy optimization (PPO).** Proximal Policy Optimization (Schulman et al., 2017) is a
clipped-surrogate policy-gradient method with a learned value baseline (GAE advantages), robust
and low-tuning, and it consumes any scalar reward. The return is *linear* in the rewards, so a
value function and advantage are additive across independent reward streams.

# Baselines

**PPO with extrinsic reward only (Schulman et al., 2017).** Optimize the clipped surrogate on the
environment reward alone, exploring via the stochastic policy and an entropy bonus. **Gap:** under
very sparse reward the advantage signal is almost always zero; on the hard-exploration Atari games
it often finds no positive reward at all.

**Forward-dynamics-error curiosity (Pathak et al., 2017; Burda et al., 2018).** Intrinsic bonus =
error of a model predicting the next observation's features from the current features and the
action. **Gap:** the noisy-TV trap — in stochastic or partially-observable environments (e.g. with
sticky actions making a room-crossing outcome random) the forward error is irreducibly high at
those transitions, so the agent oscillates there for the aleatoric reward instead of exploring.

**Pseudo-count exploration (Bellemare et al., 2016).** A density-model-derived count bonus on top
of DQN; the prior state of the art on Montezuma's Revenge (about half the rooms). **Gap:** relies
on a learned density model over images that is comparatively heavy and harder to scale to massive
parallelism.

**Prediction-improvement / learning-progress bonuses (Schmidhuber, 1991; Lopes et al., 2012).**
Reward the *gain* in prediction rather than the error, to sidestep aleatoric/misspecification
error. **Gap:** computationally expensive, hard to scale.

# Evaluation settings

The hard-exploration subset of Atari 2600 (Bellemare et al., 2016): Gravitar, Montezuma's Revenge,
Pitfall!, Private Eye, Solaris, Venture — games with sparse rewards where RL agents typically find
little or no positive reward. Montezuma's Revenge is the focus for ablations. Preprocessing:
grayscale, $84\times84$, max-and-skip 4, extrinsic reward clipped to $[-1,1]$, max 18K frames per
episode, *no* terminal-on-loss-of-life, *no* random starts, and **sticky actions** with
probability $0.25$ (following the recommendation to inject non-determinism so an agent cannot
simply memorize an action sequence) — this last point matters because it makes the environment
genuinely stochastic, which is exactly where naive prediction-error curiosity fails. Training is at
scale: rollouts of length 128, 128 parallel environments, $\sim$30K rollouts per environment
($\sim$1.97B frames). Exploration is measured by mean episodic return and by the number of distinct
rooms discovered (the latter even when training with no extrinsic reward at all, to isolate the
bonus's inductive bias). Comparisons use no expert demonstrations and no access to the emulator's
underlying RAM state.

# Code framework

A PPO agent exists: a convolutional policy/value network, GAE advantage estimation, and the
clipped-surrogate update over minibatches of rollout experience. The open slots are *the intrinsic
reward* (what extra signal is added when the environment reward is sparse, and what computes it),
*how many value heads/returns* the agent fits, and *what normalization* the inputs and rewards
receive.

```python
import torch
import torch.nn as nn

class PolicyValueNet(nn.Module):
    """Conv encoder (frozen design) + policy head + value head(s).
    How many value heads, and over which return streams, is an open slot."""
    def __init__(self, n_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
        )
        self.pi = nn.Linear(512, n_actions)
        self.value = None  # TODO: value head(s)

    def forward(self, x):
        raise NotImplementedError  # TODO

class IntrinsicBonus(nn.Module):
    """Produces a per-transition novelty bonus from observations.
    The prediction problem it solves is the open slot."""
    def __init__(self):
        super().__init__()
        # TODO

    def bonus(self, obs):
        raise NotImplementedError  # TODO: the intrinsic reward i_t

    def loss(self, obs):
        raise NotImplementedError  # TODO: how (if at all) it is trained

def normalize_observation(x, stats):
    raise NotImplementedError  # TODO: what normalization, for which networks

def combine_rewards_and_advantages(ext, intr):
    raise NotImplementedError  # TODO: how the streams combine
```
