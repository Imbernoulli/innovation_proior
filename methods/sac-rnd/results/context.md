# Context

## Research question

Offline reinforcement learning learns a policy entirely from a fixed dataset `D` of transitions
collected by an unknown behavior policy `π_β`, with no further environment interaction. The defining
hazard is **out-of-distribution (OOD) action overestimation**: a bootstrapped critic, asked to value
an action the policy proposes but the dataset never contained, extrapolates — almost always upward —
and the policy, trained to maximize the critic, is pulled toward those inflated values and degrades,
with no fresh data to correct the mistake.

A natural way to express the remedy is *uncertainty*: if the agent had a reliable estimate of how
unfamiliar a given `(s, a)` is, it could penalize the value (and the policy) in proportion to that
unfamiliarity — the exact mirror image of the *intrinsic-exploration* bonus used online, where novelty
is rewarded. Online RL has a cheap, ensemble-free novelty estimator that works at scale: Random
Network Distillation. The question is whether RND's distillation error can serve as an anti-exploration
penalty for offline RL — a single, ensemble-free OOD-action detector that suppresses overestimation.

## Background

**Anti-exploration framing.** Online exploration adds an intrinsic bonus `+b(s,a)` to reward novel
states; offline, the same novelty signal is *subtracted* — `−b(s,a)` — to push the agent away from
unfamiliar actions. This duality (Rezaeifar et al. 2022) reframes behavior regularization as an
uncertainty penalty: the better the OOD detector `b`, the better the conservatism.

**Random Network Distillation (Burda et al. 2018).** Keep a fixed, randomly-initialized *target*
network `g(x)` and train a *predictor* `ĝ(x)` to match it by regression on observed inputs. On inputs
seen often the predictor matches the target (low error); on rarely- or never-seen inputs it has not
been trained and the error is large. The squared error `‖ĝ(x) − g(x)‖²` is therefore a cheap,
ensemble-free novelty signal. Online it drives exploration.

**Soft Actor-Critic (Haarnoja et al. 2018).** The off-policy continuous-control base: a Tanh-Gaussian
stochastic actor, twin critics with a `min` target (clipped double-Q, Fujimoto et al. 2018), and an
entropy term whose temperature `α` is auto-tuned to hold a target entropy (here `−dim(A)`). SAC is the
natural base for an action-conditioned penalty because its stochastic actor already samples actions to
evaluate.

**FiLM — Feature-wise Linear Modulation (Perez et al. 2018).** A conditioning mechanism: a *context*
input produces per-feature affine parameters `(γ, β)` that scale and shift the hidden activations of a
network processing a separate *feature* input, `h ← γ ⊙ h + β`. It is a more expressive way to inject
one input into another's computation than plain concatenation.

**Overestimation control baselines.** Clipped double-Q and Polyak target networks are carried in from
TD3/SAC. The alternative high-performance offline family is *ensembles* — SAC-N and EDAC (An et al.
2021) keep `N` critics (often 10–50) and use their spread as the uncertainty; very strong on D4RL.

## Baselines

**Policy-constraint methods — BCQ, BEAR, BRAC, TD3+BC, ReBRAC.** Keep the policy near `π_β` via
generative models, MMD/KL penalties, or behavior-cloning terms (Fujimoto et al. 2019; Kumar et al.
2019; Wu et al. 2019; Fujimoto & Gu 2021; Tarasov et al. 2023). ReBRAC is the strongest *ensemble-free*
member, a decoupled, LayerNorm-regularized TD3+BC.

**Value-regularization methods — CQL, Fisher-BRC.** Push `Q` down on OOD actions and up on dataset
actions (Kumar et al. 2020; Kostrikov et al. 2021).

**Expectile / one-step — IQL.** In-sample expectile value learning that never queries an OOD action
(Kostrikov et al. 2022).

**Ensemble pessimism — SAC-N, EDAC.** Many critics; disagreement = uncertainty (An et al. 2021).
State-of-the-art on D4RL locomotion; the cost of the uncertainty signal scales with `N`.

**Naive offline RND.** RND error can be fed directly as an anti-exploration penalty (Rezaeifar et al.
2022), with concatenated `[s, a]` conditioning.

## Evaluation settings

D4RL (Fu et al. 2020): MuJoCo Gym locomotion (halfcheetah / hopper / walker2d, in
medium / medium-replay / medium-expert / etc.), AntMaze, and Adroit. Performance is the normalized
score (0 = random, 100 = expert) averaged over evaluation rollouts and several seeds. The relevant
comparison is to ensemble-free methods (TD3+BC, IQL, CQL, ReBRAC) on Gym average, and to ensemble
methods (SAC-N, EDAC) on score-vs-cost. Ablations sweep the RND conditioning architecture
(concat / gated / bilinear / FiLM, at the first / last / all hidden layers) and the penalty coefficient.

## Code framework

The primitives that already exist: a Tanh-Gaussian SAC actor, a twin LayerNorm critic with a `min`
target and Polyak updates, automatic entropy tuning, Adam, and an offline loop sampling minibatches
from a static dataset. What does not yet exist is the RND module — a frozen target, a trained
predictor, the conditioning of the prior on the action, the running normalization of the bonus — and
the wiring of the resulting penalty into both the actor objective and the critic target.

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class TanhGaussianActor(nn.Module):
    """SAC policy: trunk -> (mu, log_sigma), tanh-squashed action + log-prob."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        # TODO: 3x256 trunk, mu head, log_sigma head (clip -5, 2)
        pass
    def sample(self, s):
        # TODO: rsample, tanh-squash, log-prob with the tanh correction
        pass


class Critic(nn.Module):
    """Q(s, a), 3x256 with LayerNorm."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = None  # TODO
    def forward(self, s, a):
        pass


class RND(nn.Module):
    """Frozen target + trained predictor over (s, a); error = OOD-action novelty."""
    def __init__(self, obs_dim, act_dim, embedding_dim=32):
        super().__init__()
        # TODO: predictor and target; condition the prior on the action
        pass
    def bonus(self, s, a):
        # TODO: ||predictor - target||^2, normalized by a running std
        pass


def anti_exploration_penalty(*args):
    # TODO: subtract beta * bonus from BOTH the actor loss and the critic target
    pass
```
