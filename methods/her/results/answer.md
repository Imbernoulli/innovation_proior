# Hindsight Experience Replay (HER)

## Problem

Learn goal-reaching tasks from a **sparse, binary** reward — `r(s, a, g) = -[ f_g(s') = 0 ]`, i.e. `-1` until the goal is satisfied and `0` there — with no reward shaping and no domain knowledge. The difficulty is that an exploring agent essentially never reaches the goal, so it almost never sees a reward other than `-1`; the Bellman target is constant and value learning has nothing to bootstrap from. Reward shaping fixes the signal but reintroduces per-task engineering, can distort exploration into degenerate policies, and optimizes a proxy rather than the binary success metric.

## Key idea

A trajectory that *failed* to reach its goal `g` still *succeeded* at reaching whatever state it actually ended in. So **relabel** the transition: pretend an achieved goal `g'` (some state the trajectory actually reached) had been the target all along, and recompute the reward against `g'`. Against `g'` the reward is no longer stuck at `-1` — at least one transition per episode is rewarding — which gives the learner a usable, dense-enough signal from experience it already has.

Two ingredients make this legal:
- **Goal-conditioned value function (UVFA).** Learn `Q(s, a, g)` and `pi(s, g)` with the goal as an input. A single network generalizes value estimates across goals, so a transition is valid input under *any* goal we pair with it.
- **Off-policy learning.** Relabeling changes the reward and the effective goal the data is "about". Only off-policy algorithms (DQN, DDPG, NAF, SDQN) can learn from such data, because their Bellman target depends only on `(s, a, r_{g'}, s')` and the environment dynamics `p(s'|s,a)` are independent of the goal.

## Setup

Goal space `G`; each goal has a predicate `f_g: S -> {0,1}`; reward `r_g(s,a) = -[f_g(s')=0]`. A map `m: S -> G` with `f_{m(s)}(s) = 1` lets us name, for any reached state, a goal it satisfies (identity when `G=S`; `m(s)=s_object` for object-position goals `G=R^3`, `f_g(s)=[||g - s_object|| <= eps]`).

Goal-conditioned Bellman equation (bootstrapped exactly like DQN/DDPG):
```
Q*(s, a, g) = E_{s'}[ r_g(s, a) + gamma * max_{a'} Q*(s', a', g) ].
```
DDPG target with `g` held fixed across the backup: `y = r + gamma * Q'(s', pi'(s', g), g)`, clipped to the achievable range `[-1/(1-gamma), 0]`.

## Replay strategies S (which achieved goals to relabel with)

- **final** — the achieved goal of the last state, `m(s_T)`. Simplest.
- **future** — `k` achieved goals from states observed *after* the transition, in the same episode. Near-horizon achieved goals give tight, attributable credit and form an automatic easy-to-hard curriculum as the policy improves; the implementation commonly uses this strategy with `k = 4`.
- **episode** — `k` achieved goals from anywhere in the same episode.
- **random** — `k` achieved goals from anywhere in training so far.

`k` is the ratio of relabeled to original-goal replay. In the lazy sampler this becomes the relabeled fraction `future_p = 1 - 1/(1+k)`, leaving an original-goal fraction `1/(1+k)` so the real goal distribution is still represented. Very large `k` would make original-goal updates too rare.

## Algorithm

```
Given: off-policy algorithm A, replay strategy S, reward function r.
Initialize A and replay buffer R.
for episode = 1, M:
    sample a goal g and initial state s_0
    for t = 0 .. T-1:                     # roll out with behavioral pi(s_t || g)
        a_t = pi_b(s_t || g); execute; observe s_{t+1}
    for t = 0 .. T-1:
        store (s_t||g, a_t, r(s_t,a_t,g), s_{t+1}||g) in R    # standard replay
        for g' in S(current episode):                        # HER relabeling
            store (s_t||g', a_t, r(s_t,a_t,g'), s_{t+1}||g') in R
    for t = 1 .. N:
        sample minibatch B from R; one optimization step of A on B
```

## Code

The relabeling is done lazily at sampling time (store each episode once as full trajectories `{o, ag, g, u}` where `ag_t = m(s_t)`; relabel a `future_p` fraction of sampled transitions). This is the crux of the implementation:

```python
import numpy as np

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """HER transition sampler.
    replay_strategy: 'future' (HER) or 'none' (plain off-policy replay).
    replay_k: ratio of HER (relabeled) to regular replays.
    reward_fun(ag_2, g, info) -> binary reward against the (possibly new) goal.
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))   # relabeled:real = k:1
    else:
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        # u/g have T action steps; o/ag have T+1 states, so future_t may hit T.
        # 'ag' is the achieved-goal sequence m(s_t).
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # sample a random (episode, timestep) per transition
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # pick the transitions that get a hindsight (future) goal
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = (np.random.uniform(size=batch_size) * (T - t_samples)).astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # relabel: goal <- achieved goal at a future step of the same episode
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # recompute reward against the (possibly substituted) goal
        info = {k.replace('info_', ''): v for k, v in transitions.items()
                if k.startswith('info_')}
        transitions['r'] = reward_fun(ag_2=transitions['ag_2'],
                                      g=transitions['g'], info=info)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert transitions['u'].shape[0] == batch_size_in_transitions
        return transitions

    return _sample_her_transitions
```

The off-policy learner underneath is a goal-conditioned DDPG: actor `pi(s||g)` (MLP, ReLU, `tanh` output rescaled to the control range, with a small penalty on the preactivations to avoid `tanh` saturation), critic `Q(s, a, g)` (MLP, ReLU), target networks updated by polyak averaging, Adam, `gamma = 0.98`, critic targets clipped to `[-1/(1-gamma), 0]`, observations normalized to zero-mean/unit-variance and clipped to `[-5, 5]`, exploration via 20% uniform-random actions plus Gaussian noise on the rest.

## Why it works

HER converts every episode — including the failures, which under a sparse reward are almost all of them — into rewarding experience for *some* goal, breaking the constant-`-1` Bellman target. Because the relabel goals are states the agent actually reaches, they shift from easy to hard as the policy improves, acting as an implicit curriculum with no hand-design and no control over the initial-state distribution. It plugs into any off-policy RL algorithm and learns from sparse, binary rewards where shaping is unavailable or counterproductive.
