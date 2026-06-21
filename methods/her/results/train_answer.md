We want an agent that learns to reach a goal from a reward that is honest and cheap to specify: minus one on every step until the goal is satisfied, and zero once it is. This binary reward is the metric we actually care about and it asks for no domain knowledge, but in any reasonably large state space it is nearly useless for learning. A randomly exploring agent almost never lands exactly on the goal, so almost every transition it observes carries reward minus one. The Bellman target becomes a constant, the value regression has nothing to distinguish good actions from bad ones, and no gradient points toward the goal. Better exploration does not solve the problem: the agent is not stuck revisiting states, it is stuck receiving no reward signal for the states it does visit. Reward shaping replaces the sparse signal with a smooth distance-to-goal surrogate, which restores learning but reintroduces per-task engineering, can suppress useful behavior by punishing movement in the wrong direction, and optimizes a proxy rather than the true success criterion. What we need is a way to learn from the sparse binary reward itself.

The method is Hindsight Experience Replay, or HER. The core observation is that a trajectory which fails to reach its intended goal still succeeds at reaching wherever it actually ended up. If the agent was trying to push a puck to the left net and it ended up on the right, then the very same trajectory is a perfect demonstration of pushing the puck to the right. HER exploits this by relabeling transitions: after an episode is collected, it pretends that an achieved state from the trajectory had been the target goal all along, and recomputes the reward against that achieved goal. Against the relabeled goal, the final transition of the episode is rewarding by construction, so the constant-minus-one curse is broken without changing the environment, without adding dense shaping, and without hand-designed curricula.

This relabeling is possible because the value function is goal-conditioned. Instead of learning a single Q function, HER learns a universal value function approximator that takes both the state and the goal as inputs, Q of s, a, g, and a policy pi of s, g that is also conditioned on the goal. The same network shares parameters across all goals, so value estimates generalize across the goal space. When a transition is relabeled with a new goal, the reward is recomputed consistently and the Bellman backup uses that same goal throughout. The dynamics of the environment do not depend on the goal, only the reward does, so the relabeled transition remains a valid one-step sample. This means HER must sit on top of an off-policy learner such as DQN or DDPG, whose target depends on the transition tuple and the target policy rather than on which behavioral policy generated the data.

The most effective relabeling strategy is called future. Rather than only relabeling each transition with the final achieved goal of the episode, future samples an achieved goal from some state that occurs after the transition in the same episode. This has two advantages. First, it provides tight, near-horizon credit: the relabeled goal is reachable from the current state in just a few steps, which gives the value function an attributable learning signal. Second, because the relabeled goals are states the agent actually reaches, they automatically form an easy-to-hard curriculum. Early in training the policy reaches only nearby states, so the hindsight goals are easy. As the policy improves, the achieved states spread farther, and the hindsight goals become harder. This curriculum emerges for free from the data, with no ordering of tasks and no privileged knowledge of difficulty.

To keep the real objective from being drowned out, only a fraction of sampled transitions are relabeled. If the ratio of relabeled to original-goal replay is k to one, then the relabeled fraction is k divided by one plus k. A common choice is k equals four, meaning eighty percent of sampled transitions use a hindsight goal and twenty percent keep the original goal. The relabeling is done lazily at sampling time: the buffer stores each episode once as full trajectories containing observations, achieved goals, desired goals, and actions, and the sampler swaps in a future achieved goal on the fly before recomputing the reward. This saves memory and makes the strategy easy to change.

The off-policy learner underneath is a goal-conditioned DDPG. The actor maps a concatenation of state and goal to an action through an MLP with ReLU activations and a tanh output rescaled to the control range. The critic maps a concatenation of state, goal, and action to a scalar Q value. Target networks are updated by polyak averaging. The critic target is clipped to the achievable range, since the binary reward lives in minus one to zero. Observations are normalized to zero mean and unit variance and clipped. The behavioral policy mixes in uniform random actions and adds Gaussian noise to encourage exploration, because hindsight relabeling can only teach the agent about states it already reaches.

```python
import numpy as np

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Build a transition sampler for Hindsight Experience Replay.

    replay_strategy: 'future' enables HER relabeling, 'none' disables it.
    replay_k: ratio of relabeled transitions to original-goal transitions.
    reward_fun(ag_2, g, info) -> sparse binary reward for the chosen goal.
    """
    if replay_strategy == 'future':
        future_p = 1.0 - (1.0 / (1.0 + replay_k))
    else:
        future_p = 0.0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        # episode_batch keys:
        #   'o'  : observations, shape (num_episodes, T+1, obs_dim)
        #   'ag' : achieved goals, shape (num_episodes, T+1, goal_dim)
        #   'g'  : desired goals,  shape (num_episodes, T,   goal_dim)
        #   'u'  : actions,        shape (num_episodes, T,   action_dim)
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Sample one (episode, timestep) pair per transition in the minibatch.
        episode_idxs = np.random.randint(0, rollout_batch_size, size=batch_size)
        t_samples = np.random.randint(0, T, size=batch_size)
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }

        # Decide which transitions get a hindsight goal.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)[0]

        # For those, pick a future timestep strictly after t in the same episode.
        future_offset = (
            np.random.uniform(size=batch_size) * (T - t_samples)
        ).astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Relabel: replace the desired goal with the achieved goal at future_t.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Recompute the reward against the possibly relabeled goal.
        info = {
            k.replace('info_', ''): v
            for k, v in transitions.items()
            if k.startswith('info_')
        }
        transitions['r'] = reward_fun(
            ag_2=transitions['ag_2'],
            g=transitions['g'],
            info=info,
        )

        # Flatten the leading batch dimension for training.
        transitions = {
            k: v.reshape(batch_size, *v.shape[1:])
            for k, v in transitions.items()
        }
        return transitions

    return _sample_her_transitions
```

HER turns the central weakness of sparse reward learning into a strength. Every episode, however badly it fails at the original task, becomes a source of informative supervision for some achievable goal. By learning a universal goal-conditioned value function and replaying experience with hindsight goals, the agent can bootstrap from sparse binary rewards and solve continuous-control manipulation tasks without reward shaping or domain-specific engineering.
