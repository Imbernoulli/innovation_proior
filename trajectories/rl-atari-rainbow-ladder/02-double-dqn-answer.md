# Double DQN — decouple selection from evaluation

**Problem.** DQN's target $R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta^-)$ uses one set of estimates to both
*select* the greedy next action and *evaluate* it. Because $\max$ is convex, any estimation error biases
the target *upward* (Jensen: $\mathbb{E}[\max_a Q(s',a)]\ge\max_a\mathbb{E}[Q(s',a)]$). The bias is
non-uniform across states, so it corrupts the relative action ordering the greedy policy reads off, and it
grows with the action count — broad, systematic damage on every game with more than one action.

**Key idea.** Decouple selection from evaluation, reusing the target network already in the floor as the
second estimator: the **online** net $\theta$ selects the greedy next action, the **target** net
$\theta^-$ scores it,
$$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big).$$
One line changed versus DQN — whose $\arg\max$ — no new parameters; replay, $\epsilon$-greedy, and the
target-sync rule are untouched.

**Why it works.** In an all-tied-true-value state with balanced errors of mean-squared spread $C$ over
$m$ actions, the single-estimator max overshoots by at least $\sqrt{C/(m-1)}$ (tight); an independent
evaluator drops that floor to $0$. The reuse is imperfect — $\theta^-$ is a stale copy of $\theta$
(correlated errors; right after a sync it reverts to plain Q-learning) — so it removes much, not all, of
the bias, most where the action count is largest.

**What stays open.** Only the target changed. The value object is still a scalar mean per action,
exploration is still $\epsilon$-greedy, replay is still uniform.

```python
# Double DQN target: online net selects, target net evaluates. The ONLY change from DQN is the argmax net.
# Code home: vwxyzjn/cleanrl (cleanrl/ddqn_atari.py); excerpted from methods/double-dqn/results/answer.md.
import torch


def double_dqn_target(rewards, discounts, next_obs, online_net, target_net):
    """y = r + discount * Q_target(s', argmax_a Q_online(s', a)).

    `discounts` is gamma for nonterminal transitions and 0 for terminal ones.
    """
    with torch.no_grad():
        selector_q = online_net(next_obs)
        next_actions = selector_q.argmax(dim=1, keepdim=True)        # SELECT with online theta
        evaluator_q = target_net(next_obs)
        next_q = evaluator_q.gather(1, next_actions).squeeze(1)      # EVALUATE with target theta-
        return rewards + discounts * next_q


def clipped_td_error_loss(td_error, grad_error_bound=1.0):
    """Huber/clipped-error loss matching the DQN clipped TD-error gradient (cross-suite stability)."""
    abs_error = td_error.abs()
    quadratic = torch.minimum(abs_error, torch.full_like(abs_error, grad_error_bound))
    linear = abs_error - quadratic
    return (0.5 * quadratic.pow(2) + grad_error_bound * linear).mean()


def double_dqn_loss(batch, gamma, online_net, target_net):
    obs, actions, rewards, next_obs, dones = batch
    discounts = gamma * (1.0 - dones.float())
    target = double_dqn_target(rewards, discounts, next_obs, online_net, target_net)
    q_sa = online_net(obs).gather(1, actions.long().view(-1, 1)).squeeze(1)
    return clipped_td_error_loss(target - q_sa)
```
