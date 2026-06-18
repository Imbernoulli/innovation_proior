# Double DQN

## Problem

DQN uses the target
$$
Y_t^{\mathrm{DQN}} =
R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta_t^-)
=R_{t+1}+\gamma Q\left(S_{t+1},
\arg\max_a Q(S_{t+1},a;\theta_t^-);\theta_t^-\right).
$$
The target-network values both select the next greedy action and evaluate it.
With estimation error this max tends to select positively mistaken values, so
the target is biased upward. The bias is non-uniform, so it can alter the
greedy policy rather than merely shift all values.

## Key Idea

Separate action selection from action evaluation while leaving the rest of DQN
unchanged. Use the online network to select the next action and the target
network to evaluate that selected action:
$$
Y_t^{\mathrm{DoubleDQN}} =
R_{t+1}+\gamma Q\left(S_{t+1},
\arg\max_a Q(S_{t+1},a;\theta_t);\theta_t^-\right).
$$
This reuses the target network already present in DQN. It adds no new value
network and does not change replay, acting, the target-copy rule, or the DQN
optimizer/loss machinery.

## Math Check

If all true action values in a state are tied at $V_*(s)$, the estimates are
balanced,
$$
\sum_a(Q_t(s,a)-V_*(s))=0,
$$
and their mean squared error over $m\ge2$ actions is $C>0$, then
$$
\max_a Q_t(s,a)\ge V_*(s)+\sqrt{\frac{C}{m-1}}.
$$
The bound is tight at
$$
\epsilon_a=\sqrt{\frac{C}{m-1}}\quad(a=1,\ldots,m-1),\qquad
\epsilon_m=-\sqrt{(m-1)C}.
$$
For i.i.d. uniform errors in $[-1,1]$,
$$
\mathbb E[\max_a\epsilon_a]=\frac{m-1}{m+1},
$$
so the discounted uniform-error overestimation term is
$\gamma\epsilon(m-1)/(m+1)$ for errors scaled to
$[-\epsilon,\epsilon]$.

## Algorithm

For each replay transition $(S_t,A_t,R_{t+1},S_{t+1})$:

1. Select the next action with the online network:
   $$a^*=\arg\max_a Q(S_{t+1},a;\theta_t).$$
2. Evaluate that action with the target network:
   $$v=Q(S_{t+1},a^*;\theta_t^-).$$
3. Form the one-step target:
   $$y=R_{t+1}+\gamma_t v,$$
   where $\gamma_t=0$ on terminal transitions and $\gamma_t=\gamma$ otherwise.
4. Update $\theta_t$ from the TD error
   $$\delta_t=y-Q(S_t,A_t;\theta_t).$$
5. Copy $\theta^-\leftarrow\theta$ on the same schedule as DQN.

The untuned Atari condition keeps $\tau=10{,}000$ target-copy steps. The tuned
condition increases this to $30{,}000$ and also changes exploration and the final
layer bias, so those are tuning choices rather than the core method.

## Code

The faithful implementation is the target/loss slot. The rest of the DQN
training loop remains unchanged.

```python
import torch


def double_dqn_target(rewards, discounts, next_obs, online_net, target_net):
    """Return y = r + discount * Q_target(s', argmax_a Q_online(s', a)).

    `discounts` is gamma for nonterminal transitions and 0 for terminal ones.
    """
    with torch.no_grad():
        selector_q = online_net(next_obs)
        next_actions = selector_q.argmax(dim=1, keepdim=True)
        evaluator_q = target_net(next_obs)
        next_q = evaluator_q.gather(1, next_actions).squeeze(1)
        return rewards + discounts * next_q


def clipped_td_error_loss(td_error, grad_error_bound=1.0):
    """Huber/clipped-error loss matching the DQN clipped TD-error gradient."""
    abs_error = td_error.abs()
    quadratic = torch.minimum(
        abs_error, torch.full_like(abs_error, grad_error_bound)
    )
    linear = abs_error - quadratic
    return (0.5 * quadratic.pow(2) + grad_error_bound * linear).mean()


def double_dqn_loss(batch, gamma, online_net, target_net):
    obs, actions, rewards, next_obs, dones = batch
    discounts = gamma * (1.0 - dones.float())
    target = double_dqn_target(rewards, discounts, next_obs, online_net, target_net)
    q_sa = online_net(obs).gather(1, actions.long().view(-1, 1)).squeeze(1)
    td_error = target - q_sa
    return clipped_td_error_loss(td_error)
```

Compared with DQN, the only target-side change is that the `argmax` is computed
from `online_net(next_obs)` while the selected value is gathered from
`target_net(next_obs)`.
