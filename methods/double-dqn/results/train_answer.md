The problem is the overestimation bias that appears when Q-learning bootstraps from a maximum over its own action-value estimates. DQN already uses a target network, but the target network both selects the next greedy action and evaluates that action. Because the maximum of a set of noisy estimates is, on average, larger than the maximum of their true values, the bootstrap target is systematically too high. The error is not uniform across states and actions, so it warps the relative action ordering that the greedy policy depends on and propagates backward through the temporal-difference updates.

Existing ideas address parts of the problem but do not remove the source of the bias inside the deep DQN framework. Tabular Double Q-learning decouples selection and evaluation by keeping two independent value tables, which shows that the bias can be eliminated in principle, but it is not obvious how to transfer that idea to a single deep network trained with experience replay and target-network updates. Simply adding a second full network would change the training pipeline, memory footprint, and update dynamics. What is needed is a minimal change that keeps the DQN architecture, replay buffer, optimizer, and target-copy schedule intact while making the online network select the next action and the target network evaluate it.

The method is Double DQN, also referred to as Deep Reinforcement Learning with Double Q-learning. It reuses the two weight sets that DQN already maintains: the online parameters theta and the periodically copied target parameters theta_minus. The only algorithmic change is the next-state target. Instead of taking the maximum over target-network values, Double DQN selects the next action using the online network and then asks the target network for the value of that selected action. The target becomes r + gamma Q(s', argmax_a Q(s', a; theta); theta_minus). When the two networks differ, the action was not chosen because the target network happened to overestimate it, so the positive floor created by self-evaluation disappears. Right after a target copy the two networks are identical, so the first target reverts briefly to ordinary DQN, but as soon as the online network takes gradient steps the decoupling returns. This design adds no new parameters, no extra forward pass at training time beyond computing the online Q-values that are already needed for the loss, and no change to exploration or replay.

The effect is most visible where the action space is large and approximation error is uneven. In states where the true action values are tied, a single estimator can be proved to overshoot by at least the root of the mean-squared spread divided by the number of actions minus one, and uniform errors give an expected overestimation proportional to the number of actions. Splitting selection and evaluation removes that forced lower bound. Empirically, the predicted value of the greedy policy moves closer to the discounted return actually obtained, and the normalized Atari scores improve over DQN without any hyperparameter tuning. A tuned variant can copy the target network less frequently to keep the selector and evaluator further apart, but the core method works with the standard DQN copy interval.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


def double_dqn_target(rewards, discounts, next_obs, online_net, target_net):
    """y = r + discount * Q_target(s', argmax_a Q_online(s', a))."""
    with torch.no_grad():
        selector_q = online_net(next_obs)
        next_actions = selector_q.argmax(dim=1, keepdim=True)
        evaluator_q = target_net(next_obs)
        next_q = evaluator_q.gather(1, next_actions).squeeze(1)
        return rewards + discounts * next_q


def clipped_td_loss(td_error, grad_error_bound=1.0):
    abs_error = td_error.abs()
    quadratic = torch.minimum(
        abs_error, torch.full_like(abs_error, grad_error_bound)
    )
    linear = abs_error - quadratic
    return (0.5 * quadratic.pow(2) + grad_error_bound * linear).mean()


def train_step(batch, gamma, online_net, target_net, optimizer):
    obs, actions, rewards, next_obs, dones = batch
    discounts = gamma * (1.0 - dones.float())
    target = double_dqn_target(rewards, discounts, next_obs, online_net, target_net)
    q_sa = online_net(obs).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    loss = clipped_td_loss(target - q_sa)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```

The training loop remains the standard DQN loop: act epsilon-greedily with the online network, store transitions in replay, sample minibatches, run the update above, and periodically copy the online weights into the target network. The only difference from DQN is that argmax is computed on the online network while the corresponding value is gathered from the target network. That single-line change is what turns a self-referential maximum into a decoupled selection-evaluation pair and removes the systematic overestimation bias.