Offline reinforcement learning tries to learn a good policy from a fixed dataset of transitions without any further interaction with the environment. The fundamental difficulty is distributional shift on actions: standard off-policy algorithms like Q-learning or actor-critic fit a Q-function to a Bellman target that queries Q at next-state actions chosen by the policy being learned, but those actions lie outside the data distribution. Because the Q-function is only trained on actions taken by the logging policy, its values at these out-of-distribution actions are unconstrained, and since the policy is trained to maximize Q, it drifts toward actions that happen to have spuriously high values. Without online rollouts to correct the mistake, these inflated values enter the bootstrap target and compound until the estimates diverge.

Earlier methods attempt to prevent this by constraining the learned policy to stay close to the behavior policy, using generative models of the behavior policy, support constraints, or explicit divergence penalties. These approaches have two weaknesses. First, they require estimating the behavior policy, which is awkward when data comes from mixed or unknown sources. Second, and more importantly, they constrain the policy but leave the Q-function itself unregularized. Neural function approximation couples values across nearby state-action pairs, so even a constrained policy can still see the Q-function inflate at out-of-distribution actions. The real source of the instability is in the critic, not the actor.

The method I propose is Conservative Q-Learning, abbreviated CQL. The central idea is to stop trying to fence in the policy and instead make the Q-function conservative: learn a value estimate that lower-bounds the true policy value, so that optimizing against it cannot be fooled by hallucinated out-of-distribution actions. CQL adds a regularizer to the standard Bellman loss that pushes down Q under a distribution over actions that the Q-function currently finds attractive, while pushing up Q under the data distribution. The combined effect expands the gap between in-distribution and out-of-distribution actions and makes the learned value pessimistic where data is scarce.

In the simplest form, one adds a penalty alpha times the expected Q under a chosen action distribution mu, which in the tabular case yields an iterate that is the Bellman target minus a non-negative correction, giving a pointwise lower bound. A tighter and more useful bound is obtained by also adding a bonus for Q under the behavior policy, since the algorithm only needs the policy value, not every individual Q-value, to be a lower bound. With mu set to the current policy, the per-iteration value correction is alpha times the non-negative quantity D_CQL equal to the sum over actions of pi times pi over pi_beta minus one, which simplifies to a sum of squared deviations from the behavior policy divided by pi_beta. This shows the correction is always non-negative and zero only when the learned policy equals the behavior policy.

The practical CQL objective, called CQL(H), chooses mu adaptively by maximizing the expected Q under mu plus an entropy regularizer. The inner optimization has a closed-form solution: mu is a Boltzmann distribution proportional to exp Q, and the resulting term collapses to log of the sum of exponentials of Q-values minus the expected Q under the data distribution. This gives a clean loss: alpha times the average over dataset states of log-sum-exp_a Q(s,a) minus Q(s, a_data), added to the usual temporal-difference error. The log-sum-exp naturally targets whichever action currently has the highest Q, pushing down hardest on the most overestimated action, while the data term keeps up the values of actions that actually appeared in the data. No behavior-policy model is needed.

The code below implements the continuous-control version of CQL on top of a standard SAC agent. Only the critic loss changes. It keeps the twin-Q temporal-difference loss and adds the CQL regularizer estimated by importance sampling over random uniform actions, current-policy actions at the observed state, and current-policy actions at the next state. A Lagrange variant can adapt the conservative coefficient automatically. The actor uses the standard SAC objective but should be updated with a much smaller learning rate than the critic, because the theoretical guarantee that the value remains a lower bound relies on the policy changing slowly.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self):
        return self.value

class ContinuousCQL:
    """SAC critic with the CQL(H) regularizer added to the critic loss."""

    def __init__(self, actor, critic_1, critic_2, target_critic_1, target_critic_2,
                 discount=0.99, cql_n_actions=10, cql_temp=1.0, cql_alpha=5.0,
                 cql_lagrange=False, cql_target_action_gap=-1.0,
                 cql_importance_sample=True, cql_clip_diff_min=-np.inf,
                 cql_clip_diff_max=np.inf, backup_entropy=False, qf_lr=3e-4):
        self.actor = actor
        self.critic_1, self.critic_2 = critic_1, critic_2
        self.target_critic_1, self.target_critic_2 = target_critic_1, target_critic_2
        self.discount, self.backup_entropy = discount, backup_entropy
        self.cql_n_actions, self.cql_temp, self.cql_alpha = cql_n_actions, cql_temp, cql_alpha
        self.cql_lagrange, self.cql_target_action_gap = cql_lagrange, cql_target_action_gap
        self.cql_importance_sample = cql_importance_sample
        self.cql_clip_diff_min, self.cql_clip_diff_max = cql_clip_diff_min, cql_clip_diff_max
        if cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=qf_lr)

    def _critic_regularizer(self, obs, actions, next_obs, q1_data, q2_data):
        B, action_dim = actions.shape[0], actions.shape[-1]
        rand_actions = actions.new_empty((B, self.cql_n_actions, action_dim)).uniform_(-1, 1)
        current_actions, current_logp = self.actor(obs, repeat=self.cql_n_actions)
        next_actions, next_logp = self.actor(next_obs, repeat=self.cql_n_actions)
        current_actions, current_logp = current_actions.detach(), current_logp.detach()
        next_actions, next_logp = next_actions.detach(), next_logp.detach()

        q1_rand = self.critic_1(obs, rand_actions)
        q2_rand = self.critic_2(obs, rand_actions)
        q1_current = self.critic_1(obs, current_actions)
        q2_current = self.critic_2(obs, current_actions)
        q1_next = self.critic_1(obs, next_actions)
        q2_next = self.critic_2(obs, next_actions)

        cat_q1 = torch.cat([q1_rand, q1_next, q1_current], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_next, q2_current], dim=1)
        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cat_q1 = torch.cat([q1_rand - random_density,
                                q1_next - next_logp,
                                q1_current - current_logp], dim=1)
            cat_q2 = torch.cat([q2_rand - random_density,
                                q2_next - next_logp,
                                q2_current - current_logp], dim=1)

        q1_ood = torch.logsumexp(cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        q2_ood = torch.logsumexp(cat_q2 / self.cql_temp, dim=1) * self.cql_temp
        q1_diff = torch.clamp(q1_ood - q1_data, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        q2_diff = torch.clamp(q2_ood - q2_data, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime().exp(), min=0.0, max=1e6)
            min_q1 = alpha_prime * self.cql_alpha * (q1_diff - self.cql_target_action_gap)
            min_q2 = alpha_prime * self.cql_alpha * (q2_diff - self.cql_target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            (-(min_q1 + min_q2) * 0.5).backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            min_q1 = self.cql_alpha * q1_diff
            min_q2 = self.cql_alpha * q2_diff
        return min_q1 + min_q2

    def _critic_loss(self, obs, actions, next_obs, rewards, dones, ent_alpha):
        q1 = self.critic_1(obs, actions)
        q2 = self.critic_2(obs, actions)
        next_a, next_logp = self.actor(next_obs)
        target_q = torch.min(self.target_critic_1(next_obs, next_a),
                             self.target_critic_2(next_obs, next_a))
        if self.backup_entropy:
            target_q = target_q - ent_alpha * next_logp
        td_target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * self.discount * target_q.detach()
        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        return td_loss + self._critic_regularizer(obs, actions, next_obs, q1, q2)

    def _actor_loss(self, obs, ent_alpha):
        a, logp = self.actor(obs)
        q = torch.min(self.critic_1(obs, a), self.critic_2(obs, a))
        return (ent_alpha * logp - q).mean()
```
