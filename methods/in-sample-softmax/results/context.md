## Research question

We have a fixed offline dataset `D = {(s_i, a_i, r_i, s'_i)}` collected by an
unknown behavior policy `pi_D`, and there is no further environment interaction.
The goal is to learn the best policy that the data can justify, even when the
data came from a good controller, a mediocre controller, or a mixture of
operators.

The central difficulty is the interaction between temporal-difference
bootstrapping and poor action coverage. A fitted-Q target such as

```text
r(s,a) + gamma max_{a'} q(s',a')
```

can select an action that never appears in the batch at `s'`. The value of that
action is then an extrapolation of the function approximator rather than a
quantity trained by data. If it is overestimated, the bootstrap target pushes
nearby values upward, and the effect can feed back through later updates. An
offline actor-critic has the same failure mode in a softer form: if the actor
drifts toward high predicted values in uncovered parts of action space, the
critic can start evaluating and backing up those unsupported actions.

A useful solution should keep the backup and policy extraction tied to actions
the dataset can support, while still allowing improvement over a suboptimal
behavior policy. Staying exactly behavior-cloned is safe but often too weak;
free maximization is strong but unsafe.

## Background

**Finite MDPs and Bellman updates.** An MDP has states `S`, actions `A`,
transition kernel `P`, reward `r`, and discount `gamma < 1`. For policy `pi`,

```text
v^pi(s) = E_pi[sum_t gamma^t r(s_t,a_t) | s_0=s]
q^pi(s,a) = r(s,a) + gamma E_{s'}[v^pi(s')]
```

The optimal action-value function satisfies

```text
q*(s,a) = r(s,a) + gamma E_{s'}[max_{a'} q*(s',a')].
```

That last maximization is harmless in online RL with exploration, but in offline
RL it can query state-action pairs that have no data support.

**Entropy-regularized RL and SAC.** Maximum-entropy RL augments reward by
discounted entropy. For a temperature `tau > 0`, the one-state identity is

```text
max_{p in Delta(A)} sum_a p(a) q(a) + tau H(p)
  = tau log sum_a exp(q(a)/tau),
```

with maximizer `p(a) proportional exp(q(a)/tau)`. SAC turns this into an
actor-critic with a stochastic actor, a double-Q critic, and a soft value target
`q(s,a) - tau log pi(a|s)`. The log-sum-exp form is the soft Bellman optimality
operator. It should not be confused with the different Boltzmann-policy Bellman
operator that backs up an expectation under `softmax(q)`.

**Offline overestimation.** Prior offline-RL work documents that fitted
Q-learning on a fixed batch can severely overestimate values when action
coverage is incomplete. The failure is clearest in small gridworld diagnostics:
optimistic initialization plus a narrow dataset leaves many actions uncorrected,
and the `max` keeps selecting them. The same logic carries to continuous control,
where a neural critic can assign high values to actions never sampled in the
batch.

**Support and behavior density.** Let `pi_D(a|s)` denote the conditional action
distribution behind the dataset. In tabular settings, support can be counted.
In large or continuous settings, only samples are available, so algorithms often
learn a behavior model `pi_omega(a|s)` by maximum likelihood. Such a model is
good for scoring dataset actions, but it is hard to make its generated samples
have exactly the same support as the true data distribution.

## Baselines

**Behavior-regularized policy improvement.** AWR, AWAC, one-step policy
improvement methods, and TD3+BC keep the learned policy close to the behavior
policy. A typical one-state objective is

```text
max_pi E_{a~pi}[q(a)] - tau KL(pi || pi_D),
```

whose closed form is `pi'(a) proportional pi_D(a) exp(q(a)/tau)`. This is safe
where `pi_D(a)=0`, but it also preserves the behavior distribution's shape. If
the batch is skewed toward mediocre actions, the learned policy remains biased
toward them.

**Conservative value estimation.** CQL, Fisher-BRC, ensemble penalties, and
related pessimistic methods push down values of unsupported or uncertain
actions. They reduce the damage caused by erroneous maximization, but they still
require tuning how conservative to be. Too little pessimism lets overestimation
return; too much suppresses useful improvement.

**Support-constrained hard maximization.** BCQ expresses the ideal hard backup
as a maximum over supported actions, then approximates the candidate set with a
learned generative model. This removes unsupported actions only to the extent
that the generator's support is accurate. In continuous spaces, a slightly leaky
generator can still provide actions the dataset never really supports.

**Expectile-based value learning.** IQL avoids generated actions by fitting a
high expectile of `q(s,a)` over dataset actions, then learning `q` from
`r + gamma v(s')` and extracting a policy by advantage-weighted regression. It
keeps the critic's target tied to dataset actions, but the expectile depends on
the behavior action distribution, not only on which actions are possible. It is
also less algebraically transparent than a closed-form Bellman operator.

## Evaluation settings

The natural checks for this problem separate action coverage from ordinary
function-approximation noise.

```text
Tabular diagnostic:
  - four-room gridworld, sparse goal reward, gamma around 0.9
  - datasets with narrow expert coverage, broad random coverage, mixed coverage,
    and deliberately missing actions
  - optimistic value initialization to expose unsupported-action overestimation

Continuous control:
  - D4RL MuJoCo tasks such as HalfCheetah, Hopper, Walker2d, and Ant
  - expert, medium-expert, medium, and medium-replay datasets
  - fixed offline training budget and normalized rollout scores

Discrete control:
  - Acrobot, Lunar Lander, and Mountain Car
  - expert and mixed datasets from a trained controller

Offline-to-online:
  - initialize from the offline learner
  - keep the offline data in replay while appending new online transitions
```

These settings specify where action coverage, behavior skew, and fine-tuning
behavior can be observed.

## Code framework

The pieces below are already available: a replayed transition batch, MLP bodies,
a stochastic actor, a double-Q critic, a scalar value network, a behavior policy
model trained by likelihood, Adam optimizers, and Polyak target networks. The
open slots are the offline actor target, the value target, and the critic
bootstrap.

```python
import torch
import torch.nn as nn

class MLPBody(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256)):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(din, dout), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x):
        return self.net(x)

class StochasticPolicy(nn.Module):
    def forward(self, s):
        """Return sampled action and log pi(a|s)."""
        raise NotImplementedError

    def get_logprob(self, s, a):
        """Score supplied actions under the policy."""
        raise NotImplementedError

class DoubleCritic(nn.Module):
    def forward(self, s, a):
        """Return q1(s,a), q2(s,a)."""
        raise NotImplementedError

    def min(self, s, a):
        q1, q2 = self.forward(s, a)
        return torch.minimum(q1, q2).squeeze(-1)

class ValueNet(nn.Module):
    def forward(self, s):
        raise NotImplementedError

def polyak_update(target, source, polyak=0.995):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(polyak)
            p_t.data.add_((1.0 - polyak) * p.data)

class OfflineSoftActorCritic:
    def __init__(self, actor, critic, target_critic, value, behavior,
                 tau, gamma=0.99, lr=3e-4):
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.value = value
        self.behavior = behavior
        self.tau = tau
        self.gamma = gamma
        self.opt_actor = torch.optim.Adam(actor.parameters(), lr)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr)
        self.opt_value = torch.optim.Adam(value.parameters(), lr)
        self.opt_behavior = torch.optim.Adam(behavior.parameters(), lr)

    def loss_behavior(self, s, a):
        return -self.behavior.get_logprob(s, a).mean()

    def loss_value(self, s):
        # TODO: define a soft value target that is compatible with offline data.
        raise NotImplementedError

    def loss_critic(self, s, a, r, s2, done):
        # TODO: define a TD target that avoids a free max over unsupported actions.
        raise NotImplementedError

    def loss_actor(self, s, a):
        # TODO: extract an improved stochastic actor without sampling unsupported actions.
        raise NotImplementedError
```
