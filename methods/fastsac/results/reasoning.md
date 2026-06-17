Let me start from what actually hurts. I want to train continuous-control policies — torques
for a simulated humanoid, say — and I have two families of methods, each broken in a different
way. The on-policy policy gradients, TRPO and PPO, are stable and I can usually get them to
work, but they are ravenous: every gradient step wants fresh trajectories, and once I've used
a batch I throw it away. On the hardest tasks, the 21-dimensional Humanoid, PPO needs gigantic
batches just to stay stable, so the sample bill is enormous. The off-policy side, DDPG, reuses
a replay buffer and so is far thriftier with data, but it's a nightmare to stabilize: I tune
the learning rates, the exploration noise schedule, the target-update rate, and it still
diverges or stalls, and on Humanoid it often makes no progress at all. So I have efficiency
without stability, or stability without efficiency, and I want both.

Let me understand *why* DDPG is brittle, because that's the thing to fix. DDPG has a
deterministic actor mu_phi(s) and a critic Q_theta(s,a). The critic learns the usual Bellman
residual with a target net, and the actor is pushed straight uphill on the critic by the
deterministic policy gradient, grad_phi Q(s, mu_phi(s)). So the actor's entire job is to find
the argmax of the current critic. Now, the critic is learned from bootstrapped targets, and
bootstrapped value estimates overestimate — this is old (Thrun & Schwartz), and it's been
shown to carry over to continuous actor-critics: the target involves a max-like operation over
a noisy Q, noise plus max gives a consistent upward bias, and the Bellman backup propagates
that bias forward. A deterministic actor whose only instruction is "maximize Q" will happily
walk into wherever Q is most overestimated. That's the instability: actor and critic feed each
other's errors. And on top of that, a deterministic policy has no exploration of its own, so I
have to inject and schedule an external noise process, one more fragile knob.

TD3 patches the overestimation cleverly — keep two critics, train them independently, and
bootstrap from the *minimum* of the two. The intuition: a value that's overestimated in one
network is unlikely to also be the smaller of the two, so the min systematically prefers
underestimation, and underestimation, unlike overestimation, doesn't get chased and
amplified. It also delays the actor updates and smooths the target action with noise. Good —
that fixes the value side. But the policy is still deterministic, and the exploration is still
an external noise process bolted on; the stochasticity TD3 uses is a regularizer on the
*target*, not something the policy itself wants. So the "exploration is a separate brittle
subsystem" problem is untouched. I'd like exploration to fall out of the objective, not be
grafted on.

That phrase — exploration falling out of the objective — points me somewhere. There's a body
of work that changes the objective itself: maximize reward *and* the entropy of the policy.
The standard objective is sum_t E[r(s,a)]; the augmented one is

  J(pi) = sum_t E_{(s,a)~rho_pi} [ r(s,a) + alpha * H(pi(.|s)) ],

with H the entropy and alpha a temperature. As alpha -> 0 I get standard RL back, so this is a
strict generalization. Why would I want this? Because a policy rewarded for entropy won't
collapse onto a single action prematurely — it keeps probability mass on all the actions that
look comparably good, which is exactly exploration, and it's exploration the policy is
*choosing* because the objective pays for it. Ziebart argued these policies are also robust:
hedging across near-equal-value actions makes you less sensitive to errors in your value
estimates. So if I build my method around the max-entropy objective, exploration and a degree
of robustness come for free, baked in, no separate noise schedule.

The trouble is that the existing max-entropy off-policy method, soft Q-learning, took this and
made it complicated. It noted that the optimal max-entropy policy is energy-based,
pi(a|s) proportional to exp(Q(s,a)/alpha), and went after the *optimal* soft Q directly,
Q-learning style. But to actually act in a continuous space you need to sample from that
exp(Q/alpha) distribution, which is intractable, so they train a separate sampling network by
Stein variational gradient descent to approximate draws from it. Now the whole thing's
correctness rides on how well that sampler tracks the true energy-based posterior, the
inference machinery is heavy, and empirically it doesn't even beat DDPG from scratch. The
actor there isn't really an actor — it's an approximate sampler glued to a value function.

So here's what I actually want: keep the max-entropy objective for its free exploration and
robustness, but get it into a clean *actor-critic* — evaluate the Q of the *current* policy
and improve the policy against it, off-policy, with a genuine policy network — and avoid all
the approximate-inference baggage. Let me try to build that from policy iteration and see if
it holds together.

First I have to redefine value to include entropy, or none of this is consistent. In ordinary
RL, V(s) = E_{a~pi}[Q(s,a)]. If my objective pays alpha*H at every step, then the value of a
state under pi should also credit the entropy I'll collect. So define the soft state value as

  V(s) = E_{a~pi}[ Q(s,a) - alpha * log pi(a|s) ],

because -log pi(a|s) is exactly the per-sample entropy contribution (its expectation over a~pi
is H(pi(.|s))). And the soft Q stays a one-step-ahead object that bootstraps on this soft V:

  T^pi Q(s,a) = r(s,a) + gamma * E_{s'~p}[ V(s') ].

Note the entropy lives *inside* V and therefore inside the bootstrap — the future entropy I'll
earn shows up in the value of acting now. That's the key difference from just sprinkling an
entropy bonus on the actor's loss: there, the critic never hears about entropy, so it only
shapes the immediate action; here it propagates through the Bellman recursion and shapes
long-horizon behavior.

Does iterating this T^pi actually converge? Let me check, because if the soft backup isn't a
contraction the whole policy-evaluation step is meaningless. Take two candidate Q-functions,
Q_1 and Q_2, and apply the same soft backup to both. The reward term is the same, and the
entropy term is the same too because the policy is fixed during evaluation. Only the Q term can
change:

  T^pi Q_1(s,a) - T^pi Q_2(s,a)
    = gamma * E_{s',a'~pi}[ Q_1(s',a') - Q_2(s',a') ].

Now take absolute values and the sup norm:

  ||T^pi Q_1 - T^pi Q_2||_infty
    <= gamma * ||Q_1 - Q_2||_infty.

So the soft Bellman operator is a gamma-contraction. If I want to see it as ordinary policy
evaluation, I can fold the fixed entropy term into a bounded one-step quantity,

  r_pi(s,a) = r(s,a) + gamma * alpha * E_{s'~p}[ H(pi(.|s')) ],

leaving the remaining part as gamma * E_{s',a'~pi} Q(s',a'). The reward is bounded by
assumption, and for the tabular proof |A| < infinity bounds the entropy term. Banach's fixed
point theorem then gives a unique fixed point, and repeated soft backups from any starting Q^0
converge to the soft Q of pi. Soft policy evaluation: done, and it converges. That's Lemma 1.

Now the improvement step. In ordinary policy iteration I'd set the new policy greedy on Q. In
the soft world the analogous "greedy" move is toward the energy-based policy exp(Q/alpha)/Z —
that's the distribution that, for a fixed Q, maximizes E[Q] + alpha*H, by the usual
Gibbs/variational fact. But I can't represent an arbitrary energy-based density with my Gaussian
actor; I'm restricted to some tractable family Pi. So instead of setting pi to exp(Q/alpha)/Z,
I *project* exp(Q/alpha)/Z onto Pi. Which projection? The information projection — minimize the
KL from my candidate to the target:

  pi_new(.|s) = argmin_{pi' in Pi} KL( pi'(.|s) || exp( (1/alpha) Q^{pi_old}(s,.) ) / Z^{pi_old}(s) ).

And here's a piece of luck: the partition function Z^{pi_old}(s) is intractable, but it depends
only on s, not on pi', so it's an additive constant in the objective and drops out of the
gradient with respect to pi'. So I never have to compute Z. Good.

But does this projection actually *improve* the policy? Greedy improvement is exact; an
I-projection onto a restricted family is not obviously monotone. I need to prove
Q^{pi_new} >= Q^{pi_old} pointwise, or I don't have policy iteration. Let me work it out.
Write the KL objective (times alpha, dropping the log Z constant) as

  J_{pi_old}(pi'(.|s)) = E_{a~pi'}[ alpha log pi'(a|s) - Q^{pi_old}(s,a) + alpha log Z^{pi_old}(s) ].

Since pi_new minimizes this over pi' in Pi, and pi_old is itself in Pi, I can always fall back
to pi_old, so

  J_{pi_old}(pi_new(.|s)) <= J_{pi_old}(pi_old(.|s)).

Write that inequality out and notice the alpha log Z term is the same on both sides (it depends
only on s), so it cancels. Rearranging what's left:

  E_{a~pi_new}[ Q^{pi_old}(s,a) - alpha log pi_new(a|s) ] >= E_{a~pi_old}[ Q^{pi_old}(s,a) - alpha log pi_old(a|s) ].

And the right-hand side is, by definition, V^{pi_old}(s). So I've got the soft-value bound

  E_{a~pi_new}[ Q^{pi_old}(s,a) - alpha log pi_new(a|s) ] >= V^{pi_old}(s).   (*)

Now I bootstrap this through the soft Bellman equation for pi_old. Start from

  Q^{pi_old}(s,a) = r(s,a) + gamma * E_{s'}[ V^{pi_old}(s') ],

and replace V^{pi_old}(s') using the bound (*) — V^{pi_old}(s') is *no larger than* the same
expectation taken under pi_new:

  Q^{pi_old}(s,a) <= r(s,a) + gamma * E_{s'}[ E_{a'~pi_new}[ Q^{pi_old}(s',a') - alpha log pi_new(a'|s') ] ].

The right-hand side is one soft-Bellman backup of Q^{pi_old} under pi_new. Apply the same
substitution again to the Q^{pi_old}(s',a') inside, and again, unrolling the recursion; each
step only ever uses the bound (*), which points the same direction, so the inequality is
preserved at every level. In the limit the right side converges (by the contraction I just
proved for T^{pi_new}) to Q^{pi_new}(s,a). Therefore

  Q^{pi_old}(s,a) <= Q^{pi_new}(s,a)   for all (s,a).

That's the improvement guarantee — Lemma 2. The projection onto the restricted family does
improve the policy, precisely because falling back to pi_old is always an option inside the
argmin, which is what makes (*) hold.

Chaining the two: alternate soft evaluation and soft improvement. By Lemma 2 the sequence
Q^{pi_i} is monotonically nondecreasing in i; it's bounded above because reward and entropy
are both bounded; a monotone bounded sequence converges, to some pi*. At convergence the
argmin can't strictly improve, so for every other pi in Pi the same unrolling argument gives
Q^{pi*} >= Q^{pi} pointwise — pi* is optimal within Pi. That's soft policy iteration, Theorem
1. It's tabular and exact only in principle, but it tells me the *shape* of a correct
algorithm: evaluate the soft Q of the current policy, then KL-project toward exp(Q/alpha). Now
I make it practical with function approximators and SGD.

Take a critic Q_theta(s,a) and a policy pi_phi(a|s), both neural nets. The critic should
minimize the soft Bellman residual — the squared gap between Q_theta and its bootstrap target:

  J_Q(theta) = E_{(s,a)~D}[ 1/2 ( Q_theta(s,a) - ( r(s,a) + gamma * E_{s'}[ V_thetabar(s') ] ) )^2 ],

where D is the replay buffer (off-policy data, that's the whole point) and the target uses a
slow target network thetabar. A separate value network for V is tempting because V is smoother
than Q and avoids sampling inside the target, but it is also an extra approximation to a
quantity that is already defined by Q and pi:
V(s') = E_{a'~pi}[ Q(s',a') - alpha log pi(a'|s') ]. I can estimate that expectation with a
single reparameterized action sampled from the *current* policy. So I can fold V straight into
the Q target and drop the extra network. The target action a' is drawn from pi_phi (the current
policy, not the buffer's behavior action — this is what makes it evaluate the current policy),
and the target becomes

  y = r(s,a) + gamma * ( Q_thetabar(s', a') - alpha * log pi_phi(a'|s') ),   a' ~ pi_phi(.|s').

That second term, -alpha log pi(a'|s'), is the entropy bonus entering the bootstrap exactly as
the theory said it must. So my critic target is just the TD3 target with an extra
-alpha*log pi piece. (I'll come back to the twin-critic min.)

Now the actor. The improvement step says minimize the KL, which I showed equals, up to the
log Z constant, the expected

  J_pi(phi) = E_{s~D}[ E_{a~pi_phi}[ alpha log pi_phi(a|s) - Q_theta(s,a) ] ].

How do I take the gradient of an expectation whose sampling distribution depends on phi? The
generic answer is the likelihood-ratio / REINFORCE estimator — but that's high-variance, and
worse, it doesn't exploit a thing I have: the integrand contains Q_theta(s,a), and Q_theta is
a neural net, so it's *differentiable in a*. When the thing inside the expectation is
differentiable in the sample, I should use the reparameterization trick instead and get a
pathwise, low-variance gradient. So write the action as a deterministic function of state,
parameters, and fixed noise:

  a = f_phi(eps; s),   eps ~ N(0, I).

Then the actor objective becomes an expectation over the *fixed* noise:

  J_pi(phi) = E_{s~D, eps~N}[ alpha log pi_phi(f_phi(eps;s) | s) - Q_theta(s, f_phi(eps;s)) ],

and I can just backprop. The gradient flows two ways — through the explicit log pi_phi, and
through both log pi_phi and Q_theta via the action f_phi:

  grad_phi J_pi = grad_phi (alpha log pi_phi(a|s))
                + ( grad_a (alpha log pi_phi(a|s)) - grad_a Q_theta(s,a) ) * grad_phi f_phi(eps;s),

with a evaluated at f_phi(eps;s). This is just DDPG's pathwise policy gradient — the
grad_a Q * grad_phi f part is the deterministic policy gradient — extended to a stochastic
actor, with the extra entropy term. So I get DDPG's efficiency of gradient signal but with a
stochastic, exploring policy. That's the unification I was hoping for: max-entropy
exploration with an actor-critic that's as cheap to differentiate as DDPG.

Now the twin-critic min. Even with entropy regularization, my critic still bootstraps off
itself and can overestimate — that pathology doesn't disappear just because the policy is
stochastic. TD3's fix transfers directly: keep two independently initialized critics
Q_{theta_1}, Q_{theta_2}, train each on the same soft Bellman target, and wherever a single
Q value would drive bootstrapping or policy improvement, use the smaller estimate. So the
critic target uses min(Q_thetabar_1, Q_thetabar_2) at (s', a'), and the actor maximizes
min(Q_theta_1, Q_theta_2) at (s, a). The min deliberately trades a little underestimation for
protection against the actor chasing a positive error spike. Targets are kept by Polyak
averaging, thetabar_i <- tau*theta_i + (1-tau)*thetabar_i, with a small tau — a slow target
reduces TD target drift; push tau too high and bootstrapping follows its own noise too quickly,
too low and the target lags behind the learned critic.

Now the actor's distribution. I want a tractable pi_phi I can both sample (reparameterized) and
evaluate log pi for. A diagonal Gaussian is the obvious choice: the net outputs a mean mu_phi(s)
and a log-std log sigma_phi(s), and a = mu + sigma * eps with eps ~ N(0,I) is exactly a
reparameterized sample. But there's a problem: my actions must live in a bounded box
(torque/position limits), and a Gaussian has infinite support. If I just clip, the log-prob is
wrong at the boundary and the gradient there is garbage. So I squash first and then rescale to
the environment bounds: let u ~ N(mu, sigma) be the unbounded pre-activation, set y = tanh(u)
elementwise, and map a = action_scale * y + action_bias. tanh maps the real line into (-1,1),
and the affine map puts that into the actual action box, so I can get the exact density of a
by change of variables — which I need, because both the entropy bonus in the critic target and
the actor objective require log pi(a|s).

Let me derive that correction, because getting it wrong corrupts the entropy term silently.
For a scalar invertible transform a = scale * tanh(u) + bias with base density mu(u|s), the
change-of-variables formula gives

  pi(a|s) = mu(u|s) * | da/du |^{-1}.

For a vector u with elementwise tanh and per-dimension scaling, the Jacobian da/du is diagonal
with entries scale_i * (1 - tanh^2(u_i)), so its determinant is the product of those, and
taking logs turns the product into a sum and the inverse into a minus sign:

  log pi(a|s) = log mu(u|s) - sum_i log( scale_i * (1 - tanh^2(u_i)) ).

So I compute the ordinary Gaussian log-prob of u, then subtract
sum_i log(scale_i * (1 - tanh^2(u_i))). In code 1 - tanh^2(u_i) can hit zero at the saturated
boundary and blow up the log, so I add a small epsilon inside the correction. And I keep the
network's log sigma in a sane band, say [-5, 2], so the std neither collapses (which would make
the correction term and the log-prob explode) nor runs away.

There's one knob left that I haven't pinned down, and it turns out to be the one that decides
whether this whole thing is usable across tasks: the temperature alpha. In standard RL the
optimal policy is invariant to scaling the reward; in max-entropy RL it is *not* — alpha sets
the relative weight of entropy versus reward, so it's effectively a reward-scale parameter, and
the right value differs from task to task. Worse than that: even within a single run the right
alpha drifts. Early on the policy is bad and uncertain and wants lots of entropy to explore;
late on it's good and should commit, wanting little. Pinning entropy to a fixed value is
wrong — I want freedom to be stochastic where the best action is genuinely ambiguous and nearly
deterministic where it's clear. So I shouldn't hand-set alpha at all; I should let it adapt.

Let me reformulate. Instead of adding alpha*H to the objective with a hand-picked alpha, I'll
*constrain* the expected entropy to stay at least some target H_bar, and maximize plain return
subject to that:

  max_{pi_{0:T}} E[ sum_t r(s_t,a_t) ]   s.t.   E_{(s,a)~rho_pi}[ -log pi_t(a_t|s_t) ] >= H_bar  for all t.

Because for a fully observed MDP the return-optimal policy is deterministic, this entropy
constraint will generically be tight, so I don't need an upper bound on entropy. Now solve it
with a Lagrangian, and here alpha reappears — but as a *dual variable*, not a hyperparameter.
Since the policy at time t only affects current-and-future return, I can go backward in time,
dynamic-programming style. At the last step T, with the constraint, strong duality (the
objective is linear in pi_T and the entropy constraint is convex in pi_T) lets me swap the
constrained max for

  max_{pi_T} E[ r(s_T,a_T) ]  =  min_{alpha_T >= 0} max_{pi_T} E[ r(s_T,a_T) - alpha_T log pi_T(a_T|s_T) ] - alpha_T H_bar.

The inner max over pi_T is just the max-entropy problem at temperature alpha_T — so the optimal
last-step policy is the max-entropy policy for that temperature, and given that optimal policy,
the optimal dual variable is

  alpha_T* = argmin_{alpha_T} E_{a_T~pi_T*}[ -alpha_T log pi_T*(a_T|s_T) - alpha_T H_bar ].

Recurse: define the soft Q backward, Q_t*(s,a) = E[r] + E[ Q_{t+1}* - alpha_{t+1}* log pi_{t+1}* ],
and at each step the same dual manipulation gives the same form for alpha_t*. So in the
stationary limit, dropping the time indices, the temperature is whatever solves

  alpha* = argmin_alpha E_{a~pi}[ -alpha ( log pi(a|s) + H_bar ) ].

This is beautiful operationally: it's just another scalar I descend by SGD alongside the actor
and critic. Read off its gradient sign — if the current expected entropy -E[log pi] is below
the target H_bar (too deterministic), the bracket (log pi + H_bar) is positive on average, so
to minimize I push alpha up, raising the entropy reward and forcing more exploration; if
entropy is above target, alpha falls and the policy is allowed to commit. It's a thermostat. I
parameterize it as log alpha (so alpha stays positive) and take one gradient step per update —
dual gradient descent with a single inner step; the convexity that justifies it doesn't
literally hold with neural nets, so this becomes the practical approximation. The objective for
the temperature is

  J(alpha) = E_{a~pi}[ -alpha ( log pi(a|s) + H_bar ) ],

and I treat log pi as a constant (detached) in that step — I'm adjusting the multiplier, not the
policy. The one remaining choice is H_bar. A simple, scale-aware heuristic: set the target
entropy to -dim(A), one unit per action dimension — it scales with how many actions there are,
and it's negative because the differential entropy of a tightly-peaked continuous policy is
negative. With that, alpha self-tunes and I no longer have a per-task temperature to sweep,
which was the last brittle knob.

Let me also nail down the moving parts of the loop so they're concrete. I take an environment
step by *sampling* from pi_phi (stochastic — that's the exploration, no extra noise process),
store (s, a, r, s') in the replay buffer, then take gradient steps on minibatches from the
buffer: update the two critics on the soft Bellman residual toward the min-target with the
-alpha log pi entropy term; update the actor on the reparameterized
E[alpha log pi - min Q]; update log alpha on the dual objective; Polyak-update the target
critics. At evaluation time I don't want the exploration randomness anymore, so I act with the
*mean* action, tanh(mu_phi(s)) — the policy was trained to also maximize entropy, so its mean
is a good deterministic readout but isn't itself what the stochastic objective optimizes.

Now let me write it as the code it actually becomes — a tanh-squashed Gaussian actor, twin
soft critics, the min-target with the entropy bonus, the reparameterized actor loss, and the
automatic temperature, each block tied back to the step that motivated it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, action_low=None, action_high=None, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, n_act)
        self.fc_logstd = nn.Linear(hidden_dim, n_act)

        if action_low is None or action_high is None:
            action_scale = torch.ones(n_act)
            action_bias = torch.zeros(n_act)
        else:
            low = torch.as_tensor(action_low, dtype=torch.float32)
            high = torch.as_tensor(action_high, dtype=torch.float32)
            action_scale = (high - low) / 2.0
            action_bias = (high + low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        u = normal.rsample()                       # reparameterized: mean + std * eps
        y = torch.tanh(u)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(u)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def build_algorithm(n_obs, n_act, device, action_low=None, action_high=None,
                    policy_lr=3e-4, q_lr=1e-3):
    actor = Actor(n_obs, n_act, action_low, action_high).to(device)
    qf1, qf2 = QNetwork(n_obs, n_act).to(device), QNetwork(n_obs, n_act).to(device)
    qf1_target = QNetwork(n_obs, n_act).to(device)
    qf2_target = QNetwork(n_obs, n_act).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    return {
        "actor": actor, "qf1": qf1, "qf2": qf2,
        "qf1_target": qf1_target, "qf2_target": qf2_target,
        "log_alpha": log_alpha, "target_entropy": -float(n_act),
        "actor_opt": torch.optim.Adam(actor.parameters(), lr=policy_lr),
        "q_opt": torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr),
        "alpha_opt": torch.optim.Adam([log_alpha], lr=q_lr),
    }


def update_critic(batch, c, gamma):
    actor = c["actor"]
    qf1, qf2 = c["qf1"], c["qf2"]
    qf1_target, qf2_target = c["qf1_target"], c["qf2_target"]
    alpha = c["log_alpha"].exp().detach()
    obs, action, reward, next_obs, done = batch

    with torch.no_grad():
        next_action, next_logp, _ = actor.get_action(next_obs)
        qf1_next = qf1_target(next_obs, next_action)
        qf2_next = qf2_target(next_obs, next_action)
        min_q_next = torch.min(qf1_next, qf2_next) - alpha * next_logp
        next_q = reward + (1.0 - done) * gamma * min_q_next

    qf1_a = qf1(obs, action)
    qf2_a = qf2(obs, action)
    q_loss = F.mse_loss(qf1_a, next_q) + F.mse_loss(qf2_a, next_q)
    c["q_opt"].zero_grad(); q_loss.backward(); c["q_opt"].step()
    return q_loss


def update_actor(batch, c):
    actor, qf1, qf2 = c["actor"], c["qf1"], c["qf2"]
    alpha = c["log_alpha"].exp().detach()
    obs = batch[0]

    pi, logp, _ = actor.get_action(obs)
    min_q_pi = torch.min(qf1(obs, pi), qf2(obs, pi))
    actor_loss = (alpha * logp - min_q_pi).mean()
    c["actor_opt"].zero_grad(); actor_loss.backward(); c["actor_opt"].step()

    with torch.no_grad():
        _, logp_alpha, _ = actor.get_action(obs)
    alpha_loss = (-c["log_alpha"].exp() * (logp_alpha + c["target_entropy"])).mean()
    c["alpha_opt"].zero_grad(); alpha_loss.backward(); c["alpha_opt"].step()
    return actor_loss, alpha_loss


@torch.no_grad()
def soft_update(src, tgt, tau):
    for p, p_t in zip(src.parameters(), tgt.parameters()):
        p_t.mul_(1.0 - tau).add_(p, alpha=tau)


def train_step(batch, c, gamma, tau, global_step,
               policy_frequency=2, target_network_frequency=1):
    q_loss = update_critic(batch, c, gamma)
    actor_loss = alpha_loss = None
    if global_step % policy_frequency == 0:
        for _ in range(policy_frequency):
            actor_loss, alpha_loss = update_actor(batch, c)
    if global_step % target_network_frequency == 0:
        soft_update(c["qf1"], c["qf1_target"], tau)
        soft_update(c["qf2"], c["qf2_target"], tau)
    return q_loss, actor_loss, alpha_loss
```

So the causal chain: I wanted off-policy efficiency *and* stability, and traced DDPG's
brittleness to a deterministic actor maximizing an overestimated critic with bolted-on
exploration. Putting entropy into the objective makes exploration something the policy
optimizes rather than something I schedule, and pushing it inside the soft value function
makes it propagate through the bootstrap. Soft policy iteration — soft evaluation as a direct
gamma-contraction and soft improvement as a KL-projection that's provably monotone because the
old policy is always a fallback — gives the correct skeleton, which I approximate with a
soft-Bellman critic and a KL-minimizing actor. The
reparameterization trick turns the stochastic-actor update into DDPG's cheap pathwise gradient,
TD3's twin-min curbs the residual overestimation, tanh squashing plus affine action rescaling
with the change-of-variables correction handles bounded actions exactly, and recasting the
temperature as a dual variable that tracks a target entropy of -dim(A) removes the last
per-task knob. The result is an off-policy actor-critic that explores by design, resists
overestimation, and tunes its own stochasticity.
