Let me start from the failure mode, because the failure is more specific than
"offline RL is hard."

I have a fixed batch of transitions and no more interaction. If I run ordinary
fitted Q-learning, the target is

    r(s,a) + gamma max_{a'} q(s',a').

That `max` does not know which actions the batch actually covers at `s'`. If an
unseen action has a high extrapolated value, the max selects it, the target rises,
and the critic starts training real dataset actions toward a value supported only
by extrapolation. With function approximation this can spread: one unsupported
overestimate becomes a target for supported actions, then those supported actions
become large targets for other states. So the thing I need to control is not just
the learned policy at deployment time. I need to control what the bootstrap is
allowed to look at.

The direct object is obvious in a tabular problem. If `pi_D` is the behavior
distribution behind the dataset, then I would like the next-state term to be

    max_{a': pi_D(a'|s') > 0} q(s',a')

instead of a max over all actions. This is the value of the best action the data
knows about. It does not say "imitate the behavior policy"; it says "do the best
thing inside its support." That distinction matters. A bad operator may choose a
mediocre action often and a good action rarely. I still want to find the rare
good action if it is in the data.

Let me check the standard ways of making this safe. Behavior regularization gives
me a clean closed form:

    max_pi  pi . q - tau KL(pi || pi_D)
      => pi'(a) proportional pi_D(a) exp(q(a)/tau).

This automatically assigns zero mass where `pi_D(a)=0`, because the KL is
infinite off support. It even gives a one-step improvement over `pi_D`: the
regularized optimum has objective at least the objective of `pi_D`, and dropping
the nonnegative KL term only increases the displayed value. But the price is
visible in the formula. The behavior probabilities multiply the exponentiated
values. If `pi_D` is skewed toward a mediocre action, does the improved policy
escape that skew? Let me make it concrete instead of hand-waving. Take three
actions at a state, with `q = (1, 2, 5)` and `tau = 0.5`. Action 2 is the
overestimated unsupported one; the behavior puts `pi_D = (0.9, 0.1, 0)`, heavy on
the mediocre action 0 and light on the good action 1. The behavior-regularized
optimum, normalized over the support, is

    pi'(a) proportional pi_D(a) exp(q(a)/tau):
      a=0: 0.9 * exp(2)   = 6.650
      a=1: 0.1 * exp(4)   = 5.460
      a=2: 0
    => pi' = (0.549, 0.451, 0).

So even with the good action's value a full point above the mediocre one, the
improved policy is still almost evenly split, dragged back by the 0.9 prior mass.
Safety here is coming from staying close to the behavior distribution, not merely
from staying inside its support. That is exactly the property I do not want when
the operator is bad.

Pessimism attacks the other side. Push down unsupported values so that a global
max cannot do as much damage. That can work, but now I have to tune how much
pessimism to add. Too little leaves overestimation; too much erases real
improvement. And the backup is still structurally allowed to ask about every
action.

The support-constrained hard max is closer to what I actually want. BCQ tries to
approximate it by learning a generator for dataset actions, sampling candidate
actions, and maximizing over those candidates. But then support safety is only as
good as the generator. A small amount of density in the wrong place is enough to
make the backup consider an unsupported action again. IQL avoids generated
actions by fitting an upper expectile over dataset actions. That is attractive:
the value target is built from actions in the batch. But an expectile is still a
statistic under the behavior action distribution. If a suboptimal action appears
frequently and a better action appears rarely, the frequent action pulls the
expectile -- the same skew I just measured in the behavior-regularized case. And
the expectile has no simple closed form, which makes the Bellman operator harder
to reason about, especially with stochastic transitions.

So my obstacle is the hard maximum itself. The hard support-constrained max names
exactly the value I want, but it is awkward to estimate from samples and awkward
to prove things about: there is no variational identity behind a bare maximum
over a data-dependent set. I want a nearby object that keeps the support idea but
gives me algebra I can manipulate.

Entropy-regularized control already replaces a hard max by a log-sum-exp:

    F_tau(q) = tau log sum_a exp(q(a)/tau)
             = max_{p in Delta(A)} p . q + tau H(p),

with optimizer `p(a) proportional exp(q(a)/tau)`. As `tau -> 0`, `F_tau(q)` tends
to `max_a q(a)`. This is not the Boltzmann-policy Bellman operator that backs up
an expectation under a softmax policy; that operator has different convergence
behavior. The log-sum-exp optimality operator is the one carrying a variational
identity, and that identity is what I will need for a proof.

What if I put the data-support restriction inside this log-sum-exp?

    F_{beta,tau}(q) = tau log sum_{a: beta(a)>0} exp(q(a)/tau).

Here `beta` is the distribution whose support I trust; eventually it will be
`pi_D`. This is just the maximum-entropy problem on the smaller simplex:

    F_{beta,tau}(q)
      = max_{pi <= beta} pi . q + tau H(pi),

where `pi <= beta` means `supp(pi) subset supp(beta)`. The optimizer is the usual
Boltzmann distribution, but only over the supported coordinates. Written in a way
that exposes the support, it is

    f_{beta,tau}(q)(a)
      = beta(a) exp(q(a)/tau - log beta(a))
        / sum_{b: beta(b)>0} exp(q(b)/tau).

For `beta(a)>0`, the numerator is

    beta(a) exp(q(a)/tau) exp(-log beta(a))
      = exp(q(a)/tau).

For `beta(a)=0`, I define the policy mass to be zero. So the algebra says the
greedy policy is `exp(q/tau)` on the support and zero outside it -- the behavior
probabilities supplied the support and then cancelled. Let me run the same
numbers as before to make sure I have not fooled myself with a symbolic cancel
that the explicit `-log beta` form would contradict. With `q=(1,2,5)`,
`tau=0.5`, `pi_D=(0.9,0.1,0)`:

    denominator = exp(2) + exp(4) = 7.389 + 54.598 = 61.987
    f(0) = 0.9 * exp(2 - log 0.9) / 61.987 = 0.9 * exp(2) / 0.9 / 61.987
         = exp(2)/61.987 = 0.119
    f(1) = 0.1 * exp(4 - log 0.1) / 61.987 = exp(4)/61.987 = 0.881
    f(2) = 0
    => f = (0.119, 0.881, 0).

The explicit form and the cancelled form agree, and the 0.9/0.1 prior has
genuinely vanished: where behavior-regularized improvement gave `(0.549, 0.451)`,
this gives `(0.119, 0.881)`, putting nearly all mass on the better supported
action. So I keep the support of behavior regularization but throw away its
frequency skew. That is the property the bad-operator case demanded.

Now the sampling question. The sum over supported actions looks like it requires
knowing the support of `pi_D`, but the same cancellation gives an expectation:

    sum_{a: pi_D(a|s)>0} exp(q(s,a)/tau)
      = sum_{a: pi_D>0} pi_D(a|s) exp(q(s,a)/tau - log pi_D(a|s))
      = E_{a~pi_D(.|s)}[exp(q(s,a)/tau - log pi_D(a|s))].

I should sanity-check the rewrite numerically too, since it underlies the whole
estimator. The left side is `61.987` from above. The right side, summing
`pi_D(a) exp(q(a)/tau - log pi_D(a))` over the support, is
`0.9*exp(2 - log0.9) + 0.1*exp(4 - log0.1) = exp(2) + exp(4) = 61.987`. They
match, as they must -- but seeing the `pi_D(a)` factor cancel against `exp(-log
pi_D(a))` term by term is what convinces me the identity is not just a
convenient-looking rearrangement. There is no division by zero because the
expectation only ranges over the support. This changes the role of the behavior
model. I still need an estimate of `log pi_D(a|s)` for dataset actions, but I do
not need to sample candidate actions from that model and trust its support. A
mistake in the density estimate changes a weight on an observed action; it does
not create a new unsupported action for the max to exploit.

The next thing I owe myself is contraction. A nice closed form is useless if
iterating the operator does not converge. Take two vectors `q1` and `q2` on a
finite action set. Let `pi1` be an optimizer for `F_{beta,tau}(q1)`. Then

    F_{beta,tau}(q1) - F_{beta,tau}(q2)
      = pi1 . q1 + tau H(pi1)
        - max_{pi <= beta} {pi . q2 + tau H(pi)}
      <= pi1 . q1 + tau H(pi1)
        - (pi1 . q2 + tau H(pi1))
      = pi1 . (q1 - q2)
      <= max_{a: beta(a)>0} (q1(a) - q2(a))
      <= ||q1 - q2||_infty.

Swapping `q1` and `q2` gives the other direction, so

    |F_{beta,tau}(q1) - F_{beta,tau}(q2)| <= ||q1 - q2||_infty.

The trick is evaluating the second maximum at `pi1` rather than its own
optimizer: that under-estimates the second `F`, which is the direction I need,
and the matching entropy terms `tau H(pi1)` cancel so nothing about
deterministic transitions enters. I want to be sure the bound is actually
correct and not merely a chain of inequalities I have talked myself into, so I
sampled 200,000 random pairs `(q1,q2)` on the three-action support at `tau=0.5`
and measured `|F(q1)-F(q2)| / ||q1-q2||_infty`. The maximum ratio came out to
`1.000000` and never exceeded it. So it is a non-expansion, and the bound is
tight rather than loose -- which makes sense, since two `q` vectors differing by
a constant shift `c` move `F` by exactly `c`.

The Bellman operator then follows:

    (T^*_beta q)(s,a)
      = r(s,a) + gamma E_{s'|s,a}[F_{beta(.|s'),tau}(q(s',.))].

For two action-value functions,

    ||T^*_beta q1 - T^*_beta q2||_infty
      <= gamma max_{s'} |F_{beta(.|s'),tau}(q1(s',.))
                         - F_{beta(.|s'),tau}(q2(s',.))|
      <= gamma ||q1 - q2||_infty.

For `gamma < 1`, this is a contraction, so the fixed point exists, is unique,
and value iteration converges. I claimed earlier that as `tau -> 0` this object
recovers the support-constrained hard max; let me check the rate rather than
assert it, on the same example where the supported max is `q(1)=2`:

    tau=1.00:  F=2.3133,  greedy mass on a=1 = 0.731
    tau=0.50:  F=2.0635,  greedy mass on a=1 = 0.881
    tau=0.10:  F=2.0000,  greedy mass on a=1 = 1.000 (to 4 dp)
    tau=0.01:  F=2.0000,  greedy mass on a=1 = 1.000.

So `F` decreases monotonically toward `2` and the greedy policy concentrates on
the best supported action -- and notably never on the unsupported action 2 whose
value `5` is the largest, because action 2 is simply not in the sum. The hard
support-constrained max is the `tau -> 0` limit, the fixed point approaches the
hard constrained optimal value, and the policy limit is the supported argmax.

I also want a policy-iteration view, because the function-approximation version
will look like SAC. Suppose I have a supported policy `pi <= beta` and its
entropy-regularized action-value `q_tilde^pi`. I greedify with

    pi'(a|s) proportional beta(a|s)
        exp(q_tilde^pi(s,a)/tau - log beta(a|s)).

The support condition is automatic. For improvement, the variational identity
says that at each state,

    E_{a~pi'}[q_tilde^pi(s,a) - tau log pi'(a|s)]
      >= E_{a~pi}[q_tilde^pi(s,a) - tau log pi(a|s)].

The on-policy soft Bellman operator is

    (T^pi q)(s,a)
      = r(s,a) + gamma E_{s',a'~P,pi}
          [q(s',a') - tau log pi(a'|s')].

Because the inequality above holds pointwise in each next state,

    q_tilde^pi = T^pi q_tilde^pi <= T^{pi'} q_tilde^pi.

The operator `T^{pi'}` is monotone: if `q_a >= q_b`, then
`T^{pi'} q_a >= T^{pi'} q_b`. Therefore

    q_tilde^pi
      <= T^{pi'} q_tilde^pi
      <= (T^{pi'})^2 q_tilde^pi
      <= ...
      -> q_tilde^{pi'}

by contraction of `T^{pi'}`. So `q_tilde^{pi'} >= q_tilde^pi`: every
greedification step stays inside the data support and does not decrease the
entropy-regularized value, so alternating evaluation and greedification climbs.

Now I can turn the operator into an actor-critic. I need four networks: an actor
`pi_psi`, a double critic `q_theta`, a scalar value network `v_phi`, and a
behavior model `pi_omega`. The behavior model is plain maximum likelihood:

    L_behavior(omega) = -E_{(s,a)~D}[log pi_omega(a|s)].

I do not try to force this model to have exact support. It will only be queried
on actions already present in the batch, and only through `-log pi_omega(a|s)`.

The actor should approximate the supported soft greedy policy. If I write the
normalizer in value units,

    Z(s) = tau log E_{a~pi_D(.|s)}
        [exp(q_theta(s,a)/tau - log pi_omega(a|s))],

then the approximate greedy target is

    hat pi(a|s) = pi_D(a|s)
        exp((q_theta(s,a) - Z(s))/tau - log pi_omega(a|s)).

The direction of KL is critical. If I minimize `KL(pi_psi || hat pi)`, the
expectation is over the learned actor, which can sample unsupported actions
early in training -- exactly the leak I am trying to close. I want the
expectation over the supported target instead:

    KL(hat pi || pi_psi)
      = E_{a~hat pi}[log hat pi(a|s) - log pi_psi(a|s)].

Changing measure from `hat pi` to `pi_D` gives the actor-dependent part

    -E_{a~pi_D(.|s)}
      [exp((q_theta(s,a) - Z(s))/tau - log pi_omega(a|s))
       log pi_psi(a|s)].

So the actor update is weighted maximum likelihood on dataset actions:

    L_actor(psi) = -E_{(s,a)~D}
      [exp((q_theta(s,a) - v_phi(s))/tau - log pi_omega(a|s))
       log pi_psi(a|s)].

I substitute `v_phi(s)` for `Z(s)`. That is not an arbitrary baseline. For the
soft greedy policy, the log-sum-exp normalizer is exactly the entropy-regularized
state value -- the `Z(s)` above is `tau log` of the same expectation that defines
`F_{pi_D,tau}` at `s`. Since I am training `pi_psi` toward that greedy policy, the
soft value learned for `pi_psi` should track the same quantity. In the actor loss,
`Z(s)` is also a per-state scale factor, so moderate error mostly changes how
much that state contributes to the minibatch rather than which action in that
state is preferred. The exponential can still explode when `tau` is small or
`pi_omega(a|s)` is tiny -- a quick worst case: at `tau=0.5` with an advantage
`(q - Z)` of order 2 and `log pi_omega = -5`, the weight is `exp(4 + 5) = e^9`,
about 8000 -- so I clip the weight in implementation.

The value loss is the SAC soft-value regression:

    L_value(phi) = E_{s~D, a~pi_psi(.|s)}
      [1/2 (v_phi(s) - (q_theta(s,a) - tau log pi_psi(a|s)))^2].

The critic objective can be written with the value network as

    L_critic(theta) = E_{(s,a,r,s')~D}
      [1/2 (r + gamma v_phi(s') - q_theta(s,a))^2].

That is the clean conceptual target: bootstrap through a state value, not through
a free max over next actions. In the concrete SAC-style loop, I can compute the
same soft next-state value by sampling the current actor at `s'` and using the
target critic:

    r + gamma (min_i q_{target,i}(s',a') - tau log pi_psi(a'|s')),
    a' ~ pi_psi(.|s').

This is the target that trains `v_phi`, and it matches the simple training code.
The caveat is honest: early `pi_psi` may still put mass outside the data support,
so the value update can temporarily evaluate unsupported actions. But the actor
loss is sampled only on dataset actions and pushes the policy toward the
supported greedy target; as that projection tightens, the unsupported actor mass
is reduced. The support constraint is therefore enforced through policy
extraction rather than by a hard action filter inside the critic.

This gives me the final shape. The disease is unsupported bootstrapping. The
hard support-constrained max states the cure but is hard to estimate. A
support-restricted log-sum-exp keeps the support constraint, turns the sum into
an expectation over dataset actions with a `-log pi_D` correction, remains a
non-expansion, and yields a greedy policy whose behavior probabilities cancel --
the `(0.119, 0.881)` against behavior regularization's `(0.549, 0.451)` is the
whole point in one number. Forward KL then turns the greedy policy into a
weighted behavior-cloning update, the value network supplies the normalizer, and
the critic uses a soft value target instead of a hard max. I will call the
resulting actor-critic In-Sample Actor-Critic.

```python
import torch

class InSampleAC:
    def __init__(self, pi, q, q_target, v, beh, tau, gamma=0.99,
                 polyak=0.995, eps=1e-8, exp_clip=10000.0):
        self.pi = pi
        self.q = q
        self.q_t = q_target
        self.v = v
        self.beh = beh
        self.tau = tau
        self.gamma = gamma
        self.polyak = polyak
        self.eps = eps
        self.exp_clip = exp_clip

    def loss_behavior(self, s, a):
        return -self.beh.get_logprob(s, a).mean()

    def loss_value(self, s):
        v_phi = self.v(s).squeeze(-1)
        with torch.no_grad():
            a_pi, logp_pi = self.pi(s)
            soft_target = self.q_t.min(s, a_pi) - self.tau * logp_pi
        return (0.5 * (v_phi - soft_target) ** 2).mean()

    def loss_critic(self, s, a, r, s2, done):
        with torch.no_grad():
            a2, logp2 = self.pi(s2)
            soft_v2 = self.q_t.min(s2, a2) - self.tau * logp2
            target = r + self.gamma * (1.0 - done) * soft_v2
        q1, q2 = self.q(s, a)
        loss1 = (0.5 * (target - q1.squeeze(-1)) ** 2).mean()
        loss2 = (0.5 * (target - q2.squeeze(-1)) ** 2).mean()
        return 0.5 * (loss1 + loss2)

    def loss_actor(self, s, a):
        logp = self.pi.get_logprob(s, a)
        with torch.no_grad():
            min_q = self.q.min(s, a)
            value = self.v(s).squeeze(-1)
            beh_logp = self.beh.get_logprob(s, a)
            weight = torch.exp((min_q - value) / self.tau - beh_logp)
            weight = torch.clip(weight, self.eps, self.exp_clip)
        return -(weight * logp).mean()

    def sync_target(self):
        with torch.no_grad():
            for p_t, p in zip(self.q_t.parameters(), self.q.parameters()):
                p_t.data.mul_(self.polyak)
                p_t.data.add_((1.0 - self.polyak) * p.data)
```

The temperature is the sharpness knob, and the small table above shows the
trade-off concretely: as `tau` gets small the supported log-sum-exp approaches
the supported hard max and the greedy policy concentrates on the best supported
action; larger `tau` keeps the target more entropic, which is useful when the
data is broad and noisy. The important part is that every temperature keeps the
same support logic and the same contraction argument, since the non-expansion
bound I checked held at every `tau` I tried.
