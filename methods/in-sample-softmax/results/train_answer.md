We are given a fixed offline dataset $D = \{(s_i, a_i, r_i, s'_i)\}$ collected by an unknown behavior policy $\pi_D$, with no further interaction allowed, and we want to extract the best policy the data can justify — better than a mediocre operator if a better action is present in the batch, but never reaching for actions the data cannot support. The disease that makes this hard is unsupported bootstrapping. An ordinary fitted-Q target $r(s,a) + \gamma \max_{a'} q(s',a')$ takes a maximum that does not know which actions the batch actually covers at $s'$. If an unseen action has a high extrapolated value, the max selects it, the target rises, and the critic begins training real dataset actions toward a value that is supported only by extrapolation. With function approximation this spreads: one unsupported overestimate becomes a target for supported actions, which then become inflated targets for other states. So the object I actually need to control is not just the deployed policy — it is what the bootstrap is allowed to look at.

The standard remedies each miss in a specific way. Behavior regularization, $\max_\pi \, \pi\cdot q - \tau\,\mathrm{KL}(\pi\,\|\,\pi_D)$, has the clean closed form $\pi'(a)\propto \pi_D(a)\exp(q(a)/\tau)$, which is automatically zero off support because the KL is infinite there; but the behavior probabilities multiply the exponentiated values, so if $\pi_D$ is skewed toward a mediocre action the improved policy keeps that skew unless the value gap is enormous. Its safety comes from staying close to the behavior distribution, not merely inside its support. Conservative value estimation (CQL, Fisher-BRC, ensembles) pushes down unsupported values, but trades overestimation for a pessimism coefficient that must be tuned, and the backup is still structurally allowed to query every action. The support-constrained hard max is the object I really want — $\max_{a':\,\pi_D(a'|s')>0} q(s',a')$, the value of the best action the data knows about — and BCQ approximates it by sampling candidates from a learned generator, but then support safety is only as good as that generator; a little density in the wrong place readmits an unsupported action. IQL fits an upper expectile over dataset actions, which keeps the target in-sample, but an expectile is a statistic under the behavior action distribution — a frequent suboptimal action pulls it — and it has no clean closed form, which makes the resulting Bellman operator awkward to reason about under stochastic transitions. The wall is therefore the hard maximum itself: it is exactly the right limiting object, but it is awkward to estimate and awkward to prove things about. I want a nearby object that keeps the support idea but gives me algebra.

I propose the In-Sample Softmax operator and the actor-critic built from it, In-Sample Actor-Critic (InAC). Entropy-regularized control already replaces a hard max by a log-sum-exp, $F_\tau(q) = \tau\log\sum_a \exp(q(a)/\tau) = \max_{p\in\Delta(A)} p\cdot q + \tau H(p)$ with optimizer $p(a)\propto\exp(q(a)/\tau)$ and the $\tau\to 0$ limit $\max_a q(a)$. The move is to put the data-support restriction inside this log-sum-exp:
$$F_{\beta,\tau}(q) = \tau\log\sum_{a:\,\beta(a)>0}\exp\!\big(q(a)/\tau\big),$$
where $\beta$ is the distribution whose support I trust, eventually $\pi_D$. This is exactly the maximum-entropy problem on the smaller simplex, $F_{\beta,\tau}(q) = \max_{\pi\le\beta}\,\pi\cdot q + \tau H(\pi)$ with $\pi\le\beta$ meaning $\mathrm{supp}(\pi)\subset\mathrm{supp}(\beta)$, and its optimizer is the Boltzmann distribution over only the supported coordinates,
$$f_{\beta,\tau}(q)(a) = \frac{\beta(a)\exp\!\big(q(a)/\tau - \log\beta(a)\big)}{\sum_{b:\,\beta(b)>0}\exp(q(b)/\tau)}.$$
For $\beta(a)>0$ the numerator collapses to $\beta(a)\exp(q(a)/\tau)\exp(-\log\beta(a)) = \exp(q(a)/\tau)$, and for $\beta(a)=0$ I set the mass to zero. This is the first real payoff: the greedy policy is $\exp(q/\tau)$ on the support and zero outside it — the behavior distribution supplies the support but its probabilities cancel through the $-\log\beta$ term. I get the safety of behavior regularization without inheriting the behavior policy's action frequencies, which is exactly where $\pi'(a)\propto\beta(a)\exp(q(a)/\tau)$ fails.

The same cancellation solves sampling. The sum over the (unknown) support is an expectation over dataset actions,
$$\sum_{a:\,\pi_D(a|s)>0}\exp\!\big(q(s,a)/\tau\big) = \mathbb{E}_{a\sim\pi_D(\cdot|s)}\Big[\exp\!\big(q(s,a)/\tau - \log\pi_D(a|s)\big)\Big],$$
with no division by zero because the expectation only ranges over the support. This changes the role of the behavior model: I need an estimate of $\log\pi_D(a|s)$ for dataset actions, but I never sample candidate actions from it and trust its support. A mistake in the density estimate merely re-weights an action that was actually observed; it cannot manufacture a new unsupported action for the max to exploit. The behavior model $\pi_\omega$ is therefore plain maximum likelihood, $L_{\text{behavior}}(\omega) = -\mathbb{E}_{(s,a)\sim D}[\log\pi_\omega(a|s)]$, queried only through $-\log\pi_\omega(a|s)$ on batch actions.

What makes the soft operator usable as a Bellman operator is that the support restriction does not break the contraction. For two vectors $q_1,q_2$ on a finite action set, letting $\pi_1$ optimize $F_{\beta,\tau}(q_1)$,
$$F_{\beta,\tau}(q_1) - F_{\beta,\tau}(q_2) \le \big(\pi_1\cdot q_1 + \tau H(\pi_1)\big) - \big(\pi_1\cdot q_2 + \tau H(\pi_1)\big) = \pi_1\cdot(q_1-q_2) \le \max_{a:\,\beta(a)>0}\big(q_1(a)-q_2(a)\big) \le \|q_1-q_2\|_\infty,$$
where the second maximum is evaluated at the first optimizer so the entropy terms cancel; swapping $q_1,q_2$ gives $|F_{\beta,\tau}(q_1)-F_{\beta,\tau}(q_2)|\le\|q_1-q_2\|_\infty$. This is the precise advantage over the expectile: nothing about deterministic transitions is needed. The in-sample soft Bellman optimality operator $(T^*_\beta q)(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'|s,a}[F_{\beta(\cdot|s'),\tau}(q(s',\cdot))]$ then inherits $\|T^*_\beta q_1 - T^*_\beta q_2\|_\infty \le \gamma\|q_1-q_2\|_\infty$, so for $\gamma<1$ the fixed point is unique and value iteration converges; as $\tau\to 0$ the operator approaches the support-constrained hard max, and its fixed point the hard constrained optimum. A policy-iteration view follows the same algebra: greedifying a supported $\pi\le\beta$ via $\pi'(a|s)\propto\beta(a|s)\exp(\tilde q^\pi(s,a)/\tau - \log\beta(a|s))$ keeps support automatically, the restricted maximum-entropy identity gives $\mathbb{E}_{\pi'}[\tilde q^\pi - \tau\log\pi'] \ge \mathbb{E}_\pi[\tilde q^\pi - \tau\log\pi]$ pointwise, hence $\tilde q^\pi = T^\pi\tilde q^\pi \le T^{\pi'}\tilde q^\pi$, and monotonicity plus contraction of $T^{\pi'}$ telescope to $\tilde q^{\pi'}\ge\tilde q^\pi$ — every greedification stays in-support and improves the entropy-regularized value.

Turning this into an actor-critic gives InAC, with four networks: a stochastic actor $\pi_\psi$, a double critic $q_\theta$ (using $\min(q_1,q_2)$), a scalar value network $v_\phi$, and the behavior model $\pi_\omega$. Writing the normalizer in value units, $Z(s) = \tau\log\mathbb{E}_{a\sim\pi_D(\cdot|s)}[\exp(q_\theta(s,a)/\tau - \log\pi_\omega(a|s))]$, the approximate greedy target is $\hat\pi(a|s) = \pi_D(a|s)\exp((q_\theta(s,a)-Z(s))/\tau - \log\pi_\omega(a|s))$. The direction of the KL projection is critical. Minimizing $\mathrm{KL}(\pi_\psi\,\|\,\hat\pi)$ takes the expectation over the learned actor, which can sample unsupported actions early in training; instead I use the forward KL, $\mathrm{KL}(\hat\pi\,\|\,\pi_\psi) = \mathbb{E}_{a\sim\hat\pi}[\log\hat\pi - \log\pi_\psi]$, whose actor-dependent part, after changing measure from $\hat\pi$ to $\pi_D$, is an expectation over dataset actions. The actor update is therefore weighted maximum likelihood on batch actions:
$$L_{\text{actor}}(\psi) = -\mathbb{E}_{(s,a)\sim D}\Big[\exp\!\big((q_\theta(s,a)-v_\phi(s))/\tau - \log\pi_\omega(a|s)\big)\,\log\pi_\psi(a|s)\Big].$$
Here $v_\phi(s)$ stands in for $Z(s)$: for the soft greedy policy the log-sum-exp normalizer is exactly the entropy-regularized state value, and since $\pi_\psi$ is trained toward that greedy policy its soft value tracks the same quantity. In the loss $Z(s)$ is a per-state scale, so moderate error mostly changes how much a state contributes to the minibatch rather than which action it prefers; because the exponential can still explode when $\tau$ is small or $\pi_\omega(a|s)$ tiny, the weight is clipped. The value network learns the SAC soft value, $L_{\text{value}}(\phi) = \mathbb{E}_{s\sim D,\,a\sim\pi_\psi(\cdot|s)}[\tfrac12(v_\phi(s) - (\min_i q_{\theta,i}(s,a) - \tau\log\pi_\psi(a|s)))^2]$, and the critic bootstraps through that value rather than a free max,
$$L_{\text{critic}}(\theta) = \mathbb{E}_{(s,a,r,s')\sim D}\Big[\tfrac12\big(r + \gamma\,v_\phi(s') - q_\theta(s,a)\big)^2\Big],$$
which in the concrete SAC-style loop is computed as $r + \gamma(1-\text{done})(\min_i q_{\text{target},i}(s',a') - \tau\log\pi_\psi(a'|s'))$ with $a'\sim\pi_\psi(\cdot|s')$. Both forms avoid a hard max over next actions. The honest caveat is that an early $\pi_\psi$ may still place mass outside the data support, so the value update can momentarily evaluate unsupported actions; but the actor loss is sampled only on dataset actions and pulls $\pi_\psi$ toward the supported greedy target, so as that projection tightens the unsupported mass shrinks. The support constraint is enforced through policy extraction, not by a hard filter inside the critic. Temperature $\tau$ is the sharpness knob: small $\tau$ drives the supported log-sum-exp toward the supported hard max and concentrates the policy on the best supported action, larger $\tau$ keeps a more entropic target for broad noisy data, and every $\tau$ keeps the same support logic and the same contraction.

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
            target = self.q_t.min(s, a_pi) - self.tau * logp_pi
        return (0.5 * (v_phi - target) ** 2).mean()

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
