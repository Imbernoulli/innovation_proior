The trouble starts with the objective I usually optimize. Maximizing expected return, $\max_\pi \sum_t \mathbb{E}_{(s,a)\sim\rho_\pi}[\,r(s,a)\,]$, has — under full observability — a deterministic optimum: there is a single best action at every state, so the policy collapses to a delta. Whatever stochasticity I have is bolted on as an exploration trick: I inject action noise, or initialize a Gaussian policy with large variance and watch it anneal away. But there are tasks where I genuinely want the *solution* itself to be stochastic and, more than that, multimodal. When several distinct behaviors are roughly equally good — four symmetric goals around the agent, two passages through a maze, swimming in either direction toward a finish line — a unimodal policy has to commit to one mode early, and if it guesses wrong it never recovers. A policy that keeps every good mode alive explores far better because it keeps sampling all the options until it is sure. And there is a second payoff: a policy that has learned *all* the ways to move forward is a far better initialization for finetuning into a specific gait than a near-deterministic expert that knows only one. Near-optimal policies under the standard objective are near-deterministic, which makes them poor initializers.

The existing options each fail one of the two requirements this demands. DDPG backpropagates the critic's action-gradient $\nabla_a Q$ into a deterministic actor, which chases the single $\arg\max_a Q(s,a)$ — a unimodal MAP action that cannot represent multiple modes. Normalized advantage functions force $Q$ to be quadratic in the action so the max is closed-form, but the implied action distribution is then a single Gaussian by construction. Entropy-regularized policy gradients like PGQ add a per-step entropy bonus, but they maximize entropy only at the *current* timestep rather than over the trajectory, so they never plan toward future states where many good options will remain. And the obvious fix — letting a neural net output the *parameters* of a stochastic family — does not help: a net that emits $\mu(s)$ and $\Sigma(s)$ still produces $\mathcal{N}(\mu(s),\Sigma(s))$, which is structurally unimodal no matter how much capacity sits in the heads. The expressive maximum-entropy methods that do capture rich distributions (Z-learning, MaxEnt IRL, message passing) live in discrete or tabular settings and do not scale to high-dimensional continuous actions. What is missing is a method that is genuinely multimodal in continuous spaces *and* fast enough to sample online.

I propose Soft Q-Learning. The objective is the maximum-entropy one, which augments reward with the policy's entropy at every visited state, $\pi^*_{\text{MaxEnt}} = \arg\max_\pi \sum_t \mathbb{E}_{(s,a)\sim\rho_\pi}[\,r(s,a) + \alpha\,\mathcal{H}(\pi(\cdot|s))\,]$, where $\alpha$ trades entropy against reward. Keeping the entropy *inside* the long-horizon sum is the whole point: it makes the agent value reaching states where it will *also* have many good options, unlike myopic Boltzmann exploration which maximizes entropy only at the current leaf. The defining structural fact is that the optimal policy of this objective is forced to be an energy-based policy whose energy is a soft $Q$-function. To see that it is forced rather than assumed, define the soft $Q$-value $Q^\pi_{\text{soft}}(s,a)$ as the entropy-augmented return from taking $a$ in $s$, and ask for the one-step policy-improvement move: pick the new action distribution $\rho$ at each state to maximize entropy plus value, $\mathcal{H}(\rho(\cdot|s)) + \mathbb{E}_{a\sim\rho}[Q^\pi_{\text{soft}}(s,a)]$. Recognizing that $\mathcal{H}(\rho) + \mathbb{E}_\rho[Q] = \mathbb{E}_\rho[Q - \log\rho]$ and defining $\tilde\pi(a|s)\propto\exp(Q^\pi_{\text{soft}}(s,a))$ with normalizer $Z(s)$, the objective rewrites as

$$\mathcal{H}(\rho) + \mathbb{E}_{a\sim\rho}[Q^\pi_{\text{soft}}] = \mathbb{E}_\rho\!\left[\log\frac{Z\,\tilde\pi}{\rho}\right] = -\,\mathrm{KL}\big(\rho \,\|\, \tilde\pi\big) + \log Z(s),$$

so maximizing entropy-plus-value is exactly minimizing $\mathrm{KL}(\rho\,\|\,\tilde\pi)$, whose minimum is $\rho=\tilde\pi$. The improving policy is the Boltzmann form $\tilde\pi(a|s)\propto\exp(Q^\pi_{\text{soft}}(s,a))$ — the $Q$-function emerges as the negative energy, not by fiat. This improvement step genuinely raises the soft value everywhere: telescoping the entropy-plus-value bound forward, replacing $\pi$ by $\tilde\pi$ at each future step, gives $Q^\pi_{\text{soft}}\le Q^{\tilde\pi}_{\text{soft}}$, so iterating $\pi_{i+1}\propto\exp(Q^{\pi_i}_{\text{soft}})$ converges to a fixed point that is itself energy-based. Hence the *optimal* maxent policy must have this shape.

The normalizer is the soft value, defined as the log-partition over actions so the policy is properly normalized:

$$V^*_{\text{soft}}(s) = \alpha \log \int_{\mathcal{A}} \exp\!\big(Q^*_{\text{soft}}(s,a')/\alpha\big)\, da', \qquad \pi^*(a|s) = \exp\!\big((Q^*_{\text{soft}}(s,a) - V^*_{\text{soft}}(s))/\alpha\big).$$

This $V$ is a log-sum-exp — a *soft* maximum over actions — and as $\alpha\to 0$ it concentrates on $\max_a Q$, recovering the ordinary greedy value, so the soft theory contains the hard theory as a limit. Substituting $\pi=\exp(Q_{\text{soft}}-V_{\text{soft}})$ into the definition of $Q^\pi_{\text{soft}}$, the next-state entropy term $\mathcal{H}(\pi(\cdot|s')) = V_{\text{soft}}(s') - \mathbb{E}_{a'\sim\pi}[Q_{\text{soft}}(s',a')]$ combines with the expected-$Q$ term, and everything collapses into the clean soft Bellman equation $Q_{\text{soft}}(s,a) = r + \gamma\,\mathbb{E}_{s'\sim p}[V_{\text{soft}}(s')]$ — the ordinary Bellman recursion with the hard $\max$ replaced by log-sum-exp. The backup operator $\mathcal{T}Q = r + \gamma\,\mathbb{E}_{s'}[\alpha\log\int\exp(Q/\alpha)]$ is a $\gamma$-contraction in sup-norm: if $\|Q_1-Q_2\|_\infty=\varepsilon$, then $Q_1\le Q_2+\varepsilon$ pointwise, the $\alpha$ outside the log cancels the $1/\alpha$ inside the exponent so the soft values differ by at most $\varepsilon$, and the $\gamma$ discount yields $\|\mathcal{T}Q_1-\mathcal{T}Q_2\|_\infty\le\gamma\varepsilon$. So there is a unique fixed point and soft $Q$-iteration converges to it.

Two intractabilities remain in a continuous, high-dimensional space with a neural-net $Q_\theta$: the integral inside $V$, and sampling from $\pi\propto\exp(Q/\alpha)$. I dispatch them one at a time. For the value integral, any intractable integral becomes an expectation under something I can sample: introduce an arbitrary positive proposal $q_{a'}$ and importance-weight, $V^\theta_{\text{soft}}(s) = \alpha\log\mathbb{E}_{a'\sim q_{a'}}[\exp(Q^\theta_{\text{soft}}(s,a')/\alpha)/q_{a'}(a')]$. With a uniform proposal $q_{a'}=(1/2)^d$ on $[-1,1]^d$, the estimate is $V(s)=\alpha(\operatorname{logsumexp}_j Q(s,a_j)/\alpha - \log N + d\log 2)$ over $N$ uniform samples. For the critic, the fixed point "$Q=\mathcal{T}Q$ at *all* $(s,a)$" is an infinite set of equality constraints, not something I can train by SGD; I convert it using the identity that $g_1(x)=g_2(x)\;\forall x \iff \mathbb{E}_{x\sim q}[(g_1-g_2)^2]=0$ for any strictly positive $q$. Driving that squared-error expectation to zero gives the soft-Bellman-error regression

$$J_Q(\theta) = \mathbb{E}_{s\sim q_s,\,a\sim q_a}\!\Big[\tfrac{1}{2}\big(\hat Q^{\bar\theta}_{\text{soft}}(s,a) - Q^\theta_{\text{soft}}(s,a)\big)^2\Big], \qquad \hat Q^{\bar\theta}_{\text{soft}}(s,a)=r+\gamma\,\mathbb{E}_{s'\sim p}[V^{\bar\theta}_{\text{soft}}(s')],$$

with $\bar\theta$ the *delayed target parameters* — a frozen copy refreshed periodically, exactly as in deep $Q$-learning, because a bootstrapped target that moves with the network being trained is unstable. The critic is then ordinary bootstrapped value regression with the log-sum-exp soft value in the target.

The hard half is sampling from $\pi\propto\exp(Q^\theta/\alpha)$, needed both to act online and to generate the action samples. Running MCMC to convergence at every timestep is hopeless for real-time control, so I train a *separate sampling network* that emits an approximate draw in one forward pass: $a = f^\phi(\xi;s)$ with $\xi\sim\mathcal{N}(0,I)$, a state-conditioned net mapping noise to actions whose induced distribution $\pi^\phi(\cdot|s)$ should match the target EBM, minimizing $J_\pi(\phi;s)=\mathrm{KL}(\pi^\phi(\cdot|s)\,\|\,\exp((Q^\theta_{\text{soft}}-V^\theta_{\text{soft}})/\alpha))$. Minimizing a KL to an unnormalized target that I can only evaluate through $Q$ is exactly what Stein variational gradient descent solves: the perturbation $a\leftarrow a+\Delta(a)$ that most reduces $\mathrm{KL}(q\,\|\,p)$ within the unit ball of an RKHS with kernel $\kappa$ is a functional of the score $\nabla\log p$. Written for a particle $a=f^\phi(\xi;s)$ with $\log p(a)=Q^\theta_{\text{soft}}(s,a)/\alpha$ up to a constant, the descent direction is

$$\Delta f^\phi(\cdot\,;s) = \mathbb{E}_{a\sim\pi^\phi}\!\Big[\,\kappa(a,f^\phi(\cdot\,;s))\,\nabla_{a'}Q^\theta_{\text{soft}}(s,a')\big|_{a'=a} + \alpha\,\nabla_{a'}\kappa(a',f^\phi(\cdot\,;s))\big|_{a'=a}\,\Big].$$

The first term is the $\kappa$-weighted score: each particle drags its neighbors toward higher $Q$. The second, $\alpha\,\nabla\kappa$, is a *repulsion* that pushes nearby particles apart so they do not all collapse onto the single highest-$Q$ point — and that repulsive term is precisely what gives multimodality, spreading the particle cloud to cover every mode of $\exp(Q/\alpha)$. The $\alpha$ riding on the repulsion makes the balance correct: more temperature means stronger repulsion means a broader, higher-entropy spread, exactly as a larger entropy weight should. Since $\Delta f^\phi$ perturbs particle *positions* but I want to train network *weights*, I use the amortized trick: backpropagate the Stein direction into $\phi$ through the chain rule, $\partial J_\pi(\phi;s)/\partial\phi \propto \mathbb{E}_\xi[\Delta f^\phi(\xi;s)\cdot\partial f^\phi(\xi;s)/\partial\phi]$, optimized as a surrogate whose gradient ascent moves the sampler outputs along $\Delta f^\phi$. The result is a feed-forward sampler queryable in $O(1)$, and it is in fact an actor — a state-to-action net trained by backpropagating a gradient of the critic $Q$ into its weights, which is precisely DDPG's structure. The *only* difference is the $\alpha\,\nabla\kappa$ repulsion: drop it and keep a single particle and the actor estimates the MAP action and *becomes* DDPG with a soft critic; keep it and the actor captures the whole multimodal EBM. So an entropy-regularized actor-critic is just approximate soft $Q$-learning, and DDPG is its degenerate single-particle MAP case — which also explains why DDPG works off-policy, since as an approximate $Q$-learning maximizer it needs no on-policy data.

A few practical points make the implementation correct. The action space is bounded to $[-1,1]^d$ but SVGD lives on $\mathbb{R}^d$, so $f^\phi$ emits unbounded raw values and I squash with $\tanh$; the change of variables for $a=\tanh(u)$ adds $\sum_i \log(1-a_i^2)$ to $\log\pi$, so the log-density I differentiate for the score is $Q_{\text{soft}}(s,a) + \sum_i\log(1-a_i^2+\varepsilon)$, the $\varepsilon$ being numerical safety as $a\to\pm1$. The kernel is an RBF $\kappa(a,a')=\exp(-\|a-a'\|^2/h)$ with a median-heuristic bandwidth $h$ set so each particle's kernel neighborhood holds about the right number of particles, $h=(\text{median pairwise squared distance})/\log K$ clamped to a floor, giving $\nabla_a\kappa = -2(a-a')/h\cdot\kappa$ and a per-state adaptive scale. For the SVGD update I draw a set of particles per state and split it into $n_{\text{fixed}}$ particles (gradient stopped, used for the score and the first kernel argument) and $n_{\text{updated}}$ sampler outputs that receive the backpropagated direction. Because scaling the reward scales $1/\alpha$, the reward scale is simply the inverse-temperature knob, and with $\alpha$ folded into it the value estimate is just $\operatorname{logsumexp}$ over the uniform samples minus $\log N$ plus $d\log 2$.

```python
import numpy as np, tensorflow as tf
EPS = 1e-6

class StochasticNNPolicy:
    """Sampler/actor f^φ: (state, gaussian noise) → action."""
    def __init__(self, env_spec, hidden_layer_sizes, squash=True, name='policy'):
        self._action_dim = flat_dim(env_spec.action_space)
        self._observation_dim = flat_dim(env_spec.observation_space)
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._name = name

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples, self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)
        latents = tf.random_normal(latent_shape)
        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net((observations, latents),
                layer_sizes=self._layer_sizes, activation_fn=tf.nn.relu,
                output_nonlinearity=None)
        return tf.tanh(raw_actions) if self._squash else raw_actions


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    Kx, D = xs.get_shape().as_list()[-2:]
    Ky, D2 = ys.get_shape().as_list()[-2:]
    leading_shape = tf.shape(xs)[:-2]
    diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)
    dist_sq = tf.reduce_sum(diff**2, axis=-1)
    input_shape = tf.concat((leading_shape, [Kx * Ky]), axis=0)
    values, _ = tf.nn.top_k(tf.reshape(dist_sq, input_shape), k=(Kx * Ky // 2 + 1), sorted=True)
    medians_sq = values[..., -1]
    h = tf.maximum(medians_sq / np.log(Kx), h_min)
    h = tf.stop_gradient(tf.expand_dims(tf.expand_dims(h, -1), -1))
    kappa = tf.exp(-dist_sq / h)
    kappa_grad = -2 * diff / tf.expand_dims(h, -1) * tf.expand_dims(kappa, -1)
    return {"output": kappa, "gradient": kappa_grad}


class SQL:
    def _create_td_update(self):
        with tf.variable_scope('target'):
            target_actions = tf.random_uniform((1, self._value_n_particles, self._action_dim), -1, 1)
            q_value_targets = self.qf.output_for(self._next_observations_ph[:, None, :], target_actions)
        self._q_values = self.qf.output_for(self._observations_ph, self._actions_pl, reuse=True)
        next_value = tf.reduce_logsumexp(q_value_targets, axis=1)
        next_value -= tf.log(tf.cast(self._value_n_particles, tf.float32))   # − log N
        next_value += self._action_dim * np.log(2)                          # + d log 2 (uniform proposal)
        ys = tf.stop_gradient(self._reward_scale * self._rewards_pl
                              + (1 - self._terminals_pl) * self._discount * next_value)
        bellman_residual = 0.5 * tf.reduce_mean((ys - self._q_values)**2)
        if self._train_qf:
            self._training_ops.append(tf.train.AdamOptimizer(self._qf_lr).minimize(
                bellman_residual, var_list=self.qf.get_params_internal()))

    def _create_target_ops(self):
        source_params = self.qf.get_params_internal()
        target_params = self.qf.get_params_internal(scope='target')
        self._target_ops = [tf.assign(tgt, src) for tgt, src in zip(target_params, source_params)]

    def _create_svgd_update(self):
        actions = self.policy.actions_for(self._observations_ph,
            n_action_samples=self._kernel_n_particles, reuse=True)
        n_updated = int(self._kernel_n_particles * self._kernel_update_ratio)
        n_fixed = self._kernel_n_particles - n_updated
        fixed_actions, updated_actions = tf.split(actions, [n_fixed, n_updated], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        svgd_target_values = self.qf.output_for(self._observations_ph[:, None, :], fixed_actions, reuse=True)
        squash_correction = tf.reduce_sum(tf.log(1 - fixed_actions**2 + EPS), axis=-1)
        log_p = svgd_target_values + squash_correction
        grad_log_p = tf.stop_gradient(tf.expand_dims(tf.gradients(log_p, fixed_actions)[0], axis=2))
        kernel = self._kernel_fn(xs=fixed_actions, ys=updated_actions)
        kappa = tf.expand_dims(kernel["output"], dim=3)
        action_gradients = tf.reduce_mean(kappa * grad_log_p + kernel["gradient"], reduction_indices=1)
        gradients = tf.gradients(updated_actions, self.policy.get_params_internal(),
                                 grad_ys=action_gradients)
        surrogate_loss = tf.reduce_sum([tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self.policy.get_params_internal(), gradients)])
        if self._train_policy:
            self._training_ops.append(tf.train.AdamOptimizer(self._policy_lr).minimize(
                -surrogate_loss, var_list=self.policy.get_params_internal()))
```
