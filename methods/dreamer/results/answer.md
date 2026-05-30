# Dreamer

## Problem

Control an agent from raw images while spending as few real environment steps as
possible. Model-free agents are sample-inefficient on visual control for two reasons:
data is consumed once per on-policy update, and the score-function (REINFORCE) policy
gradient has variance that grows with the horizon. Dreamer attacks both by learning a
differentiable latent world model, generating imagined rollouts inside it, and training
the policy with the *analytic* gradient of multi-step returns propagated back through
the learned dynamics.

## Key idea

Three learned components and three interleaved loops:

1. **World model (RSSM).** A recurrent state-space model with a deterministic recurrent
   state and a stochastic state:
   - recurrent (memory): $h_t = f_\theta(h_{t-1}, s_{t-1}, a_{t-1})$ (a GRU),
   - prior / transition: $s_t \sim q_\theta(s_t \mid h_t)$,
   - posterior / representation: $s_t \sim p_\theta(s_t \mid h_t, o_t)$,
   - reward $q_\theta(r_t\mid s_t,h_t)$ and observation decoder $q_\theta(o_t\mid s_t,h_t)$.

   The model state fed to everything downstream is $\big[s_t;h_t\big]$. It is trained on
   real sequences by maximizing a variational / information-bottleneck lower bound:
   $$\mathcal{J}=\mathbb{E}\Big[\sum_t \ln q(o_t\mid s_t)+\ln q(r_t\mid s_t)-\beta\,\mathrm{KL}\big(p(s_t\mid h_t,o_t)\,\|\,q(s_t\mid h_t)\big)\Big],$$
   with the KL clipped at free nats. The split state solves the tension between
   remembering (deterministic $h$) and representing uncertain, multimodal futures
   (stochastic $s$, which the KL regularizes). A contrastive (NCE) variant replaces the
   image decoder with a state model $q(s_t\mid o_t)$ and the term
   $\ln q(s_t\mid o_t)-\ln\sum_{o'}q(s_t\mid o')$ to avoid pixel prediction.

2. **Imagination.** From posterior states $s_t$ of real sequence batches, roll the prior
   forward $H$ steps under the actor $a_\tau\sim q_\phi(a_\tau\mid s_\tau)$, predicting
   rewards and values — entirely in latent space, with no image decoding.

3. **Actor and value via analytic value gradients.**
   - $\lambda$-return target over the imagined rollout (bias/variance dial, bootstraps
     past the horizon with $v_\psi$):
     $$\mathrm{V}_{\mathrm N}^k(s_\tau)=\mathbb{E}\Big[\textstyle\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n+\gamma^{h-\tau}v_\psi(s_h)\Big],\ h=\min(\tau+k,t+H),$$
     $$\mathrm{V}_\lambda(s_\tau)=(1-\lambda)\textstyle\sum_{n=1}^{H-1}\lambda^{n-1}\mathrm{V}_{\mathrm N}^n(s_\tau)+\lambda^{H-1}\mathrm{V}_{\mathrm N}^H(s_\tau),$$
     computed by the backward recursion
     $\mathrm{V}_\lambda(s_\tau)=r_\tau+\gamma\big[(1-\lambda)v_\psi(s_{\tau+1})+\lambda\,\mathrm{V}_\lambda(s_{\tau+1})\big]$,
     $\mathrm{V}_\lambda(s_{t+H})=v_\psi(s_{t+H})$.
   - Value: $\min_\psi \mathbb{E}\big[\sum_\tau \tfrac12\|v_\psi(s_\tau)-\mathrm{sg}(\mathrm{V}_\lambda(s_\tau))\|^2\big]$ (semi-gradient, stop-grad target).
   - Actor: $\max_\phi \mathbb{E}\big[\sum_\tau \mathrm{V}_\lambda(s_\tau)\big]$, optimized by the
     **analytic** gradient $\nabla_\phi\mathbb{E}[\sum_\tau\mathrm{V}_\lambda(s_\tau)]$, where the
     gradient flows through the reward/value heads, through the reparameterized imagined
     states, through the transition, into the reparameterized actions
     $a_\tau=\tanh(\mu_\phi(s_\tau)+\sigma_\phi(s_\tau)\,\epsilon)$, and into $\phi$.

A state value (not an action value) suffices because the differentiable transition itself
supplies $\partial(\text{return})/\partial(\text{action})$ by the chain rule — unlike
DDPG/SAC, which need $Q(s,a)$ and differentiate only one step. The world model $\theta$ is
fixed during behavior learning. Terms are weighted by the cumulative product of $\gamma$
(or a predicted discount for early-terminating tasks). Discrete actions use straight-through
gradients.

## Algorithm

```
Initialize dataset D with S random seed episodes; init theta, phi, psi.
while not converged:
  for c in 1..C:                                    # dynamics + behavior learning
    draw B sequences {(a,o,r)} of length L from D
    s_t ~ p_theta(s_t | s_{t-1}, a_{t-1}, o_t)        # filter -> posterior states
    update theta on the variational bound             # reconstruct o,r + KL(post||prior)
    imagine {(s_tau, a_tau)} for H steps from each s_t under q_phi
    predict rewards and values; compute V_lambda(s_tau)
    phi <- phi + alpha * grad_phi  sum_tau V_lambda(s_tau)            # analytic, through dynamics
    psi <- psi - alpha * grad_psi  sum_tau 1/2 || v_psi(s_tau) - V_lambda(s_tau) ||^2
  o_1 = env.reset()                                 # environment interaction
  for t in 1..T:
    s_t ~ p_theta(s_t | s_{t-1}, a_{t-1}, o_t); a_t ~ q_phi(a_t | s_t) + exploration noise
    r_t, o_{t+1} = env.step(a_t)
  D <- D u {(o_t, a_t, r_t)}
```

Hyperparameters (continuous control): 30-dim diagonal-Gaussian latents, 200-dim
deterministic state, dense nets of 300 ELU units; batches of 50 sequences of length 50;
Adam at $6\times10^{-4}$ (model), $8\times10^{-5}$ (actor, value); grad-norm clip 100;
$\beta=1$, free nats 3; horizon $H=15$, $\gamma=0.99$, $\lambda=0.95$; action repeat 2;
$S=5$ seed episodes; Gaussian action noise $\mathcal{N}(0,0.3)$. Discrete control: categorical
actions with straight-through gradients, $H=10$, $\beta=0.1$, reward tanh-bounded, a predicted
discount head, $\epsilon$-greedy noise $0.4\to0.1$.

## Code

TensorFlow.

```python
class RSSM(tools.Module):
  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    self._stoch_size, self._deter_size, self._hidden_size = stoch, deter, hidden
    self._activation = act
    self._cell = tfkl.GRUCell(self._deter_size)

  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)      # model state = [s_t ; h_t]

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  def img_step(self, prev_state, prev_action):                  # PRIOR / transition
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']]); deter = deter[0]   # h_t = GRU(...)
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1); std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()  # reparameterized
    return {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}

  def obs_step(self, prev_state, prev_action, embed):           # POSTERIOR
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1); std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  def observe(self, embed, action, state=None):                 # filter a real sequence
    if state is None: state = self.initial(tf.shape(action)[0])
    embed, action = tf.transpose(embed, [1,0,2]), tf.transpose(action, [1,0,2])
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post  = {k: tf.transpose(v, [1,0,2]) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1,0,2]) for k, v in prior.items()}
    return post, prior


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
  # backward recursion: return_t = reward_t + pcont_t*[(1-l)*v_{t+1} + l*return_{t+1}]
  if isinstance(pcont, (int, float)): pcont = pcont * tf.ones_like(reward)
  next_values = tf.concat([value[1:], bootstrap[None]], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = tools.static_scan(
      lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
      (inputs, pcont), bootstrap, reverse=True)
  return returns


class Dreamer(tools.Module):
  def _imagine_ahead(self, post):
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in post.items()}            # roots = posterior states
    policy = lambda state: self._actor(
        tf.stop_gradient(self._dynamics.get_feat(state))).sample()   # detach state-as-input
    states = tools.static_scan(
        lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
        tf.range(self._c.horizon), start)                       # H latent steps, no pixels
    return self._dynamics.get_feat(states)

  def _train(self, data):
    with tf.GradientTape() as model_tape:                       # 1) world model
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      likes = tools.AttrDict()
      likes.image  = tf.reduce_mean(self._decode(feat).log_prob(data['image']))
      likes.reward = tf.reduce_mean(self._reward(feat).log_prob(data['reward']))
      div = tf.reduce_mean(tfd.kl_divergence(
          self._dynamics.get_dist(post), self._dynamics.get_dist(prior)))
      div = tf.maximum(div, self._c.free_nats)                  # free-nats clip
      model_loss = self._c.kl_scale * div - sum(likes.values())

    with tf.GradientTape() as actor_tape:                       # 2) actor: analytic value grad
      imag_feat = self._imagine_ahead(post)
      reward = self._reward(imag_feat).mode()
      pcont  = self._c.discount * tf.ones_like(reward)
      value  = self._value(imag_feat).mode()
      returns = lambda_return(
          reward[:-1], value[:-1], pcont[:-1],
          bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
      actor_loss = -tf.reduce_mean(discount * returns)          # grad flows through dynamics

    with tf.GradientTape() as value_tape:                       # 3) value: regress target
      value_pred = self._value(imag_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

    self._model_opt(model_tape, model_loss)                     # theta fixed in 2) and 3)
    self._actor_opt(actor_tape, actor_loss)
    self._value_opt(value_tape, value_loss)
```
