# Dreamer

## Method

Dreamer learns a latent world model from replay, then learns an actor and state-value
function entirely inside imagined latent rollouts. The central update is an analytic
pathwise actor gradient: value estimates are backpropagated through predicted rewards,
latent transitions, sampled latent states, and sampled actions.

The world model is an RSSM with deterministic memory and stochastic latent state:

$$h_t=f_\theta(h_{t-1},s_{t-1},a_{t-1}),$$

$$s_t\sim q_\theta(s_t\mid h_t)\quad\text{(prior / transition)},\qquad
s_t\sim p_\theta(s_t\mid h_t,o_t)\quad\text{(posterior / representation)}.$$

The feature used by reward, value, and actor heads is $[s_t;h_t]$. With image
reconstruction, the representation objective is

$$
\mathcal J_{\rm REC}=
\mathbb E\!\left[\sum_t
\log q_\theta(o_t\mid s_t,h_t)+\log q_\theta(r_t\mid s_t,h_t)
-\beta\,\mathrm{KL}\!\left(
p_\theta(s_t\mid h_t,o_t)\,\|\,q_\theta(s_t\mid h_t)
\right)\right].
$$

The contrastive variant replaces the image decoder term with

$$\mathcal J_{\rm S}^t=\log q_\theta(s_t\mid o_t)-\log\sum_{o'}q_\theta(s_t\mid o').$$

Starting from posterior states of replayed sequences, Dreamer rolls the prior forward
for horizon $H$ under $a_\tau\sim q_\phi(a_\tau\mid s_\tau,h_\tau)$, predicts rewards
and values in latent space, and does not decode images during behavior learning.

For an imagined trajectory, define

$$
\mathrm V_{\mathrm N}^k(s_\tau)=
\mathbb E\!\left[
\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n+
\gamma^{h-\tau}v_\psi(s_h)
\right],
\qquad h=\min(\tau+k,t+H).
$$

Dreamer uses the finite-horizon lambda return

$$
\mathrm V_\lambda(s_\tau)=
(1-\lambda)\sum_{k=1}^{H-1}\lambda^{k-1}\mathrm V_{\mathrm N}^k(s_\tau)
+\lambda^{H-1}\mathrm V_{\mathrm N}^H(s_\tau),
$$

equivalently computed backward as

$$
G_\tau=r_\tau+\gamma\big((1-\lambda)v_\psi(s_{\tau+1})+\lambda G_{\tau+1}\big),
\qquad G_{t+H}=v_\psi(s_{t+H}).
$$

With learned continuation probabilities, $\gamma$ is replaced by the predicted
per-step continuation-discount `pcont`. The edge cases are correct: $\lambda=0$
gives a one-step bootstrap, and $\lambda=1$ gives the discounted Monte Carlo return
with final bootstrap.

The value update regresses to stopped lambda-return targets:

$$
\min_\psi\ \mathbb E\!\left[
\sum_\tau \frac12\left\|v_\psi(s_\tau)-\mathrm{sg}(\mathrm V_\lambda(s_\tau))\right\|^2
\right].
$$

The actor update maximizes imagined values:

$$
\max_\phi\ \mathbb E\!\left[\sum_\tau \mathrm V_\lambda(s_\tau)\right].
$$

For continuous actions,

$$a_\tau=\tanh(\mu_\phi(s_\tau,h_\tau)+\sigma_\phi(s_\tau,h_\tau)\epsilon),
\qquad \epsilon\sim\mathcal N(0,I),$$

so the actor gradient is reparameterized. Discrete actions use straight-through
gradients. The world model is fixed during actor and value updates.

## Algorithm

```
Initialize replay D with S random seed episodes; initialize theta, phi, psi.
while training:
  repeat C update steps:
    draw B real sequences of length L from D
    infer posterior states and transition priors with the RSSM
    update theta on reconstruction/reward likelihoods minus posterior-prior KL
    imagine H latent steps from posterior states under the actor
    compute rewards, values, continuation weights, and lambda returns
    update phi by ascending imagined lambda returns through the dynamics
    update psi by regressing stopped lambda-return targets
  collect one real episode with actor actions plus exploration noise
  append the episode to D
```

Hyperparameters for continuous control: $S=5$ seed episodes, batches of 50
sequences of length 50, $H=15$, $\gamma=0.99$, $\lambda=0.95$, $\beta=1$, 3 free
nats, action repeat 2, Gaussian exploration noise $\mathcal N(0,0.3)$, Adam learning
rates $6\times10^{-4}$ for the model and $8\times10^{-5}$ for actor and value, and
gradient-norm clip 100. The reported configuration uses three dense layers of 300 ELU units for
non-convolutional functions; the Danijar TensorFlow 2 reference implementation uses
`num_units=400` while keeping stochastic size 30 and deterministic size 200.

## Reference Code

Code target: `methods/dreamer/code/danijar_dreamer`, commit `56d4d44`. The original
Google TensorFlow 1 release is also retrieved under `methods/dreamer/code/google_research_dreamer`.
The snippet below preserves the TF2 mechanics that matter for faithfulness.

```python
class RSSM(tools.Module):
  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  def img_step(self, prev_state, prev_action):
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    return {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}

  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior
```

```python
def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
  assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * tf.ones_like(reward)
  dims = list(range(reward.shape.ndims))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = tf.transpose(reward, dims)
    value = tf.transpose(value, dims)
    pcont = tf.transpose(pcont, dims)
  if bootstrap is None:
    bootstrap = tf.zeros_like(value[-1])
  next_values = tf.concat([value[1:], bootstrap[None]], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = tools.static_scan(
      lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
      (inputs, pcont), bootstrap, reverse=True)
  if axis != 0:
    returns = tf.transpose(returns, dims)
  return returns
```

```python
def _imagine_ahead(self, post):
  if self._c.pcont:
    post = {k: v[:, :-1] for k, v in post.items()}
  flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
  start = {k: flatten(v) for k, v in post.items()}
  policy = lambda state: self._actor(
      tf.stop_gradient(self._dynamics.get_feat(state))).sample()
  states = tools.static_scan(
      lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
      tf.range(self._c.horizon), start)
  return self._dynamics.get_feat(states)

def _train_behavior(self, post):
  imag_feat = self._imagine_ahead(post)
  reward = self._reward(imag_feat).mode()
  pcont = self._pcont(imag_feat).mean() if self._c.pcont else (
      self._c.discount * tf.ones_like(reward))
  value = self._value(imag_feat).mode()
  returns = tools.lambda_return(
      reward[:-1], value[:-1], pcont[:-1],
      bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
  discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
      [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
  actor_loss = -tf.reduce_mean(discount * returns)
  value_pred = self._value(imag_feat)[:-1]
  value_loss = -tf.reduce_mean(
      discount * value_pred.log_prob(tf.stop_gradient(returns)))
  return actor_loss, value_loss
```

The model loss in the TF2 reference is
`kl_scale * max(mean(KL(post || prior)), free_nats) - (image_like + reward_like + optional_pcont_like)`.
The original TensorFlow 1 release implements free nats differently for its objective terms
(`max(0, KL - free_nats)`), so this answer names the TF2 code path explicitly rather than
conflating the two implementations.
