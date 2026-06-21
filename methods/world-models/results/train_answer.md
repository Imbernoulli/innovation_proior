An agent that acts from pixels has to solve two problems at once: it must compress the current scene into something compact, and it must predict how that scene will change when it acts. A large recurrent predictor is the natural instrument for the second job, because the environment is a stream of images and actions and a single frame cannot carry the temporal context. The trouble is training that predictor. Standard reinforcement learning drives everything through reward, and a sparse end-of-episode return is a terrible signal for fixing millions of recurrent weights — it tells me only whether a whole behavior worked, never what should have been predicted at any given step. That is why practical model-free agents keep their networks small enough to be optimized quickly, and pay for it in capacity. The internal-model precedent shows the right separation — learn a model from interaction, use it to reason about the future, train a controller against it — but it also exposes the central danger: if the learned model is imperfect, a controller optimized against it will exploit the model's errors and produce behavior that looks good in the model and fails in reality. Model-based policy search like PILCO mitigates model error with uncertainty estimates, but Gaussian-process dynamics do not scale to long histories of high-dimensional frames. What is missing is a way to let a high-capacity predictive model help control without forcing that whole model through the reward signal.

I propose World Models. The idea is to split the agent into a large part that learns from dense prediction and a tiny part that learns from reward, so that capacity and credit assignment never collide. Three components carry this: a convolutional variational autoencoder $V$ that compresses frames, a mixture-density recurrent network $M$ that predicts how the latent evolves under actions, and a small linear controller $C$ that is the only piece trained from return.

The first split comes from the image stream. Predicting in pixel space is expensive and tied to irrelevant visual detail, so I learn a compact latent state first. A plain autoencoder would compress frames but leave a jagged, unconstrained code, and that matters here because the temporal model will later sample latent vectors and feed them back as if they were real observations — so off-training-point latents must stay meaningful. A variational autoencoder gives me exactly that. Each $64 \times 64 \times 3$ frame maps to a Gaussian posterior; I sample $z = \mu + \exp(\tfrac12\log\sigma^2)\,\epsilon$, reconstruct, and pay a KL penalty toward $N(0,I)$. The reconstruction term preserves the visible state while the KL term keeps the code smooth and capacity-limited so generated latents are not immediately nonsensical. Concretely the loss is a squared-error reconstruction plus a KL with a tolerance floor,
$$\mathrm{KL} = -\tfrac12\sum_i\big(1 + \log\sigma_i^2 - \mu_i^2 - \exp(\log\sigma_i^2)\big),\qquad \mathrm{KL} \leftarrow \max(\mathrm{KL},\ \texttt{kl\_tolerance}\cdot z_{\text{size}}),$$
where the floor stops the model from spending effort driving the KL below a level that buys no reconstruction gain. The implementation parameterizes the posterior by $\log\sigma^2$ (`logvar`), so $\sigma = \exp(\log\sigma^2/2)$ throughout.

The temporal model is where the second failure mode bites. If I let an RNN predict a single next latent vector, it averages distinct futures — and in the survival task "a projectile is launched" and "no projectile is launched" are genuinely separate outcomes, not two noisy samples around one mean. Averaging them yields a state the controller can learn nothing useful from. So the next-latent prediction must be a distribution, and a multimodal one. A mixture-density output layer on top of the LSTM provides it: given the current latent $z_t$, the action $a_t$, and the recurrent state, the network emits mixture weights, means, and log standard deviations for the next latent, and training is just negative log-likelihood of the observed next latent under that mixture. The precise form matters. I do not use one joint mixture component shared across the whole latent vector; instead each scalar coordinate gets its own $K$-component one-dimensional Gaussian mixture, and the scalar losses are averaged over coordinates, time, and batch. For a scalar target $y$,
$$\mathcal{L} = -\log \sum_{k=1}^{K} \pi_k\, \frac{1}{\sqrt{2\pi}\,\sigma_k}\exp\!\left(-\tfrac12\Big(\frac{y-\mu_k}{\sigma_k}\Big)^2\right),$$
with $\log\pi$ normalized by a log-sum-exp over components and $\sigma_k = \exp(\mathrm{logstd}_k)$. This factorized mixture is deliberately not a full diagonal-Gaussian mixture that would sum log-density across coordinates inside one shared component index; it is simpler, and crucially the sampling code matches it by drawing a mixture component separately per coordinate. For the survival task $M$ also emits a restart/done logit alongside the latent prediction, trained with a sigmoid cross-entropy that up-weights the rare positive (restart) class by a `restart_factor` of 10 so termination events are not drowned out.

The controller can now stay tiny, and that is the whole point. It does not need to own visual or temporal capacity; it reads features that the big models already compute. It sees the current latent $z_t$, which says what the present frame contains, and the recurrent state, which summarizes the predictive context. A single linear map suffices: for car racing the features are $z$ concatenated with the LSTM output $h$, giving input width $32+256=288$ and $288\cdot 3 + 3 = 867$ parameters mapping to three bounded actions; for the survival task the features are $z$, the LSTM cell $c$, and the LSTM output $h$, giving input width $64+512+512=1088$ and a single-output map with 1088 weights and no bias. Because this search space is so small, CMA-ES is a reasonable optimizer — it needs only a scalar return per rollout, evaluates each candidate policy independently and in parallel, and never needs gradients through the environment or through time. That is what keeps reward credit assignment away from $V$ and $M$: the large models get dense unsupervised losses, and the controller gets black-box selection pressure through average rollout return, with a population of 64 candidates each scored on 16 seeded rollouts.

The same $M$ can serve as a latent environment, which makes controller training nearly free — no pixel renderer, no game engine. The virtual episode samples the next latent from the mixture, feeds it back into the RNN, returns a reward of one for each step survived, and terminates when the restart logit crosses zero or a 2100-step cap is reached. But this is exactly where model exploitation appears: a controller optimized inside an approximate simulator will hunt the simulator's weaknesses, and if $M$ is too deterministic the policy can find hidden-state trajectories where projectiles vanish or never launch — a simulator exploit, not a survival skill. The mixture model hands me the control knob to fight this. Sampling divides the mixture logits by a temperature $\tau$ and multiplies the Gaussian noise by $\sqrt{\tau}$, so $z_{\text{next}} = \mu_{k} + \exp(\mathrm{logstd}_{k})\sqrt{\tau}\,\epsilon$ with the component $k$ chosen from a temperature-softened categorical. Low $\tau$ collapses toward deterministic modes and makes the dream easy to exploit; raising $\tau$ makes generated futures noisier and forces the controller to survive a harder, less predictable simulation; too much $\tau$ destroys useful structure. The knob therefore trades realism against exploitability, and the transfer that works best uses $\tau = 1.15$.

Put together, the recipe is: train the frame VAE on random-policy observations; encode recorded frames into latent statistics and resample $z$ while training the MDN-RNN so it does not overfit a single latent draw; then train the small linear controller with CMA-ES — directly in the real environment using $[z, h]$ features for car racing, or inside the sampled dream for the survival task before transferring back. Car racing reaches $906 \pm 21$ over 100 trials; the survival task reaches $1092 \pm 556$ real-environment steps. The predictive model is trained from dense observation data, while the reward-trained search space stays small enough for black-box optimization — which is the constraint that made the whole thing work.

```python
mu = dense(enc, z_size)
logvar = dense(enc, z_size)
sigma = exp(logvar / 2.0)
z = mu + sigma * epsilon

r_loss = mean(sum((x - y) ** 2, axis=[1, 2, 3]))
kl = -0.5 * sum(1 + logvar - mu ** 2 - exp(logvar), axis=1)
kl = maximum(kl, kl_tolerance * z_size)
loss = r_loss + mean(kl)
```

```python
# car
NOUT = OUTWIDTH * K * 3
output = affine(lstm_output, NOUT)
output = reshape(output, [-1, K * 3])

logmix, mean, logstd = split(output, 3, axis=1)
logmix = logmix - reduce_logsumexp(logmix, axis=1, keepdims=True)
target = reshape(next_z, [-1, 1])

log_normal = -0.5 * ((target - mean) / exp(logstd)) ** 2
log_normal = log_normal - logstd - log(sqrt(2 * pi))
nll = -mean(reduce_logsumexp(logmix + log_normal, axis=1))
```

```python
NOUT = WIDTH * K * 3 + 1
restart_logit = output[:, 0]
mdn_output = output[:, 1:]
restart_loss = sigmoid_cross_entropy(labels=target_restart, logits=restart_logit)
restart_loss *= 1 + target_restart * (restart_factor - 1)  # restart_factor = 10
loss = z_nll + mean(restart_loss)
```

```python
weights = softmax(logmix / tau)          # per latent coordinate
k_j = sample_categorical(weights[j])     # one component per coordinate
z_next[j] = mean[j, k_j] + exp(logstd[j, k_j]) * sqrt(tau) * eps_j
```

```python
features = concat([z_t, h_t])            # 32 + 256 = 288
raw = features @ W + b                   # W: 288 x 3, b: 3
action = tanh(raw)
action[1] = (action[1] + 1.0) / 2.0      # gas in [0, 1]
action[2] = clip(action[2], 0.0, 1.0)    # brake in [0, 1]
```

```python
features = concat([z_t, c_t, h_t])       # 64 + 512 + 512 = 1088
action = tanh(features @ W)              # W: 1088 x 1, no bias
```
