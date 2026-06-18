# World Models

## Method

Split the agent into a large prediction-trained model and a tiny
reward-trained controller:

1. Train a convolutional VAE \(V\) on resized \(64 \times 64 \times 3\) frames.
2. Use \(V\) to encode recorded rollouts into latent statistics.
3. Train an MDN-RNN \(M\) to predict the next latent distribution from current
   latent, action, and recurrent state.
4. Train a small linear controller \(C\) with CMA-ES using rollout return.
5. For the survival task, wrap \(M\) as a latent environment and train \(C\)
   inside sampled model rollouts before transferring it to the real environment.

## V: Frame VAE

The reference implementation uses TensorFlow `logvar`, not `logsigma`:

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

Architecture: four stride-2 conv layers with channels 32, 64, 128, 256; dense
heads for \(\mu\) and \(\log\sigma^2\); dense decoder to \(1 \times 1 \times
1024\); four stride-2 transpose conv layers with channels 128, 64, 32, 3 and a
sigmoid output. Latent size is 32 for car racing and 64 for Doom.

## M: MDN-RNN

Car racing uses input width \(32+3=35\), output width 32, LSTM size 256, and
\(K=5\) mixtures. Doom uses latent width 64, LSTM size 512, and \(K=5\); its RNN
input includes latent \(z_t\), scalar action \(a_t\), and restart flag.

The official MDN output is a factorized scalar mixture, not one shared joint
mixture component for the whole latent vector:

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

Doom adds one restart/done logit:

```python
NOUT = WIDTH * K * 3 + 1
restart_logit = output[:, 0]
mdn_output = output[:, 1:]
restart_loss = sigmoid_cross_entropy(labels=target_restart, logits=restart_logit)
restart_loss *= 1 + target_restart * (restart_factor - 1)  # restart_factor = 10
loss = z_nll + mean(restart_loss)
```

Sampling uses the same factorization:

```python
weights = softmax(logmix / tau)          # per latent coordinate
k_j = sample_categorical(weights[j])     # one component per coordinate
z_next[j] = mean[j, k_j] + exp(logstd[j, k_j]) * sqrt(tau) * eps_j
```

For Doom, the virtual episode terminates when `restart_logit > 0`, or when the
2100-step cap is reached. The virtual reward is 1 for each alive step.

## C: Linear Controller

Car racing uses \(z_t\) and the LSTM output state \(h_t\):

```python
features = concat([z_t, h_t])            # 32 + 256 = 288
raw = features @ W + b                   # W: 288 x 3, b: 3
action = tanh(raw)
action[1] = (action[1] + 1.0) / 2.0      # gas in [0, 1]
action[2] = clip(action[2], 0.0, 1.0)    # brake in [0, 1]
```

Parameter count: \(288 \cdot 3 + 3 = 867\).

Doom uses \(z_t\), LSTM cell \(c_t\), and LSTM output \(h_t\):

```python
features = concat([z_t, c_t, h_t])       # 64 + 512 + 512 = 1088
action = tanh(features @ W)              # W: 1088 x 1, no bias
```

Parameter count: 1088. In the real Doom wrapper, actions below \(-1/3\) move
left, actions above \(1/3\) move right, and the middle interval stays still.

## Training And Evaluation

The paper-level protocol collects 10,000 random-policy rollouts, trains \(V\),
trains \(M\) from resampled VAE latents and recorded actions, then trains \(C\)
with CMA-ES. CMA-ES uses a population of 64; each candidate is evaluated on 16
rollouts with different seeds; fitness is average cumulative return.

Car racing trains \(C\) in the real environment with features \([z,h]\), reaching
906 +/- 21 over 100 trials in the reported result. Doom trains \(C\) in the
latent dream environment, using temperature to reduce model exploitation. Very
low temperature collapses the dream toward deterministic easy modes; a moderate
temperature makes the simulator harder and transfers better. The reported best
transfer uses \(\tau=1.15\), with 1092 +/- 556 real-environment survival steps
over 100 trials.

## Reference-Faithfulness Notes

The current artifact follows the official `hardmaru/WorldModelsExperiments`
implementation for signs, shapes, and cases. The largest correction is the MDN
factorization: the code optimizes a scalar mixture likelihood independently over
latent coordinates, while a joint diagonal-Gaussian mixture would instead sum
log-density over coordinates inside each shared mixture component. The VAE KL
uses `logvar` and a KL tolerance floor. Doom predicts restart/done, not reward;
the reward in the dream environment is the survival reward supplied by the
wrapper.
