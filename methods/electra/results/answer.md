# ELECTRA — Replaced Token Detection

## Problem

Masked-language-model (MLM) pre-training corrupts ~15% of input tokens to a
`[MASK]` placeholder and trains a Transformer encoder to reconstruct them. The
loss is computed *only at the masked positions*, so per example the encoder pays
the full cost of processing all tokens but learns from only ~15% of them. This
makes pre-training compute-inefficient. A secondary defect: `[MASK]` appears in
pre-training but never in fine-tuning, a pre-train/fine-tune input mismatch.

## Key idea

Replace the generative reconstruction task with a discriminative one —
**replaced token detection (RTD)** — whose loss is defined at *every* position:

1. A small **generator** $G$ (a masked LM) proposes plausible replacements at the
   masked positions by sampling from its output distribution.
2. The **discriminator** $D$ (the encoder we keep) reads the resulting sequence
   and predicts, for *every* token, whether it is the original token or a
   generator replacement.

Because the loss is a per-position binary classification over all $n$ tokens
rather than a softmax over only the ~15% masked subset, RTD extracts a much
denser learning signal per step. The discriminator also never sees `[MASK]`,
removing the mismatch. It is *not* a GAN: $G$ is trained by maximum likelihood,
not adversarially.

## Method

Given $\bm{x}=[x_1,\dots,x_n]$, sample a masked set $\bm{m}$ of size
$k=\lceil 0.15n\rceil$, form $\bm{x}^{\text{masked}}=\textsc{replace}(\bm{x},\bm{m},\texttt{[MASK]})$.

Generator (vocabulary softmax over its hidden states $h_G$):
$$p_G(x_t\mid \bm{x}^{\text{masked}}) = \frac{\exp(e(x_t)^\top h_G(\bm{x}^{\text{masked}})_t)}{\sum_{x'}\exp(e(x')^\top h_G(\bm{x}^{\text{masked}})_t)}.$$

Sample replacements $\hat{x}_i\sim p_G(x_i\mid \bm{x}^{\text{masked}})$ for $i\in\bm{m}$,
form $\bm{x}^{\text{corrupt}}=\textsc{replace}(\bm{x},\bm{m},\hat{\bm{x}})$.

Discriminator (sigmoid over its hidden states $h_D$):
$$D(\bm{x}^{\text{corrupt}},t)=\mathrm{sigmoid}(w^\top h_D(\bm{x}^{\text{corrupt}})_t).$$

Losses ($\mathbf{1}(\cdot)$ keyed on the *outcome*, so a resampled-correct token is "real"):
$$\mathcal{L}_{\text{MLM}}=\mathbb{E}\Big[\textstyle\sum_{i\in\bm{m}}-\log p_G(x_i\mid \bm{x}^{\text{masked}})\Big],$$
$$\mathcal{L}_{\text{Disc}}=\mathbb{E}\Big[\textstyle\sum_{t=1}^{n}-\mathbf{1}(x^{\text{corrupt}}_t=x_t)\log D(\bm{x}^{\text{corrupt}},t)-\mathbf{1}(x^{\text{corrupt}}_t\neq x_t)\log(1-D(\bm{x}^{\text{corrupt}},t))\Big].$$

Combined objective over corpus $\mathcal{X}$, expectations estimated with a single sample:
$$\min_{\theta_G,\theta_D}\sum_{\bm{x}\in\mathcal{X}}\mathcal{L}_{\text{MLM}}(\bm{x},\theta_G)+\lambda\,\mathcal{L}_{\text{Disc}}(\bm{x},\theta_D),\qquad \lambda=50.$$

The discriminator loss is **not** back-propagated into $G$ (the discrete sample
blocks the gradient). After pre-training, $G$ is discarded; only $D$ is
fine-tuned downstream (linear classifier for GLUE, span head for SQuAD).

Design choices and reasons:
- **Loss over all tokens** (not just 15%) is the main efficiency win; restricting
  $\mathcal{L}_{\text{Disc}}$ to the masked subset removes most of the gain.
- **$\lambda=50$**: the binary cross-entropy is numerically much smaller than the
  ~30k-way MLM cross-entropy; $\lambda$ rescales them onto comparable footing.
- **Train $G$ by MLE, not adversarially**: sampling is non-differentiable; the
  REINFORCE workaround is sample-inefficient (worse generator) and yields
  low-entropy samples — adversarial training underperforms MLE.
- **Share token + positional embeddings only**: $G$'s full-vocab softmax densely
  updates the whole embedding table, making it a better teacher; tying all
  weights would force $G$ and $D$ to be the same size.
- **$G$ at 1/4–1/2 of $D$'s width**: a too-strong $G$ makes detection unlearnable
  and forces $D$ to spend capacity modeling $G$; same-size $G$ doubles compute.
- **Joint training**: provides a curriculum — $G$ starts weak (easy fakes) and
  sharpens — avoiding the cold-start collapse of staged training.

## Code

The generator/discriminator construction, Gumbel-max sampling with
stop-gradient, the outcome-keyed labels, and the weighted combined loss:

```python
import tensorflow as tf
import collections


class PretrainingModel:
    """Replaced-token-detection pre-training: small generator + discriminator."""

    def __init__(self, config, features, is_training):
        self._config = config
        self._bert_config = get_bert_config(config)          # discriminator config
        embedding_size = config.embedding_size or self._bert_config.hidden_size

        # Corrupt: pick mask_prob (=0.15) of positions, place [MASK].
        inputs = features_to_inputs(features)
        masked_inputs = mask(config, inputs, config.mask_prob)

        # Generator = a small masked LM; embeddings tied with the discriminator.
        gen_config = get_generator_config(config, self._bert_config)
        generator = build_transformer(
            config, masked_inputs, is_training, gen_config,
            embedding_size=embedding_size, scope="generator")
        mlm_output = self._get_masked_lm_output(masked_inputs, generator)

        # Sample replacements and splice them in -> x^corrupt.
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)

        # Discriminator labels every position real/replaced.
        discriminator = build_transformer(
            config, fake_data.inputs, is_training, self._bert_config,
            embedding_size=embedding_size, scope="discriminator")
        disc_output = self._get_discriminator_output(
            fake_data.inputs, discriminator, fake_data.is_fake_tokens)

        # Combined loss: gen_weight=1.0, disc_weight=50.0.
        self.total_loss = (config.gen_weight * mlm_output.loss
                           + config.disc_weight * disc_output.loss)

    def _get_masked_lm_output(self, inputs, model):
        reprs = gather_positions(model.get_sequence_output(),
                                 inputs.masked_lm_positions)
        logits = get_token_logits(reprs, model.get_embedding_table(),
                                  self._bert_config)
        return get_softmax_output(logits, inputs.masked_lm_ids,
                                  inputs.masked_lm_weights,
                                  self._bert_config.vocab_size)

    def _get_fake_data(self, inputs, mlm_logits):
        inputs = unmask(inputs)
        sampled = tf.stop_gradient(                          # no gradient to G via the sample
            sample_from_softmax(mlm_logits / self._config.temperature))
        sampled_ids = tf.argmax(sampled, -1, output_type=tf.int32)
        updated_ids, masked = scatter_update(
            inputs.input_ids, sampled_ids, inputs.masked_lm_positions)
        # "Replaced" only where the sampled token differs from the original.
        is_fake = masked * (1 - tf.cast(
            tf.equal(updated_ids, inputs.input_ids), tf.int32))
        FakedData = collections.namedtuple(
            "FakedData", ["inputs", "is_fake_tokens"])
        return FakedData(
            inputs=get_updated_inputs(inputs, input_ids=updated_ids),
            is_fake_tokens=is_fake)

    def _get_discriminator_output(self, inputs, discriminator, labels):
        hidden = tf.layers.dense(
            discriminator.get_sequence_output(),
            units=self._bert_config.hidden_size,
            activation=get_activation(self._bert_config.hidden_act))
        logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
        labels_real = 1.0 - tf.cast(labels, tf.float32)      # 1 = original, 0 = replaced
        weights = tf.cast(inputs.input_mask, tf.float32)     # ignore padding
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels_real) * weights
        loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
        DiscOutput = collections.namedtuple("DiscOutput", ["loss"])
        return DiscOutput(loss=loss)


def sample_from_softmax(logits):
    """Exact categorical sample from softmax(logits) via the Gumbel-max trick."""
    u = tf.random.uniform(get_shape_list(logits), minval=0, maxval=1)
    gumbel = -tf.log(-tf.log(u + 1e-9) + 1e-9)
    return tf.one_hot(
        tf.argmax(tf.nn.softmax(logits + gumbel), -1, output_type=tf.int32),
        logits.shape[-1])


def get_generator_config(config, bert_config):
    """Generator is narrower than the discriminator (~1/4 hidden width)."""
    gen = bert_config.copy()
    gen.hidden_size = int(round(bert_config.hidden_size
                                * config.generator_hidden_size))   # 0.25
    gen.intermediate_size = 4 * gen.hidden_size
    gen.num_attention_heads = max(1, gen.hidden_size // 64)
    return gen
```

After pre-training, discard the generator and fine-tune only the discriminator
(the ELECTRA encoder) on downstream tasks.
