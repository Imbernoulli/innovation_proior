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
- **$G$ smaller than $D$**: 1/4–1/2 width works best
  (Small/Large use 1/4; Base uses 1/3). A too-strong $G$ makes detection
  unlearnable and forces $D$ to spend capacity modeling $G$; same-size $G$
  doubles compute.
- **Joint training**: provides a curriculum — $G$ starts weak (easy fakes) and
  sharpens — avoiding the cold-start collapse of staged training.

## Code

The equations above define $D$ as probability "real." The
TensorFlow implementation uses the equivalent opposite convention in code:
the discriminator sigmoid is trained with `labels = is_fake_tokens`, so it is
probability "fake." The core ELECTRA path is:

```python
import tensorflow as tf
import collections


class PretrainingModel(object):
    """Transformer pre-training using replaced-token detection."""

    def __init__(self, config, features, is_training):
        self._config = config
        self._bert_config = training_utils.get_bert_config(config)
        embedding_size = (self._bert_config.hidden_size
                          if config.embedding_size is None
                          else config.embedding_size)

        # Dynamic masking records original ids; the helper masks 85% of the
        # selected positions and leaves the remaining selected positions unchanged.
        unmasked_inputs = pretrain_data.features_to_inputs(features)
        masked_inputs = pretrain_helpers.mask(
            config, unmasked_inputs, config.mask_prob)

        # Default ELECTRA path: a smaller, untied generator with tied embeddings.
        generator_config = get_generator_config(config, self._bert_config)
        generator = build_transformer(
            config, masked_inputs, is_training, generator_config,
            embedding_size=(None if config.untied_generator_embeddings
                            else embedding_size),
            untied_embeddings=config.untied_generator_embeddings,
            scope="generator")
        mlm_output = self._get_masked_lm_output(masked_inputs, generator)

        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)

        self.total_loss = config.gen_weight * mlm_output.loss
        discriminator = build_transformer(
            config, fake_data.inputs, is_training, self._bert_config,
            reuse=not config.untied_generator, embedding_size=embedding_size)
        disc_output = self._get_discriminator_output(
            fake_data.inputs, discriminator, fake_data.is_fake_tokens)
        self.total_loss += config.disc_weight * disc_output.loss

    def _get_masked_lm_output(self, inputs, model):
        reprs = pretrain_helpers.gather_positions(
            model.get_sequence_output(), inputs.masked_lm_positions)
        logits = get_token_logits(reprs, model.get_embedding_table(),
                                  self._bert_config)
        return get_softmax_output(logits, inputs.masked_lm_ids,
                                  inputs.masked_lm_weights,
                                  self._bert_config.vocab_size)

    def _get_fake_data(self, inputs, mlm_logits):
        inputs = pretrain_helpers.unmask(inputs)
        disallow = tf.one_hot(
            inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
            dtype=tf.float32) if self._config.disallow_correct else None
        sampled_tokens = tf.stop_gradient(pretrain_helpers.sample_from_softmax(
            mlm_logits / self._config.temperature, disallow=disallow))
        sampled_ids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_ids, masked = pretrain_helpers.scatter_update(
            inputs.input_ids, sampled_ids, inputs.masked_lm_positions)
        # Outcome-keyed label: a correct resample is real, not fake.
        labels = masked * (1 - tf.cast(
            tf.equal(updated_ids, inputs.input_ids), tf.int32))
        FakedData = collections.namedtuple(
            "FakedData", ["inputs", "is_fake_tokens", "sampled_tokens"])
        return FakedData(
            inputs=pretrain_data.get_updated_inputs(inputs, input_ids=updated_ids),
            is_fake_tokens=labels,
            sampled_tokens=sampled_tokens)

    def _get_discriminator_output(self, inputs, discriminator, labels):
        hidden = tf.layers.dense(
            discriminator.get_sequence_output(),
            units=self._bert_config.hidden_size,
            activation=modeling.get_activation(self._bert_config.hidden_act),
            kernel_initializer=modeling.create_initializer(
                self._bert_config.initializer_range))
        logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
        labelsf = tf.cast(labels, tf.float32)      # 1 = replaced/fake, 0 = original
        weights = tf.cast(inputs.input_mask, tf.float32)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labelsf) * weights
        loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
        DiscOutput = collections.namedtuple("DiscOutput", ["loss"])
        return DiscOutput(loss=loss)


def sample_from_softmax(logits, disallow=None):
    """Exact categorical sample from softmax(logits) via the Gumbel-max trick."""
    if disallow is not None:
        logits -= 1000.0 * disallow
    u = tf.random.uniform(modeling.get_shape_list(logits), minval=0, maxval=1)
    gumbel = -tf.log(-tf.log(u + 1e-9) + 1e-9)
    return tf.one_hot(
        tf.argmax(tf.nn.softmax(logits + gumbel), -1, output_type=tf.int32),
        logits.shape[-1])


def get_generator_config(config, bert_config):
    """Scale generator depth and width relative to the discriminator."""
    gen = modeling.BertConfig.from_dict(bert_config.to_dict())
    gen.hidden_size = int(round(bert_config.hidden_size
                                * config.generator_hidden_size))
    gen.num_hidden_layers = int(round(bert_config.num_hidden_layers
                                      * config.generator_layers))
    gen.intermediate_size = 4 * gen.hidden_size
    gen.num_attention_heads = max(1, gen.hidden_size // 64)
    return gen
```

After pre-training, discard the generator and fine-tune only the discriminator
(the ELECTRA encoder) on downstream tasks.
