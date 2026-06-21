The problem is pre-training compute efficiency for a Transformer text encoder. The standard approach is masked language modeling: hide roughly 15% of tokens behind a `[MASK]` placeholder and train the encoder to reconstruct only those hidden tokens. This works well for learning bidirectional representations, but it is structurally wasteful. The encoder pays the full forward and backward cost of processing every token in the sequence, yet the loss is computed at only the masked positions. Per example, about 85% of positions are encoded and then discarded as far as the gradient is concerned. There is also a pre-train/fine-tune mismatch: the `[MASK]` symbol appears during pre-training but never during downstream fine-tuning.

The reason masked reconstruction cannot simply be densified is that the task "predict the hidden token" is only meaningful where something was actually hidden. At an unmasked position the answer is sitting in the input, so a generative objective is welded to a small corrupted subset. Raising the mask rate much above 15% degrades the context and makes reconstruction harder, so the density is capped by the task formulation itself. What is needed is a self-supervised task whose label is non-trivial at every position, so the loss can be summed over all tokens rather than a small subset.

The method I propose is ELECTRA, which stands for Efficiently Learning an Encoder that Classifies Token Replacements Accurately. The core idea is to replace generative reconstruction with a discriminative task called replaced token detection. Instead of asking the model to produce the missing token, ELECTRA asks it to judge, at every position, whether the token that is currently visible is the original token or a replacement. This judgment is well-defined at all positions, because every token can be either original or corrupted.

ELECTRA uses two networks. A small generator is a masked language model that sees the sequence with some positions replaced by `[MASK]` and produces a distribution over tokens at those positions. From that distribution it samples a replacement token for each masked position. Because the generator is itself a language model, the samples are contextually plausible, which forces the second network to actually understand language rather than rely on shallow local cues. The second network is the discriminator, which is the encoder we keep for downstream tasks. It reads the corrupted sequence, containing original tokens mixed with generator samples, and predicts for every position whether the token is real or replaced. The discriminator never sees the `[MASK]` symbol, which removes the pre-train/fine-tune mismatch as a side effect.

The generator is trained by ordinary maximum likelihood on the masked positions, just like a standard masked language model. The discriminator is trained by binary cross-entropy over all positions. Importantly, this is not a GAN. The generator is not trained to fool the discriminator, because sampling a discrete token blocks the gradient from the discriminator back into the generator. One could use a REINFORCE estimator to train the generator adversarially, but policy gradient over a large vocabulary is sample-inefficient and adversarial text generators tend to collapse to low-entropy, low-diversity outputs, which gives the discriminator poor negatives. Training the generator with maximum likelihood is simpler, more stable, and produces more diverse plausible replacements. The two losses are combined with a large weight on the discriminator loss, around 50, because a 30,000-way cross-entropy is numerically much larger than a single binary cross-entropy and the discriminator is the network we actually care about.

Several design choices matter. The generator should be smaller than the discriminator, typically one-quarter to one-third the width. A generator that is too strong makes detection too hard and forces the discriminator to model the generator rather than the language; a same-size generator also roughly doubles compute. Token and positional embeddings are shared between generator and discriminator, because the generator's full-vocabulary softmax densely updates the whole embedding table, giving the discriminator better embeddings than it would learn from its sparse token exposures. The two networks are trained jointly so that the generator starts weak and produces easy fakes early, giving the discriminator a natural curriculum. After pre-training, the generator is discarded and only the discriminator is fine-tuned.

The discriminator output has a clean interpretation related to density ratio estimation. At optimum, its score for a token, together with the generator's probability for that token, contains enough information to recover the true conditional token distribution. So the binary task is not merely a heuristic for detecting awkward phrases; it remains tied to modeling the data distribution. Empirically, the main gain comes from computing the loss over all positions; restricting the discriminator loss back to the masked subset removes most of the improvement.

```python
import tensorflow as tf
import collections


class PretrainingModel(object):
    """Transformer pre-training using replaced-token detection (ELECTRA)."""

    def __init__(self, config, features, is_training):
        self._config = config
        self._bert_config = training_utils.get_bert_config(config)
        embedding_size = (self._bert_config.hidden_size
                          if config.embedding_size is None
                          else config.embedding_size)

        # Prepare masked inputs; the helper records which positions are masked.
        unmasked_inputs = pretrain_data.features_to_inputs(features)
        masked_inputs = pretrain_helpers.mask(
            config, unmasked_inputs, config.mask_prob)

        # Smaller generator with tied embeddings.
        generator_config = get_generator_config(config, self._bert_config)
        generator = build_transformer(
            config, masked_inputs, is_training, generator_config,
            embedding_size=(None if config.untied_generator_embeddings
                            else embedding_size),
            untied_embeddings=config.untied_generator_embeddings,
            scope="generator")
        mlm_output = self._get_masked_lm_output(masked_inputs, generator)

        # Sample replacements and build the corrupted sequence.
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)

        # Combined loss: weighted generator MLM + weighted discriminator RTD.
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
        # Outcome-keyed label: correct resamples count as real.
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
        labelsf = tf.cast(labels, tf.float32)      # 1 = replaced, 0 = original
        weights = tf.cast(inputs.input_mask, tf.float32)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labelsf) * weights
        loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
        DiscOutput = collections.namedtuple("DiscOutput", ["loss"])
        return DiscOutput(loss=loss)


def sample_from_softmax(logits, disallow=None):
    """Exact categorical sample via the Gumbel-max trick."""
    if disallow is not None:
        logits -= 1000.0 * disallow
    u = tf.random.uniform(modeling.get_shape_list(logits), minval=0, maxval=1)
    gumbel = -tf.log(-tf.log(u + 1e-9) + 1e-9)
    return tf.one_hot(
        tf.argmax(tf.nn.softmax(logits + gumbel), -1, output_type=tf.int32),
        logits.shape[-1])


def get_generator_config(config, bert_config):
    """Make the generator smaller than the discriminator."""
    gen = modeling.BertConfig.from_dict(bert_config.to_dict())
    gen.hidden_size = int(round(bert_config.hidden_size
                                * config.generator_hidden_size))
    gen.num_hidden_layers = int(round(bert_config.num_hidden_layers
                                      * config.generator_layers))
    gen.intermediate_size = 4 * gen.hidden_size
    gen.num_attention_heads = max(1, gen.hidden_size // 64)
    return gen
```

After pre-training, the generator is discarded and only the discriminator is fine-tuned on downstream tasks such as GLUE or SQuAD.
