## Research question

The goal is to pre-train a Transformer text encoder — a network that maps a
sequence of tokens to a sequence of contextualized vector representations — on
unlabeled text, so that the encoder can later be fine-tuned on downstream
language-understanding tasks (entailment, question answering, sentiment,
paraphrase) and beat training those tasks from scratch. Self-supervised
denoising pre-training of this kind has become the dominant recipe:
representations learned by corrupting text and asking the network to recover it
transfer well. State-of-the-art encoders are pre-trained for large amounts of
compute.

The question is one of **pre-training compute efficiency**: for a fixed model
size, fixed corpus, and fixed compute budget, what self-supervised objective
should the encoder be trained on, and how much useful supervision per step does
that objective provide? The natural place to look is the structure of the
prevailing objective and the supervision signal it produces per training
example.

The dominant pre-training task selects a small subset of the input (typically
15% of tokens), hides them, and trains the network to reconstruct only those
hidden tokens. The loss is computed *only at the hidden positions*. Per training
example, ~15% of the sequence positions produce a learning signal; the other
~85% are encoded but do not contribute to the loss. The hiding is done with a
special placeholder symbol that appears during pre-training but not during
fine-tuning.

## Background

**Denoising-autoencoder pre-training.** The modern recipe casts representation
learning as denoising (Vincent et al. 2008): corrupt an input, then train a
network to undo the corruption. For text the corruption hides a subset of tokens
and the network reconstructs them from the surviving context. Because the
network sees both left and right context when reconstructing, it learns
*bidirectional* representations.

**The masked-reconstruction objective.** Concretely, pick a random set of
positions m = [m_1, ..., m_k] (with k = ceil(0.15n) for a length-n sequence),
replace the tokens there with a placeholder, and train the encoder to predict
each original token via a softmax over the vocabulary at those positions. The
training loss is a sum of cross-entropy terms over the $k$ masked positions. The
task is "what token was hidden here?", which is well defined at positions that
were hidden; the network still reads all tokens when forming the contextual
representations used to make those predictions.

**Properties of that objective.** Two facts about the existing
masked-reconstruction system frame the work. (1) The loss covers the 15% of
positions that were hidden; the network reads all tokens but is scored on that
subset. (2) The placeholder symbol differs between pre-training and fine-tuning.
The standard objective uses a corruption recipe: of the 15% chosen positions,
80% are replaced with the placeholder, 10% with a random token, and 10% left
unchanged.

**Noise-contrastive estimation.** Instead of fitting a normalized distribution,
one can learn it implicitly by training a binary classifier to separate genuine
data points from samples drawn from a known noise distribution (Gutmann &
Hyvärinen 2010). The classifier's logits, at optimum, encode the ratio between
the data density and the noise density, so a discriminative real-vs-noise task
can teach a model about the data distribution without an explicit softmax over
all outcomes.

**Contrastive / negative-sampling word representations.** Early word-embedding
methods used this move. The continuous-bag-of-words model with negative sampling
(Mikolov et al. 2013) predicts a token from its surrounding context, but rather
than a full softmax it trains a binary classifier to distinguish the true token
from tokens drawn from a simple unigram-frequency proposal distribution. The
encoder there is a bag of vectors and the proposal is fixed (corpus
frequencies).

**Generative adversarial networks.** GANs (Goodfellow et al. 2014) couple a
generator that produces synthetic samples with a discriminator that tells real
from synthetic, training the generator to *fool* the discriminator by
back-propagating the discriminator's gradient through the generator's output.
This requires the generator's output to be differentiable. For discrete data
such as text, sampling a token is non-differentiable, so the
generator-fooling gradient does not flow through the sampled token. GANs for
text exist; at the time they typically lag plain maximum-likelihood training
(Caccia et al. 2018), with reported failure modes of poor sample quality and
low-entropy, low-diversity (mode-collapsed) generators. Separately, using a GAN
discriminator's learned features for downstream tasks had been proposed (Radford
et al. 2015). A related text model trains a generator to fill in deleted tokens
(MaskGAN, Fedus et al. 2018).

**Transformer encoders.** The backbone is the Transformer (Vaswani et al. 2017):
stacked multi-head self-attention and feed-forward blocks producing one
contextualized vector per input position. All of the above pre-training recipes
sit on top of this encoder; the contribution being sought is in the
*pre-training objective*, not the architecture.

**Score-function reinforcement learning for discrete generators.** When a
training signal must pass through a discrete sampling step, the score-function /
REINFORCE estimator (Williams 1992) gives an unbiased gradient of an expected
reward with respect to the sampler's parameters, typically with a learned
baseline for variance reduction. This is one route to training a discrete text
generator against a non-differentiable reward.

## Baselines

**Masked language modeling (Devlin et al. 2019).** Corrupt 15% of positions to a
placeholder and train the encoder to reconstruct the originals with a
per-position vocabulary softmax. *Core math:* for masked set $\bm{m}$, minimize
$\sum_{i\in\bm{m}} -\log p(x_i\mid x^{\text{masked}})$, where
$p(x_t\mid x)=\mathrm{softmax}(e(\cdot)^\top h(x)_t)$. The loss is defined over
the ~15% hidden positions; the placeholder is handled with the 80/10/10
corruption recipe.

**Permutation / two-stream language modeling (Yang et al. 2019).** Avoids the
placeholder by factorizing the joint over a random permutation of positions with
two attention streams, so the input is auto-regressively generated in random
order. It predicts ~15% of the positions per example.

**Word2Vec CBOW with negative sampling (Mikolov et al. 2013).** Predict a token
from context as a binary real-vs-proposal classification with a unigram noise
distribution. The encoder is a shallow bag-of-vectors and the proposal is a
fixed unigram distribution; the learned representations are static word vectors.

**Noise-contrastive estimation (Gutmann & Hyvärinen 2010).** Learn a
distribution by classifying data against samples from a fixed noise
distribution. Formulated for density estimation with a fixed noise distribution.

**Adversarial text generation / GAN discriminators (Goodfellow et al. 2014;
Radford et al. 2015; Fedus et al. 2018).** Train a generator to fool a
discriminator, reuse the discriminator's features. The generator-fooling
gradient does not back-propagate through discrete token sampling; the RL route
trains the generator against the discriminator's reward.

## Evaluation settings

Pre-training data is large unlabeled English text: a Wikipedia + BooksCorpus
corpus of ~3.3B tokens (Zhu et al. 2015), with a larger ~33B-token corpus
(adding ClueWeb, CommonCrawl, Gigaword) available for the largest models. The
downstream yardsticks are the GLUE benchmark (Wang et al. 2018) — nine
sentence/sentence-pair classification and regression tasks spanning entailment
(MNLI, RTE, QNLI), paraphrase (MRPC, QQP), similarity (STS), sentiment (SST),
and linguistic acceptability (CoLA) — and the SQuAD question-answering datasets
1.1 (answer-span selection) and 2.0 (with unanswerable questions) (Rajpurkar et
al. 2016). Metrics: accuracy for most GLUE tasks, Matthews correlation for CoLA,
Spearman correlation for STS, typically averaged; Exact-Match and F1 for SQuAD.
The fine-tuning protocol adds a small linear classifier (GLUE) or a span head
(SQuAD) on the pre-trained encoder. Because several datasets are small,
fine-tuned scores vary with random seed, so reporting the median over multiple
fine-tuning runs from a fixed checkpoint is the natural protocol. Because the
question is about *efficiency*, the natural comparison axis is downstream score
versus pre-training **compute** (FLOPs), not just versus training steps —
counting any auxiliary network's compute as part of the cost. Natural ablation
axes: which positions the loss covers, how the corruption is produced, the
relative weight of competing loss terms, and the size of any auxiliary proposal
network.

## Code framework

The primitives that already exist: a Transformer encoder module that maps token
ids to per-position hidden states, a masking utility that picks random positions
and records the original ids, a vocabulary-softmax reconstruction head, a
standard optimizer with warmup and linear decay, and a pre-training loop over a
large corpus. The open objective hook receives an unlabeled batch and returns the
training loss plus the encoder whose states will later be fine-tuned.

```python
import tensorflow as tf

class TransformerEncoder:
    """Maps input_ids -> per-position contextual hidden states h[1..n].
    Standard stacked self-attention + feed-forward encoder."""
    def __init__(self, config, input_ids, input_mask, segment_ids, scope): ...
    def get_sequence_output(self): ...      # [B, n, hidden]
    def get_embedding_table(self): ...      # [vocab, embedding_size]

def mask_inputs(config, inputs, mask_prob):
    """Pick k = ceil(mask_prob * n) random positions; record their ids;
    apply the existing dynamic-masking recipe. Returns inputs plus the
    chosen positions and original ids."""
    ...

def softmax_reconstruction_head(hidden_at_positions, embedding_table, target_ids):
    """Vocabulary softmax + cross-entropy at the given positions (the existing
    masked-reconstruction loss)."""
    ...

def optimizer(loss, lr, num_steps, warmup):
    """AdamW with linear warmup and linear decay."""
    ...

def pretraining_objective(config, features, is_training):
    """Return (loss, encoder_to_finetune) for a self-supervised batch."""
    # TODO: choose the corruption process, any auxiliary computation, and the loss.
    pass

def pretrain(config, dataset):
    for features in dataset:
        loss, encoder = pretraining_objective(config, features, is_training=True)
        optimizer(loss, config.learning_rate, config.num_train_steps,
                  config.warmup_steps)
    return encoder
```
