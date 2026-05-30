## Research question

The goal is to pre-train a Transformer text encoder — a network that maps a
sequence of tokens to a sequence of contextualized vector representations — on
unlabeled text, so that the encoder can later be fine-tuned on downstream
language-understanding tasks (entailment, question answering, sentiment,
paraphrase) and beat training those tasks from scratch. Self-supervised
denoising pre-training of this kind has become the dominant recipe, and it
works: representations learned by corrupting text and asking the network to
recover it transfer well. But it is *expensive*. State-of-the-art encoders are
pre-trained for enormous amounts of compute, which both raises their cost and
puts them out of reach for anyone without a large accelerator budget.

The precise problem is therefore one of **pre-training compute efficiency**: for
a fixed model size, fixed corpus, and fixed compute budget, can we extract more
useful supervision per step than the prevailing objective does? A satisfying
answer has to identify *where* the prevailing objective wastes signal and design
a self-supervised task that does not waste it — without sacrificing (ideally
improving) the quality of the representations that transfer downstream.

A diagnostic fact about the prevailing objective sharpens this. The dominant
pre-training task selects a small subset of the input (typically 15% of tokens),
hides them, and trains the network to reconstruct only those hidden tokens. The
loss is computed *only at the hidden positions*. So per training example, only
~15% of the sequence positions ever produce a learning signal; the other ~85%
are encoded but never contribute to the loss. The network pays the full
forward/backward cost of processing every token but learns from a small fraction
of them. This is the inefficiency a better objective must remove.

There is also a secondary defect worth fixing. The hiding is done with a special
placeholder symbol that appears during pre-training but never during fine-tuning,
creating a mismatch between the input distribution the encoder is trained on and
the one it is later used on.

## Background

**Denoising-autoencoder pre-training.** The modern recipe casts representation
learning as denoising (Vincent et al. 2008): corrupt an input, then train a
network to undo the corruption. For text the corruption hides a subset of tokens
and the network reconstructs them from the surviving context. Because the
network sees both left and right context when reconstructing, it learns
*bidirectional* representations, which is what made this recipe more effective
than left-to-right language modeling for transfer.

**The masked-reconstruction objective and its loss-density limit.** Concretely,
pick a random set of positions m = [m_1, ..., m_k] (with
k = ceil(0.15n) for a length-n sequence), replace the tokens there with a
placeholder, and train the encoder to predict each original token via a softmax
over the vocabulary at those positions. The training loss is a sum of
cross-entropy terms over the $k$ masked positions only. The structural reason it
*must* be restricted to a small subset: the task is "what token was hidden
here?", which is only non-trivial at positions you actually hid — asking it at a
visible position is trivial because the answer is sitting in the input. So a
generative reconstruction task is intrinsically tied to a small corrupted subset,
and that is exactly why ~85% of positions per example carry no loss. This is the
load-bearing diagnostic.

**Diagnostic facts about that objective.** Two empirical observations about the
existing masked-reconstruction system frame the work. (1) Restricting the loss to
15% of positions is suspected to be wasteful, but it is not obvious — the network
*does* still read all tokens, so maybe the representations at unsupervised
positions still benefit. Whether the 15% restriction actually costs accuracy is
an empirical question that a controlled ablation can answer. (2) The placeholder
symbol creates a pre-train/fine-tune input mismatch. The standard objective
already includes a partial patch — of the 15% chosen positions, 80% are replaced
with the placeholder, 10% with a random token, and 10% left unchanged — but
whether this heuristic fully removes the mismatch is, again, an empirical
question.

**Noise-contrastive estimation.** Instead of fitting a normalized distribution,
one can learn it implicitly by training a binary classifier to separate genuine
data points from samples drawn from a known noise distribution (Gutmann &
Hyvärinen 2010). The classifier's logits, at optimum, encode the ratio between
the data density and the noise density, so a discriminative real-vs-noise task
can teach a model about the data distribution without an explicit softmax over
all outcomes. This reframing — *generative estimation recast as discriminative
classification against a proposal distribution* — is a key conceptual tool.

**Contrastive / negative-sampling word representations.** Early word-embedding
methods used exactly this move. The continuous-bag-of-words model with negative
sampling (Mikolov et al. 2013) predicts a token from its surrounding context, but
rather than a full softmax it trains a binary classifier to distinguish the true
token from tokens drawn from a simple unigram-frequency proposal distribution.
The encoder there is a bag of vectors and the proposal is fixed (corpus
frequencies); the structure — *binary classification of real token vs.
proposal-distribution token, conditioned on context* — is the template.

**Generative adversarial networks.** GANs (Goodfellow et al. 2014) couple a
generator that produces synthetic samples with a discriminator that tells real
from synthetic, training the generator to *fool* the discriminator by
back-propagating the discriminator's gradient through the generator's output.
This requires the generator's output to be differentiable. For discrete data such
as text this breaks: sampling a token is non-differentiable, so the
generator-fooling gradient cannot flow through the sampled token. GANs for text
exist but, at the time, consistently lag plain maximum-likelihood training
(Caccia et al. 2018); two recurring failure modes are poor sample quality and
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
baseline for variance reduction. This is the only obvious route to training a
discrete text generator against a non-differentiable downstream reward.

## Baselines

**Masked language modeling (Devlin et al. 2019).** Corrupt 15% of positions to a
placeholder and train the encoder to reconstruct the originals with a
per-position vocabulary softmax. *Core math:* for masked set $\bm{m}$, minimize
$\sum_{i\in\bm{m}} -\log p(x_i\mid x^{\text{masked}})$, where
$p(x_t\mid x)=\mathrm{softmax}(e(\cdot)^\top h(x)_t)$. *Gaps:* (a) the loss is
defined over only ~15% of positions, so most of the per-step compute produces no
learning signal; (b) the placeholder creates a pre-train/fine-tune mismatch that
the 80/10/10 heuristic only partly addresses.

**Permutation / two-stream language modeling (Yang et al. 2019).** Avoids the
placeholder by factorizing the joint over a random permutation of positions with
two attention streams, so the input is auto-regressively generated in random
order. *Gap:* it still only predicts ~15% of the positions per example, so it
inherits the same loss-density inefficiency; and it adds the machinery of two
attention streams.

**Word2Vec CBOW with negative sampling (Mikolov et al. 2013).** Predict a token
from context as a binary real-vs-proposal classification with a unigram noise
distribution. *Gap:* the encoder is a shallow bag-of-vectors with no
contextualization, and the proposal is a fixed unigram distribution, so the
negatives are easy and the learned representations are static word vectors, not
deep contextual ones.

**Noise-contrastive estimation (Gutmann & Hyvärinen 2010).** Learn a
distribution by classifying data against samples from a fixed noise distribution.
*Gap:* formulated for density estimation with a *fixed* noise distribution; it
does not, by itself, specify a deep bidirectional text encoder or a *learned*,
context-dependent proposal distribution that keeps the negatives challenging.

**Adversarial text generation / GAN discriminators (Goodfellow et al. 2014;
Radford et al. 2015; Fedus et al. 2018).** Train a generator to fool a
discriminator, reuse the discriminator's features. *Gap:* the generator-fooling
gradient cannot back-propagate through discrete token sampling; the RL workaround
is sample-inefficient and adversarial text generators tend to collapse to
low-entropy outputs, so adversarial text models underperform maximum-likelihood
ones.

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
fine-tuning runs from a fixed checkpoint is the natural protocol. Crucially for
this work, because the question is about *efficiency*, the natural comparison
axis is downstream score versus pre-training **compute** (FLOPs), not just versus
training steps — counting the generator's compute as part of the cost. Natural
ablation axes: which positions the loss covers, how the corruption is produced,
the relative weight of competing loss terms, and the size of any auxiliary
proposal network.

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
