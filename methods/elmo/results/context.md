# Context: contextual word representations for NLP

## Research question

Across nearly every neural NLP system, the first thing that happens to a sentence is that each word is looked
up in a table of pretrained vectors. These vectors are the workhorse input feature for question answering,
textual entailment, semantic role labeling, coreference, named-entity recognition, and sentiment. But the
lookup table assigns exactly one vector to each word *type*: the string "play" maps to a single point in space
no matter what sentence it sits in. That point has to simultaneously stand for the theatrical play, the sports
play, the children's play, and the verb — so it ends up an averaged blur of all of them, and a downstream model
can never recover which sense was meant. Polysemy is structurally unrepresentable. Worse, a single vector also
fuses two very different kinds of information — the word's syntactic role (is it a noun here? a verb?) and its
fine-grained meaning — into one undifferentiated point, with no way for a task to ask for one without the other.

The problem, stated precisely: produce a representation of a word that is a function of the *entire sentence it
appears in*, so that the same string gets different vectors in different contexts; that captures both the
syntactic and the semantic facets of word use; that can be computed for any token including ones never seen in
training; and that can be dropped into an existing supervised model with minimal surgery and a modest amount of
labeled data. A solution has to come from somewhere other than the labeled task data itself, because the labeled
datasets for most of these tasks are small — the knowledge of what words mean in context has to be transferred
in from large amounts of unlabeled text.

## Background

**Word-type embeddings.** The dominant input feature is a pretrained, context-independent word vector. word2vec
(Mikolov et al., 2013) learns vectors by predicting context words from a target (skip-gram) or vice versa; GloVe
(Pennington et al., 2014) factorizes global co-occurrence counts. Both produce one dense vector per vocabulary
entry, capturing distributional similarity. They are pretrained once on large corpora and reused everywhere
(Turian et al., 2010, established this transfer recipe). Their structural limit is exactly the research question:
one vector per type, no context, and nothing for out-of-vocabulary words.

**Subword and character input.** To attack the OOV and morphology problems, a line of work represents a word
from its characters or character n-grams: fastText (Bojanowski et al., 2017), charagram (Wieting et al., 2016),
and compositional character models (Ling et al., 2015). A character-level CNN over a word, in particular, has
been used as the input layer of neural language models (Kim et al., 2015; Jozefowicz et al., 2016): a small
convolution over character embeddings, max-pooled and projected, yields a token vector that exists for *any*
string, in or out of vocabulary, and that encodes morphological regularities (prefixes, suffixes, capitalization).

**Language models and LSTMs.** A language model factorizes the probability of a token sequence
p(t_1,...,t_N) = prod_k p(t_k | t_1,...,t_{k-1}) and is trained by maximum likelihood on raw text — no labels
needed, so it can consume effectively unlimited data. Modern neural LMs (Jozefowicz et al., 2016, the
CNN-BIG-LSTM; Merity et al., 2017; Melis et al., 2017) stack LSTM layers (Hochreiter & Schmidhuber, 1997) on a
token-embedding or character-CNN input and predict the next token with a softmax over the vocabulary on the top
layer. The internal hidden states of such a model are, by construction, functions of the whole left context.

**What different layers of a deep RNN encode.** A recurring empirical finding is that stacked recurrent layers
specialize. Supervising low-level tasks (e.g., POS tagging) at the *lower* layers of a deep network improves
high-level tasks at the upper layers (Sogaard & Goldberg, 2016; Hashimoto et al., 2017). In a two-layer LSTM
encoder of a machine-translation system, the first-layer representations predict POS tags better than the
second-layer ones, even though the deeper system has higher BLEU (Belinkov et al., 2017). The top layer of a
biLSTM around a pivot word has been shown to learn word sense (context2vec; Melamud et al., 2016). The
through-line: syntactic information concentrates in lower layers, semantic/word-sense information in higher
layers, so no single layer is best for everything.

**Diagnostic measurements on existing contextual encoders.** One can probe a frozen contextual encoder by
freezing it and using its per-layer activations as input to a tiny classifier. On a fine-grained word-sense
disambiguation suite (a 1-nearest-neighbor classifier over averaged per-sense representations) and on PTB
part-of-speech tagging (a linear classifier), the two layers of a deep biLSTM encoder behave differently: the
*lower* layer scores higher on POS, the *upper* layer scores higher on WSD. An MT-encoder's two layers show the
same ordering but at a markedly lower absolute level than a language-model encoder. These are facts about
encoders already on the table, and they say plainly that a downstream system restricted to a single layer is
leaving information on the table.

## Baselines

**word2vec / GloVe (context-independent embeddings).** Core idea: learn a fixed vector per word type from
distributional statistics over a large corpus; reuse as frozen (or fine-tuned) input features. Gap: a single
vector per type cannot disambiguate polysemy, conflates syntax and semantics, and provides nothing for OOV
tokens. This is the feature these systems are trying to improve on.

**CoVe — contextualized vectors from machine translation (McCann et al., 2017).** Core idea: train an
attentional sequence-to-sequence neural MT system on parallel (e.g., English→German) data; the *encoder* is a
two-layer biLSTM. After training, freeze the encoder and use its TOP-layer hidden states as contextual word
vectors, concatenated with GloVe and fed into downstream task models (e.g., a biattentive classification
network for sentiment, an entailment model). This genuinely makes word vectors context-dependent and improves
several tasks. Gaps: (1) it needs *parallel* translation data, which is far scarcer than monolingual text, so
the encoder is trained on a comparatively small corpus; (2) it uses only the top encoder layer, discarding the
lower-layer (more syntactic) information; (3) the supervised MT objective may bias the representations.

**TagLM / biLM features for tagging (Peters et al., 2017).** Core idea: pretrain a *bidirectional* LM (forward
and backward LMs, independently parameterized) on unlabeled text; concatenate the TOP-layer biLM hidden state
into a sequence tagger's token representation. Establishes that LM pretraining on monolingual text transfers and
that bidirectional and large-scale LMs help. Gaps: only the top layer is exposed (again discarding lower layers),
and the two directions use fully independent parameters.

**context2vec (Melamud et al., 2016).** Core idea: a biLSTM reads the left and right context around a pivot word
and produces a context embedding; the top layer learns word sense. Gap: again a single (top) layer, and the
representation is of the *context*, not a deep multi-layer function of the token.

**Pretrain-then-finetune encoders (Dai & Le, 2015; Ramachandran et al., 2017).** Core idea: pretrain an
encoder-decoder with a language-model or autoencoder objective, then fine-tune the whole thing on the
supervised task. Gap: this couples the size of the supervised model to the pretrained model and requires
fine-tuning the large network on each task, rather than reusing a fixed universal representation as a feature.

## Evaluation settings

The natural yardstick is a spread of supervised NLP benchmarks, each with an existing strong neural baseline
into which a new word feature can be substituted:

- **Question answering** — SQuAD (Rajpurkar et al., 2016), 100K+ crowd-sourced question/answer pairs where the
  answer is a span in a Wikipedia paragraph; metric Exact Match and span F1. Strong baselines use bidirectional
  attention (BiDAF; Seo et al., 2017) plus self-attention.
- **Textual entailment** — SNLI (Bowman et al., 2015), ~550K premise/hypothesis pairs labeled
  entailment/contradiction/neutral; metric accuracy. Strong baseline: ESIM (Chen et al., 2017).
- **Semantic role labeling** — OntoNotes / CoNLL-2012 (Pradhan et al., 2013), BIO tagging of predicate-argument
  structure; metric span F1. Strong baseline: a deep interleaved biLSTM tagger (He et al., 2017).
- **Coreference resolution** — OntoNotes / CoNLL-2012; metric average F1 over MUC/B3/CEAF. Strong baseline: an
  end-to-end span-ranking neural model (Lee et al., 2017).
- **Named-entity recognition** — CoNLL-2003 (Tjong Kim Sang & De Meulder, 2003), four entity types; metric
  span F1, averaged over random seeds because the test set is small. Strong baseline: biLSTM-CRF (Lample et al.,
  2016; Ma & Hovy, 2016).
- **Sentiment** — fine-grained SST-5 (Socher et al., 2013), five-way sentence classification; metric accuracy,
  averaged over seeds. Strong baseline: the biattentive classification network of McCann et al. (2017).

Intrinsic probing tasks for *analyzing* representations: fine-grained all-words WSD (Raganato et al., 2017,
evaluation framework; 1-NN over per-sense averages) and PTB POS tagging (Marcus et al., 1993; a linear probe).
Unlabeled pretraining corpus: the One Billion Word Benchmark (Chelba et al., 2014), ~30M sentences. Standard
training machinery: Adam (Kingma & Ba, 2015), Adadelta (Zeiler, 2012), dropout (Srivastava et al., 2014) and
variational recurrent dropout (Gal & Ghahramani, 2016), highway connections (Srivastava et al., 2015), layer
normalization (Ba et al., 2016), CRF decoding (Lafferty et al., 2001).

## Code framework

Available scaffold: a character-CNN token encoder, stacked LSTM layers, and a softmax LM head — the standard
neural-language-model stack — plus downstream task models that consume a per-token feature. The empty slot is a
generic adapter from a frozen sentence encoder's layer activations to one feature per token.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterCNNEncoder(nn.Module):
    """Map a word's character ids to a context-independent token
    vector. Char embeddings -> parallel Conv1d of several widths -> max-pool over
    characters -> concat -> highway layers -> linear projection. Exists for any
    string, in or out of vocabulary (Kim 2015, Jozefowicz 2016)."""
    def __init__(self, n_chars, char_dim, filters, n_highway, proj_dim):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, char_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(char_dim, n, width) for (width, n) in filters])
        total = sum(n for (_, n) in filters)
        self.highways = Highway(total, n_highway, activation=F.relu)
        self.proj = nn.Linear(total, proj_dim)

    def forward(self, char_ids):
        # char_ids: (batch, seq_len, max_chars)
        ...  # embed, convolve per width, max over chars, concat, highway, project
        return token_repr  # (batch, seq_len, proj_dim)


class StackedLSTM(nn.Module):
    """L LSTM layers producing a hidden state per token per layer."""
    def __init__(self, input_dim, hidden_dim, proj_dim, n_layers):
        super().__init__()
        # ... standard projected LSTM layers with residual connections ...

    def forward(self, x):
        # returns the per-layer hidden states, not just the top one
        ...
        return layer_outputs  # list of length n_layers


class LanguageModel(nn.Module):
    """A neural LM. Char-CNN input -> stacked LSTM -> softmax over
    the vocabulary on the top layer. Trained by maximum likelihood on raw text."""
    def __init__(self, ...):
        super().__init__()
        # TODO: decide how to train a sentence encoder and expose its useful internals
        # as reusable token features.
        pass

    def forward(self, char_ids):
        pass


class ContextualTokenFeature(nn.Module):
    """Given a pretrained sentence encoder's per-layer activations
    for each token, produce a SINGLE per-token vector to hand to a downstream model.
    What combination of the encoder's internal layers should this be?"""
    def __init__(self, ...):
        super().__init__()
        # TODO: choose the layer-combination rule
        pass

    def forward(self, per_layer_activations):
        # per_layer_activations: list over layers of (batch, seq_len, dim)
        # TODO: collapse into (batch, seq_len, out_dim)
        pass


def add_feature_to_task_model(task_token_repr, contextual_feature):
    """A downstream model forms a token representation x_k and a
    context-sensitive h_k via biRNN/CNN. A new pretrained feature is concatenated
    into x_k (and optionally h_k). The encoder providing the feature is frozen."""
    # TODO: decide where to inject the feature in the task model
    return torch.cat([task_token_repr, contextual_feature], dim=-1)
```
