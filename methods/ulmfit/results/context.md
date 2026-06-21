# Context: whole-model transfer for text classification

## Research question

Computer vision has a reliable transfer recipe: pretrain a deep model once on a
large source corpus, then fine-tune its weights for a new supervised task. Text
classification does not yet have that recipe. The common reusable object is a word
embedding table, which initializes only the first layer and leaves the sequence
model and classifier random. The question is whether a whole pretrained text model
can be adapted to many target classification tasks - short or long documents, few
or many labels, small or large labeled sets - with one architecture, one training
process, no hand-built task features, and no requirement for millions of
additional in-domain documents.

## Background

**What vision suggests.** Deep visual features become more task-specific in higher
layers, so transfer usually keeps low layers stable and adapts upper layers first
(Yosinski et al. 2014; Donahue et al. 2014; Long et al. 2015). End-to-end
fine-tuning has mostly replaced fixed feature extraction in that setting.

**Why text transfer is harder.** Word2vec-style transfer only moves an embedding
matrix. Hypercolumn-style NLP transfer carries richer contextual vectors from a
pretrained model, but feeds them as fixed features into a new task-specific model
that is still trained from scratch. Multi-task approaches train the auxiliary and
target objectives together, so they must be rerun for every task and require
objective balancing. Earlier attempts at full NLP fine-tuning across unrelated
tasks were reported to fail, and language-model fine-tuning needed millions of
in-domain documents while still overfitting around the 10k-label regime.

**The useful source signal.** Next-word prediction is attractive because unlabeled
text is abundant and the objective forces a model to encode syntax, long-range
dependencies, topical regularities, and sentiment-relevant cues. Strong recurrent
language models already exist: the AWD-LSTM is a regular stacked LSTM with no
attention or shortcut connections, but with weight-drop on recurrent matrices,
locked/variational dropout, activation regularization, temporal activation
regularization, randomized-length BPTT, and averaged-SGD training.

**The optimization failure modes.** A small target corpus can make a pretrained
language model overfit. A random classifier head can send large noisy gradients
through all recurrent layers and erase useful source knowledge. Freezing almost
everything, the common vision shortcut, can underfit because the recurrent text
model has only a few layers to adapt. Any successful recipe has to balance these
three cases rather than solve just one of them.

## Baselines

**Embedding transfer.** Initialize words from a pretrained embedding table, then
train the rest of the classifier from scratch. This is simple and widely useful,
but it does not transfer the sequence model.

**Hypercolumn and contextual-feature transfer.** CoVe, ELMo-style, and related
approaches use representations from an auxiliary model as fixed inputs to a task
model. They avoid catastrophic forgetting by not fine-tuning the source model, but
that also leaves the main classifier randomly initialized and often task-specific.

**Language-model fine-tuning before this point.** Dai and Le showed the direction
was plausible, but relied on very large in-domain document collections and did not
give a generally stable small-data fine-tuning recipe.

**Training text classifiers from scratch.** Character CNNs, word CNNs, recurrent
classifiers, and deep pyramid CNNs were strong on large benchmarks, but they paid
for each task with fresh supervised training and many labels.

## Evaluation settings

The target surface is text classification across sentiment, topic, and question
classification: IMDb, TREC-6, AG News, DBpedia, Yelp binary, and Yelp full. The
general source corpus is WikiText-103, with 28,595 Wikipedia articles and about
103M words. Results are reported as test error rates, and low-shot experiments
vary labeled examples down to 100 to test whether transfer changes sample
efficiency rather than merely improving large-data accuracy.

## Code framework

The scaffold already has a pretrained recurrent encoder and a next-word training
loop. The open slots are how to adapt that encoder to a target corpus, how to
summarize a long document for classification, and how to train the new head
without destroying the transferred encoder.

```python
import torch
import torch.nn as nn

class RecurrentLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_size=400, hidden_size=1150, n_layers=3):
        super().__init__()
        # Embedding, stacked recurrent layers, recurrent weight-drop, locked
        # dropout, and tied decoder exist in the surrounding framework.

    def forward(self, tokens, state=None):
        # tokens: [time, batch]
        # returns per-layer raw/dropped hidden sequences and the next state.
        raise NotImplementedError

def pretrain_next_word(model, wiki_text_loader):
    """Standard BPTT language-model pretraining already exists."""
    raise NotImplementedError

def adapt_encoder_to_target_text(model, target_text_loader):
    # TODO: choose a fine-tuning rule that adapts without overfitting or erasing.
    raise NotImplementedError

class TextClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        # TODO: attach a task head and decide how a whole document becomes one
        # classification vector.

    def forward(self, document_tokens):
        raise NotImplementedError

def train_target_classifier(classifier, labeled_loader):
    # TODO: train new head and transferred encoder without catastrophic forgetting.
    raise NotImplementedError
```
