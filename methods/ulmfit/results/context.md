# Context: transfer learning for text the way ImageNet transfer works for vision

## Research question

In computer vision, almost nobody trains from scratch: a network pretrained on a
large labeled corpus (ImageNet) is *fine-tuned* — its weights carried over and
adapted — to detection, segmentation, or a new classification task, and this works
robustly across tasks. NLP has no equivalent. Deep models for text classification
are trained from scratch, need large labeled datasets, and take days to converge.
The one piece of transfer that *is* universal in NLP — pretrained word embeddings —
only initializes the model's first layer; everything above it is still random. The
precise question is whether the whole model can be pretrained once and then
fine-tuned to *any* target task — varying in document length, number of examples,
and label type — with a single architecture and training process, no custom
feature engineering, and no requirement for extra in-domain data, so that even a
task with very few labeled examples becomes learnable.

## Background

**Transfer in vision sets the template.** Features in a deep network move from
*general* (edges, textures) in early layers to *task-specific* in late layers
(Yosinski et al. 2014). Vision exploits this by carrying over the early/most layers
of a pretrained model and fine-tuning the last one or several layers while leaving
the rest frozen (Donahue et al. 2014; Long et al. 2015) — end-to-end fine-tuning
has largely superseded using fixed features.

**Why NLP transfer lagged.** Fine-tuning pretrained word embeddings (Mikolov et al.
2013) targets only the first layer and treats the rest as random init. The
prevailing way to transfer more — *hypercolumns* — pretrains representations on some
auxiliary task (language modeling, machine translation, entailment) and feeds those
embeddings, concatenated with the word embeddings or injected at intermediate
layers, as *fixed* features into a task model that is still trained from scratch
(Peters et al. 2017; McCann et al. 2017, CoVe; Peters et al. 2018, ELMo). Several of
these require custom-engineered architectures per task. Multi-task learning trains
a language-modeling objective jointly with the task (Rei 2017), but must train from
scratch every time and needs careful objective weighting. Crucially, full *fine-
tuning* of a pretrained model had been tried and reported to *fail* for NLP between
unrelated tasks (Mou et al. 2016); and fine-tuning a language model (Dai & Le 2015)
required millions of in-domain documents and still overfit with around 10k labeled
examples.

**The two failure modes of naive fine-tuning.** A language model fine-tuned on a
small target set *overfits*. And when its weights are then adapted to a classifier,
aggressive fine-tuning of all layers at once causes *catastrophic forgetting* — the
general linguistic knowledge captured during pretraining is overwritten, and the
error, after dropping early, climbs back up as training continues. NLP models are
also typically shallower than vision models, so the vision recipe of "fine-tune
only the last layer" *underfits* them. These observations — overfitting, forgetting,
underfitting-from-last-layer-only — are what any working method must defeat.

**Language modeling as the ideal source task.** Predicting the next word captures
long-range dependencies (Linzen et al. 2016), hierarchical structure (Gulordava et
al. 2018), and even sentiment (Radford et al. 2017); it has near-unlimited unlabeled
data in every domain and language; and it is already a component of machine
translation and dialogue. It is the natural ImageNet analog for text.

**The base language model.** The strongest LSTM language model of the time is the
AWD-LSTM (Merity et al. 2017): a regular three-layer LSTM with *no* attention or
shortcut connections, regularized heavily — DropConnect applied to the
hidden-to-hidden recurrent weight matrix (the same dropout mask reused across all
time steps), variational dropout elsewhere, activation and temporal-activation
regularization, randomized-length backpropagation-through-time — and optimized with
non-monotonically-triggered averaged SGD. It reaches state-of-the-art word-level
perplexity (57.3 on Penn Treebank, 65.8 on WikiText-2). A learning-rate schedule
that warms up briefly then decays is known to help training (triangular learning
rates, Smith 2017; cosine annealing, Loshchilov & Hutter 2017).

## Baselines

**CoVe / hypercolumn transfer (McCann et al. 2017).** Pretrain an encoder (here on
7M machine-translation sentence pairs), then feed its contextual vectors as fixed
features into a task-specific model trained from scratch, often with custom
attention. Gap: the main model is still trained from scratch, needs a large
auxiliary supervised corpus, and the transferred part is frozen.

**Dai & Le (2015) LM fine-tuning.** Pretrain a language model and fine-tune it for
the task — the right idea — but requires millions of *in-domain* documents and still
overfits with around 10k labels. Gap: no general-domain pretraining and no recipe to
fine-tune without overfitting/forgetting.

**Train-from-scratch text classifiers.** Char-level and word-level CNNs (Zhang et
al. 2015; Johnson & Zhang 2016, 2017) are the state of the art on the large topic/
sentiment datasets. Gap: need large labeled data; no transfer; sample-inefficient.

## Evaluation settings

Six widely-used text-classification datasets spanning three task types: sentiment
(IMDb binary movie reviews, 25k; Yelp binary and five-class, 560k/650k), question
classification (TREC-6, six classes, 5.5k), and topic classification (AG news, four
classes, 120k; DBpedia ontology, 14 classes, 560k). Preprocessing follows prior work
(Johnson & Zhang; McCann et al.). General-domain pretraining uses WikiText-103
(28,595 Wikipedia articles, 103M words). Metric is test error rate (lower is
better). A low-shot protocol varies the number of labeled examples (down to 100) in
supervised and semi-supervised settings to measure sample efficiency.

## Code framework

The substrate is a recurrent language model (the AWD-LSTM: embedding → 3-layer LSTM
with the regularizers above → tied softmax over the vocabulary) and a standard
training loop with backpropagation-through-time. What is *not* fixed: how to carry
the pretrained LM into a small target dataset without overfitting or forgetting,
and how to attach and train a classifier head over a recurrent encoder of long
documents. The scaffold leaves those slots.

```python
import torch, torch.nn as nn

class AWD_LSTM(nn.Module):
    """Embedding -> 3-layer LSTM with DropConnect/variational dropout (exists)."""
    def __init__(self, vocab, emb=400, hidden=1150, n_layers=3): ...
    def forward(self, tokens, hidden=None):
        # returns per-timestep hidden states H = {h_1..h_T} and final state
        ...

class LMHead(nn.Module):
    """Tied softmax over the vocabulary for next-word prediction (exists)."""
    def __init__(self, emb, vocab): ...
    def forward(self, H): ...

def train_lm(model, corpus):  # standard BPTT next-word training (exists)
    ...

# --- adapting the pretrained LM to a small target set: TO DECIDE ---
def finetune_lm(model, target_corpus):
    # TODO: adapt the pretrained LM to the target text without overfitting
    #       or destroying pretrained knowledge.
    pass

# --- classifier head over a recurrent encoder of long documents: TO DECIDE ---
class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        # TODO: build a classifier on top of the recurrent encoder.
        pass
    def forward(self, document):
        pass

def finetune_classifier(clf, labeled_data):
    # TODO: fine-tune the classifier without catastrophic forgetting.
    pass
```
