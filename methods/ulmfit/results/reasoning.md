I start with the uncomfortable fact that text already has a cheap form of transfer, but it is too shallow. A word embedding table gives me useful lexical geometry, yet the recurrent layers above it and the classifier head still start random. In vision I get to reuse a whole hierarchy, not just the first lookup table. So the thing I need is not another fixed feature to concatenate onto a task model; I need a way to carry a whole sequence model into a new text task and still let it move.

The source task has to be broad enough that one trained model is worth reusing. A supervised NLP task is too narrow or too expensive, and machine translation or entailment would build in a task-specific bias. Language modeling is the clean source: predict the next token from previous context. The data is nearly unlimited, the objective forces syntax and long-range information into the hidden state, and a target corpus can always be viewed as unlabeled language-model data before I look at labels. That gives me the same basic structure as vision transfer: pretrain once on a large general corpus, adapt to the target distribution, then train the classifier.

The problem is that naive fine-tuning breaks in three different ways. If I adapt a language model on a tiny target corpus, it can overfit. If I attach a random classifier head and train every layer with the same learning rate, the first gradients are noisy and large, so they overwrite the general language knowledge I wanted to keep. If I freeze everything except the last layer, I protect the source knowledge but underfit because a three-layer recurrent model does not have many layers to spare. The recipe has to let the upper layers adapt while keeping the lower layers conservative, and it has to make the learning rate high only when that helps.

For the base model I choose the strongest plain recurrent language model available: an AWD-LSTM. It is still just an embedding, three LSTM layers, and a tied decoder, but it is heavily regularized: weight-drop on hidden-to-hidden recurrent weights, locked dropout on recurrent activations, embedding dropout, activation regularization, temporal activation regularization, and randomized-length BPTT. That matters because I am about to fine-tune on small corpora. I do not want the architecture itself to solve the target task; I want the transfer and fine-tuning procedure to be the thing that works.

Now I need to decide how much each layer should move. The vision evidence says early layers tend to be more general and late layers more task-specific. The same idea should hold in a stacked language model: the bottom has lexical and local syntactic features, and the top is closest to the language-model objective and the target head. A single learning rate is wrong because it treats these layers as equally disposable. I split the parameters into layer groups θ^1,...,θ^L with learning rates η^1,...,η^L. The ordinary update is θ_t = θ_{t-1} - η ∇_θ J(θ); with layer groups it becomes θ_t^l = θ_{t-1}^l - η^l ∇_{θ^l} J(θ). The sign is still descent, so the only change is the per-layer scale. I choose the top-layer rate η^L and shrink as I go down, using η^{l-1} = η^l / 2.6 in the paper-level rule. In the classifier training code that becomes five groups with [lr/2.6^4, lr/2.6^3, lr/2.6^2, lr/2.6, lr]; the language-model training code uses a four-group fixture [lr/6, lr/3, lr, lr/2], so I should not pretend the code and the prose are identical in that one detail.

The schedule within a fine-tuning run should also be asymmetric. At the very beginning I want a short burst that moves the model into a good region for the target distribution; after that I want a long decay that refines without thrashing the pretrained weights. A triangular schedule has the right two phases, but I want the up-slope much shorter than the down-slope. With T total updates, cut = floor(T * cut_frac). For t < cut, p = t / cut. Otherwise, p = 1 - (t - cut) / (cut * (1/cut_frac - 1)). Then η_t = η_max * (1 + p * (ratio - 1)) / ratio. At p=0 the rate is η_max/ratio; at p=1 it is η_max; by the final iteration it falls back near the floor. I use cut_frac = 0.1, ratio = 32, and η_max = 0.01 as the method constants. The training code calls the same short-up/long-down CLR family through `use_clr`, with the language-model run using `(32,10)` and classifier phases using smaller tuples such as `(8,3)` and `(8,8)`.

After adapting the language model to the target text, I need a classifier head. A label can depend on a few words anywhere in a long review or question, so the final recurrent state alone is too brittle. I keep the sequence of final-layer hidden states H = {h_1,...,h_T}. The classifier vector should be h_c = [h_T, maxpool(H), meanpool(H)]. The last state keeps the left-to-right summary, the max-pool catches a salient feature wherever it appears, and the mean-pool preserves the document's average content. This is exactly what the reference classifier does: it pools the last output sequence, concatenates last/max/average, then sends that 3*hidden vector through two linear blocks with batch normalization and dropout, with ReLU only between the hidden block and the final class logits.

Long documents force one more engineering choice. I cannot backpropagate through every token of every document in one shot. I split the document into BPTT chunks, reset the hidden state at the start of the document, carry the recurrent state from chunk to chunk, and keep the hidden sequences from the suffix that fits in memory. Then the pooling layer sees the accumulated hidden states and gradients flow back through the chunks that contributed. That is BPTT for text classification, not a new model.

The classifier fine-tuning order is the delicate part. If the head is random, the lower encoder layers should not be exposed to its first gradients. I unfreeze from the top. First I train only the classifier-side/top group for one epoch. Then I unfreeze one more lower group and train the now-unfrozen set for another epoch. I repeat until the whole model is unfrozen, and only then continue full fine-tuning. This is different from chain-thaw because the thawed set grows; I am not training one isolated layer and then freezing it again. The bottom layers see gradients only after the upper layers and head have started to align with the target labels.

A forward language model sees only left context, so it can miss right-context evidence. I can repeat the same pipeline with a backward language model, train a backward classifier independently, and average the forward and backward classifier predictions. That is an ensemble cost, not a change in the core method.

Putting the pieces together gives me the method: pretrain a regular AWD-LSTM language model on a general corpus, adapt it to target text with discriminative fine-tuning and a slanted triangular schedule, attach a concat-pooling classifier, then train that classifier with gradual unfreezing plus the same careful rates and schedule. The name for the package is ULMFiT, but the important causal chain is the three failure modes and the matching controls: regularized language-model adaptation for overfitting, layerwise rates and short-up/long-down learning rates for stable adaptation, concat pooling for long documents, and top-down unfreezing for catastrophic forgetting.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def paper_stlr(t, T, eta_max=0.01, cut_frac=0.1, ratio=32):
    cut = math.floor(T * cut_frac)
    if cut <= 0:
        return eta_max
    if t < cut:
        p = t / cut
    else:
        p = 1 - (t - cut) / (cut * (1 / cut_frac - 1))
    return eta_max * (1 + p * (ratio - 1)) / ratio

def classifier_lrs(lr, n_groups=5, factor=2.6):
    return [lr / (factor ** (n_groups - 1 - i)) for i in range(n_groups)]

def v072_lm_lrs(lr):
    return [lr / 6, lr / 3, lr, lr / 2]

def concat_pool_seq_first(output):
    # output: [time, batch, hidden], matching the v0.7 fastai text code.
    avg_pool = output.mean(dim=0)
    max_pool = output.max(dim=0).values
    return torch.cat([output[-1], max_pool, avg_pool], dim=1)

class PoolingLinearClassifier(nn.Module):
    def __init__(self, hidden_size, n_classes, inner=50, drops=(0.4, 0.1)):
        super().__init__()
        self.block1 = nn.Sequential(nn.BatchNorm1d(3 * hidden_size),
                                    nn.Dropout(drops[0]),
                                    nn.Linear(3 * hidden_size, inner),
                                    nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(nn.BatchNorm1d(inner),
                                    nn.Dropout(drops[1]),
                                    nn.Linear(inner, n_classes))

    def forward(self, raw_outputs, outputs):
        pooled = concat_pool_seq_first(outputs[-1])
        return self.block2(self.block1(pooled))

def bpt3c(encoder, tokens, bptt=70, max_seq=20 * 70):
    # tokens: [time, batch]. Hidden state is reset once, then carried by encoder.
    encoder.reset()
    raw_all, out_all = [], []
    sl = tokens.size(0)
    for i in range(0, sl, bptt):
        raw, out = encoder(tokens[i:min(i + bptt, sl)])
        if i > sl - max_seq:
            raw_all.append(raw)
            out_all.append(out)
    cat = lambda xs: [torch.cat([x[layer] for x in xs], dim=0)
                      for layer in range(len(xs[0]))]
    return cat(raw_all), cat(out_all)
```
