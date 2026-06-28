I start with the uncomfortable fact that text already has a cheap form of transfer, but it is too shallow. A word embedding table gives me useful lexical geometry, yet the recurrent layers above it and the classifier head still start random. In vision I get to reuse a whole hierarchy, not just the first lookup table. So the thing I want is not another fixed feature to concatenate onto a task model; I want a way to carry a whole sequence model into a new text task and still let it move.

The source task has to be broad enough that one trained model is worth reusing. A supervised NLP task is too narrow or too expensive, and machine translation or entailment would build in a task-specific bias. Language modeling is the clean source: predict the next token from previous context. The data is nearly unlimited, the objective forces syntax and long-range information into the hidden state, and a target corpus can always be viewed as unlabeled language-model data before I look at labels. That gives me the same basic structure as vision transfer: pretrain once on a large general corpus, adapt to the target distribution, then train the classifier.

The problem is that naive fine-tuning breaks in a few different ways, and I should be honest about which. If I adapt a language model on a tiny target corpus, it can overfit. If I attach a random classifier head and train every layer with the same learning rate, the first gradients are noisy and large, so they overwrite the general language knowledge I wanted to keep. If I freeze everything except the last layer, I protect the source knowledge but underfit, because a three-layer recurrent model does not have many layers to spare. So I am looking for a procedure that lets the upper layers adapt while keeping the lower layers conservative, and that makes the learning rate high only when that helps. Each of those three failure modes wants its own control; I will see what controls fall out.

For the base model I want the strongest plain recurrent language model available, and the AWD-LSTM fits. It is still just an embedding, three LSTM layers, and a tied decoder, but it is heavily regularized: weight-drop on hidden-to-hidden recurrent weights, locked dropout on recurrent activations, embedding dropout, activation regularization, temporal activation regularization, and randomized-length BPTT. That matters because I am about to fine-tune on small corpora and want the regularization to live in the encoder rather than ask the architecture to solve the target task for me. The transfer and fine-tuning procedure should be the thing that does the work.

Now I need to decide how much each layer should move. The vision evidence says early layers tend to be more general and late layers more task-specific. The same idea should hold in a stacked language model: the bottom has lexical and local syntactic features, and the top is closest to the language-model objective and the target head. A single learning rate treats these layers as equally disposable, which is exactly what overwrites the general knowledge. So I split the parameters into layer groups θ^1,...,θ^L with learning rates η^1,...,η^L. The ordinary update is θ_t = θ_{t-1} - η ∇_θ J(θ); with layer groups it becomes θ_t^l = θ_{t-1}^l - η^l ∇_{θ^l} J(θ). The sign is still descent, so the only change is the per-layer scale. I shrink the rate as I go down with a constant factor, η^{l-1} = η^l / 2.6.

I want to know what that factor actually does to the span, so I expand it for the classifier's five groups with a top rate of 0.01. Bottom to top that gives [0.000219, 0.000569, 0.001479, 0.003846, 0.01], each step a clean 2.6x, and the top group ends up 2.6^4 ≈ 45.7 times faster than the bottom. That is a large spread but not a freeze: the bottom embedding layer still moves at roughly 2e-4, so it adapts to target vocabulary without being thrashed. The 2.6 factor is doing the "conservative below, plastic above" job I asked for, with one knob. In the classifier training code that becomes the five-group list [lr/2.6^4, lr/2.6^3, lr/2.6^2, lr/2.6, lr]; the language-model fine-tuning code in the v0.7.2 snapshot uses a different four-group fixture [lr/6, lr/3, lr, lr/2], so I should not pretend the code and the prose are identical in that one detail.

The schedule within a single fine-tuning run is the second control. The first gradients from a fresh adaptation are the dangerous ones, so I want a short burst early that moves the model into a good region for the target distribution, then a long decay that refines without thrashing the pretrained weights. A triangular rate has two phases, but a symmetric triangle spends as long climbing as descending, which is not what I want — I want the up-slope much shorter than the down-slope. So I parameterize the cut point. With T total updates, cut = floor(T * cut_frac); for t < cut, p = t / cut; otherwise p = 1 - (t - cut) / (cut * (1/cut_frac - 1)); and η_t = η_max * (1 + p * (ratio - 1)) / ratio.

Before I trust this I should actually evaluate it, because the algebra in the decay branch is easy to get wrong. Take T = 1000 with the constants I have in mind, cut_frac = 0.1, ratio = 32, η_max = 0.01. Then cut = 100. At t = 0, p = 0 and η = 0.01 * 1/32 = 0.000313 — the floor, η_max/ratio, as I wanted. Stepping up: t = 50 gives p = 0.5 and η = 0.00516, about 16.5x the floor; t = 99 gives p = 0.99 and η = 0.00990; t = 100 gives p = 1 and η = 0.01 exactly, the peak η_max. So the climb takes 10% of training and lands on η_max at the cut, which checks out. Now the decay branch: t = 101 gives p = 0.9989 and η = 0.00999, just below the peak — good, it is continuous across the cut and turning over. At t = 500, p = 0.556 and η = 0.00569; at t = 999, p = 0.0011 and η = 0.000323, essentially back at the floor. So the realized shape is a fast 10% ramp from floor to peak and a 90% linear glide back to the floor, peak-to-floor ratio 32. The decay-branch formula is the part I was unsure of, and the numbers come out monotone and continuous, so I will keep it. The training code calls this same short-up/long-down CLR family through `use_clr`, with the language-model run using `(32,10)` and the classifier phases using smaller tuples such as `(8,3)` and `(8,8)`.

After adapting the language model to the target text, I need a classifier head, and this is where I have to think about what a document-level decision actually depends on. A label can hinge on a few words anywhere in a long review or question. The obvious summary is the final recurrent state h_T, but a left-to-right state is biased toward the end of the document and can forget a decisive phrase that appeared early. Let me make that concrete. Suppose the last-layer hidden sequence over four steps, in one feature dimension, is [1, 9, 1, 1] — a strong activation at step 2 and quiet elsewhere. Then h_T reports 1.0 and the spike is simply gone. A max over the sequence reports 9.0 and recovers it. A mean reports 3.0 — it dilutes the spike but preserves the document's average level, which a pure max would throw away. So the three pools genuinely carry different information; they only coincide when the salient feature happens to sit at the end (if the sequence is monotone increasing, last and max are identical and the extra term buys nothing). Since I cannot assume that, I keep all three and concatenate: h_c = [h_T, maxpool(H), meanpool(H)] over H = {h_1,...,h_T}. That 3*hidden vector then goes through two linear blocks with batch normalization and dropout, with ReLU only between the hidden block and the final class logits.

Long documents force one more engineering choice. I cannot backpropagate through every token of every document in one shot — memory will not hold it. So I split the document into BPTT chunks, reset the hidden state once at the start of the document, carry the recurrent state from chunk to chunk, and keep the hidden sequences from the suffix that fits in memory. The pooling layer then sees the accumulated hidden states, and gradients flow back through the chunks that contributed. The state is reset only at the document boundary, not at every chunk, so the recurrence still sees the whole document forward even though backprop only reaches the retained suffix. That is BPTT adapted to text classification, not a new model.

The classifier fine-tuning order is the part I am least sure about, so I reason through what breaks first. If the head is random, its first gradients are large and meaningless, and I just argued that those are exactly the gradients that overwrite general knowledge — so the lower encoder layers should not be exposed to them yet. The cleanest way to enforce that is to unfreeze from the top. I train only the classifier-side top group for one epoch; then unfreeze one more lower group and train the now-unfrozen set for another epoch; I repeat until the whole model is unfrozen, then continue full fine-tuning. I considered the chain-thaw style of training one isolated layer and then re-freezing it, but that is not what I want here: re-freezing the upper layer once I move down would throw away the alignment it just learned, and the head needs to keep adapting as the lower layers come online. So the thawed set grows monotonically and nothing is re-frozen. The effect is that the bottom layers see gradients only after the upper layers and head have started to align with the target labels — which is the catastrophic-forgetting control, now matched to the same discriminative rates and slanted schedule the encoder adaptation used.

A forward language model sees only left context, so it can miss right-context evidence. I can repeat the same pipeline with a backward language model, train a backward classifier independently, and average the forward and backward classifier predictions. That is an ensemble cost, not a change in the core method.

Putting the pieces back together, the procedure is: pretrain a regular AWD-LSTM language model on a general corpus; adapt it to the target text with discriminative per-layer rates and the short-up/long-down schedule I checked above; attach a concat-pooling classifier; then train that classifier with top-down gradual unfreezing under the same rates and schedule. The causal chain is the three failure modes and the matching controls — regularized language-model adaptation for overfitting, layerwise rates plus the slanted schedule for stable adaptation, concat pooling for long documents, and top-down unfreezing for catastrophic forgetting — and the package is what I would call ULMFiT.

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
