Computer vision has a dependable transfer recipe — pretrain one deep model on a large source corpus, then fine-tune the whole thing on a new supervised task — and text classification has nothing of the kind. The reusable object everyone reaches for is a word embedding table, but that initializes only the first lookup layer and leaves the entire sequence model and classifier head random. We want the vision-grade version: a single architecture and single training process that adapts a *whole* pretrained text model to many target tasks — short or long documents, few or many labels, small or large labeled sets — with no hand-built features and no demand for millions of in-domain documents. The existing options each fall short of that. Word2vec-style transfer moves only the embedding matrix. Hypercolumn and contextual-feature methods (CoVe, ELMo-style) carry richer vectors from a pretrained model but feed them as *fixed* inputs into a task model that is still trained from scratch, so the main classifier stays random and task-specific. Multi-task setups train auxiliary and target objectives jointly, so they must be rerun and rebalanced per task. And earlier full language-model fine-tuning (Dai and Le) needed very large in-domain corpora and still overfit around the 10k-label regime. Next-word prediction is the right source signal because unlabeled text is essentially unlimited and the objective forces a model to encode syntax, long-range dependencies, topical structure, and sentiment cues; the missing piece is purely a *fine-tuning procedure* that does not break.

It breaks in three distinct ways, and that is the key to the design. Adapting a language model on a tiny target corpus can overfit. Attaching a random classifier head and training every layer at one learning rate lets the first noisy, large gradients overwrite the general knowledge we wanted to keep — catastrophic forgetting. Freezing everything except the last layer protects that knowledge but underfits, because a three-layer recurrent model has almost no spare capacity to adapt. A working recipe has to handle all three at once, not just one. I propose ULMFiT — Universal Language Model Fine-tuning — which is exactly a matched set of controls, one per failure mode, wrapped around three stages: pretrain a regular AWD-LSTM language model on a general corpus (WikiText-103, about 103M words), fine-tune that language model on the target task's *unlabeled* text, then replace the decoder with a concat-pooling classifier head and fine-tune it with gradual unfreezing. I deliberately use the strongest *plain* recurrent language model — the AWD-LSTM, just an embedding, three LSTM layers, and a tied decoder, but heavily regularized with weight-drop on the recurrent matrices, locked/variational dropout, embedding dropout, activation and temporal-activation regularization, and randomized-length BPTT — so that the regularization, not some exotic architecture, is what makes small-corpus adaptation safe. The architecture should not solve the target task; the transfer procedure should.

The first control is discriminative fine-tuning, which addresses how much each layer is allowed to move. The vision evidence says early layers are more general and late layers more task-specific, and the same holds in a stacked language model: the bottom holds lexical and local-syntactic features, the top sits closest to the language-model objective and the target head. A single learning rate is wrong because it treats every layer as equally disposable. So I split the parameters into layer groups $\theta^1,\dots,\theta^L$ with their own rates $\eta^1,\dots,\eta^L$, turning the ordinary update $\theta_t = \theta_{t-1} - \eta\,\nabla_\theta J(\theta)$ into

$$\theta_t^l = \theta_{t-1}^l - \eta^l\,\nabla_{\theta^l} J(\theta).$$

The sign is still plain gradient descent; only the per-layer scale changes. I pick the top rate $\eta^L$ and shrink as I descend, $\eta^{l-1} = \eta^l/2.6$, so lower layers move conservatively. (One honest caveat: the reference classifier code realizes this as five groups $[\,lr/2.6^4,\,lr/2.6^3,\,lr/2.6^2,\,lr/2.6,\,lr\,]$, while the language-model fine-tuning script uses a four-group fixture $[\,lr/6,\,lr/3,\,lr,\,lr/2\,]$, so the code and the $2.6$ rule are not identical in that one place.)

The second control shapes the schedule *within* a fine-tuning run, and it is the slanted triangular learning rate (STLR). At the start I want a short burst that drives the model into a good region for the target distribution; after that I want a long decay that refines without thrashing the pretrained weights. A plain triangular schedule has the two phases but symmetric slopes — I want the climb much shorter than the descent. With $T$ total updates I set $\text{cut} = \lfloor T\cdot \text{cut\_frac}\rfloor$, define the fraction completed

$$p = \begin{cases} t/\text{cut} & t < \text{cut}\\[4pt] 1 - \dfrac{t-\text{cut}}{\text{cut}\,(1/\text{cut\_frac} - 1)} & t \ge \text{cut}\end{cases}$$

and set $\eta_t = \eta_{\max}\,(1 + p\,(\text{ratio}-1))/\text{ratio}$. At $p=0$ the rate is $\eta_{\max}/\text{ratio}$, it climbs to $\eta_{\max}$ at $p=1$ (reached at the cut), then falls linearly back toward the floor by the last iteration. The defaults are $\text{cut\_frac}=0.1$, $\text{ratio}=32$, $\eta_{\max}=0.01$ — a 10% climb, a 90% decay, and a high peak only where it helps. (In the training code the same short-up/long-down family is invoked through `use_clr`, with tuples like $(32,10)$ for the LM and $(8,3)$/$(8,8)$ for the classifier phases.)

The third issue is how a whole document becomes one classification vector. A label can hinge on a few words anywhere in a long review or question, so the final recurrent state alone is too brittle. I keep the full sequence of final-layer hidden states $H=\{h_1,\dots,h_T\}$ and build the classifier input by concat pooling,

$$h_c = [\,h_T,\;\text{maxpool}(H),\;\text{meanpool}(H)\,],$$

where the last state carries the left-to-right summary, the max-pool catches a salient feature wherever it occurs, and the mean-pool preserves the document's average content. That $3\times\text{hidden}$ vector then passes through two linear blocks with batch normalization and dropout, with a ReLU only between the hidden block and the final class logits. Long documents force one more piece of engineering, BPTT for text classification: I cannot backpropagate through every token at once, so I split the document into BPTT chunks, reset the hidden state once at the start, carry the recurrent state from chunk to chunk, and keep the hidden sequences from the suffix that fits in memory. The pooling layer then sees the accumulated states and gradients flow back through the chunks that contributed — it is a training loop, not a new model.

The fourth and most delicate control is the classifier fine-tuning order, gradual unfreezing, which directly answers catastrophic forgetting. If the head is random, the lower encoder layers must not be exposed to its first gradients. So I unfreeze from the top: train only the top group (the head plus the last group) for one epoch, then unfreeze one more lower group and train the now-unfrozen set for another epoch, repeating until everything trains together, and only then continue full fine-tuning. This differs from chain-thaw because the thawed set *grows* — I never train one isolated layer and re-freeze it — so the bottom layers see gradients only after the upper layers and head have begun to align with the target labels. Finally, a forward language model sees only left context and can miss right-context evidence; when bidirectional evidence is wanted I run the identical pipeline with a backward language model, train a backward classifier independently, and average the two classifiers' predicted probabilities. That is an ensemble cost, not a change to the core method. The causal chain is the whole point: regularized language-model adaptation answers overfitting, discriminative rates plus the slanted triangular schedule give stable adaptation, concat pooling handles long documents, and top-down gradual unfreezing prevents forgetting.

```python
import math
import torch
import torch.nn as nn

def stlr(t, T, eta_max=0.01, cut_frac=0.1, ratio=32):
    cut = math.floor(T * cut_frac)
    if cut <= 0:
        return eta_max
    if t < cut:
        p = t / cut
    else:
        p = 1 - (t - cut) / (cut * (1 / cut_frac - 1))
    return eta_max * (1 + p * (ratio - 1)) / ratio

def classifier_group_lrs(lr, factor=2.6):
    return [lr / (factor ** 4), lr / (factor ** 3), lr / (factor ** 2), lr / factor, lr]

def lm_group_lrs_v072(lr):
    return [lr / 6, lr / 3, lr, lr / 2]

def concat_pool(output):
    # fastai v0.7 uses sequence-first tensors: [time, batch, hidden].
    avg_pool = output.mean(dim=0)
    max_pool = output.max(dim=0).values
    return torch.cat([output[-1], max_pool, avg_pool], dim=1)

class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, p):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.drop = nn.Dropout(p)
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))

class PoolingLinearClassifier(nn.Module):
    def __init__(self, hidden_size, n_classes, inner=50, drops=(0.4, 0.1)):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(3 * hidden_size, inner, drops[0]),
            LinearBlock(inner, n_classes, drops[1]),
        ])

    def forward(self, raw_outputs, outputs):
        x = concat_pool(outputs[-1])
        for layer in self.layers:
            logits = layer(x)
            x = torch.relu(logits)
        return logits, raw_outputs, outputs

class MultiBatchEncoder(nn.Module):
    def __init__(self, encoder, bptt=70, max_seq=20 * 70):
        super().__init__()
        self.encoder, self.bptt, self.max_seq = encoder, bptt, max_seq

    def forward(self, tokens):
        # tokens: [time, batch]. The encoder carries hidden state across chunks.
        self.encoder.reset()
        raw_chunks, out_chunks = [], []
        sl = tokens.size(0)
        for i in range(0, sl, self.bptt):
            raw, out = self.encoder(tokens[i:min(i + self.bptt, sl)])
            if i > sl - self.max_seq:
                raw_chunks.append(raw)
                out_chunks.append(out)
        return self._concat(raw_chunks), self._concat(out_chunks)

    @staticmethod
    def _concat(chunks):
        return [torch.cat([chunk[layer] for chunk in chunks], dim=0)
                for layer in range(len(chunks[0]))]

def gradual_unfreezing_training(learner, lrs, final_cycles):
    learner.freeze_to(-1)
    learner.fit(lrs, 1, use_clr=(8, 3))
    learner.freeze_to(-2)
    learner.fit(lrs, 1, use_clr=(8, 3))
    learner.unfreeze()
    learner.fit(lrs, final_cycles, use_clr=(8, 8))
```
