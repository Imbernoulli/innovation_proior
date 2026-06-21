The field has been moving fast — ELMo, GPT, BERT, then a permutation-language-model called XLNet — and each new method announces a new pretraining objective or architecture together with a new state of the art on GLUE, SQuAD, and RACE. What unsettles me about this sequence is that every head-to-head comparison changes several things at once. XLNet does not merely swap the objective for masked language modeling; it also trains on roughly ten times more data, with batches eight times larger for half the optimization steps, so it sees about four times as many sequences. When XLNet beats BERT, I genuinely cannot say what did the work: the permutation objective, or the simple fact that XLNet is a much larger training run. The two are tangled. And the structural reason this confusion persists is that pretraining is expensive — people tune lightly, run once, ship a bundle of changes, and frequently use private corpora of differing size. That is a recipe for misattributing gains. So before I credit any new objective for beating masked language modeling, I want to ask a sharper question: how good is masked language modeling itself when it is actually trained carefully? My suspicion is that the baseline everyone keeps beating is simply undertrained.

I propose RoBERTa, a robustly optimized recipe for the original masked-LM pretraining. The crucial discipline is that I hold the architecture and the objective completely fixed — the same bidirectional Transformer encoder, the same masked-language-model loss in which I select 15% of tokens and, of those, replace 80% with `[MASK]`, leave 10% unchanged, and swap 10% for a random vocabulary token, predicting the originals under cross-entropy — and I move only the training procedure. That way any improvement I observe cannot be credited to a different or bigger network or to a new objective; it must come from the recipe. Five changes make up the method, each one motivated by a specific failure of the original recipe.

The first is dynamic masking. The original implementation masks once, during preprocessing, producing a single static mask per sequence; to avoid the model seeing the identical mask every epoch, the data is duplicated ten times with ten different masks and training runs forty epochs, so each particular mask is seen four times. This bakes the masking pattern into the data and couples it to the epoch count, which is incoherent the moment I want to train much longer or on much more data — I would either have to re-duplicate or let the model keep re-seeing stale masks. The clean alternative is to generate the mask on the fly, freshly, every single time a sequence is fed to the model, so the model essentially never sees the same masked instance twice. This is strictly more varied supervision over the same data, it costs almost nothing because I am already iterating the data and just sample the 15% at load time, and on a matched comparison dynamic masking is comparable to or slightly better than static. It is also exactly what makes "train much longer" coherent, because the supervision keeps refreshing.

The second change is dropping next-sentence prediction in favor of a FULL-SENTENCES input. The original report claims that NSP — a binary classifier on whether two concatenated segments were consecutive in the corpus — is important, that removing it hurts NLI, QNLI, and SQuAD. Yet several recent efforts quietly dropped NSP and did not suffer. Both cannot be casually true, so the original ablation is being misread. The resolution is that there is a confound hidden in how the input is built, separate from the loss. When the original "removed NSP," it also changed the input format rather than only deleting the loss term. To untangle objective from input format I lay out four constructions: the original multi-sentence segment pairs with NSP on; single-sentence pairs with NSP on (and a raised batch size, since single sentences are far shorter than 512, to keep the token count comparable); full sentences sampled contiguously and packed up to 512 across document boundaries with a separator, NSP off; and the same packing forbidden to cross document boundaries, NSP off. The tell is that the single-sentence setting, with NSP still on, hurts relative to the segment-pair setting — so what was costing performance in the original "no NSP" ablation was not the missing loss but a degraded input that no longer offered long contiguous spans for learning long-range dependencies. When I instead keep the input long and contiguous and only delete the NSP head, downstream quality is matched or slightly improved. NSP, given proper long contiguous input, is redundant: the model already gets whatever inter-span signal it needs from predicting masked tokens inside long passages. Staying within a single document is marginally better but forces variable batch sizes because of the short tail near document ends, which is annoying to compare against; so I take full-sentences-no-NSP, packing contiguous full sentences to 512 and allowing document crossing with a separator, leaving exactly one pretraining loss, the masked-token cross-entropy.

The third change is large batches, and it is the one that forces me to touch the optimizer. I have a prior from machine translation: training with very large minibatches improves both optimization speed and final quality, provided the learning rate is scaled up to match, because a larger batch yields a less noisy gradient estimate per step — fewer steps, each more confident — and it parallelizes cleanly across devices. By gradient accumulation, the original million steps at batch 256 is the same compute as 125k steps at batch 2k or 31k steps at batch 8k. At batch 2k the masked-LM perplexity and downstream accuracy improve clearly; at batch 8k the perplexity is still better than the 256-sequence run, downstream quality stays comparable, and the run becomes far easier to parallelize. So I train with 8k sequences per batch. But the large batch exposes a stability problem in Adam. With batch 8k the gradient is much less noisy than at 256, and the default $\beta_2 = 0.999$ maintains a very long running average of squared gradients — a window of roughly a thousand steps. With a huge batch and an aggressive learning rate, that long, slow-moving estimate of the gradient scale reacts too sluggishly; the denominator lags the scale of the current gradients, the effective step size goes wrong, and training destabilizes. Shortening the window to $\beta_2 = 0.98$ — roughly fifty steps — lets the second-moment estimate adapt to the now lower-variance gradients, and stability returns. Concretely the update keeps Adam's form,

$$\theta_t = \theta_{t-1} - \alpha\,\hat m_t / (\sqrt{\hat v_t} + \epsilon),$$

with $\hat m_t$ and $\hat v_t$ the bias-corrected first and second moments under $\beta_1 = 0.9$, $\beta_2 = 0.98$, and $\epsilon = 1\mathrm{e}{-6}$; the change is solely the shortened second-moment window, made specifically because I went to large batches. With this in place I find I do not need gradient clipping at all — warmup plus the stabilized Adam keeps the updates bounded — so clipping is off, with a peak learning rate of $4\mathrm{e}{-4}$ (large) under linear warmup over 30k steps and then linear decay.

The fourth change is the tokenizer. The original uses a 30K character-level byte-pair-encoding vocabulary, but only after heuristic, language- and domain-specific tokenization rules, and any character outside the learned vocabulary becomes an unknown token — both of which are problems the moment I train on huge, messy, diverse web text. The cleaner idea is byte-level BPE: build the subword merges over bytes rather than unicode characters. Because every possible input is a sequence of bytes, a modest 50K-unit byte-level vocabulary can encode literally any text — every script, every emoji, every stray symbol — with zero unknown tokens and no preprocessing at all. It costs roughly 15M extra parameters at base and 20M at large, and in early comparisons it is even slightly worse on a few tasks, but the price is small and the universality is exactly what a single pipeline over 160GB of heterogeneous text needs, so I use the 50K byte-level BPE.

The fifth change addresses the two knobs the field treated as fixed but which I now suspect dominate: how much data and how long. To match the conflation that drove XLNet's edge, I first match data to the original BookCorpus plus Wikipedia (16GB) under my improved recipe, then add more — gathering over 160GB across five corpora: the original 16GB, a fresh CC-News collection (76GB of English news after filtering), an open recreation of the Reddit-link WebText (38GB), and a story-style CommonCrawl subset (31GB) — and separately I push the number of pretraining passes from 100k to 300k to 500k steps. Throughout, the architecture is frozen at the large configuration: 24 layers, hidden 1024, FFN inner 4096, 16 heads, head size 64, about 355M parameters; the base configuration is 12 layers, hidden 768, FFN inner 3072, 12 heads. What settles the attribution question is that quality improves monotonically as I add data and steps, with no sign of overfitting even at the longest run — meaning the objective was never the bottleneck, the budget was, and a well-trained masked-LM is competitive with everything published after it under the same architecture. The masked-LM head reprojects the selected positions back to the vocabulary through a dense layer, GELU, layer norm, and a projection tied to the input embeddings; a small classification head over the first-token representation handles finetuning.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ---- tokenizer: byte-level BPE, ~50K units, no preprocessing, no UNK ----
class Tokenizer:
    def __init__(self, encoder_json, vocab_bpe):
        self.bpe = load_byte_bpe(encoder_json, vocab_bpe)
    def encode(self, text):
        return self.bpe.encode(text)

# ---- full-sentences packing (no segment pairs, no NSP) ----
def build_pretraining_inputs(documents, max_len=512, sep_id=2):
    seqs, buf = [], []
    def flush():
        nonlocal buf
        if buf:
            seqs.append(buf)
            buf = []
    for doc in documents:
        if buf:
            if len(buf) + 1 > max_len:
                flush()
            else:
                buf.append(sep_id)
        for sent in doc:
            if len(sent) > max_len:
                sent = sent[:max_len]
            if len(buf) + len(sent) > max_len:
                flush()
            buf.extend(sent)
    flush()
    return seqs

# ---- dynamic masking: fresh each feed; 80/10/10 split over 15% ----
def mask_tokens(tokens, mask_id, vocab_size, *, ignore_index=-100, p=0.15,
                special_ids=()):
    out, labels = tokens.clone(), torch.full_like(tokens, ignore_index)
    eligible = torch.ones_like(tokens, dtype=torch.bool)
    for special_id in special_ids:
        eligible &= tokens.ne(special_id)
    sel = (torch.rand(tokens.shape, device=tokens.device) < p) & eligible
    labels[sel] = tokens[sel]
    r = torch.rand(tokens.shape, device=tokens.device)
    out[sel & (r < 0.8)] = mask_id                                  # 80% -> [MASK]
    rand = sel & (r >= 0.9)
    if rand.any():
        out[rand] = torch.randint(vocab_size, (int(rand.sum().item()),),
                                  device=tokens.device)             # 10% random
    return out, labels, sel                                         # 10% unchanged

# ---- masked-LM head ----
class MLMHead(nn.Module):
    def __init__(self, hidden, vocab_size, embed_weight):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.layer_norm = nn.LayerNorm(hidden)
        self.weight = embed_weight                       # tied to input embeddings
        self.bias = nn.Parameter(torch.zeros(vocab_size))
    def forward(self, features, masked_tokens=None):
        if masked_tokens is not None:
            features = features[masked_tokens]            # project only selected positions
        x = self.layer_norm(F.gelu(self.dense(features)))
        return F.linear(x, self.weight, self.bias)

def mlm_loss(logits, labels, masked_tokens=None):
    if masked_tokens is not None:
        labels = labels[masked_tokens]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100)

# ---- finetuning head over the first token ----
class ClassificationHead(nn.Module):
    def __init__(self, hidden, inner, n_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, inner)
        self.out_proj = nn.Linear(inner, n_classes)
    def forward(self, features):
        x = self.dropout(features[:, 0, :])              # <s> representation
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))

# ---- large-batch-stabilized Adam ----
def make_optimizer(params, peak_lr=4e-4):
    return torch.optim.Adam(params, lr=peak_lr, betas=(0.9, 0.98),
                            eps=1e-6, weight_decay=0.01)  # beta2=0.98; clipping off
```
