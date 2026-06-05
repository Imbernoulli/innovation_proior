OK, let me think about this from the ground up. The field is moving fast: ELMo, then GPT, then BERT, then a permutation-LM called XLNet, each one announcing a new pretraining objective or architecture and a new state of the art on GLUE and SQuAD. Each new method says "our new idea beats the last." And I keep getting an uneasy feeling reading them, because each comparison changes several things at once. XLNet doesn't just swap the objective — it also trains on roughly ten times more data than BERT, with batches eight times larger for half the optimization steps, which means it sees about four times as many sequences. So when XLNet beats BERT, what did the work? The permutation objective, or the fact that XLNet is simply a much bigger training run? Nobody can say, because the two are tangled together.

Pretraining is the reason this confusion persists. It's expensive, so people tune lightly and run once; the corpora differ in size and are frequently private; and the convention is to ship a bundle of changes. That's a recipe for misattributing gains. So before I believe any claim that a new objective is better than masked language modeling, I want to ask a sharper question: how good is masked language modeling itself if I actually train it carefully? Maybe the baseline everyone is beating is just under-trained.

Let me pin down exactly what the masked-LM recipe is, because the whole point is to vary it one knob at a time and keep everything else fixed. The model is a bidirectional Transformer. Input is two segments concatenated with delimiters — `[CLS] x … [SEP] y … [EOS]` — under a max length. Two losses. The masked LM loss: pick 15% of tokens, of those replace 80% with `[MASK]`, leave 10% unchanged, swap 10% for a random token, and predict the originals with cross-entropy. And a next-sentence-prediction loss: a binary classifier on whether the two segments were actually consecutive in the corpus, positives being adjacent text and negatives being segments from two different documents, fifty-fifty. The optimizer is Adam, β₁=0.9, β₂=0.999, ε=1e-6, weight decay 0.01, learning rate warmed up over 10k steps to 1e-4 then linearly decayed, dropout 0.1, GELU. The reference run is a million updates, batch 256, length 512. Data is 16GB: BookCorpus plus English Wikipedia.

Now I want to interrogate each of these as a *choice*, not a law. Let me hold the architecture completely fixed — same layers, same hidden size, same heads — so that whatever I find can't be attributed to a bigger or different network. I'll start from the base configuration, twelve layers, hidden 768, twelve heads, 110M parameters, and only touch the training procedure.

First the masking. The original implementation does something I find slightly odd once I look closely: it masks *once*, during preprocessing, producing a single static mask per sequence. To avoid feeding the identical mask every epoch, the data is duplicated ten times with ten different masks, and training runs forty epochs — so each specific mask gets seen four times. Fine for forty epochs. But it bothers me because the masking pattern is now baked into the data and coupled to the epoch count. If I want to train much longer, or on much more data, I either re-duplicate more or the model keeps re-seeing stale masks. The clean thing is to generate the mask on the fly, every single time a sequence is fed to the model — dynamic masking. The model then essentially never sees the exact same masked instance twice. Intuitively that's strictly more varied supervision for the same data, and it costs almost nothing — I'm already iterating the data, I just sample the 15% at load time. Let me check it doesn't *hurt*: comparing static (matched to the original) against dynamic, dynamic comes out comparable or a touch better. And it's the thing that makes "train much longer" even coherent, because the supervision keeps refreshing. So dynamic masking, and I'll keep it for everything downstream.

Now the part that's been nagging at me most: the next-sentence-prediction loss. The original claim is that NSP matters — that removing it noticeably hurts NLI, QNLI, SQuAD. But several recent efforts quietly dropped NSP and didn't seem to suffer. Those two things can't both be casually true, so something about the original ablation is being misread. Let me reason about what NSP could even be teaching. It's supposed to give the model a notion of inter-sentence coherence — whether two spans belong together. But there's a confound hiding in *how* the input is built, separate from the loss itself. When you remove NSP, do you also change the input format, or just delete the loss term while keeping the segment-pair construction? Those are different experiments, and I suspect the original "removing NSP hurts" actually changed the input too.

So let me untangle objective from input format by laying out four constructions and only then deciding. One: the original — pairs of multi-sentence *segments*, combined under 512, NSP on. Two: pairs of single *sentences*, NSP on; since single sentences are much shorter than 512, I'll raise the batch size to keep the token count comparable. Three: pack *full sentences* sampled contiguously from one or more documents up to 512, allowed to cross document boundaries with an extra separator token between documents, NSP off. Four: same packing but *not* allowed to cross document boundaries; near the end of a document the sequence runs short, so I dynamically grow the batch to keep tokens comparable, NSP off.

Stare at the results across these. The single-sentence setting, with NSP still on, *hurts* relative to the multi-sentence segment-pair setting. That's the tell. The thing that was actually costing performance in the "no NSP" ablation wasn't the missing loss — it was likely a degraded input that no longer gave the model long contiguous spans to learn long-range dependencies over. When I instead keep the input long and contiguous (full sentences packed to 512) and *only* delete the NSP head, downstream is matched or slightly improved. So NSP, given proper long contiguous input, is redundant: the model already gets whatever inter-span signal it needs from predicting masked tokens inside long passages. The original conclusion conflated "drop the loss" with "shorten the input." And between the two no-NSP packings, staying within a single document is marginally better — presumably the packed sequence is more coherent when it doesn't splice unrelated documents — but it forces variable batch sizes because of the short tail near document ends, which is annoying to compare against other work. So I'll take full-sentences-no-NSP for the main runs: drop NSP, pack contiguous full sentences to 512, allow crossing document boundaries with a separator. One loss now, the masked-token cross-entropy.

Next, batch size. Here I have a prior from machine translation: training with very large minibatches improves both optimization speed and final quality, provided you raise the learning rate to match. The reason is that a larger batch gives a less noisy gradient estimate per step, so you can take bigger, more confident steps — fewer steps, each more accurate — and it parallelizes cleanly across devices. The original run is a million steps at batch 256. By gradient accumulation that's the same compute as 125k steps at batch 2k, or 31k steps at batch 8k — I can trade step count for batch size at fixed number of passes over the data and check what happens to quality. At batch 2k the masked-LM perplexity and downstream accuracy improve clearly; at batch 8k the perplexity is still better than the 256-sequence run and downstream quality stays comparable, while the scaled run becomes much easier to parallelize. So large batches aren't only an engineering convenience here; they are a viable training regime. I'll train with 8k sequences per batch.

But large batches expose a stability problem in the optimizer, and this is where I have to touch Adam. With batch 8k the gradient is much less noisy than at 256, and I find training is sensitive to two things: the ε in the denominator, and the second-moment decay β₂. The default β₂=0.999 maintains a very long running average of squared gradients — a window of roughly a thousand steps. With a huge batch and an aggressive learning rate, that long, slow-moving estimate of the gradient scale reacts too sluggishly; the denominator lags the scale of the current gradients, so the effective step size can be wrong, and training becomes unstable. Shortening that window — β₂=0.98, roughly fifty steps — lets the second-moment estimate adapt faster to the now lower-variance gradients, and stability returns. So β₂=0.98, not 0.999, specifically *because* I went to large batches; and I tune ε as well. With that in place I find I don't need gradient clipping at all — warmup plus the stabilized Adam keeps the updates bounded — so clipping is off.

Now the tokenizer. The original uses a character-level byte-pair-encoding vocabulary of 30K, but only after running heuristic tokenization rules to preprocess the text first. That preprocessing is a problem the moment I want to train on huge, messy, diverse web text — news, Reddit-linked pages, story-style crawl — because the rules are language- and domain-specific and any character outside the learned vocabulary becomes an unknown token. There's a cleaner idea: byte-level BPE. Instead of building subwords over unicode *characters*, build them over *bytes*. Every possible input is a sequence of bytes, so a modest 50K-unit byte-level vocabulary can encode literally any text — every script, every emoji, every stray symbol — with zero unknown tokens and no preprocessing at all. That's exactly what I want for a universal pipeline over 160GB of heterogeneous text: one tokenizer, no per-corpus cleaning, no UNKs. It costs something — the larger vocabulary adds roughly 15M parameters at base and 20M at large, and in early comparisons the byte-level encoding is even *slightly* worse than the character-level one on a few tasks. But the price is small and the universality is worth it, so I'll use the 50K byte-level BPE.

Two more knobs that the field has been treating as fixed but that I now suspect dominate: how much data, and how long. The original is 16GB and effectively a fixed budget. XLNet's edge over the baseline came partly from ~10× the data and ~4× the sequences seen — that's the conflation again. So let me first match data to the original (BookCorpus + Wikipedia) and run my improved recipe; then add more. I'll gather five corpora totalling over 160GB: the original BookCorpus + Wikipedia (16GB), a fresh CommonCrawl-News collection I build (CC-News, 76GB of English news after filtering, tens of millions of articles), an open recreation of the Reddit-link WebText (38GB), and a story-style CommonCrawl subset (31GB). And separately I'll push the number of pretraining passes: take the same step budget out to 100k, then 300k, then 500k.

Let me now aggregate every one of these into a single configuration and reason about whether it's coherent. Dynamic masking — fresh mask each time. Full-sentences input with no NSP — one loss, long contiguous spans. Large batches of 8k with the stabilized Adam (β₂=0.98). The 50K byte-level BPE. Then scale data and steps. To disentangle data and duration from the modeling choices, I keep the architecture exactly at the large configuration — twenty-four layers, hidden 1024, sixteen heads, 355M parameters — so nothing I observe can be credited to a different network. Pretrain 100k steps on the original-sized data; then the same 100k steps on the full 160GB; then 300k; then 500k.

What I'd want to see, and what would settle the attribution question, is monotone improvement as I add data and steps, with no sign of overfitting even at the longest run — which would mean the objective was never the bottleneck, the *budget* was, and a well-trained masked-LM is competitive with everything published after it under the same architecture. The whole exercise only makes sense because I held the architecture and the objective fixed and moved the recipe; any gain has to come from the recipe.

Let me now write this as code, filling exactly the slots I left open: the tokenizer, the input construction and masking, the loss, the optimizer settings, and the heads. The model body is a stock bidirectional Transformer encoder; the contribution lives in how I feed and optimize it.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- tokenizer: byte-level BPE, 50K units, encodes ANY text, no preprocessing, no UNK ---
class Tokenizer:
    def __init__(self, encoder_json, vocab_bpe):
        # bytes -> printable unicode mapping, then BPE merges over the byte stream
        self.bpe = load_byte_bpe(encoder_json, vocab_bpe)  # ~50265 symbols
    def encode(self, text):                # never emits an unknown token
        return self.bpe.encode(text)

# --- pretraining input: full-sentences packing, dynamic masking, no NSP ---
def build_pretraining_inputs(documents, max_len=512, sep_id=2):
    # pack contiguous tokenized sentences up to max_len; cross documents with a separator
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
    return seqs                            # no segment-pair labels

def mask_tokens(tokens, mask_id, vocab_size, *, ignore_index=-100, p=0.15,
                special_ids=()):
    # fresh mask every time this sequence is fed to the model
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
    # remaining 10% of selected positions: left unchanged
    return out, labels, sel

# --- masked-LM head: dense -> gelu -> layernorm -> tied projection ---
class MLMHead(nn.Module):
    def __init__(self, hidden, vocab_size, embed_weight):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.layer_norm = nn.LayerNorm(hidden)
        self.weight = embed_weight                       # tie to input embeddings
        self.bias = nn.Parameter(torch.zeros(vocab_size))
    def forward(self, features, masked_tokens=None):
        if masked_tokens is not None:
            features = features[masked_tokens]            # project only selected positions
        x = self.layer_norm(F.gelu(self.dense(features)))
        return F.linear(x, self.weight, self.bias)

# the only pretraining loss: masked-token cross-entropy (NSP removed)
def mlm_loss(logits, labels, masked_tokens=None):
    if masked_tokens is not None:
        labels = labels[masked_tokens]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100)

# --- finetuning head over the first token (<s>) ---
class ClassificationHead(nn.Module):
    def __init__(self, hidden, inner, n_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, inner)
        self.out_proj = nn.Linear(inner, n_classes)
    def forward(self, features):
        x = features[:, 0, :]                            # <s>/CLS representation
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))

# --- optimizer: large-batch-stabilized Adam ---
def make_optimizer(params, peak_lr=4e-4):
    return torch.optim.Adam(params, lr=peak_lr,
                            betas=(0.9, 0.98),  # beta2=0.98 for large-batch stability
                            eps=1e-6, weight_decay=0.01)
# gradient clipping OFF; LR linear warmup (30k steps) then linear decay
# batch 8k sequences, length 512, up to 500k steps, 160GB across 5 corpora
```

So the causal chain is: I distrusted the head-to-head comparisons because each new method changed objective, data, and budget together, so I fixed the architecture and the objective and asked what a *well-trained* masked-LM does. Static masking coupled the mask to epoch count, so I made it dynamic. The "NSP helps" claim turned out to conflate dropping the loss with shortening the input; with long contiguous full-sentence input the NSP loss is redundant, so I removed it and kept one loss. Large batches improve the masked-LM objective and can keep downstream quality intact while making the run parallel enough to scale, but they break Adam's default long second-moment window, so I shortened it to β₂=0.98 and dropped clipping. A byte-level 50K BPE gives a universal, preprocessing-free, UNK-free tokenizer for messy web text at a small parameter cost. And the two knobs the field had under-emphasized — data quantity and training duration — I scaled out to 160GB and 500k steps with the architecture frozen, so that whatever improvement appears is attributable to the recipe and not to a new objective or a bigger network.
