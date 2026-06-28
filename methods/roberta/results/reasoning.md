OK, let me think about this from the ground up. The field is moving fast: ELMo, then GPT, then BERT, then a permutation-LM called XLNet, each one announcing a new pretraining objective or architecture and a new state of the art on GLUE and SQuAD. Each new method says "our new idea beats the last." And I keep getting an uneasy feeling reading them, because each comparison changes several things at once. XLNet doesn't just swap the objective — it also trains on roughly ten times more data than BERT, with batches eight times larger for half the optimization steps. Let me actually multiply that out before I trust my own unease: 8× the batch over 0.5× the steps is 8 × 0.5 = 4× as many sequences seen. So XLNet sees about four times as much supervision as BERT, on top of ten times more distinct data. When XLNet beats BERT, what did the work? The permutation objective, or the fact that XLNet is simply a much bigger training run? I can't tell from the numbers as reported, because the two are tangled together.

Pretraining is the reason this confusion persists. It's expensive, so people tune lightly and run once; the corpora differ in size and are frequently private; and the convention is to ship a bundle of changes. That's a recipe for misattributing gains. So before I believe any claim that a new objective is better than masked language modeling, I want to ask a sharper question: how good is masked language modeling itself if I actually train it carefully? Maybe the baseline everyone is beating is just under-trained — that's a hypothesis I should hold loosely and try to break, not a conclusion.

Let me pin down exactly what the masked-LM recipe is, because the whole point is to vary it one knob at a time and keep everything else fixed. The model is a bidirectional Transformer. Input is two segments concatenated with delimiters — `[CLS] x … [SEP] y … [EOS]` — under a max length. Two losses. The masked LM loss: pick 15% of tokens, of those replace 80% with `[MASK]`, leave 10% unchanged, swap 10% for a random token, and predict the originals with cross-entropy. And a next-sentence-prediction loss: a binary classifier on whether the two segments were actually consecutive in the corpus, positives being adjacent text and negatives being segments from two different documents, fifty-fifty. The optimizer is Adam, β₁=0.9, β₂=0.999, ε=1e-6, weight decay 0.01, learning rate warmed up over 10k steps to 1e-4 then linearly decayed, dropout 0.1, GELU. The reference run is a million updates, batch 256, length 512. Data is 16GB: BookCorpus plus English Wikipedia.

Now I want to interrogate each of these as a *choice*, not a law. Let me hold the architecture completely fixed — same layers, same hidden size, same heads — so that whatever I find can't be attributed to a bigger or different network. I'll start from the base configuration, twelve layers, hidden 768, twelve heads, 110M parameters, and only touch the training procedure.

First the masking. The original implementation does something I find slightly odd once I look closely: it masks *once*, during preprocessing, producing a single static mask per sequence. To avoid feeding the identical mask every epoch, the data is duplicated ten times with ten different masks, and training runs forty epochs — so each specific mask gets seen 40/10 = 4 times. Fine for forty epochs. But it bothers me because the masking pattern is now baked into the data and coupled to the epoch count. If I want to train much longer, or on much more data, I either re-duplicate more or the model keeps re-seeing stale masks. The clean thing is to generate the mask on the fly, every single time a sequence is fed to the model — sample the 15% at load time instead of in preprocessing. Then the model essentially never sees the exact same masked instance twice. I should be careful here: the worry with sampling at load time is getting the 80/10/10 split wrong, since I'm now drawing it on every feed rather than once. Let me write it as a single uniform draw `r` per token and trace the thresholds. A selected token goes to `[MASK]` when `r<0.8`, stays unchanged when `0.8≤r<0.9`, becomes random when `r≥0.9`. Sampling `r` uniform on [0,1) over two million positions, the empirical fractions among selected tokens come out 0.800 / 0.100 / 0.100 — so the thresholds reproduce the intended 80/10/10 exactly, and dynamic generation doesn't quietly distort the corruption distribution. Whether dynamic helps or merely matches static I can't settle without a pretraining run; my expectation is it's at worst neutral, since it's strictly more varied supervision for the same data, and I'd want to confirm by running static (matched to the original) against dynamic and comparing masked-LM perplexity. The reason I'm keeping it regardless is structural: dynamic masking is what makes "train much longer" even coherent, because the supervision keeps refreshing instead of going stale after the fixed number of duplicates. So dynamic masking, carried forward into everything downstream.

Now the part that's been nagging at me most: the next-sentence-prediction loss. The original claim is that NSP matters — that removing it noticeably hurts NLI, QNLI, SQuAD. But several recent efforts quietly dropped NSP and didn't seem to suffer. Those two things can't both be casually true, so something about the original ablation is being misread. Let me reason about what NSP could even be teaching. It's supposed to give the model a notion of inter-sentence coherence — whether two spans belong together. But there's a confound hiding in *how* the input is built, separate from the loss itself. When you remove NSP, do you also change the input format, or just delete the loss term while keeping the segment-pair construction? Those are different experiments, and I suspect the original "removing NSP hurts" actually changed the input too.

So let me untangle objective from input format by laying out four constructions and only then deciding which one to keep. One: the original — pairs of multi-sentence *segments*, combined under 512, NSP on. Two: pairs of single *sentences*, NSP on; since single sentences are much shorter than 512, I'll raise the batch size to keep the token count comparable. Three: pack *full sentences* sampled contiguously from one or more documents up to 512, allowed to cross document boundaries with an extra separator token between documents, NSP off. Four: same packing but *not* allowed to cross document boundaries; near the end of a document the sequence runs short, so I dynamically grow the batch to keep tokens comparable, NSP off.

The discriminating comparison is one versus two, because they hold the loss fixed (NSP on in both) and change only the input length. If single-sentence input underperforms multi-sentence segment-pair input, then input length matters on its own, independent of NSP — and the original "no NSP hurts" result becomes suspect, because that ablation likely shortened the input at the same time. That is what I'd expect to see and would want to verify: single-sentence, NSP-on input losing to segment-pair, NSP-on input. If it holds, the thing costing performance in the "no NSP" ablation wasn't the missing loss — it was a degraded input that no longer gave the model long contiguous spans to learn long-range dependencies over. The clean test of NSP itself is then construction three or four against one: keep the input long and contiguous and *only* delete the NSP head. My prediction is that downstream is matched or slightly improved, which would mean that given proper long contiguous input the model already extracts whatever inter-span signal it needs from predicting masked tokens inside long passages, and NSP is redundant. The original conclusion would then be conflating "drop the loss" with "shorten the input." Between the two no-NSP packings, I'd lean toward three over four: four staying within a single document is plausibly marginally more coherent, but it forces variable batch sizes because of the short tail near document ends, and that variable batch size is exactly the kind of confound I'm trying to eliminate when comparing against other work. So I'll take full-sentences-no-NSP (construction three) for the main runs: drop NSP, pack contiguous full sentences to 512, allow crossing document boundaries with a separator. One loss now, the masked-token cross-entropy — conditional on the one-versus-two comparison coming out as I expect, which is the load-bearing check here.

Next, batch size. Here I have a prior from machine translation: training with very large minibatches improves both optimization speed and final quality, provided you raise the learning rate to match. The reason is that a larger batch gives a less noisy gradient estimate per step, so you can take bigger, more confident steps — fewer steps, each more accurate — and it parallelizes cleanly across devices. Let me get the bookkeeping exactly right so I'm comparing equal compute, not accidentally giving the big-batch run more data. The original run is a million steps at batch 256, which is 1,000,000 × 256 = 256,000,000 sequences seen. To hold that constant: at batch 2,000 it's 256,000,000 / 2,000 = 128,000 steps; at batch 8,000 it's 256,000,000 / 8,000 = 32,000 steps. So the trade is 1M steps × 256 ≡ 128k steps × 2k ≡ 32k steps × 8k, same number of passes over the data. Now I can ask, at fixed passes, what happens to quality as I move along that line. My expectation from the MT prior is that masked-LM perplexity and downstream accuracy improve at batch 2k, and that at batch 8k perplexity stays better than the 256-sequence run while downstream quality holds — and crucially the 8k run parallelizes far better. I can't read those perplexities off the page without the runs, so I'll treat "8k holds downstream quality" as the thing to verify; what I *can* commit to now is that 8k is a legitimate equal-compute point on this line, not a free lunch from extra data. I'll train with 8k sequences per batch and check the quality claim empirically.

But large batches expose a stability problem in the optimizer, and this is where I have to touch Adam, and where I can actually compute the thing I'm worried about rather than just assert it. With batch 8k the gradient is much less noisy than at 256, and the suspect parameter is the second-moment decay β₂. Adam keeps an exponential moving average of squared gradients, `v_t = β₂·v_{t-1} + (1-β₂)·g_t²`. The weight this EMA places on the gradient from `k` steps ago is `(1-β₂)·β₂^k`, a geometric decay, whose effective averaging window — the mean lag — is `1/(1-β₂)`. Plugging in: for the default β₂=0.999 that's 1/(1-0.999) = 1000 steps; for β₂=0.98 it's 1/(1-0.98) = 50 steps. So the default averages the gradient scale over a thousand-step window. Let me make the contrast sharper by asking how far back I have to reach to accumulate most of the EMA's mass: the cumulative weight over the most recent `n` steps is `1-β₂^n`, so reaching 90% needs `n = ln(0.1)/ln(β₂)`. For β₂=0.999 that's about 2301 steps; for β₂=0.98 it's about 114 steps. That's the concrete problem: with a huge batch and an aggressive learning rate, an estimate of the gradient scale that's pinned to what the gradients looked like two-to-three thousand steps ago reacts far too sluggishly to the current, lower-variance gradients — the denominator lags the true scale, the effective step size is wrong, and training destabilizes. Shrinking the window to ~114 steps by setting β₂=0.98 lets the second-moment estimate track the current gradient scale. So β₂=0.98, not 0.999, *specifically because* I went to large batches and the long EMA window is the wrong tool there; and I'll tune ε alongside it. With that in place I expect not to need gradient clipping — warmup plus the now-responsive second moment should keep the updates bounded — and I'd confirm that by watching the gradient-norm trace, but my plan is clipping off.

Now the tokenizer. The original uses a character-level byte-pair-encoding vocabulary of 30K, but only after running heuristic tokenization rules to preprocess the text first. That preprocessing is a problem the moment I want to train on huge, messy, diverse web text — news, Reddit-linked pages, story-style crawl — because the rules are language- and domain-specific and any character outside the learned vocabulary becomes an unknown token. There's a cleaner idea: byte-level BPE. Instead of building subwords over unicode *characters*, build them over *bytes*. The reason this guarantees no unknown tokens is a counting fact, not a hope: there are exactly 256 possible byte values, so a base vocabulary that includes all 256 single bytes can represent any byte sequence whatsoever before a single merge is learned; the BPE merges on top only ever compress, they never introduce a symbol that can't fall back to its constituent bytes. So a modest 50K-unit byte-level vocabulary encodes literally any text — every script, every emoji, every stray symbol — with zero UNKs and no preprocessing. That's exactly what I want for a universal pipeline over heterogeneous web text: one tokenizer, no per-corpus cleaning. It costs something — the larger vocabulary adds roughly 15M parameters at base and 20M at large, and I'd expect byte-level to be a touch worse than character-level on a few end tasks in a head-to-head, which I'd want to check. But the parameter price is small relative to a 355M model and the universality is what makes a 160GB multi-corpus run feasible at all, so I'll use the 50K byte-level BPE and accept a small possible end-task cost.

Two more knobs that the field has been treating as fixed but that I now suspect dominate: how much data, and how long. The original is 16GB and effectively a fixed budget. Recall the XLNet arithmetic from the start — its edge came partly from ~10× the data and the 4× more sequences I computed above; that's the conflation again. So let me first match data to the original (BookCorpus + Wikipedia) and run my improved recipe, to see how much of any gain is recipe rather than data; then add more. I'll gather five corpora totalling over 160GB: the original BookCorpus + Wikipedia (16GB), a fresh CommonCrawl-News collection I build (CC-News, 76GB of English news after filtering, tens of millions of articles), an open recreation of the Reddit-link WebText (38GB), and a story-style CommonCrawl subset (31GB). 16 + 76 + 38 + 31 = 161GB, so "over 160GB" checks out. And separately I'll push the number of pretraining passes: take the same step budget out to 100k, then 300k, then 500k.

Let me now aggregate every one of these into a single configuration and reason about whether it's coherent. Dynamic masking — fresh mask each feed, with the 80/10/10 split I traced above preserved. Full-sentences input with no NSP — one loss, long contiguous spans, conditional on the segment-length comparison confirming NSP is the redundant part. Large batches of 8k at the equal-compute step count I computed, with the stabilized Adam (β₂=0.98) justified by the EMA-window calculation. The 50K byte-level BPE with its UNK-free guarantee. Then scale data and steps. To disentangle data and duration from the modeling choices, I keep the architecture exactly at the large configuration — twenty-four layers, hidden 1024, sixteen heads, 355M parameters — so nothing I observe can be credited to a different network. Pretrain 100k steps on the original-sized data; then the same 100k steps on the full 160GB; then 300k; then 500k.

What would actually settle the attribution question is the *shape* of that last curve: monotone improvement as I add data and steps, with no sign of overfitting even at 500k. If that's what comes out, it means the objective was never the bottleneck — the *budget* was — and a well-trained masked-LM is competitive with everything published after it under the same architecture. If instead the curve flattens early or overfits, my whole hypothesis is wrong and the new objectives really are doing the work. I can't run that curve here, so I'm explicit that this is the experiment that confirms or kills the thesis, not something I've already established. The exercise only *means* something because I held the architecture and the objective fixed and moved the recipe; any gain has to come from the recipe by construction.

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

So the causal chain is: I distrusted the head-to-head comparisons because each new method changed objective, data, and budget together — and when I multiplied out XLNet's batch and step changes it was seeing 4× the sequences on 10× the data — so I fixed the architecture and the objective and asked what a *well-trained* masked-LM does. Static masking coupled the mask to epoch count, so I made it dynamic, after checking the on-the-fly draw still hits 80/10/10. The "NSP helps" claim, I argued, conflates dropping the loss with shortening the input; the experiment that decides this is single-sentence-NSP-on versus segment-pair-NSP-on, and if input length is what matters there, then with long contiguous full-sentence input the NSP loss is redundant and I keep one loss. Large batches sit at a known equal-compute point (32k steps × 8k ≡ 1M × 256) and should improve the objective while parallelizing, but they break Adam's default second-moment window — which I computed reaches ~2300 steps back at β₂=0.999 versus ~114 at β₂=0.98 — so I shortened it to β₂=0.98 and dropped clipping. A byte-level 50K BPE gives a universal, preprocessing-free, provably UNK-free tokenizer for messy web text at a small parameter cost. And the two knobs the field had under-emphasized — data quantity and training duration — I scale out to 160GB and 500k steps with the architecture frozen, so that whatever improvement appears is attributable to the recipe and not to a new objective or a bigger network. The pieces I could check on paper, I checked; the pieces that need the pretraining runs, I've flagged as the experiments that confirm or refute the thesis rather than as settled facts.
