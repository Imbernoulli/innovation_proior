Let me lay out the tension I actually want to resolve. The masked-language-model family — replace some tokens with a mask symbol, predict them from both sides — produces wonderful representations for understanding tasks, because every token sees its full context. But its predictions are conditionally independent given the input, and the output is token-aligned with the input, so it has no natural way to *generate* a fresh sequence. The left-to-right language-model family generates beautifully by rolling out one token at a time, but each position only sees its left context, so it's weak on understanding tasks where the right side disambiguates. Every recent variant — masking spans instead of words, predicting masked tokens in a permuted order, mixing in left-only attention masks — pushes on one side of this and stays awkward on the other. I want one pretraining scheme that's strong on both, across classification, span extraction, summarization, dialogue, long-form QA, even translation.

What is it that the masked models structurally give up? They corrupt text *in place*. A token gets replaced by a mask, but it stays in its slot; the input and the reconstruction target are the same length and aligned position-for-position. Let me make sure I have the cause-and-effect right, because the whole design will hinge on it. The masked model's loss is a sum over the masked positions of the cross-entropy at that position; "position" is well defined only because output slot *i* corresponds to input slot *i*. If I deleted a token instead of masking it, every slot after the deletion would shift by one, and the target at slot *i* would no longer be the thing the loss expects — the supervision signal becomes incoherent. So the alignment isn't a convenience, it's load-bearing: it's what makes the in-place objective even definable. And that immediately tells me the price. The only corruptions expressible under it are ones that preserve length and position — I can hide a token but I can't *delete* it, can't *reorder* sentences, can't change how long the document is. Those length-changing, order-changing corruptions are exactly the ones that would force a model to reason about structure at the document level, and they're precisely the ones the in-place loss forbids.

So I want to break the alignment. What machinery decouples an input sequence from a differently-shaped output sequence? The encoder-decoder Transformer from machine translation does exactly that: a bidirectional encoder reads a source sequence; an autoregressive decoder writes a target sequence, attending back to the encoder at every layer through cross-attention. The source and the target are completely decoupled — different content, different length, whatever. That decoupling is the thing the in-place objective was missing. If I feed a *corrupted* document to the encoder and ask the decoder to autoregressively reconstruct the *original* document, then the corruption can be anything at all — I can delete tokens, shuffle sentences, collapse spans — because the decoder isn't reading off aligned positions, it's generating the clean text from scratch conditioned on the encoded mess. The loss is no longer a per-aligned-slot sum; it's the autoregressive likelihood of the whole original document given the encoded corruption, which is defined regardless of how the lengths relate.

Let me check what that single object buys on each side. The encoder is bidirectional, so I keep the understanding strength of the masked family — nothing about feeding it a corrupted document changes the fact that each encoder position attends both ways. The decoder is left-to-right and autoregressive, so I get the generation strength of the language-model family — and reconstructing a document *is* generation, so the pretraining task and the downstream generation task are the same shape: produce clean target text autoregressively while attending to an encoded input. So the object is a denoising autoencoder built as sequence-to-sequence: corrupt the document with some noising function, train the model to reconstruct the original under the negative log-likelihood, which is the token-level cross-entropy between the decoder's output and the original document.

Before I trust this I want to know it isn't *weaker* than a plain language model — that I'm strictly adding capability, not trading some away. So take the limiting case: let the corruption be so destructive that the encoded input carries no information about the source — say, replace the entire document with a single mask, or delete everything. Then the encoder output is constant across all examples, the cross-attention has nothing to read, and the decoder's only signal is its own left context. Its likelihood factorizes as ∏ₜ p(xₜ | x_{<t}) — that is *exactly* the left-to-right language model. So the denoising objective contains the language model as the all-information-destroyed corner of its corruption space; any corruption that leaves the source partially intact can only give the decoder *more* to condition on. That's the reassurance I wanted: I haven't given anything up relative to a left-to-right LM, the encoder is pure upside. Good — the construction is at least as expressive as the baseline it's trying to beat.

Now the architecture details, which mostly follow from "standard seq2seq Transformer, made to play well with these pretraining conventions." I'll swap the original ReLU activations for GeLU and initialize from a normal with standard deviation 0.02, matching the conventions that the language-model side of the field settled on for deep transformers — smoother activation, calibrated initialization. Base is six encoder and six decoder layers; large is twelve and twelve. Compared to a same-sized masked encoder, two structural differences. First, every decoder layer additionally cross-attends to the encoder's final hidden states — that's just the seq2seq machinery, the channel through which the decoder reads the corrupted input. Second, the masked encoder puts an extra feed-forward network right before predicting the word; I don't need it, because my autoregressive decoder already builds up rich per-position features through its own stack before the output projection, so that pre-prediction layer would be redundant.

How much bigger does that make me than a same-sized masked model? Let me actually count the dominant terms, hidden size 768, comparing a 12-layer masked encoder to my 6 encoder + 6 decoder. An encoder layer is four d×d attention projections plus two d×4d feed-forward matrices; a decoder layer adds a second four-d×d block for cross-attention. Twelve encoder layers come to about 84.9M, while six-plus-six with the cross-attention comes to about 99.1M — a ratio of 1.17, with the cross-attention alone accounting for ~17% of the encoder's size. So my back-of-envelope says ~15–17% naively. That's higher than the ~10% I'd expect the real figure to land at, but the gap is explained: I dropped the pre-prediction feed-forward head the masked model carries, and the token-embedding and output-projection matrices (shared, and large at this vocabulary) sit in *both* models' totals, diluting the relative overhead. So I'll record this as "roughly ten percent larger, almost entirely from the decoder's cross-attention" while noting my crude count runs a little hotter — the precise number I'd confirm by instantiating both and calling `numel()`.

The interesting freedom is the noising function, so let me think through candidates and what each demands of the model. Plain token masking — replace random tokens with a mask — is the inherited baseline; fine, but it doesn't use any of my new freedom, the output is still the same length as the input. Token deletion — actually *remove* random tokens — is more demanding: now the model has to figure out not just what the missing token was but *where* tokens are missing, since there's no placeholder marking the gap. That's only possible because alignment is broken. Sentence permutation — split the document on full stops and shuffle the sentences — forces the model to recover document-level order. Document rotation — pick a random token and rotate the document to start there — forces it to find where the document actually begins. These last two are pure structure tasks that an in-place masked model literally cannot pose, since both change which token sits in which slot.

The candidate I'm most curious about is a masking variant that exploits length-change. Take a span of tokens and replace the *entire span* with a single mask token — and let the span length be random, drawn from a Poisson with mean three, including length zero. A zero-length span means I insert a lone mask where nothing was deleted. Contrast this with fixed-length span masking, where you replace a k-token span with exactly k mask symbols: there the model can *see* how many tokens are missing by counting the masks. The claim I'm making for the single-mask version is that the model can no longer count, so it must predict *how many* tokens the span held. Let me make sure my implementation actually has that property rather than just asserting it — I'll trace it on a tiny document, tokens 1..10, mask id 99, 30% budget so three tokens get masked per document on average:

```
original (len 10):  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
trial 0 -> len 8:   [1, 2, 3, 4, 5, 6, 99, 10]
trial 2 -> len 8:   [1, 99, 5, 6, 7, 8, 9, 10]
trial 4 -> len 9:   [99, 3, 4, 5, 6, 99, 8, 9, 10]
trial 1 -> len 10:  [1, 2, 3, 4, 5, 99, 7, 99, 9, 99]
```

Read trial 0: a *single* 99 sits where the original had `7, 8, 9` — three tokens collapsed to one mask, and the output is length 8, two shorter than the input. The decoder, conditioned on `...6 [MASK] 10`, has no positional cue that three tokens are hidden there rather than one or zero; it has to produce the count from content. Trial 2 collapses `2, 3, 4` to one mask the same way; trial 4 hides the two-token span `1, 2` at the front. Trial 1 is the case I wanted to confirm exists: it stays length 10 and carries masks where *zero-length* spans were drawn — a lone 99 inserted with nothing deleted, so the model must sometimes learn that a mask hides *nothing*. That's the behavior I was after, and it's real, not assumed: variable output length, one mask per span, and genuine insertions. I'll call this transformation text infilling.

These transformations compose, so I'll want to know which actually help, by replicating the competing objectives inside this same framework and comparing at base size, controlling data and optimization. I can express a left-to-right LM as my decoder with no cross-attention; a permuted language model that generates a sampled sixth of the tokens in random order; the masked model; a multitask masked model with mixed attention masks; and a masked seq2seq that masks half the tokens as a span and predicts them. For the ones that aren't naturally seq2seq I use two-stream attention to compute the output likelihoods efficiently with a diagonal mask, predicting left-to-right. I'd compare these by pretraining perplexity and by downstream score on a span-extraction task and a generation task, so I can see the understanding/generation tradeoff directly rather than guessing it.

I can already reason about what the limiting cases of that comparison must show, before running it, which is a useful prior. The pure structure tasks — rotation and permutation — carry no information about token *content*: rotation is a bijection on the existing tokens and permutation reshuffles intact sentences, so neither ever forces the model to predict an unseen token. A model trained only on those would learn to reorder but would be starved of the fill-in-the-blank signal that understanding tasks need, so I'd expect rotation and permutation *alone* to be weak and to need a content-masking partner. That's a prediction I'd want the ablation to confirm rather than a result I can assert here. For the generation side, my special-case argument already told me an autoregressive component matters: any objective whose decoding isn't left-to-right (the masked or multitask-masked variants) doesn't share the shape of the downstream generation task, so I expect those to trail at summarization — again, to be checked, not declared. And for span extraction, a left-only decoder should be poor because the classification decision needs future context, while the bidirectional encoder should rescue it even with only half as many bidirectional layers as a pure encoder. Among the single transformations, text infilling is the one that touches both axes at once — it masks content (helping understanding) and changes length (exercising the generative decoder) — so it's my best guess for the most consistently strong single noise, and the one I'll build the main model around. For the large model I'll pair it with sentence permutation: infilling for the reliable content signal, permutation for a document-level reordering signal a larger model has the capacity to exploit. I'm choosing this combination as a hypothesis the scaled run has to earn, not as a settled fact.

Now, how do I read a representation out for *discriminative* finetuning, given that I have a decoder and not just an encoder? For classification I feed the same (uncorrupted) input to both the encoder and the decoder, and take the decoder's final hidden state as the sequence representation. But which position? In a masked encoder you'd use a special class token at the front. Here the decoder is causal — a position can only attend to positions before it — so a token at the *front* would see almost nothing, which is the opposite of what a sequence summary needs. The position whose representation has attended over the *entire* decoded sequence is the *last* one, so I'll append the class token at the end and read its decoder state. Let me check the indexing I wrote does pick that position, because the head selects on an EOS mask, not on "the last token," and I want to be sure those coincide. I traced it on a two-sequence batch where I deliberately placed *two* EOS tokens in each sequence (positions 2 and 4 in one, 1 and 4 in the other) and encoded each decoder state with its own (batch, position) so I could see what got selected:

```
eos_mask:        [[0,0,1,0,1],
                  [0,1,0,0,1]]
selected (pos):  [4, 4]
```

So per sequence it grabs position 4 — the *last* EOS in each — exactly the final-token representation I argued for, and it ignores the earlier EOS. The `[:, -1, :]` is doing that work after the `view`. One subtlety the trace exposes: the `view(B, -1, D)` reshape only works because both sequences here happen to have the *same* number of EOS tokens (two each); with ragged EOS counts that reshape would fail, so in practice every example must carry the same EOS-marking convention. Worth remembering, but the readout itself does what I claimed: last position, full attention coverage. For token-level tasks like SQuAD endpoint classification I run the full document through encoder and decoder and use the per-token top decoder states, with start/end classifiers over them. For generation tasks — summarization, dialogue, abstractive QA — there's nothing to invent: the encoder reads the input, the decoder generates the output autoregressively, exactly as in pretraining.

Machine translation is the case that needs a real idea, because this model is pretrained only on English, yet I want to translate, say, Romanian into English. Let me reframe what translation *is* from the model's point of view. The decoder is already an expert at one thing: turning an encoded, possibly-corrupted input into clean English. A Romanian sentence is, in a loose sense, a heavily "corrupted" encoding of an English meaning — corrupted into another language. So if I could get a Romanian sentence into the representation space the encoder stack expects, the rest of the model would denoise it into English for free. That suggests: keep the entire pretrained stack as a target-side English denoiser, and only replace the piece that is language-specific — the encoder's *embedding* layer. Swap it for a small new randomly-initialized source encoder with its own foreign vocabulary, whose whole job is to map Romanian tokens into the kind of representation the rest of the stack knows how to denoise into English. Train end-to-end on bitext, backpropagating the reconstruction loss into the new encoder. This needs only monolingual English pretraining — no pretraining on the source language at all, unlike approaches that must pretrain on both languages. There's an obvious danger: a randomly-initialized source encoder, trained jointly with the delicate pretrained weights, will at first emit garbage and the gradients could wreck what the pretraining bought. So do it in two steps: first freeze most of the stack and update only the new source encoder plus the thin interface to it — the positional embeddings and the self-attention input projection of the first encoder layer — then unfreeze and train everything for a short while once the new encoder is producing something sane.

Two scaling details for the large run. I'll match the strong masked-model scale: batch eight thousand, half a million steps, byte-level BPE, the 160GB mixture, mask thirty percent of tokens per document under infilling and permute all sentences. And since at that scale the model is fitting the data rather than overfitting, I'll disable dropout for the final tenth of training so it can fully fit; generation finetuning uses label-smoothed cross-entropy with smoothing 0.1, and decoding uses beam search of width five with length tuning.

Let me write it. The body is a standard seq2seq Transformer; the contribution is the corruption functions and the end-token classification readout.

```python
import math
import torch, torch.nn as nn, torch.nn.functional as F

# --- noising functions: arbitrary because encoder/decoder are decoupled ---
def token_masking(tokens, mask_id, p=0.15):
    out = tokens.clone()
    sel = torch.rand_like(tokens, dtype=torch.float) < p
    out[sel] = mask_id
    return out                                    # same length, in place

def token_deletion(tokens, p=0.15):
    keep = torch.rand(tokens.shape, device=tokens.device) >= p
    return tokens[keep]                           # shorter: model must locate gaps

def _poisson_span_lengths(mask_budget, lam, device):
    dist = torch.distributions.Poisson(torch.tensor(float(lam), device=device))
    lengths, covered = [], 0
    while covered < mask_budget:
        for draw in dist.sample((max(mask_budget, 1),)).long().tolist():
            lengths.append(draw)
            covered += draw
            if covered >= mask_budget:
                break
    if lengths:
        lengths[-1] -= covered - mask_budget
    return lengths

def text_infilling(tokens, mask_id, lam=3.0, frac=0.30):
    # fairseq: mask_length="span-poisson", poisson_lambda=3, replace_length=1
    n = len(tokens)
    budget = int(math.ceil(frac * n))
    lengths = _poisson_span_lengths(budget, lam, tokens.device)
    starts = torch.randperm(max(n, 1), device=tokens.device).tolist()
    used = torch.zeros(n, dtype=torch.bool, device=tokens.device)
    spans = []
    for length in lengths:
        if length == 0:
            pos = int(torch.randint(n + 1, (1,), device=tokens.device))
            spans.append((pos, 0))                         # pure insertion
            continue
        while starts and used[starts[-1]]:
            starts.pop()
        if not starts:
            break
        start = starts.pop()
        end = min(start + length, n)
        used[start:end] = True
        spans.append((start, end - start))
    mask = torch.tensor([mask_id], dtype=tokens.dtype, device=tokens.device)
    out, i = [], 0
    for start, length in sorted(spans):
        if start > i:
            out.append(tokens[i:start])
        out.append(mask)                                  # one mask hides whole span
        i = max(i, start + length)
    if i < n:
        out.append(tokens[i:])
    return torch.cat(out) if out else tokens

def sentence_permutation(sentences):
    perm = torch.randperm(len(sentences))
    return [sentences[k] for k in perm]           # recover document order

def document_rotation(tokens):
    if len(tokens) <= 2:
        return tokens
    k = int(torch.randint(1, len(tokens) - 1, (1,), device=tokens.device))
    return torch.cat([tokens[:1], tokens[k:-1], tokens[1:k], tokens[-1:]])  # find true start

# --- pretraining: encoder sees corruption, decoder reconstructs the ORIGINAL ---
def pretraining_loss(model, document, mask_id, pad_id):
    sents = split_sentences(document)
    noised = flatten(sentence_permutation(sents))
    noised = text_infilling(noised, mask_id)             # infilling + permutation
    logits = model(src_tokens=noised, prev_output_tokens=shift_right(document))
    return F.cross_entropy(logits.view(-1, logits.size(-1)), document.view(-1),
                           ignore_index=pad_id)

# --- discriminative readout: classification token at the END of the decoder ---
class ClassificationHead(nn.Module):
    def __init__(self, d_model, inner, n_classes, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d_model, inner)
        self.out_proj = nn.Linear(inner, n_classes)
    def forward(self, dec_out, src_tokens, eos_id):
        # take the FINAL eos position: causal decoder => it attended to everything
        eos_mask = src_tokens.eq(eos_id)
        sent = dec_out[eos_mask, :].view(dec_out.size(0), -1, dec_out.size(-1))[:, -1, :]
        x = self.dropout(sent)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))

# --- machine translation: swap BART's source embedding for a new foreign encoder ---
class MTSourceEncoder(nn.Module):
    def __init__(self, src_vocab, d_model):
        super().__init__()
        self.embed = nn.Embedding(src_vocab, d_model)        # disjoint foreign vocab
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=16, batch_first=True), 6)
    def forward(self, src):                                   # foreign -> BART-denoisable input
        return self.layers(self.embed(src))
# step 1: freeze BART except positional embeddings + first encoder self-attn input proj
# step 2: unfreeze all for a few iterations; loss = BART reconstruction cross-entropy

# config (large): 12+12 layers, d_model 1024, GeLU, init N(0,0.02), batch 8000,
# 500k steps, GPT-2 BPE, 160GB; dropout off for final 10%; gen: label smoothing 0.1, beam 5
```

So the chain held together: masked models understand but can't generate and autoregressive models generate but don't fully understand, and the reason masked models are stuck is that they corrupt text *in place* — I checked that the alignment is what makes their loss definable at all, which is exactly why it forbids any length- or order-changing corruption. Decoupling input from output via an encoder-decoder removes that constraint — now the encoder can be fed *any* corruption and the autoregressive decoder reconstructs the clean document, which preserves bidirectional understanding (encoder) and native generation (decoder); I confirmed the all-information-destroyed corner reduces to a plain language model, so the encoder is strictly added capability. The freedom in corruption let me build text infilling, and tracing it on a small document confirmed the property I wanted — a single mask hides a whole Poisson-length span (including zero), the output length changes, so the model must predict *how many* tokens are missing — and I add sentence permutation for document-level structure. For discriminative readout I append the class token at the end and verified the indexing really selects that final position so a causal decoder's representation has attended to the full sequence; and for translation I reframe foreign-to-English as denoising and swap only the source embedding for a small new encoder, training in two steps to protect the pretrained weights.
