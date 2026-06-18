Let me lay out the tension I actually want to resolve. The masked-language-model family — replace some tokens with a mask symbol, predict them from both sides — produces wonderful representations for understanding tasks, because every token sees its full context. But its predictions are conditionally independent given the input, and the output is token-aligned with the input, so it has no natural way to *generate* a fresh sequence. The left-to-right language-model family generates beautifully by rolling out one token at a time, but each position only sees its left context, so it's weak on understanding tasks where the right side disambiguates. Every recent variant — masking spans instead of words, predicting masked tokens in a permuted order, mixing in left-only attention masks — pushes on one side of this and stays awkward on the other. I want one pretraining scheme that's strong on both, across classification, span extraction, summarization, dialogue, long-form QA, even translation.

What is it that the masked models structurally give up? They corrupt text *in place*. A token gets replaced by a mask, but it stays in its slot; the input and the reconstruction target are the same length and aligned position-for-position. That alignment is the cage. It means the only corruptions you can express are ones that preserve length and position — you can hide a token but you can't *delete* it, can't *reorder* sentences, can't change how long the document is, because then the model wouldn't know which output position corresponds to which input position. And those length-changing, order-changing corruptions are exactly the ones that would force a model to reason about structure at the document level.

So break the alignment. The way to break it is already sitting in the machine-translation architecture: the encoder-decoder Transformer. A bidirectional encoder reads a source sequence; an autoregressive decoder writes a target sequence, attending back to the encoder at every layer through cross-attention. The source and the target are completely decoupled — different content, different length, whatever. That decoupling is precisely what I need. If I feed a *corrupted* document to the encoder and ask the decoder to autoregressively reconstruct the *original* document, then the corruption can be anything at all — I can delete tokens, shuffle sentences, collapse spans — because the decoder isn't reading off aligned positions, it's generating the clean text from scratch conditioned on the encoded mess.

And look at what that single object gives me for free. The encoder is bidirectional, so I keep the understanding strength of the masked family. The decoder is left-to-right and autoregressive, so I get the generation strength of the language-model family — and not in a contrived way, because reconstructing a document *is* generation. The pretraining task and the downstream generation task are the same shape: produce clean target text autoregressively while attending to an encoded input. So this is a denoising autoencoder built as sequence-to-sequence: corrupt the document with some noising function, train the model to reconstruct the original under the negative log-likelihood, which is just the token-level cross-entropy between the decoder's output and the original document. In the extreme where the corruption destroys all information about the source, the encoder is useless and the decoder degenerates to a plain language model — so this objective contains the language model as a special case, which is reassuring; I haven't lost anything.

Now the architecture details, which mostly follow from "standard seq2seq Transformer, made to play well with these pretraining conventions." I'll swap the original ReLU activations for GeLU and initialize from a normal with standard deviation 0.02, matching the conventions that the language-model side of the field settled on for deep transformers — smoother activation, calibrated initialization. Base is six encoder and six decoder layers; large is twelve and twelve. Compared to a same-sized masked encoder, two structural differences. First, every decoder layer additionally cross-attends to the encoder's final hidden states — that's just the seq2seq machinery, the channel through which the decoder reads the corrupted input. Second, the masked encoder puts an extra feed-forward network right before predicting the word; I don't need it, because my autoregressive decoder already builds up rich per-position features through its own stack before the output projection, so that pre-prediction layer would be redundant. The whole thing ends up about ten percent larger than the same-sized masked model, almost entirely from the decoder's cross-attention.

The interesting freedom is the noising function, so let me think through candidates and what each teaches. Plain token masking — replace random tokens with a mask — is the inherited baseline; fine, but it doesn't use any of my new freedom. Token deletion — actually *remove* random tokens — is more demanding: now the model has to figure out not just what the missing token was but *where* tokens are missing, since there's no placeholder marking the gap. That's only possible because alignment is broken. Sentence permutation — split the document on full stops and shuffle the sentences — forces the model to recover document-level order. Document rotation — pick a random token and rotate the document to start there — forces it to find where the document actually begins. These last two are pure structure tasks that an in-place masked model literally cannot pose.

The one I most want is a masking variant that exploits length-change. Take a span of tokens and replace the *entire span* with a single mask token — and let the span length be random, drawn from a Poisson with mean three, including length zero. A zero-length span means I insert a lone mask where nothing was deleted. Contrast this with fixed-length span masking, where you replace a k-token span with exactly k mask symbols: there the model can *see* how many tokens are missing by counting the masks. With a single mask standing in for a whole span of unknown length, the model has to predict not just the content of the span but *how many tokens* it contained — it has to reason about length, and about whether a mask even hides anything. Call it text infilling. Because alignment is broken, the decoder can emit more (or fewer) tokens than the encoder saw at that position, which is the whole point.

These transformations compose, so I'll want to know which actually help, by replicating the competing objectives inside this same framework and comparing at base size, controlling data and optimization. I can express GPT as my decoder with no cross-attention; a permuted language model that generates a sampled sixth of the tokens in random order; the masked model; a multitask masked model with mixed attention masks; and a masked seq2seq that masks half the tokens as a span and predicts them. For the ones that aren't naturally seq2seq I use two-stream attention to compute the output likelihoods efficiently with a diagonal mask, predicting left-to-right.

A few things fall out of that comparison that sharpen the design. Token masking (or deletion, or attention-mask tricks) is essential — the pure structure tasks, rotation and permutation, are poor *on their own*; they don't teach enough about token content. Deletion tends to beat masking on generation. Crucially, any objective that lacks a left-to-right autoregressive component is weaker at generation — which validates having the autoregressive decoder rather than a masked predictor. And for span extraction, a left-only decoder alone is poor because the classification decision needs future context, yet the bidirectional encoder rescues that even with only half as many bidirectional layers as a pure encoder would use. Across the board, text infilling is the most consistently strong single transformation. So for the main model I'll combine text infilling with sentence permutation: infilling is the reliable workhorse, and the document-level reordering from permutation is the kind of global signal a larger model should be able to exploit.

Now, how do I read a representation out for *discriminative* finetuning, given that I have a decoder and not just an encoder? For classification I feed the same (uncorrupted) input to both the encoder and the decoder, and take the decoder's final hidden state as the sequence representation. But which position? In a masked encoder you'd use a special class token at the front. Here the decoder is causal — a position can only attend to positions before it — so a token at the *front* would see almost nothing. I want the readout token to have attended over the *entire* decoded sequence, so I append the class token at the *end*; being last, its representation has attended to everything. For token-level tasks like SQuAD endpoint classification I run the full document through encoder and decoder and use the per-token top decoder states, with start/end classifiers over them. For generation tasks — summarization, dialogue, abstractive QA — there's nothing to invent: the encoder reads the input, the decoder generates the output autoregressively, exactly as in pretraining.

Machine translation is the case that needs a real idea, because BART is pretrained only on English, yet I want to translate, say, Romanian into English. Reframe it: translating foreign-to-English is like *denoising* — the foreign sentence is a heavily "corrupted" encoding of the English meaning, and BART is already an expert at denoising encoded input into clean English. So start from the entire pretrained BART stack as a target-side English denoiser, and only replace the piece that's language-specific: BART's encoder *embedding* layer. Replace it with a small new randomly-initialized source encoder that has its own foreign vocabulary, whose job is to map Romanian tokens into the kind of representation BART's encoder stack knows how to denoise into English. Train end-to-end on bitext, backpropagating BART's reconstruction loss into the new encoder. This needs only monolingual English pretraining — no pretraining on the source language at all, unlike approaches that must pretrain on both languages. To avoid wrecking the pretrained weights early, do it in two steps: first freeze most of BART and update only the new source encoder plus the thin interface to BART — its positional embeddings and the self-attention input projection of its first encoder layer — then unfreeze and train everything for a short while.

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

So the chain: masked models understand but can't generate and autoregressive models generate but don't fully understand, and the deeper reason masked models are stuck is that they corrupt text *in place*, which forbids any length- or order-changing corruption. Decoupling input from output via an encoder-decoder breaks that cage — now the encoder can be fed *any* corruption and the autoregressive decoder reconstructs the clean document, which simultaneously preserves bidirectional understanding (encoder) and native generation (decoder), with the plain language model as the all-information-destroyed special case. The freedom in corruption let me invent text infilling, where a single mask hides a whole Poisson-length span so the model must predict *how many* tokens are missing, and to add sentence permutation for document-level structure. For discriminative readout I append the class token at the end so a causal decoder's representation has attended to the full sequence, and for translation I reframe foreign-to-English as denoising and swap only the source embedding for a small new encoder, training in two steps to protect the pretrained weights.
