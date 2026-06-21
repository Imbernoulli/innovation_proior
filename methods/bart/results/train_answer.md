Self-supervised pretraining in NLP has split into two families that each give up something fundamental. Masked language models like BERT learn powerful bidirectional representations because every token can attend to context on both sides, which makes them excellent at classification and span extraction. But their predictions are conditionally independent given the corrupted input, and the input and target are the same length and aligned position-for-position, so they have no natural mechanism for generating a fresh sequence from left to right. Left-to-right language models like GPT generate fluently because they are trained autoregressively, yet each token only sees preceding context, which weakens them on understanding tasks that require future context. Variants such as span masking, permuted prediction orders, or mixed attention masks improve one side of this trade-off while remaining awkward for the other. The goal is a single pretraining objective that is strong across the full range of tasks: classification, span extraction, summarization, dialogue, abstractive question answering, and even machine translation.

The deeper reason masked models are stuck is that they corrupt text in place. A token is replaced by a mask symbol, but it stays in its slot, so the corruption must preserve length and position. That alignment rules out deletions, reorderings, rotations, or any length-changing noise that would force the model to reason about document structure rather than just fill fixed slots. To break the alignment, we can use the encoder-decoder Transformer architecture from machine translation: a bidirectional encoder reads a source sequence, and an autoregressive decoder writes a target sequence of arbitrary length, attending back to the encoder through cross-attention at every layer. The source and target are decoupled, so we can feed any corrupted document to the encoder and ask the decoder to reconstruct the original document token by token. The bidirectional encoder preserves understanding; the autoregressive decoder provides native generation. Reconstruction is itself a generation task, so pretraining and downstream generation have the same shape. If the corruption destroys all source information, the model reduces to a plain language model, which shows that nothing has been lost.

I propose BART, a denoising sequence-to-sequence Transformer pretrained with arbitrary noising functions. BART uses a standard seq2seq Transformer with GeLU activations and weights initialized from a normal distribution with standard deviation 0.02. The base model has six encoder and six decoder layers with hidden size 768, and the large model has twelve and twelve with hidden size 1024. Compared with a same-sized masked encoder, each decoder layer additionally cross-attends to the encoder's final hidden states, and the extra feed-forward network before the prediction layer is dropped because the decoder already builds rich per-position features. The model ends up about ten percent larger than a comparable masked encoder.

The key design freedom is the noising function, and several candidates can be composed. Token masking replaces random tokens with a mask token and serves as the inherited baseline. Token deletion actually removes random tokens, so the model must figure out where gaps exist. Sentence permutation shuffles the sentences in a document, forcing recovery of document-level order. Document rotation picks a random token and rotates the document so the model must find the true beginning. The most important transformation is text infilling: a span of tokens is replaced by a single mask token, the span length is drawn from a Poisson distribution with mean three including length zero, and the decoder must predict both the content of the missing span and how many tokens it contained. Zero-length spans insert a mask where nothing was deleted, adding further ambiguity. Because the encoder and decoder are decoupled, the decoder can emit a sequence of different length than the encoder saw, which is exactly the capability that in-place masked models lack. Empirically, text infilling is the most consistently strong single transformation, and the large model combines text infilling that masks thirty percent of tokens with full sentence permutation.

For discriminative finetuning, the presence of a decoder changes how representations are read out. For classification, the same uncorrupted input is fed to both the encoder and the decoder, and a class token is appended at the end of the decoder input. Because the decoder is causal, the final position has attended to the entire decoded sequence, so its hidden state serves as the sequence representation. For token-level tasks like span extraction, the full document is run through encoder and decoder, and start and end classifiers are applied to the top decoder states. For generation tasks, the model is used exactly as in pretraining: the encoder reads the source, and the decoder generates the target autoregressively with label-smoothed cross-entropy and beam search.

Machine translation is handled by reframing foreign-to-English translation as denoising. The foreign sentence is treated as a heavily corrupted encoding of the English meaning, and the pretrained English BART stack is reused as the target-side denoiser. Only the source-specific encoder embedding layer is replaced with a small randomly initialized foreign encoder that maps source-language tokens into a representation BART can denoise into English. Training proceeds in two stages: first freeze most of BART and update only the new encoder, BART's positional embeddings, and the first encoder layer's self-attention input projection; then briefly unfreeze all parameters and continue training end-to-end. This approach needs only monolingual English pretraining, not pretraining on the source language. The large model is pretrained with batch size 8000 for 500K steps using GPT-2 byte-level BPE on a 160GB mixture of news, books, stories, and web text, with dropout disabled for the final ten percent of training.

```python
import math
import torch, torch.nn as nn, torch.nn.functional as F

# ---- noising functions: encoder and decoder are decoupled, so corruption is arbitrary ----
def token_masking(tokens, mask_id, p=0.15):
    out = tokens.clone()
    sel = torch.rand_like(tokens, dtype=torch.float) < p
    out[sel] = mask_id
    return out

def token_deletion(tokens, p=0.15):
    return tokens[torch.rand(tokens.shape, device=tokens.device) >= p]

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
    n = len(tokens)
    budget = int(math.ceil(frac * n))
    lengths = _poisson_span_lengths(budget, lam, tokens.device)
    starts = torch.randperm(max(n, 1), device=tokens.device).tolist()
    used = torch.zeros(n, dtype=torch.bool, device=tokens.device)
    spans = []
    for length in lengths:
        if length == 0:
            pos = int(torch.randint(n + 1, (1,), device=tokens.device))
            spans.append((pos, 0))
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
        out.append(mask)
        i = max(i, start + length)
    if i < n:
        out.append(tokens[i:])
    return torch.cat(out) if out else tokens

def sentence_permutation(sentences):
    return [sentences[k] for k in torch.randperm(len(sentences))]

def document_rotation(tokens):
    if len(tokens) <= 2:
        return tokens
    k = int(torch.randint(1, len(tokens) - 1, (1,), device=tokens.device))
    return torch.cat([tokens[:1], tokens[k:-1], tokens[1:k], tokens[-1:]])

# ---- pretraining: encoder sees corruption, decoder reconstructs the original ----
def pretraining_loss(model, document, mask_id, pad_id):
    noised = flatten(sentence_permutation(split_sentences(document)))
    noised = text_infilling(noised, mask_id)
    logits = model(src_tokens=noised, prev_output_tokens=shift_right(document))
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        document.view(-1),
        ignore_index=pad_id,
    )

# ---- classification readout: class token at the end of the causal decoder ----
class ClassificationHead(nn.Module):
    def __init__(self, d_model, inner, n_classes, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d_model, inner)
        self.out_proj = nn.Linear(inner, n_classes)

    def forward(self, dec_out, src_tokens, eos_id):
        eos_mask = src_tokens.eq(eos_id)
        sent = dec_out[eos_mask, :].view(dec_out.size(0), -1, dec_out.size(-1))[:, -1, :]
        x = torch.tanh(self.dense(self.dropout(sent)))
        return self.out_proj(self.dropout(x))

# ---- machine translation: replace only the source-side encoder ----
class MTSourceEncoder(nn.Module):
    def __init__(self, src_vocab, d_model):
        super().__init__()
        self.embed = nn.Embedding(src_vocab, d_model)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=16, batch_first=True), 6)

    def forward(self, src):
        return self.layers(self.embed(src))
```
