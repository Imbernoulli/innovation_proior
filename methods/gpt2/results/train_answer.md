The problem is how to make one language system handle many different language tasks without collecting fresh labels or changing the model for each task. The standard recipe is still narrow: pick a task, gather supervised examples, and train or fine-tune a model. Transfer learning helps by pre-training on unlabeled text, but it usually ends with task-specific fine-tuning, task-specific heads, and task-specific input formatting. That means the system is not really one model doing many things; it is one initialization followed by many small specialized systems. The alternative is to treat the task description itself as part of the input text, so that p(output | input, task) becomes an ordinary next-token prediction problem. The obstacle is that this only works if the training corpus actually contains a wide variety of naturally occurring demonstrations, the tokenizer can represent any benchmark string without lossy preprocessing, and the model is large and stable enough to learn those patterns.

Existing approaches fall short in three ways. First, single-domain corpora such as books or Wikipedia produce strong language models for one style of text but contain too few natural examples of translation, summarization, question answering, or dialogue. Raw web scrapes have the variety but are noisy and duplicated. Second, standard subword tokenizers rely on a fixed vocabulary and an unknown-token escape, which makes them fragile on rare strings and benchmarks. Byte-level modeling covers everything but wastes capacity on spelling and encoding details. Third, the original Transformer block places layer normalization after the residual addition, which makes very deep stacks unstable because the residual stream is normalized at every step instead of staying an additive path.

The method is GPT-2. It trains a single left-to-right Transformer language model on a broad web corpus and evaluates it zero-shot by prompting or scoring continuations, with no parameter updates and no task-specific heads.

The data pipeline starts from outbound links posted to Reddit with at least three karma, using that as a cheap human quality filter. Article text is extracted, deduplicated, and heuristically cleaned. Documents linked after December 2017 are excluded to create a temporal cutoff, and Wikipedia documents are removed to reduce benchmark contamination. The resulting corpus contains slightly over eight million documents and about forty gigabytes of text. This is not a task dataset; it is a broad field of naturally occurring demonstrations that embed many induced tasks.

The tokenizer is byte-level byte-pair encoding. UTF-8 bytes are mapped to reversible Unicode characters, which keeps the base alphabet at 256 symbols instead of the full Unicode code-point set. BPE merges are then applied after splitting the text by character category, so letters merge with letters, numbers with numbers, punctuation with punctuation, and whitespace follows its own pattern. A leading space is allowed to merge with a word because English word boundaries are informative. This yields a vocabulary of 50,257 tokens with no unknown token and no lossy normalization, so the model can score any benchmark string directly.

The model is a decoder-only Transformer with causal masked self-attention. The residual block is changed so that layer normalization sits before the attention and feed-forward sub-blocks, and the residual addition happens after each sub-block. A final layer normalization is added at the top of the stack before the output projection. The residual branch weights are scaled at initialization by one over the square root of the number of residual layers, so that the accumulated residual signal stays controlled as depth grows. Learned token and position embeddings are used, and the output projection is tied to the token embedding matrix. The context length is 1024 tokens and the batch size is 512. A family of models is trained at log-spaced sizes: 117M, 345M, 762M, and 1542M parameters. Only the learning rate is tuned per size on held-out corpus text, not on downstream tasks, so the zero-shot claim remains valid.

Evaluation is done by conditioning on natural text. For language-modeling benchmarks the model simply computes the likelihood of the benchmark text. For cloze or multiple-choice tasks it scores candidate continuations. For generation tasks it decodes from a prompt such as a question followed by an answer marker. Because the tokenizer is reversible, no dataset-specific preprocessing is needed. N-gram overlap checks are used to monitor memorization and interpret the metrics honestly.

```python
import re
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bytes_to_unicode():
    """Reversible byte -> unicode mapping; base alphabet stays at 256 symbols."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\u00a1"), ord("\u00ac") + 1))
    bs += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ByteBPETokenizer:
    def __init__(self, merges, special=None):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.bpe_ranks = {tuple(pair.split()): i for i, pair in enumerate(merges)}
        self.cache = {}

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_idx = []
        for token in self.pat.findall(text):
            token_bytes = token.encode("utf-8")
            token = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_idx.extend(self._bpe(token))
        return bpe_idx

    def decode(self, bpe_idx):
        text = "".join(bpe_idx)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_ctx):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_ctx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_ctx)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        nn.init.normal_(self.attn.c_proj.weight, std=0.02)
        nn.init.normal_(self.mlp.c_proj.weight, std=0.02)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_embd, n_head, n_layer):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.wte = nn.Embedding(n_vocab, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_ctx) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # Scale residual projections by 1/sqrt(N) where N is the number of residual layers.
        scale = 1.0 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, std=0.02 * scale)
            nn.init.normal_(block.mlp.c_proj.weight, std=0.02 * scale)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)
        return logits


if __name__ == "__main__":
    # Example: instantiate the 117M-parameter configuration.
    hparams = dict(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12)
    model = GPT2(**hparams)
    print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    # Dummy forward pass.
    x = torch.randint(0, hparams["n_vocab"], (2, 64))
    logits = model(x)
    print("logits shape:", logits.shape)
```
