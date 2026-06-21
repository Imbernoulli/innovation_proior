## Research question

Transformer self-attention lets every token attend to every other token, which is why it captures context so well — but the time and memory cost of that all-pairs interaction grows with the square of the sequence length. On current accelerators this caps the usable input at roughly 512 tokens for the kind of pretrained encoders that dominate language understanding. Many natural-language problems, however, are intrinsically long: question answering where the answer is scattered across a multi-paragraph context, coreference resolution that spans a whole document, classification of long articles, and character-level language modeling over tens of thousands of characters. How can self-attention be made to operate effectively on long sequences?

## Background

**Self-attention and its cost.** Given an input sequence of n tokens with model dimension d, a Transformer layer projects the inputs into queries, keys, and values Q, K, V ∈ ℝ^{n×d} (per head, d_k = d/h for h heads). It computes

  Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V .

The product Q Kᵀ is an n×n matrix of scores: every token scores against every token. Materializing and softmax-normalizing it costs O(n²·d_k) time and O(n²) memory. The √d_k denominator keeps the dot products from growing with dimension and saturating the softmax.

**Attention head behavior in pretrained encoders.** Probing studies of pretrained bidirectional encoders find that a large fraction of attention heads concentrate on a narrow local neighborhood — many heads attend predominantly to the immediately previous or next token, and local context is repeatedly found to be the most informative for building token representations (Clark et al. 2019; Kovaleva et al. 2019).

**The CNN analogy.** Convolutional networks confront the same tension — a single small kernel sees only a local patch — and resolve it with depth: stacking layers of small kernels grows the *receptive field*, so deep layers form features built from a large span of the input even though each individual kernel is local. Dilated (à-trous) convolutions push this further by inserting gaps into the kernel, enlarging the receptive field exponentially with depth at no extra cost — the trick behind WaveNet's long temporal context (van den Oord et al. 2016). Work relating convolution and attention (Wu et al. 2019) showed that attention restricted to a local span behaves much like a (dynamic) convolution and can match full attention on several tasks.

**Pretrain-finetune as the dominant paradigm.** The state of the art in language understanding is built on masked-language-model (MLM) pretraining of a bidirectional encoder followed by task finetuning (Devlin et al. 2018, BERT; Liu et al. 2019, RoBERTa). These models reserve a special classification token ([CLS]) whose final representation aggregates the sequence for sentence-level prediction, and for QA they concatenate question and context so self-attention can compare them. They use learned *absolute* position embeddings with a maximum position of 512.

**Implementation context.** Full attention's memory grows quadratically and exhausts a modern GPU well before sequences reach the tens of thousands. The deep-learning frameworks (PyTorch, TensorFlow) expose dense matrix multiplication as their attention primitive. Pretrained encoders have learned a strong *local* position bias (heads keying on previous/next token), keyed to their learned absolute position embeddings.

## Baselines

**Full self-attention Transformer (Vaswani et al. 2017).** The reference point. Dense softmax(QKᵀ/√d_k)V over all pairs; maximally expressive, O(n²) time and memory.

**Left-to-right recurrence: Transformer-XL (Dai et al. 2019), Adaptive Span (Sukhbaatar et al. 2019), Compressive Transformer (Rae et al. 2019).** These process a long document in segments moving left to right, caching (and in the compressive case, summarizing) past activations so a segment can attend back into history; Adaptive Span additionally *learns* how far back each head should look. They achieve strong character-level LM results.

**Sparse attention patterns: Sparse Transformer (Child et al. 2019), Reformer (Kitaev et al. 2020), Routing Transformer (Roy et al. 2020).** These avoid the full n² matrix by attending only to a structured subset of positions. Sparse Transformer factorizes attention into strided/local block patterns — a form of dilated sliding window over 8×8 blocks — using specialized block-sparse GPU kernels; Reformer uses locality-sensitive hashing to bucket similar queries and keys; Routing uses learned clustering. They scale sub-quadratically and perform well on autoregressive generation and LM.

**Sparse attention reaching beyond LM: BP-Transformer (Ye et al. 2019), Blockwise attention (Qiu et al. 2019).** BP-Transformer uses a binary-partition sparse pattern and was evaluated on machine translation. Blockwise attention pretrained with block-local attention and was evaluated on QA tasks.

**Task-specific long-document workarounds.** Independent of attention sparsity, a family of engineering solutions sidesteps the 512 limit: truncate the document (common for classification); chunk into (possibly overlapping) 512-token windows, encode each separately, and combine the activations with a task-specific model; or, for multi-hop and open-domain QA, a two-stage retrieve-then-read pipeline.

## Evaluation settings

The natural yardsticks already exist. For autoregressive character-level language modeling: **text8** and **enwik8** (100M characters of Wikipedia, split 90M/5M/5M train/dev/test), reported in bits-per-character, with the standard protocol of evaluating on overlapping sequences and scoring only the final block of each. For long-document understanding under pretrain-finetune: multi-hop and document QA (WikiHop, TriviaQA, HotpotQA — contexts averaging thousands and up to ~17K wordpieces), coreference resolution (OntoNotes), and long-text classification (IMDB, Hyperpartisan news), with task-appropriate accuracy/F1 metrics; and for long-input sequence-to-sequence, abstractive summarization of long scientific papers (arXiv dataset, ~14.5K-token documents) scored with ROUGE. MLM pretraining quality is tracked by bits-per-character on a held-out corpus of long documents. The MLM base model to continue from is the released RoBERTa checkpoint.

## Code framework

The pieces that already exist: a pretrained bidirectional encoder stack (RoBERTa-style: token + learned absolute position embeddings capped at 512, a stack of Transformer layers each wrapping a self-attention sublayer and a feedforward sublayer), the standard scaled-dot-product attention primitive, an MLM head, and standard optimizers/training loop. The efficient-attention slot sits where full self-attention currently lives; the surrounding embedding stack, feed-forward blocks, output heads, optimizer, and training loop stay in place.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_roberta import RobertaModel


def full_attention(q, k, v):  # q,k,v: (bsz, heads, n, d_k)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class LongSequenceSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key   = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # TODO: the efficient self-attention that scales linearly with sequence length goes here.
        pass


class LongSequenceEncoder(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = LongSequenceSelfAttention(config, layer_id=i)

```
