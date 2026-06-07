# Research notes — Transformer lineage, field state, explainers, code

Main agent read the full primary paper (all .tex section files + appendices
parameter_attention.tex and sqrt_d_trick.tex + visualizations.tex). The notes
below were gathered via WebFetch on arXiv abstracts and explainer pages, then
cross-checked against the primary paper's own background/citations. No nested
general-purpose Agent tool was available in this environment, so the main agent
performed the reference research directly.

## Field state at the paper's time (2017)
- RNN/LSTM/GRU encoder-decoders were the established SOTA for sequence
  transduction and language modeling (Sutskever 2014, Bahdanau 2014, Cho 2014;
  pushed by Wu 2016 GNMT, Luong 2015, Jozefowicz 2016).
- The fundamental pain point: recurrence factors computation along positions —
  h_t = f(h_{t-1}, x_t) — which is inherently sequential, precluding
  parallelization within a training example. Critical at long sequence length
  because memory limits batching. Factorization tricks (Kuchaiev 2017) and
  conditional computation / MoE (Shazeer 2017) sped this up but the sequential
  constraint remained.
- Attention (Bahdanau, Luong) had become integral but was almost always bolted
  onto an RNN. Exception: Parikh 2016 (decomposable attention) used attention
  without recurrence for NLI.
- Parallel-by-convolution line: Extended Neural GPU, ByteNet, ConvS2S replaced
  recurrence with CNNs to compute all positions in parallel — but the number of
  ops to relate two positions grows with distance (linear ConvS2S, log ByteNet),
  making long-range dependencies hard to learn.
- Self-attention (intra-attention) already used for reading comprehension,
  summarization, entailment, sentence embeddings (Cheng 2016, Parikh 2016,
  Paulus 2017, Lin 2017). End-to-end memory networks (Sukhbaatar 2015) used
  recurrent attention. Nobody had built a transduction model relying ENTIRELY
  on self-attention with no RNN/conv.

## Load-bearing ancestors (verified against primary paper)

### Sutskever et al. 2014 — Seq2Seq with LSTMs (1409.3215)
- Two multilayer LSTMs: encoder reads source into a single fixed-length vector
  (last hidden state), decoder LSTM generates target from it. 34.8 BLEU EN-FR.
- Limitation: the fixed-length context vector is an information bottleneck —
  all source info compressed into one vector; degrades on long sentences. Also
  fully sequential. (Reversing source order helped — a hack around long-range
  dependency difficulty.)

### Bahdanau, Cho, Bengio 2014 — Additive attention for NMT (1409.0473)
- Removes the bottleneck: context vector per decoder step is a weighted sum of
  ALL encoder annotations. Exact equations:
  e_ij = v^T tanh(W_s s_{i-1} + W_h h_j)   (additive alignment, a 1-hidden-layer FFN)
  alpha_ij = softmax_j(e_ij)
  c_i = sum_j alpha_ij h_j
- Still attached to an RNN: the annotations h_j and the decoder state s_{i-1}
  are produced by recurrent nets, so sequential computation remains.

### Luong, Pham, Manning 2015 — Effective attention (1508.04025)
- Global vs local attention; three score functions: dot (q·k), general
  (q^T W k), concat (additive). Introduced the multiplicative / dot-product
  score function. Still RNN-based. This is the dot-product attention the paper
  contrasts against (vs additive).

### ByteNet — Kalchbrenner et al. 2017 (1610.10099)
- Fully convolutional, dilated convolutions, decoder stacked on encoder,
  linear-time. Dilation → path length between distant positions grows only
  logarithmically. But path length still grows with distance → long-range deps
  still harder than O(1); reduced effective parallel "reach" per layer.

### ConvS2S — Gehring et al. 2017 (1705.03122)
- Fully convolutional encoder-decoder, gated linear units (GLU), learned
  position embeddings, separate multi-step attention per decoder layer. Fully
  parallel across positions during training. Key limitation: ops to relate two
  positions grow LINEARLY with distance (stack of O(n/k) conv layers). Also the
  source of "learned vs fixed positional encoding" idea the paper compares to.

### He et al. 2016 — Residual learning (1512.03385) & Ba et al. 2016 — LayerNorm (1607.06450)
- Residual connections + layer normalization: the glue that lets deep stacks
  train. Transformer wraps every sublayer as LayerNorm(x + Sublayer(x)).

### Other plumbing the method stands on
- Adam (Kingma 2014) optimizer; Dropout (Srivastava 2014); Label smoothing
  (Szegedy 2015, Inception); BPE (Sennrich 2015) / word-piece (Wu 2016) subword
  vocab; weight tying of embeddings + pre-softmax projection (Press & Wolf 2016).

## Explainers (intuition, not just the algorithm)
- The Annotated Transformer (Harvard NLP, Rush; v2022 Huang et al.) — line-by-line
  PyTorch reimplementation. THIS is the canonical code snapshot in code/.
- The Illustrated Transformer (Jay Alammar): Q/K/V analogy (query=what I'm
  looking for, key=what's on offer, value=the content); scaling by sqrt(d_k) for
  stable gradients; multi-head = multiple representation subspaces; PE added to
  embeddings; encoder gives K,V to decoder cross-attention; masked decoder
  self-attention.

## Canonical code (code/annotated_transformer.py)
Downloaded from harvardnlp/annotated-transformer. Grounds all Phase-2 code:
- attention(q,k,v,mask): scores = q k^T / sqrt(d_k); mask fill -1e9; softmax; @ v
- MultiHeadedAttention: 4 linears (Wq,Wk,Wv combined + Wo), view into h x d_k,
  attention per head, concat, final linear.
- PositionwiseFeedForward: w_2(relu(w_1 x)).
- PositionalEncoding: div_term = exp(arange(0,d,2) * -(log(10000)/d)); sin even,
  cos odd; registered buffer; added to x.
- Embeddings scaled by sqrt(d_model).
- Encoder/Decoder = N=6 stacked layers; SublayerConnection = x + dropout(sublayer(norm(x))).
- subsequent_mask = upper-triangular triu(diagonal=1)==0 for causal masking.
- Noam LR schedule: rate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5).
- make_model wires it all (xavier_uniform init).

## V3 depth-pass: additional design-why research (web)

- ADD vs CONCAT positional encoding: concatenation inflates dim (extra cols on
  every W^Q/W^K/W^V and FFN W_1), no extra expressive power because each PE/embed
  is immediately followed by a learned linear projection — addition + projection
  can already realise any separation a concat+projection could (concat is the
  special case of a block-structured projection on disjoint channels). So add is
  the cheaper, no-loss choice given downstream learned linears.
  (forum.opennmt.net, tensor2tensor#1591, Brenndoerfer)
- LayerNorm not BatchNorm: BN normalises across the batch dimension; in NLP the
  per-position batch statistics fluctuate wildly (variable lengths, padding zeros),
  unreliable. LN normalises per-token across the d features → batch-size invariant,
  identical at train and at autoregressive one-token-at-a-time inference, no
  running-stat train/test gap. (PowerNorm 2003.07845; arxiv 2210.05153)
- W^O after concat: heads output in disjoint d_v subspaces; concat alone never lets
  heads mix; W^O (h*d_v -> d_model) both mixes the heads' perspectives and maps back
  to d_model so the residual add lines up. (Raschka FAQ; d2l 11.5; McCormick)
- FFN 4x wide: the attention sublayer is (softmax-weighted) linear in V plus the
  per-position residual; the only per-position nonlinear capacity is the FFN. Widen
  d_model->d_ff=2048 (ReLU) then project back; FFN holds ~2/3 of layer params and is
  the model's per-position "compute". 4x is the capacity/efficiency knee.
  (mbrenndoerfer FFN; apxml)
- Warmup (Noam): Adam's second-moment v_t is estimated from very few samples early
  -> high variance adaptive step (RAdam 1908.03265); attention softmax can saturate
  at init -> large noisy gradients that POISON Adam's running v_t for thousands of
  steps. Warmup keeps early steps tiny so moment stats stabilise; d_model^-0.5
  scales the peak LR down for wider models. (NeurIPS 2024 2406.09405; Tfixup)
- sqrt(d_model) embedding scaling: PE components are O(1); a tied/Glorot embedding
  row has per-component scale ~1/sqrt(d). Multiplying by sqrt(d_model) lifts the
  embedding to O(1) per component so it is comparable to PE in the sum (neither
  drowns the other). (dekut-dsail tutorial; Brenndoerfer weight tying)
- Residual gradient highway: d/dx (x + F(x)) = I + F'(x); the +I path guarantees
  the upstream gradient reaches every sublayer undiminished, so the N=6 stack
  trains. (ResNet He 2016; mbrenndoerfer residual)
