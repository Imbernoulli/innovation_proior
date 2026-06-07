# Synthesis — Sequence to Sequence Learning with Neural Networks (Seq2Seq)

## The pain point (research question)
Around 2013–2014, deep neural nets (DNNs) dominated speech and vision, but they only consume/produce **fixed-dimensional vectors**. Many of the most important problems are sequence→sequence with **variable, a-priori-unknown lengths and non-monotonic alignment**: machine translation, speech recognition, QA. There was no general, domain-independent, end-to-end neural method that maps an arbitrary input sequence to an arbitrary output sequence of a different length.

State of MT at the time: phrase-based **statistical machine translation (SMT)** ruled (Koehn et al.; Moses; Durrani et al. WMT'14 best = 37.0 BLEU; the LIUM/Schwenk baseline = 33.30 BLEU on WMT'14 En→Fr). SMT = log-linear combination of features (phrase translation probs, language model, reordering, word penalty) with beam-search decoding over phrase tables + an n-gram LM. Neural nets entered MT only **at the margins**: as features inside SMT. Neural LMs (Bengio 2003 NNLM; Mikolov 2010 RNNLM) were used mostly to **rescore n-best lists** of a strong SMT baseline (Mikolov 2012), reliably improving quality. Auli 2013 / Devlin 2014 conditioned NNLMs on source info but still lived inside the SMT decoder.

## Load-bearing ancestors (the lineage), with the gap each leaves
1. **RNN (Werbos 1990 BPTT; Rumelhart 1986)** — `h_t = sigm(W_hx x_t + W_hh h_{t-1})`, `y_t = W_yh h_t`. Natural for sequences but only maps seq→seq when input/output are **aligned and equal-length** (one output per input step). Gap: can't handle different lengths / non-monotonic alignment.
2. **Vanishing/exploding gradients (Hochreiter 1991; Bengio 1994; Hochreiter & Schmidhuber gradient-flow 2001; Pascanu 2012)** — plain RNNs can't learn long-range dependencies; gradient norms shrink/blow up over many timesteps. Gap: the very long time-lag in seq2seq (whole input must be remembered until output starts) is exactly the regime RNNs fail in.
3. **LSTM (Hochreiter & Schmidhuber 1997)** — gated cell with constant error carousel. Gates (input/forget/output) + cell state let gradients flow over hundreds of steps. "LSTM can solve hard long time lag problems" (Hochreiter & Schmidhuber 1997). This is the tool that *might* survive the long lag. Graves 2013 ("Generating sequences with RNNs") gives the concrete LSTM-LM formulation used here and the gradient-norm clipping trick.
4. **RNN/Neural language models (Bengio 2003 NNLM; Mikolov 2010 RNNLM; Sundermeyer 2012 LSTM-LM)** — model `p(w_t | w_1..w_{t-1})` autoregressively with a softmax over vocab. This is the decoder skeleton; it gives a proper distribution over variable-length sequences if you add an end-of-sequence token. Gap: a *language* model is unconditional — it generates fluent text but isn't tied to a source sentence.
5. **Kalchbrenner & Blunsom 2013 (Recurrent Continuous Translation Models)** — *first* to map an entire source sentence to a single vector and back. But they encode with a **CNN/bag-of-words-ish** map that **loses word order**, and decode with an RNN. Gap: order-insensitive encoding throws away information; not a clean RNN-in/RNN-out.
6. **Cho et al. 2014 (RNN Encoder–Decoder, GRU)** — developed in parallel; same encode-to-vector-then-decode idea, but used mainly to **score phrase pairs inside SMT**, not to translate directly, and reported poor performance on long sentences.
7. **Graves 2013 attention / Bahdanau 2014** — differentiable attention so the decoder can look back at all encoder states; Bahdanau applies it to MT to fix long-sentence degradation. (Contemporary/parallel; the fixed-vector approach here deliberately does NOT use attention.)
8. **CTC (Graves 2006)** — maps seq→seq but assumes **monotonic alignment**. Gap: translation reordering is non-monotonic.
9. **Pouget-Abadie 2014 ("curse of sentence length")** — segments source to cope with the fixed-vector memory bottleneck. Establishes the *observed phenomenon*: fixed-vector NMT degrades on long sentences.

## The precise object being modeled
Estimate `p(y_1..y_{T'} | x_1..x_T)` with **T' ≠ T allowed**. Factorize autoregressively:
`p(y_1..y_{T'} | x) = ∏_{t=1}^{T'} p(y_t | v, y_1..y_{t-1})`,
where `v` = fixed-dim representation of the whole source = **last hidden state of an LSTM that read x**. Each factor = softmax over the target vocabulary. An explicit **<EOS>** token terminates generation and lets the model define a distribution over sequences of *all* lengths (this is the trick that makes variable output length well-defined).

Training objective: maximize `(1/|S|) Σ_{(T,S)∈S} log p(T|S)` over the training set S of (source, target) pairs. Decoding: `T̂ = argmax_T p(T|S)`, approximated by left-to-right **beam search**.

## Design decisions → why (and rejected alternatives)
- **Two separate LSTMs (encoder ≠ decoder), not one shared.** Why: more parameters at ~zero extra compute; makes it natural to train on multiple language pairs at once. Rejected: single shared RNN (Cho-style) — fewer params, conflates the read and write dynamics.
- **Deep LSTM, 4 layers, not shallow.** Why: each extra layer cut perplexity ~10%; the hidden state must store the *entire* source, so a bigger/deeper state helps memory. (Tutorial uses 2 layers for tractability.) Rejected: 1-layer LSTM — insufficient capacity.
- **LSTM, not vanilla RNN, for the long time-lag.** Why: the gap between reading the last source word and producing the first target word spans the whole sentence; vanilla RNN gradients vanish over that lag; LSTM's gated cell preserves them. (They report being unable to train a plain RNN on the non-reversed task.)
- **Reverse the SOURCE words (not the target).** THE key trick. Why: concatenating source+target, each source word `a_i` is ~T steps from its translation; the *minimal time lag* (distance from the first source word to the first target word it informs) is huge, so SGD struggles to "establish communication." Reversing source maps `c,b,a → α,β,γ`: now `a` sits next to `α`, `b` near `β`, etc. **Average** distance between corresponding words is unchanged, but the **minimal** lag for the early words collapses to small — backprop gets a strong early signal, bootstraps the rest. Empirically: test perplexity 5.8→4.7, BLEU 25.9→30.6; *also* improved long-sentence performance (better memory utilization), which was surprising — they expected only early-word gains.
- **<EOS> token.** Why: defines a proper distribution over variable-length outputs; tells decoder when to stop.
- **Softmax over full target vocab (80k), naive.** Why: simplest proper output distribution. (Compute-heavy; parallelized over GPUs.) Vocab capped (src 160k / tgt 80k) because softmax/embeddings scale with vocab; OOV → UNK.
- **Encoder's final (h, c) initialize decoder.** Why: that IS the conditioning vector `v` — the only channel from source to target in the no-attention design.
- **Init params uniform[-0.08, 0.08].** Small symmetric init keeps early activations/gradients in a stable range.
- **Plain SGD, lr 0.7, no momentum; halve lr every half-epoch after epoch 5; 7.5 epochs total.** Simple, worked; LSTMs found easy to train here.
- **Gradient-norm clipping at 5 (on grad/128).** Why: LSTMs don't vanish but CAN explode; hard-clip `g ← 5g/‖g‖₂` when `‖g‖₂ > 5` keeps steps bounded (Graves 2013, Pascanu 2012). Tutorial: `clip_grad_norm_(..., clip=1.0)`.
- **Sort minibatches by length (128 same-ish-length sentences).** Why: padding waste — random batches mix length-20 and length-100; bucketing by length → ~2× speedup. No effect on the model, pure efficiency.
- **Beam search decode, small B (1–2 already most of the benefit).** Why: argmax over all sequences is intractable; greedy (B=1) is myopic; beam keeps B best prefixes, extends each by every vocab word, prunes to B by log-prob; pop hypotheses to "complete" set on <EOS>. Approximate but simple; B=2 ≈ full benefit.
- **Ensemble of 5 (random init + minibatch order).** Averages log-probs; standard variance reduction. (Eval-side; mention lightly.)

## Canonical implementation
bentrevett/pytorch-seq2seq tutorial #1 ("Sequence to Sequence Learning with Neural Networks"), De→En Multi30k. Structure:
- `Encoder`: Embedding → multi-layer `nn.LSTM` → return (hidden, cell).
- `Decoder`: Embedding → multi-layer `nn.LSTM` (init with encoder's hidden,cell) → Linear to vocab → return (prediction, hidden, cell). One step at a time.
- `Seq2Seq`: encode src, loop decoder with teacher forcing, collect logits.
- Init `uniform_(-0.08, 0.08)` (matches paper exactly); `clip_grad_norm_`; CrossEntropyLoss(ignore pad); greedy `translate_sentence`.
Hyperparams (tutorial, scaled down): emb 256, hidden 512, n_layers 2, dropout 0.5. Paper: emb 1000, hidden 1000, 4 layers, src vocab 160k / tgt 80k.
Faithful additions for the final code: source reversal at data prep; beam-search decoder (paper's actual decoder, tutorial only ships greedy).

## In-frame discipline
Method name "seq2seq"/"encoder–decoder LSTM" may be named in answer.md as the thing built. Never cite the target paper. Prior-art (Hochreiter 1997, Cho 2014, Kalchbrenner 2013, Graves 2013, Mikolov 2010, Bahdanau 2014, Pascanu 2012, CTC 2006) cited freely. No proposed-method eval results in context.md (the 25.9→30.6 reversal numbers are a *diagnostic ablation about the encoding choice* — allowed as motivating finding in reasoning, but keep eval tables out).
