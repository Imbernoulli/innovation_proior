# Synthesis V4 — notes the three V4 deliverables are composed from

Purpose of this file: persist the understanding so Phase 2 is transcription-with-voice,
not figuring-it-out-while-writing. Two parts: (A) the design-decision → why table, with
rejected alternatives and the failure mode each hits; (B) the load-bearing-ancestor
write-ups. V4-specific corrections to V3 are flagged inline.

V3 was deep and correct on the derivations; V4 keeps those. The material V4 must FIX
under the new SKILL:
- context.md Code-framework must be a MINIMAL PRE-METHOD scaffold. V3 wrote "The
  reference-implementation substrate is…" and laid out EncoderDecoder /
  SublayerConnection / subsequent_mask / residual+LayerNorm multi-head sublayers —
  that presupposes the method. At context time we do NOT yet know we'll invent
  attention, multi-head, or positional encodings. Rewrite as a bare seq2seq harness:
  nn.Embedding → `class SequenceModel: # TODO the architecture we'll design` →
  output projection + log-softmax → cross-entropy loss → Adam → batching/padding-mask
  data path. NO "reference implementation"/"official repo" wording; NO method names.
- reasoning.md: every method piece must EMERGE as the conclusion of the motivating
  reasoning (insight-before-method, local discovery order). Scan for any state-then-
  justify paragraph and flip it.

---

## PART A — Design-decision → why table (with rejected alternatives + failure modes)

Format: DECISION | WHY (in-frame, derivation-time) | REJECTED ALTERNATIVE → its failure mode

1. DECISION: Kill recurrence; build the layer so all positions compute in parallel.
   WHY: recurrence h_t=f(h_{t-1},x_t) is O(n) sequential along the sequence; a GPU
   wants one big matmul, not n dependent small ops. Batching can't recover the lost
   intra-example parallelism at long n because memory caps how many long sequences
   fit per batch.
   REJECTED: keep RNN + faster constants (Kuchaiev-Ginsburg factorization 2017,
   Shazeer MoE 2017) → shaves the constant in front of O(n) but the sequential DEPTH
   is untouched; the enemy is the depth, not the constant.

2. DECISION: Use attention as the routing primitive (any position → any position in
   one hop).
   WHY: Sutskever's single fixed context vector is a bottleneck that crushes long
   sentences; Bahdanau's per-step weighted sum over all encoder annotations gives
   O(1) path length — distance stops mattering for routing. Content-based, soft,
   differentiable lookup.
   REJECTED: fixed-context seq2seq → information bottleneck + path between aligned
   words grows with distance (they reversed source word order as a hack, which is the
   tell that path length was already biting).

3. DECISION: Score with dot product, not an additive tanh-MLP.
   WHY: a batch of dot products IS a single dense GEMM — exactly what the hardware
   runs at full throughput. We'll use attention hundreds of times in a deep stack, so
   the one-GEMM version wins decisively.
   REJECTED: additive scoring e=v^T tanh(W_s s + W_h h) (Bahdanau) → forces an
   elementwise tanh over an (n_q × n_k × d) tensor before reducing; materializes a
   giant intermediate and runs a nonlinearity over it. Same asymptotic cost, far
   worse on real hardware.

4. DECISION: Make the WHOLE encoder and decoder out of self-attention over the
   sequence itself — no recurrence, no convolution.
   WHY: only self-attention gets O(1) sequential AND O(1) path length at once.
   Complexity table per layer (n length, d width):
     recurrent:     O(n·d²) compute, O(n) sequential, O(n) path
     convolutional: O(k·n·d²) compute, O(1) sequential, O(log_k n) path (dilated) /
                    O(n/k) path (plain)
     self-attn:     O(n²·d) compute, O(1) sequential, O(1) path
   self-attn is cheaper than recurrence exactly when n<d, which subword vocab
   guarantees (n a few dozen–low hundreds, d=512).
   Second, table-independent reason: in RNN/CNN the cross-position info is squeezed
   into a fixed-width summary BEFORE the query filters by content; self-attn weights
   first, aggregates after — nothing relevant is crushed by an unfiltered summary.
   REJECTED-A: attention bolted onto RNN (every NMT system) → keeps the O(n) chain,
   merely decorates it.
   REJECTED-B: convolution (ByteNet, ConvS2S) → trades sequential tax for a path-
   length tax that grows with distance (Hochreiter 2001: longer path = harder to
   learn). Moves the cost, doesn't remove it.
   FALLBACK (knob for later, not now): if n ≫ d (long documents), restrict each query
   to a local window r → O(r·n·d) compute, O(n/r) path.

5. DECISION: Divide scores by √d_k (scaled dot-product attention).
   WHY: variance argument. q·k = Σ_{i=1}^{d_k} q_i k_i with q_i,k_i indep, mean 0,
   var 1. Mean 0 (independence). Var = E[(q·k)²] (mean is 0); cross terms i≠j vanish
   (E[q_i]E[k_i]E[q_j]E[k_j]=0); diagonal Σ E[q_i²]E[k_i²]=d_k. So std grows like
   √d_k. With d_k=64 logits sit at ±8–10. The FAILURE is in the gradient, not the
   forward pass: softmax of far-apart logits → near one-hot; its Jacobian
   ∂soft_i/∂logit_j = soft_i(δ_ij − soft_j) → 0 everywhere when near one-hot (diag
   p(1−p)→0, off-diag −p_ip_j has a near-zero factor); attention weights stop getting
   gradient, model frozen in near-argmax. Dividing by √d_k makes Var=d_k/d_k=1, logits
   O(1), softmax responsive. Why √d_k and not 1/d_k: we want STD normalized to 1, and
   std(q·k)=√d_k.
   REJECTED: unscaled dot product → exactly the saturation/Jacobian-collapse above;
   observed (Britz 2017) to degrade vs additive as d_k grows.
   (Primary appendix sqrt_d_trick.tex confirms the mean-0/var-d_k argument verbatim.)

6. DECISION: Multi-head — h parallel attentions in d_k=d/h-dim subspaces, concat, W^O.
   WHY: one softmax per query = one averaged summary; an average blurs. If position i
   must track subject (agreement) + verb (tense) + a modifier simultaneously, one
   weighting smears them. Run h heads, each free to attend to a different place for a
   different reason; each head's own softmax means blurring happens WITHIN a head, not
   ACROSS heads — distinct relations stay distinct. This buys back the "reduced
   effective resolution due to averaging" cost that single-head self-attn introduced.
   REJECTED: single full-width head → can attend to one kind of relation at a time;
   the average resolution loss.

7. DECISION: tie d_k = d_v = d_model/h (so h heads cost ≈ one full-width head).
   WHY: per head score matmul O(n²·d_k), h of them → h·n²·(d/h)=n²·d, same as one
   full-width attention; projections together are d×d, same as one d×d. Resolution is
   essentially free. d=512, h=8 → d_k=d_v=64.
   REJECTED: each head keeps full d → h heads cost h× as much; defeats the point
   (spend one budget, slice into specialists).

8. DECISION: keep the output projection W^O (not just concat the heads).
   WHY: two reasons. (i) residual stream is d_model wide; need a map h·d_v→d_model.
   (ii) the REAL reason: concat alone leaves head 3's output in its own coordinates,
   untouched by head 5's — heads never interact. W^O is the learned linear that mixes
   per-head perspectives ("syntactic head says X, semantic head says Y") into one
   vector. Integration step, not a formality.
   REJECTED: concat-only → siloed heads, no cross-head composition.

9. DECISION: one MultiHead block, three wirings of Q/K/V.
   - encoder self-attn: Q,K,V = prev encoder layer.
   - cross-attn: Q = decoder, K,V = encoder output (= Bahdanau's setup in the same
     mechanism).
   - decoder self-attn: Q,K,V = prev decoder layer, but MASKED.

10. DECISION: causal mask via setting future logits to −∞ before softmax.
    WHY: autoregression — position i may depend only on <i (at inference that's all
    we'll have generated). We compute the full n×n score matrix in one shot including
    illegal future entries. Don't SKIP them (skipping breaks the single-matmul
    parallelism that is the whole reason we're here); NEUTRALIZE them: set future
    entries to −∞ → exp(−∞)=0 → zero weight, every row still computed in parallel. In
    practice add −1e9. Plus offset the target by one position.
    REJECTED: serialize the decoder / loop over positions → kills training
    parallelism, the entire point.

11. DECISION: inject position via additive positional codes at the bottom.
    WHY: pure attention softmax(QK^T/√d_k)V is permutation-equivariant — all dot
    products and weighted sums over the SET of positions; nothing knows the order.
    "dog bit man" = "man bit dog". RNN got order from the time axis, conv from kernel
    offsets; we threw both away → must inject position or we've built a bag-of-words.

12. DECISION: ADD position to embedding, don't concatenate.
    WHY: the first thing touching the combined vector is a learned linear. Over concat
    [e;p]: W[e;p]=W_e e + W_p p (two independent blocks). Over sum e+p:
    W(e+p)=We+Wp (same two terms with W_e=W_p=W). So concat = addition with the extra
    freedom of different projections for content/position. Do we need it? Downstream
    linears can already allocate some directions to position, some to content; the sum
    is easily disentangled when PE occupies a recognizable subspace.
    REJECTED: concatenate p-dim code → every downstream matrix (W^Q/W^K/W^V in every
    head of every layer, FFN W_1) grows to (d+p) input columns — a width tax paid
    everywhere forever, for a marginal expressiveness gain. Bad trade.

13. DECISION: sinusoidal positional encoding.
    PE(pos,2i)=sin(pos/10000^{2i/d}), PE(pos,2i+1)=cos(pos/10000^{2i/d}).
    WHY: what attention does with positions is dot products / differences; want a head
    to learn "look k back" and apply the SAME relative rule everywhere. So want a code
    where shift-by-k is a position-INDEPENDENT linear map. Derive: per frequency ω,
      sin(ω(pos+k)) = sin(ωpos)cos(ωk)+cos(ωpos)sin(ωk)
      cos(ω(pos+k)) = cos(ωpos)cos(ωk)−sin(ωpos)sin(ωk)
    = rotation [[cos ωk, sin ωk],[−sin ωk, cos ωk]] · [sin ωpos; cos ωpos]. Rotation
    depends only on k, not pos. So PE_{pos+k} is a fixed linear fn of PE_{pos} at every
    absolute position. Geometric freq spread (wavelengths 2π … 10000·2π): high freq
    resolves fine distinctions, low freq encodes coarse long-range position — a
    multi-scale ruler. sin/cos defined for any real pos → code exists past training
    length → extrapolation.
    REJECTED: learned per-position embedding table (ConvS2S) → no entry beyond longest
    training sequence; cannot extrapolate at all. (In-distribution it does about as
    well, so the deciding factor is extrapolation.)

14. DECISION: scale embeddings by √d_model.
    WHY: sinusoids are O(1) (sin/cos ∈ [−1,1]). A Glorot/tied embedding row component
    is ~O(1/√d). Raw embedding would be a whisper next to O(1) PE → position drowns
    content. Multiply embedding by √d_model → each component O(1), amplitude-matched to
    PE, both signals survive the sum. Amplitude-matching, not arbitrary.
    REJECTED: no scaling → PE swamps content at the bottom of the stack.

15. DECISION: tie input embedding, output (target) embedding, pre-softmax projection
    to one matrix (Press & Wolf 2016).
    WHY: all three are the same token↔d-vector correspondence used in two directions
    (lookup vs scoring a state against each token's vector). Sharing cuts a big block
    (vocab×d) and regularizes by tying the two views. The √d_model scaling is applied
    at the embedding forward; the output projection uses the unscaled geometry for
    sensible logits.

16. DECISION: position-wise FFN between attention sublayers, 512→2048→512 (4× wide),
    ReLU.
    WHY: attention mixes ACROSS positions but within a position is weak — the value is
    a softmax-weighted LINEAR combination; the only nonlinearity in the attention
    sublayer is the softmax on the weights, no per-position nonlinear processing of
    content. Need somewhere to do genuine per-token nonlinear computation. Width = per-
    position capacity: a ReLU MLP's expressive power scales with hidden width (more
    units → more linear regions → more features carved before projecting back). Expand
    to 2048 for headroom, ReLU, project back to 512 for the residual stream. 4× is the
    capacity-vs-compute knee; FFN holds most of the layer's params. Sanity check
    (primary appendix parameter_attention.tex): ReLU(xW_1)W_2 has the shape of
    compat(x,K)V with rows of W_1 as keys, rows of W_2 as values, ReLU as compatibility
    → FFN is attention over fixed LEARNED parameters; it belongs in the same family.
    REJECTED: a single d→d linear → adds nothing the value-projection can't already do;
    no nonlinearity.

17. DECISION: wrap every sublayer as LayerNorm(x + Sublayer(x)) — residual.
    WHY: want N deep; deep stacks of anything don't train naively (vanish/explode).
    d/dx[x+F(x)] = I + F'(x); the I term means upstream gradient reaches x undiminished
    no matter how small F' is — a magnitude-1 path back through every sublayer. Also
    reframes each sublayer as learning a residual correction to identity (easier).
    REJECTED: plain Sublayer(x) stacked → gradient vanishes/explodes through depth; the
    N=6 stack won't train.

18. DECISION: LayerNorm, not BatchNorm.
    WHY: BN normalizes each feature across the BATCH using batch mean/var. Sequences
    are variable-length and padded → the set of real tokens at a position varies across
    the batch and padding injects spurious zeros → per-position batch stats fluctuate
    wildly. BN also keeps running stats for inference, but the decoder runs
    autoregressively one token at a time — no meaningful batch to normalize → built-in
    train/test mismatch. LN normalizes each token across its own d features: batch-size
    invariant (works at batch 1), identical at train and one-token decode, immune to
    padding.
    REJECTED: BatchNorm → unreliable stats on padded variable-length seqs + train/test
    mismatch at autoregressive decode.

19. DECISION: N=6 layers each in encoder and decoder; all sublayers + embeddings at
    width d=512 (so residual sums line up). Encoder layer = {self-attn, FFN}; decoder
    layer = {masked self-attn, cross-attn, FFN}.

20. DECISION: warmup-then-inverse-sqrt LR schedule (Noam).
    lrate = d_model^{−0.5} · min(step^{−0.5}, step·warmup^{−1.5}), warmup=4000.
    WHY: two compounding init problems. (i) attention: random projections → large dot-
    product logits → softmax saturates early → large noisy first gradients (same
    Jacobian collapse). (ii) Adam: scales updates by 1/√v_t; early v_t is from a handful
    of samples → high-variance adaptive step. They interact viciously: a big noisy
    early gradient gets RECORDED into the slow running average v_t and poisons the step
    size for thousands of steps; one bad early step can derail the run. Fix: keep LR
    tiny while stats settle, then raise. Branches: step<warmup → min picks
    step·warmup^{−1.5} (LINEAR ramp, LR climbs from ~0); step>warmup → min picks
    step^{−0.5} (inverse-sqrt decay); cross at step=warmup (peak). d_model^{−0.5}
    prefactor: wider model → larger activations/gradients → smaller peak LR; automatic
    downscaling so schedule transfers across sizes.
    REJECTED: constant or plain-decay LR from step 0 → early big step poisons Adam's
    v_t; run derails.
    Adam betas (0.9, 0.98), eps 1e-9 — slightly higher β_2 smooths the second-moment
    average more, same impulse.

21. DECISION: dropout 0.1 on each sublayer output (before residual add) and on the
    summed embedding+PE input.
    WHY: those are the densest, most co-adaptation-prone points. Dropping the SUBLAYER
    output (not the residual path) keeps the identity gradient highway intact.

22. DECISION: label smoothing ε_ls=0.1 (soft targets via KL).
    WHY: a one-hot target pushes the gold logit to +∞ and the rest to −∞ —
    unattainable and overconfident → brittle, miscalibrated. Soften: 1−ε on gold, ε
    spread over the rest. Deliberately RAISES perplexity (less peaked) yet improves
    BLEU, because translation has many acceptable outputs and overconfidence on one
    hurts. Worth the perplexity cost.
    REJECTED: plain one-hot cross-entropy → overconfidence, miscalibration, worse BLEU.

23. DECISION: Glorot/xavier_uniform init for matrices (dim>1).
    WHY: standard variance-preserving init so the √d_k variance argument's "components
    ~ mean 0 var 1" premise approximately holds at start.

---

## PART B — Load-bearing ancestor write-ups

### Seq2Seq with LSTMs (Sutskever et al. 2014)
Two multilayer LSTMs: encoder reads source into a single fixed-length vector (its final
hidden state); decoder LSTM generates the target from that vector. Established that a
pure neural net does competitive translation. Gaps: (1) the fixed-length context vector
is an information bottleneck — the whole source squeezed into one vector; degrades on
long sentences (they reversed source word order as a hack to shorten the path between
aligned words — the tell that path length was the real problem). (2) fully sequential in
both encoder and decoder.

### Additive attention for NMT (Bahdanau, Cho, Bengio 2014)
Removes the bottleneck: at each decoder step a fresh context = weighted sum over ALL
encoder annotations. e_ij = v^T tanh(W_s s_{i-1} + W_h h_j) (a 1-hidden-layer FFN),
α_ij = softmax_j(e_ij), c_i = Σ_j α_ij h_j. The move that made long-range routing cheap;
every output position reaches every input position in one hop. Content-based addressing:
the query decides where to look by what it is, differentiable soft lookup. Gap: h_j and
s_{i-1} are still produced by RNNs → sequential bottleneck remains; additive scoring is a
small MLP per query-key pair, not a fused matmul.

### Effective / multiplicative attention (Luong, Pham, Manning 2015)
Global vs local; introduced the dot-product (multiplicative) score q·k (and q^T W k) as a
cheaper alternative to additive. Same theoretical complexity, but maps to a single dense
matmul → much faster on hardware. Gap: still RNN-based; unscaled dot products observed
(Britz 2017) to degrade vs additive as key dim grows.

### ByteNet (Kalchbrenner et al. 2017)
Abandons recurrence for stacked DILATED convolutions, decoder stacked over encoder, all
positions in parallel. Dilation → layers to connect two positions grows only
logarithmically in distance. Gap: path length still grows with distance (O(log_k n)) →
very long-range deps still harder than one hop.

### ConvS2S (Gehring et al. 2017)
Fully convolutional encoder-decoder, gated linear units, LEARNED positional embeddings,
separate attention per decoder layer. Fully parallel across positions at training — the
strongest kill-recurrence baseline. Gap: kernel width k → connecting positions distance D
apart needs O(D/k) conv layers → ops to relate two positions grows LINEARLY with distance.
Also the source of the learned-vs-fixed positional-encoding choice.

### Supporting machinery
Residual connections (He et al. 2016) + layer normalization (Ba et al. 2016): glue for
deep stacks. Adam (Kingma & Ba 2014). Dropout (Srivastava et al. 2014). Label smoothing
(Szegedy et al. 2015). BPE (Sennrich et al. 2015) / word-piece (Wu et al. 2016) subword
vocab. Weight tying of embeddings + pre-softmax projection (Press & Wolf 2016).
Path-length-matters principle: Hochreiter et al. 2001 (longer forward/backward path →
harder to learn long-range deps). Self-attention prior art: Cheng 2016, Parikh 2016
(attention without recurrence, but NLI not autoregressive transduction), Paulus 2017,
Lin 2017; end-to-end memory networks Sukhbaatar 2015.

---

## PART C — The pre-method scaffold (context.md Code-framework) ↔ final code map

Scaffold is a BARE seq2seq harness. Derived by hollowing the final code's method-specific
bodies into # TODO. Pieces:
- token embedding: nn.Embedding(vocab, d_model)  ← KNOWN before method
- class SequenceModel: # TODO the architecture we'll design  ← the ONE empty slot
- output projection + log-softmax: nn.Linear(d_model, vocab) + log_softmax  ← KNOWN
- loss: cross-entropy (NLLLoss on log-probs) over non-pad tokens  ← KNOWN
- optimizer: Adam  ← KNOWN
- data path: ids → pad to common length → build src padding mask + tgt padding mask;
  batching loop  ← KNOWN (NO causal/subsequent mask here — that's a method discovery)
NO method names anywhere (no MultiHeadAttention/PositionalEncoding/scaled-dot-product/
encoder-decoder-with-sublayers/SublayerConnection/subsequent_mask). NO "reference
implementation"/"official repo".

Final code fills the SequenceModel TODO with the whole attention stack; the cross-entropy
becomes label-smoothed KL; the Adam gets the Noam LambdaLR; the tgt mask gains the causal
component. These are exactly the slots the scaffold left open.
