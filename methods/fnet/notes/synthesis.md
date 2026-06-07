# FNet synthesis (grounded)

## Verified
- arXiv 2105.03824, "FNet: Mixing Tokens with Fourier Transforms", Lee-Thorp, Ainslie, Eckstein, Ontanon (Google Research).
- Canonical impl: google-research/google-research/tree/master/f_net (JAX/Flax). Core code reproduced in paper appendix A.9.

## Pain point / research question
- Transformer self-attention is the expressive core but O(N^2) time & memory in sequence length N; memory-bound. Efficient transformers (Longformer, BigBird, Performer, Linformer, Linear Transformer) reduce asymptotics but hide large constants / approximate attention.
- Question: is attention's token-dependent, token-token (query-key) mixing actually *necessary*? Can a SIMPLER, possibly parameter-free, token-mixing op replace it in an encoder with limited accuracy loss and big speedups?
- Evidence it might: Synthesizer (Tay 2020) learned token-dependent weights without dot product; You 2020 replaced attention with fixed Gaussian; Raganato 2020 fixed positional patterns; MLP-Mixer (Tolstikhin 2021) replaced attention with MLPs (vision). All: attention's flexibility not always crucial.

## The derivation chain (discovery order)
1. Frame self-attention abstractly: it's a *token-mixing* sublayer — it takes (seq_len N, hidden d) and produces a new representation where each token is a (data-dependent) combination of all others, then a position-wise FFN processes each token. The FFN already mixes the hidden dim; attention's job is to mix the *sequence* dim so the FFN sees all tokens.
2. So replace attention with ANY linear operator that mixes across the sequence. Simplest: a learned matrix multiply along the sequence dim (and one along the hidden dim) — the "Linear" baseline. This is the Synthesizer-like / MLP-Mixer-like move. It works decently (Linear-B: MLM acc 0.62 vs BERT 0.68).
3. But a learned N×N seq-mixing matrix is O(N^2) params (depends on max length, doesn't generalize across lengths) and still O(N^2) compute. Want a *structured*, parameter-free linear transform that mixes all tokens and is fast. -> Fourier Transform.
4. DFT: X_k = sum_{n=0}^{N-1} x_n e^{-2πi nk/N}. Each output X_k is a sum over ALL inputs x_n (weighted by twiddle factors) -> it's a dense,全-token mixing, exactly the "give the FFN access to all tokens" property. Parameter-free (twiddle factors fixed). Computable in O(N log N) via FFT (Cooley-Tukey) or O(N^2) via DFT matrix multiply.
5. The input is 2D (N, d). Apply 2D DFT = 1D DFT along seq (F_seq) then 1D DFT along hidden (F_h): y = Re(F_seq(F_h(x))). They commute (separable). Keep ONLY the real part — at the END, after both transforms — so the rest of the network (FFN, output) stays real and unmodified.
   - Why real part only at the end (not midway): found less accurate & less stable if real part taken throughout (e.g. Re(F_seq(Re(F_h(x))))). Taking |.| (abs) also worse. So Re(.) applied once, after the full 2D transform.
   - Why 2D not 1D-along-seq-only: FFN already mixes hidden dim, so 1D-seq DFT is the "important" mixing (ablation: 1D-token-only still >> FF-only, confirming token mixing is the key). But 2D (also mixing hidden) gave best accuracy. 1D was a bit faster but less accurate.

## Why the Fourier Transform specifically (design rationale)
- Dense all-to-all mixing with ZERO learnable parameters -> smaller model, more training stability (the 3 param-free-mixing models — FNet, Random, FF-only — were most stable).
- Fast: O(N log N) FFT (vs O(N^2) attention).
- Positional info baked in: the n,k indices in e^{-2πi nk/N} encode position, so FNet works even without explicit position embeddings (they keep them only for clean BERT comparison).
- Duality intuition: alternating encoder blocks ~ alternating Fourier / inverse Fourier, hopping between "time" and "frequency" domain. Multiplying by FFN weights in frequency domain ≈ convolving in time domain -> FNet ~ alternating multiplications and (large-kernel) convolutions. (Intuition only; broken by residuals + real-part-only non-invertibility.)
- Learnable params in Fourier sublayer: tried elementwise mult, seq/hidden matmuls, complex learnable DFT weights — all detrimental or inconsequential and slower. "DFT is locally optimal in some sense."
- Alternatives tried: DCT (~4% worse), Hadamard (~2% worse, slightly faster), Hartley H = Re{F} - Im{F} (matched DFT, 76.7 vs 76.7 GLUE). So DFT chosen; Hartley a tie alternative.

## Implementation grounding
- GPU: always FFT (jnp.fft.fftn). TPU: matmul with cached DFT matrix for N<=4096 (TPUs better at matmul, worse FFT), FFT beyond. Vandermonde DFT matrix W_{nk}=e^{-2πi nk/N}/sqrt(N).
- Block (post-LN, like BERT): mix = Fourier(x); x = LayerNorm(x + mix); ff = FFN(x); out = LayerNorm(x + ff). eps=1e-12.
- FFN: Dense(d_ff) -> GELU -> Dense(d). d_ff = 4*d_h. Heads in BERT baseline = d_h/64.
- Embeddings: BERT-style (word + absolute position + token-type), then encoder.
- Pooler: Dense(d_model) on x[:,0] then tanh (for NSP/classification).
- Canonical JAX code:
  FourierTransformLayer: return jax.vmap(jnp.fft.fftn)(x).real   # vmap over batch; fftn does 2D DFT over (seq,hidden); .real keeps real part
  FNetEncoderBlock: mixing_output = fourier(x); x = LayerNorm(x+mixing_output); ff = ff_layer(x); return LayerNorm(x+ff)
- FNet-Hybrid: replace 2 Fourier sublayers with self-attention; TOP layout (last layers) best — gets 97-99% of BERT GLUE, still 40-70% faster.

## Eval settings (pre-method)
- Pretraining: masked LM (MLM) + next-sentence prediction (NSP), BERT setup, C4 dataset, 32k SentencePiece vocab. Base/Large configs from BERT/Turc 2019.
- Downstream: GLUE benchmark. Long-Range Arena (LRA) for long-sequence. Metrics: accuracy, train/inference speed (steps/s, ms/batch), peak memory.

## Design-decision -> why table
- Replace attention with a fixed linear mixer: hypothesis that attention's token-dependence isn't essential; FFN does hidden mixing, mixer just needs to expose all tokens.
- Learned dense matmul (Linear baseline) -> Fourier: structured, parameter-free, O(N log N), positional info built in. Removes O(N^2) params and quadratic compute.
- 2D DFT (seq + hidden): seq mixing replaces attention's role; hidden mixing adds a bit more accuracy; commute so order free.
- Real part only, at the end: keeps downstream real & unmodified; more accurate/stable than real-part-midway or abs.
- DFT over DCT/Hadamard/Hartley: DFT best (Hartley ties); chosen for simplicity & being the natural transform.
- No learnable params in mixer: tried, didn't help, hurt stability/speed.
- FFT on GPU / matmul on TPU: hardware-specific fastest path.
- Post-LN BERT block, GELU FFN, d_ff=4d: inherited from BERT for clean comparison.

## Scaffold correspondence
- Pre-method scaffold: BERT-style encoder = embeddings, N blocks each (TokenMixer sublayer + FFN sublayer, residual+LayerNorm), pooler. TokenMixer body = empty slot.
- Final code fills TokenMixer with the parameter-free 2D-DFT-real-part op.
