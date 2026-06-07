# ELECTRA — synthesis notes (Phase 1.5)

## Pain point at the time
- BERT-style masked language modeling (MLM): corrupt ~15% of tokens to [MASK], train the
  encoder to predict the originals. Cross-entropy loss is computed ONLY at the masked
  positions. So each training example produces gradient signal from only ~15% of the
  sequence; the other ~85% of positions are read but never contribute to the loss.
- Pretraining is the compute bottleneck for these models. If only 15% of positions teach,
  you waste ~85% of the per-step forward/backward compute on a learning signal you never use.
- Secondary defect: the [MASK] token appears in pretraining but never in fine-tuning →
  a pretrain/finetune input-distribution mismatch. (BERT patches this partially: of the
  15% chosen, 80%→[MASK], 10%→random token, 10%→unchanged. The paper's "Replace MLM"
  ablation shows this patch is insufficient.)

## Central question to derive from
"Can every token in the sequence contribute to the loss every step, instead of only 15%?"
A loss over ALL n positions, not k≈0.15n, would be ~6-7x denser supervision per step.

## The derivation chain
1. Why MLM is restricted to 15%: it's a generative reconstruction task — predict the
   identity of a hidden token from context. You can only ask "what was here?" at positions
   you actually hid; asking it at visible positions is trivial (the answer is in the input).
   So generative reconstruction is structurally tied to a small masked subset.
2. To get a loss over all positions, switch from a *generative* per-position objective
   (30k-way softmax: "which token?") to a *discriminative* per-position objective that is
   well-defined at every position: a binary label per token. The natural binary label that
   is non-trivial everywhere: "is this token the original, or has it been swapped out?" =
   Replaced Token Detection (RTD). Now the loss is sigmoid binary cross-entropy at all n
   tokens.
3. For RTD to be a real task the replacements must be plausible (else it degenerates to
   detecting type/grammar errors, not language understanding). So we need a proposal
   distribution over plausible tokens at the masked positions. The most natural such
   proposal is itself a (small) MLM: mask positions, let an MLM predict a distribution,
   SAMPLE a token from it, splice it in.
4. This yields two networks: a generator G (small MLM that proposes replacements) and a
   discriminator D (the encoder we keep, predicting original/replaced at every position).
   Structurally GAN-like, but NOT a GAN (see below).

## The losses (derive inline, verify signs)
- Positions to corrupt: m_i ~ unif{1,n}, i=1..k, k=ceil(0.15 n). x^masked = REPLACE(x, m, [MASK]).
- Generator MLM: p_G(x_t | x^masked) = softmax over vocab of e(x_t)^T h_G(x^masked)_t.
  L_MLM = E[ sum_{i in m} -log p_G(x_i | x^masked) ]   (cross-entropy at masked positions only)
- Sample replacements: x̂_i ~ p_G(x_i | x^masked) for i in m. x^corrupt = REPLACE(x, m, x̂).
- Discriminator: D(x^corrupt, t) = sigmoid(w^T h_D(x^corrupt)_t).
- RTD label at position t: 1 if x^corrupt_t == x_t (original / "real"), 0 if replaced ("fake").
  IMPORTANT subtlety: if the generator happens to sample the correct token (x̂_i == x_i),
  that position is labeled REAL, not fake. (Improves results moderately.)
- L_Disc = E[ sum_{t=1}^n  -1(x^corrupt_t = x_t) log D(x^corrupt,t)
                          -1(x^corrupt_t != x_t) log(1 - D(x^corrupt,t)) ]
  i.e. standard binary cross-entropy, label y_t = 1(original): -[ y log D + (1-y) log(1-D) ].
  Sum is over ALL n positions → this is the sample-efficiency win.
- Combined: min_{θ_G, θ_D} sum_{x in X} L_MLM(x, θ_G) + λ L_Disc(x, θ_D), λ = 50.
  Single-sample MC estimate of the expectations. NO backprop of L_Disc into G (can't —
  discrete sampling blocks the gradient; and we don't want to anyway, see non-adversarial).
- Why λ=50: L_Disc is a 2-way (binary) cross-entropy, numerically much smaller than the
  30000-way softmax cross-entropy of L_MLM. λ≈50 rescales them onto comparable footing so
  the discriminator signal isn't swamped. Searched over [1,10,20,50,100].

## Sample-efficiency argument (the core claim — re-derive carefully)
MLM produces a loss term at k ≈ 0.15n positions. RTD produces a loss term at all n
positions. Same forward/backward cost for the discriminator pass, but ~1/0.15 ≈ 6.7x more
loss terms per example. The "ELECTRA 15%" ablation isolates this: restrict L_Disc to only
the masked subset i in m (sum over i in m instead of t=1..n). Result collapses from 85.0
to 82.4 GLUE — i.e. essentially all the way back to BERT (82.2). So the gain is the
all-tokens loss, not the discriminative framing per se. "All-Tokens MLM" (a generative
model predicting all positions, with a copy mechanism) recovers 84.3 — most of the gap —
confirming "learn from all tokens" is the load-bearing idea; the discriminative cast adds a
bit more and is cheaper.

## Why NOT adversarial (NOT a GAN) — load-bearing, derive from Appendix A
- A GAN would train G to MAXIMIZE D's loss (fool the discriminator). To do that by
  backprop you'd differentiate through x̂ ~ p_G — but x̂ is a discrete sample; sampling is
  not differentiable, so the gradient path G→sample→D is severed.
- Workaround tried: REINFORCE (policy gradient). Treat generating all tokens as one action
  factorizing over positions. Simplifying assumption: D(x^corrupt,t) depends only on token t
  and the unreplaced context, not other replaced tokens (few tokens replaced, so OK; also
  decouples credit assignment). Then maximize E[ sum_{t in m} E_{x̂_t~p_G} R(x̂_t, x) ]
  with R = -log D(x̂_t|x^masked) if x̂_t = x_t else -log(1 - D(x̂_t|x^masked)).
  REINFORCE gradient: ∇_θG ≈ E sum_{t in m} E_{x̂_t} ∇ log p_G(x̂_t|x^masked) [R - b],
  with learned baseline b(x^masked,t) = -log sigmoid(w^T h_G(x^masked)_t).
- Outcome: adversarial G reaches only 58% MLM accuracy vs 65% for MLE G — RL is sample-
  inefficient in a ~30k action space. Also adversarial G produces low-entropy (peaked,
  low-diversity) samples. Both known GAN-for-text failure modes (Caccia et al. 2018).
  Adversarial ELECTRA still beats BERT but underperforms MLE-trained ELECTRA. → train G by
  plain MLE.
- Also: no noise vector input to G (unlike a GAN). And the "correct token = real" relabel
  further departs from a GAN.

## Joint vs two-stage training
- Two-stage: train G alone n steps, init D from G's weights, then train D n steps with G
  frozen. Needs G and D same size (for the init). Without the weight init, D sometimes
  fails to learn beyond majority class (G too far ahead). Joint training gives a natural
  CURRICULUM: G starts weak (easy fakes) and improves, so D faces progressively harder
  negatives. Joint ≥ two-stage. → joint training.

## Weight sharing (derive the why)
- Share embeddings (token + positional) between G and D. If G smaller, its hidden size <
  D's, so make embeddings D-sized and add a linear projection in G from embedding-size to
  generator-hidden-size. (Output/input embeddings of G tied as in BERT.)
- Why tie embeddings specifically: the generator's softmax over the whole vocab densely
  updates ALL token embeddings every step (every vocab row gets gradient), whereas D only
  touches embeddings for tokens actually present/sampled. So G is a far better teacher of
  embeddings; sharing lets D inherit them. Ablation (same-size G): no tying 83.6, tie token
  embeddings 84.3, tie all weights 84.4 — tying embeddings gets almost all the benefit, and
  tying everything forces same-size G (expensive). → tie embeddings only.

## Generator size (derive the why)
- Make G smaller than D (reduce hidden/FFN/heads, keep depth). Best at 1/4–1/2 of D's
  hidden size. Too-strong G hurts: it makes the RTD task too hard (replacements too
  plausible), and D wastes capacity modeling G's distribution rather than the data. Also a
  same-size G doubles per-step compute. (Base uses 1/3; Small/Large use 1/4.) Unigram
  generator (sample by corpus frequency) is a cheap baseline but weaker.
- ELECTRA-Large uses 25% mask instead of 15% because at 15% the strong large generator was
  too accurate → too few actual replacements → too few "fake" labels.

## Sampling mechanism in code
- sample_from_softmax: Gumbel-max trick. x̂ = onehot(argmax(softmax(logits) + Gumbel)). This
  draws a categorical sample from p_G. tf.stop_gradient wraps it — confirms no gradient to G
  from the sampling path. Optional temperature; optional disallow-correct (didn't help).
- Discriminator head: dense(hidden_size, act) → dense(1) → squeeze → sigmoid_cross_entropy.
- Loss masked by input_mask (ignore padding), normalized by number of real tokens.

## Downstream
- Throw away G. Keep only D (the ELECTRA encoder). Fine-tune D like BERT: linear classifier
  on top for GLUE, span head for SQuAD. Architecture/hyperparams mostly = BERT's. No NSP.

## Lineage / load-bearing ancestors
- BERT (Devlin et al. 2019): MLM denoising autoencoder; the thing we react to. The 15%
  loss-density limit + [MASK] mismatch are its gaps.
- Denoising autoencoders (Vincent et al. 2008): corrupt-then-reconstruct framing.
- GANs (Goodfellow et al. 2014): generator/discriminator structure; what we resemble but
  reject (discrete text + adversarial instability).
- GAN-for-text difficulty (Caccia et al. 2018): why adversarial text gen lags MLE.
- Noise-Contrastive Estimation (Gutmann & Hyvärinen 2010): binary classifier real vs noise;
  RTD is an NCE-flavored objective with a learned noise distribution.
- word2vec CBOW + negative sampling (Mikolov et al. 2013): predict token from context recast
  as binary real-vs-proposal classification; ELECTRA = scaled-up CBOW-with-neg-sampling
  (transformer encoder + learned generator instead of bag-of-vectors + unigram noise).
- MaskGAN (Fedus et al. 2018): generator fills in deleted tokens (reminiscent of G).
- Transformer encoder (Vaswani et al. 2017): the backbone.
- REINFORCE (Williams 1992): the RL fallback for adversarial training.

## ELECTRA-as-MLM (Appendix) — optimal-discriminator derivation (NCE link)
For one masked token, writing p_mask=mask rate, p_data the true conditional, p_G the
generator: the population L_Disc has optimal D* (critical point wrt D):
  D(x,c) = p_data(x|c)(a + p_G(x|c)) / (a p_data(x|c) + p_G(x|c)),  a=(1-p_mask)/p_mask.
Invert to read p_data off D and p_G:
  p_data(x|c) = D p_G / (a(1-D) + p_G).
This shows D learns information about p_data relative to the noise p_G — the NCE structure.
(Used only to probe ELECTRA as an MLM; BERT slightly better at MLM, 77.9 vs 75.5, expected.)

## Design-decision → why table
| decision | why / rejected alternative |
|---|---|
| discriminative RTD over all tokens | loss at all n positions vs MLM's 15% → denser signal; isolated by ELECTRA-15% ablation |
| generator = small MLM proposing samples | replacements must be plausible; an MLM is the natural plausible-token proposal |
| label correct-resample as "real" | moderately better downstream; avoids penalizing D for a coincidentally-correct guess |
| train G by MLE not adversarially | discrete sampling blocks backprop; REINFORCE is sample-inefficient (58% vs 65% MLM acc) and yields low-entropy samples; MLE G is a better, more diverse teacher |
| no noise vector to G | not a GAN; G is just an MLM |
| joint training | provides curriculum (weak→strong G); two-stage needs same-size G and can fail w/o weight-init |
| λ=50 | binary CE ≪ 30k-way CE magnitude; rescale so disc signal isn't swamped; searched [1,10,20,50,100] |
| share (only) embeddings | G's vocab softmax densely updates all embeddings (better teacher); tying-all forces same-size G; ablation 83.6→84.3→84.4 |
| G = 1/4–1/2 of D | too-strong G makes task too hard + D wastes capacity modeling G; same-size doubles compute |
| 25% mask for Large | strong large G too accurate at 15% → too few replacements/fakes |
| keep only D downstream | D is the representation we want; G was scaffolding |
| Gumbel-max sampling, stop_gradient | draw categorical sample from p_G; confirm no gradient to G via sampling |
