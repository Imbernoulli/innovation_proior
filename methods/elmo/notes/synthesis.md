# ELMo synthesis notes

## Pain point
Pretrained word representations (word2vec, GloVe) assign ONE vector per word TYPE. "play" gets a single
vector regardless of whether it's a theatrical play, a sports play, or the verb. Polysemy is unrepresentable;
syntax-vs-semantics is conflated into one point. A good representation should model (1) complex characteristics
of word use (syntax + semantics) and (2) how those vary with context. Goal: per-TOKEN representation that is a
function of the entire input sentence.

## Load-bearing ancestors
- word2vec (Mikolov 2013) / GloVe (Pennington 2014): context-INDEPENDENT type embeddings. Limitation: one vector
  per word type; can't disambiguate; OOV words have no vector.
- Subword: Bojanowski 2017 (fastText), Wieting 2016 (charagram), Ling 2015 (char compositional). Char-level
  input → handles OOV + morphology. ELMo uses char-CNN input for the same reason.
- char-CNN LM input: Kim 2015 (character-aware neural LM), Jozefowicz 2016 (CNN-BIG-LSTM, "Exploring the limits
  of LM"). ELMo halves the dims of CNN-BIG-LSTM.
- LSTM (Hochreiter 1997). Neural LMs: Jozefowicz 2016, Merity 2017 (AWD-LSTM), Melis 2017.
- CoVe (McCann 2017): contextual vectors = TOP-LAYER hidden states of a 2-layer biLSTM ENCODER of a supervised
  NMT system. Limitation: needs parallel MT data (limited corpus size); only top layer used; supervised signal.
- TagLM / Peters 2017 (semi-supervised seq tagging w/ biLM): contextual embeddings from a pretrained biLM, but
  only the TOP layer, concatenated into a tagger. ELMo generalizes: use ALL layers, learned weighting; share
  token+softmax params across directions.
- context2vec (Melamud 2016): biLSTM around a pivot word, top layer learns word sense.
- Probing what layers encode: Belinkov 2017 (lower MT encoder layer better at POS than higher), Sogaard 2016 /
  Hashimoto 2017 (supervise low-level tasks at lower layers). Motivates: different layers → different info,
  so expose ALL.
- Dai & Le 2015 / Ramachandran 2017: pretrain then FINE-TUNE encoder-decoder. ELMo contrast: FREEZE biLM, add
  task capacity on top, use as features. Lets a small supervised model leverage a big universal biLM.

## Core method
Forward LM: p(t_1..t_N) = prod_k p(t_k | t_1..t_{k-1}). Char-CNN gives context-indep token rep x_k^LM; pass
through L forward LSTM layers; top layer + softmax predicts t_{k+1}.
Backward LM: p(t_1..t_N) = prod_k p(t_k | t_{k+1}..t_N). Same but reversed.
biLM jointly maximizes:
  sum_k [ log p(t_k | t_1..t_{k-1}; Θ_x, Θ_LSTM→, Θ_s) + log p(t_k | t_{k+1}..t_N; Θ_x, Θ_LSTM←, Θ_s) ]
Tie token-embedding params Θ_x and softmax params Θ_s across directions; separate LSTM params per direction.

For each token, L-layer biLM gives 2L+1 reps:
  R_k = { x_k^LM, h→_{k,j}^LM, h←_{k,j}^LM | j=1..L } = { h_{k,j}^LM | j=0..L }
where h_{k,0}^LM = x_k^LM (token layer, duplicated for both dirs), h_{k,j}^LM = [h→_{k,j}; h←_{k,j}].

ELMo collapses R into ONE task-specific vector:
  ELMo_k^task = E(R_k; Θ^task) = γ^task sum_{j=0}^L s_j^task h_{k,j}^LM
s^task = softmax(w^task) (normalized layer weights, sum to 1, nonneg); γ^task scalar scale.
Optional layer-norm per layer before weighting (different layers have different activation distributions).
Special case E(R_k)=h_{k,L}^LM (top only) recovers TagLM/CoVe.

Why all layers + learned weighting beats top-only: lower layers → syntax (POS), higher → word sense (WSD).
Diagnostic: biLM 1st-layer POS acc 97.3 > 2nd-layer 96.8; biLM 2nd-layer WSD F1 69.0 > 1st-layer 67.4.
(CoVe shows same pattern but weaker: POS 93.3/92.8, WSD 59.4/64.7.) So no single layer is universally best;
let the task learn a soft mixture.

Why γ: biLM internal activations and task activations have very different distributions; γ rescales the whole
ELMo vector to aid optimization. Without γ, top-only case performs well below baseline (SNLI) / fails (SRL).

Usage: freeze biLM, run it, record all layer activations. Concatenate ELMo_k^task with task token rep x_k:
[x_k; ELMo_k^task] into the task RNN. Optionally also at output: replace h_k with [h_k; ELMo_k^task] (with a
separate set of weights). Add dropout to ELMo; optionally L2-regularize the weights λ||w||^2 (large λ → average
of layers; small λ → free weights). Larger λ for small datasets (NER).

## biLM architecture (pretrained)
L=2 biLSTM, 4096 units, 512-dim projections, residual connection layer1→layer2. Input: char-CNN with 2048
char n-gram conv filters + 2 highway layers + linear projection to 512. 3 layers of reps per token. Trained
10 epochs on 1B Word Benchmark, avg fwd/bwd perplexity 39.7. Char input → reps for any token incl OOV.

## Canonical impl (allennlp ScalarMix + char encoder)
ScalarMix: normed = softmax(cat(scalar_params)); out = gamma * sum(normed_j * tensor_j); optional layer_norm
per tensor first. Char encoder: char emb → list of Conv1d (widths 1..7, filter counts) → relu → max over char
dim → concat → Highway(n=2) → Linear projection to 512.
