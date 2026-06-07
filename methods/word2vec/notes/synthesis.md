# Word2Vec (CBOW + Skip-gram, with NEG / HS / subsampling) — synthesis notes

## Pain point / research question (in-frame, ~2012)
Want **high-quality continuous word vectors** that (a) put similar words near each other and (b) preserve *linear regularities* (king−man+woman≈queen), trainable on **billions of words** with **millions of vocab**, in <1 day on commodity hardware. Existing neural LMs that yield good vectors are too slow: nobody had trained on >~hundreds of millions of words with D∈[50,100]. The bottleneck is the model's compute per example, dominated by (1) a non-linear hidden layer and (2) the full-vocabulary softmax normalization.

## Complexity accounting (the framing device)
Training cost O = E·T·Q. Per-example Q for each architecture:
- **Bengio NNLM (feedforward):** Q = N·D + N·D·H + H·V. Dominant = H·V (output softmax) and N·D·H (dense projection→hidden). With HS, H·V → H·log₂V, leaving N·D·H as the bottleneck.
- **RNNLM:** Q = H·H + H·V → with HS, H·H dominates (recurrent matrix).
- **CBOW:** Q = N·D + D·log₂V. No hidden layer; sum/avg of N context vectors, then HS output.
- **Skip-gram:** Q = C·(D + D·log₂V). C = max window; for each center word pick R∈[1,C], use 2R context words as targets.

Key observation: kill the non-linear hidden layer → the model becomes log-linear → can pour the saved compute into way more data + bigger D. The hypothesis: a *simpler* model on *much more* data beats a complex model on little data, especially for the *linear-regularity* property (a linear model should produce linearly-structured vectors).

## The two architectures
- **CBOW:** average the input vectors of the 2c context words (shared projection — order-independent, hence "bag of words"; "continuous" because distributed reps), predict the center word. Q = N·D + D·log₂V.
- **Skip-gram:** use the center word's input vector to predict each surrounding word independently. Objective: (1/T)Σ_t Σ_{−c≤j≤c, j≠0} log p(w_{t+j}|w_t). Dynamic window: sample R∈[1,c] uniformly → distant words sampled less → soft down-weighting of far context (cheap, no extra weights).

Two vector tables per word: **input** v_w (the "embedding" we keep) and **output** v'_w (used only in the prediction layer). Separate because a word is rarely its own neighbor → if shared, the dot product v_w·v_w (always large positive) corrupts the geometry; separate tables make the optimization clean. The kept embedding is syn0 (input side).

## Softmax & its cost
p(w_O|w_I) = exp(v'_{w_O}·v_{w_I}) / Σ_{w=1}^W exp(v'_w·v_{w_I}). ∇log p costs O(W), W∈[1e5,1e7]. Two escapes: hierarchical softmax (HS) and negative sampling (NEG).

## Hierarchical softmax (full math)
Binary tree, words at leaves. Path root→w has nodes n(w,1)=root … n(w,L(w))=w. Each inner node n has a vector v'_n and a learned left/right split via σ. Define [[x]]=+1 if x true else −1.
p(w|w_I) = Π_{j=1}^{L(w)−1} σ( [[n(w,j+1)=ch(n(w,j))]] · v'_{n(w,j)}·v_{w_I} ).
Because σ(x)+σ(−x)=1 and at each node the two children's probs sum to 1, the leaf probs sum to 1 over the whole vocab (Σ_w p(w|w_I)=1). Cost of log p and ∇ = L(w_O) ≈ log₂W. **Huffman tree** for the structure: frequent words get short codes → expected path length ≈ log₂(unigram-perplexity), faster than balanced log₂V. Parameters: one v_w per word (input), one v'_n per inner node (W−1 of them).

Gradient (matches C code, syn1 = inner-node vectors): for each node d on the path, let f = σ(v_{w_I}·v'_{n_d}); the binary "code" bit code_d∈{0,1} encodes which child; g = (1 − code_d − f)·α. Update v'_{n_d} += g·v_{w_I}; accumulate input grad += g·v'_{n_d}. (Label for the node is (1−code_d); turning point: g is (label − σ)·α, standard logistic-regression gradient.)

## Negative sampling derived from NCE (full math — the core derivation)
**NCE (Gutmann–Hyvärinen 2010; Mnih–Teh 2012 for LMs):** to fit an unnormalized model p_θ without computing Z, turn density estimation into binary classification: mix true data with k noise samples per datum from p_n. Posterior that a sample came from data:
P(D=1|w,c) = p_d(c|w) / ( p_d(c|w) + k·p_n(c) ).
Plug the model in for p_d (treat partition Z as a free parameter / fold into the score, can set Z=1): with score s(w,c)=exp(v'_c·v_w),
P(D=1|w,c) = exp(v'_c·v_w) / ( exp(v'_c·v_w) + k·p_n(c) ).
NCE maximizes Σ over data E[log P(D=1)] + k·E_{noise}[log P(D=0)]; it provably ≈ maximizes the softmax log-likelihood and needs the numeric values of p_n.

**NEG simplification:** we only want good *vectors*, not a calibrated LM, so drop the requirement that NCE recover the softmax. Set **k·p_n(c) = 1** in the posterior:
P(D=1|w,c) = exp(v'_c·v_w)/(exp(v'_c·v_w)+1) = σ(v'_c·v_w).
Then per positive (w_O,w_I) the objective is
log σ(v'_{w_O}·v_{w_I}) + Σ_{i=1}^k E_{w_i∼P_n}[ log σ(−v'_{w_i}·v_{w_I}) ].
This replaces every log p(w_O|w_I) term. Difference from NCE: NEG uses only *samples* (not p_n values) and no longer maximizes softmax likelihood — fine for our goal. k = 5–20 (small data), 2–5 (big data).

**Noise distribution:** P_n(w) = U(w)^{3/4}/Z (unigram raised to 3/4). Beats unigram and uniform on every task. Intuition: 3/4 power flattens the unigram — boosts rare-word sampling relative to U(w) (so rares appear as negatives often enough to get pushed apart) while still sampling frequent words more than uniform would.

Gradient (matches C code, syn1neg = output vectors): for each of the 1 positive + k negatives, label∈{1,0}; f=σ(v_{w_I}·v'_target); g=(label − f)·α. Update v'_target += g·v_{w_I}; input grad += g·v'_target; finally v_{w_I} += accumulated grad.

## Subsampling of frequent words (full math)
Frequent words ("the", "a", "in") give little info and dominate counts. Discard each token w_i with probability
P(discard w_i) = 1 − √( t / f(w_i) ),  f = relative frequency, t ≈ 1e−5 (paper) / 1e−3 (code default).
Aggressively drops words with f>t, preserves frequency ranking. Speeds training 2–10×, and *improves* rare-word vectors (removes drowning-out frequent neighbors, effectively widening the window over content words). C-code keep-prob variant: keep with prob √(t/fr)+(t/fr) — same √(t/f) leading term plus a small linear term, clamp ≤1.

## Design decisions → why
- **Drop non-linear hidden layer (log-linear model):** hidden layer is the compute hog (N·D·H); linearity also matches the goal of *linear* analogical structure; trade representational power for ≫ more data.
- **CBOW averages context (shared projection):** order-free, cheap, robust; predicts center.
- **Skip-gram predicts context from center:** more training pairs per word, much better on *semantic* analogies and rare words.
- **Dynamic window R∈[1,c]:** soft distance weighting for free; near words count more.
- **Two vector spaces (in/out):** avoids the self-dot-product pathology; clean gradients.
- **HS with Huffman tree:** O(log V) output; short codes for frequent words → ≈log(unigram-PPL).
- **NEG over HS/NCE:** even simpler, no tree, no p_n values; great for frequent words; k controls cost/quality.
- **U^{3/4} noise:** empirically best balance between unigram and uniform.
- **Subsampling 1−√(t/f):** down-weight frequent tokens; speed + better rares.
- **Linear LR decay, SGD, init U[−0.5/D,0.5/D] for input, zeros for output:** standard; output zeros so early σ≈0.5 (no spurious push), small input init keeps dot products in σ's linear regime.
- **expTable:** precompute σ on [−6,6], clamp outside (σ≈0/1) → no per-step exp().

## Canonical implementation (grounded)
tmikolov/word2vec `word2vec.c`. syn0=input vectors (the embeddings), syn1=HS inner-node vectors, syn1neg=NEG output vectors. InitUnigramTable() builds the 1e8 table with power 0.75. CreateBinaryTree() = Huffman. TrainModelThread: subsample → dynamic window b=rand%window → CBOW(avg)/Skip-gram → HS loop over Huffman code + NEG loop over 1+negative samples → update. Defaults: D=100, window=5, negative=5, sample=1e-3, alpha=0.025 linear decay, iter=5, min_count=5.

## Evaluation settings (pre-method, no outcomes)
Analogy task: a:b::c:? answered by argmax_x cos(x, vec(b)−vec(a)+vec(c)), exclude input words; exact-match accuracy; split semantic vs syntactic. Corpora: Google News (~6B), LDC (320M), 1B-word news. Baselines to beat: Collobert-Weston SENNA, Turian, Mnih HLBL, Mikolov RNNLM, Huang. MSR Sentence Completion. (Outcomes excluded.)
