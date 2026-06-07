# MoE synthesis

## Pain point
Capacity (params) bounded by compute when whole net fires per example → quadratic blow-up (model size × #examples). Want >>params without >>compute/example. Big text corpora (1B word LM, WMT) reward capacity.

## Ancestors (load-bearing)
- **Conditional computation** (Davis&Arel 13; Bengio,Léonard,Courville 13 "Estimating or propagating gradients through stochastic neurons"; Bengio,Bacon,Pineau,Precup 15 "Conditional computation in NN for faster models"; Cho&Bengio 14; Eigen,Ranzato,Sutskever 13): turn parts of net on/off per example. Promise unrealized. Failure modes: GPUs slow at branching → must gate big chunks; large batches amortize param loads, conditional compute shrinks per-chunk batch; network bandwidth (embedding-like patterns bottlenecked); need extra loss terms for sparsity/balance (Bengio15 uses 3 losses); small datasets (≤600k imgs) can't train huge param counts. Bengio15 uses BOOLEAN gates + REINFORCE — high variance, hard.
- **Classic mixtures of experts** (Jacobs,Jordan,Nowlan,Hinton 91 "Adaptive mixtures of local experts"; Jordan&Jacobs 94 HME): the *whole* model is a mixture; gating softmax over experts; experts specialize. y=Σ g_i(x) E_i(x). Dense (all experts run). Top-level only.
- **Eigen,Ranzato,Sutskever 13**: stacked MoEs as *components* inside a deep net, each with own gate; allude to sparsity. Use a hard constraint at start of training to avoid collapse.
- Testbed: stacked LSTM LMs (Jozefowicz/RafalNoam 16 "exploring limits of LM"), GNMT (Wu et al 16) for MT.

## Method derived
- Layer: n experts E_1..E_n (identical FFN arch, separate params) + gating net G; y = Σ_i G(x)_i E_i(x). Skip E_i wherever G(x)_i=0 → compute ∝ k (active experts), not n.
- Softmax gating (Jordan&Jacobs): G_σ(x)=Softmax(x·W_g). Dense — no savings.
- **Noisy top-k gating**:
  H(x)_i = (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i)
  KeepTopK(v,k)_i = v_i if in top-k else −∞
  G(x) = Softmax(KeepTopK(H(x), k))
  k>1 so top-k gate values have nonzero grad w.r.t. gating weights (back-prop, NOT REINFORCE). −∞ → softmax gives 0 → expert skipped.
- Why noise: (a) load balancing (breaks symmetry, exploration so non-favored experts get tried), (b) makes the "in-top-k" event have a differentiable probability → Load loss.
- **Collapse problem**: gate converges to favoring few experts; self-reinforcing (favored trained more → selected more). Eigen used hard constraint.
- **Importance loss**: Importance(X)=Σ_{x∈X} G(x) (per-expert batchwise sum of gates). L_importance = w_importance · CV(Importance(X))² . CV=std/mean; CV²=var/mean². Pushes equal importance.
- Importance ≠ equal #examples (one expert: few big gates; another: many small). Distributed hardware needs balanced counts. #examples is discrete → no grad.
- **Load loss**: smooth estimator. P(x,i)=Pr(noise makes H(x)_i land in top-k) = probability G(x)_i nonzero given fresh noise on i, others fixed. G(x)_i>0 iff H(x)_i > kth-greatest of H excluding i. So
  P(x,i)=Φ( ((x·W_g)_i − kth_excluding(H(x),k,i)) / Softplus((x·W_noise)_i) ), Φ=std normal CDF.
  Load(X)_i = Σ_x P(x,i). L_load = w_load · CV(Load(X))². 
  Impl detail: compare clean value to the kth threshold; "if I'm already in top-k" uses the kth-excluding-self = the (k+1)th value when self is in; the code computes threshold_if_in (k-th of the top k+1, i.e. position k) and threshold_if_out (position k-1), picks per whether currently in.
- Init W_g=W_noise=0 → at start logits 0 + noise only → roughly equal load, avoid OOM before soft constraints kick in.
- Loss table: any of the two losses ≈ same quality (ppl 35.6-35.7); none → 39.8 and CV 3.0, max/mean load 17.8. w_load lowers max/mean. Use w_imp=w_load=0.1 (LM), 0.01 (MT).

## Performance challenges
- **Shrinking batch**: k of n per example → each expert gets ~kb/n ≪ b. Fix: make batch huge via *mixed data+model parallelism*: d devices, data-parallel on standard layers+gate, but ONE shared copy of each expert (model-parallel shards); each expert gets combined batch ≈ kbd/n → factor d bigger. Also convolutional trick: apply MoE to all timesteps at once (×#timesteps). Recurrent MoE breaks conv trick → recompute activations (Gruslys16).
- **Bandwidth**: experts stationary; send only inputs/outputs. Compute/io ratio = hidden size; raise efficiency by bigger hidden layer.

## Hierarchical MoE
a groups × b experts; primary gate G_primary picks group, secondary G_i picks within. y_H=ΣΣ G_primary(x)_i G_i(x)_j E_{i,j}(x). Importance_H = Σ G_primary_i G_i_j. Load_H = Load_primary_i · Load_i(X^(i))_j / |X^(i)| (so grad flows to primary; naive Load_i(X_i)_j wouldn't).

## Code (davidmrau, based on tensor2tensor expert_utils.py)
- SparseDispatcher: dispatch (build per-expert batches from nonzero gates), combine (index_add weighted sum).
- MoE.noisy_top_k_gating: clean_logits = x@w_gate; noise_stddev=softplus(x@w_noise)+eps; noisy=clean+randn*stddev; softmax; topk(k+1); normalize top-k; scatter into zeros; load = _prob_in_top_k(...) sum if noisy+train else (gates>0).sum.
- _prob_in_top_k: uses top k+1 values; threshold_if_in=position k, threshold_if_out=position k-1; prob_if_in=Φ((clean−thr_in)/stddev), prob_if_out=Φ((clean−thr_out)/stddev); where(is_in,...).
- cv_squared = var/(mean²+eps).
- forward: gates,load = gate(x); importance=gates.sum(0); loss=(cv²(importance)+cv²(load))*coef; dispatch→experts→combine.
- Note: davidmrau softmaxes BEFORE topk; paper does KeepTopK then softmax. Equivalent up to which values feed softmax — paper softmaxes the top-k raw logits; davidmrau softmaxes all then renormalizes top-k. I'll follow the canonical impl but note both. Experts here have Softmax output (MLP.soft) — that's task-specific; the generic FFN is Linear-ReLU-Linear. I'll present generic FFN expert.
