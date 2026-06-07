# ALBERT synthesis (verified vs arXiv 1909.11942 source + google-research/albert)

## Verified core
- Question: "Is having better NLP models as easy as having larger models?"
- Obstacle: memory limits of hardware; communication overhead in distributed training proportional to #params.
- KEY observation (motivating, about existing systems): doubling BERT-large hidden size to BERT-xlarge (H=2048, ~1.27-1.37B params) gives WORSE performance (RACE 54.3% vs BERT-large 73.9% in commented table; main Table shows BERT-xlarge SQuAD1.1 86.3/77.9, RACE 39.7, Avg 76.7 vs BERT-large Avg 85.1). No sign of overfitting -> model degradation / harder to optimize, not capacity.
- ALBERT = A Lite BERT. Two parameter-reduction techniques + new SOP loss.

### Technique 1: Factorized embedding parameterization
- BERT ties WordPiece embedding size E to hidden size H (E ≡ H).
- Argument: WordPiece embeddings are context-INDEPENDENT; hidden layers are context-DEPENDENT. Power of BERT comes from context. So want H >> E.
- Practical: vocab V large (V=30000). If E≡H, embedding matrix V×E grows with H => billions of params, sparsely updated.
- Fix: factorize. Project one-hot -> low-dim E, then E -> H. Params O(V×H) -> O(V×E + E×H). Significant when H>>E.
- Same E for all wordpieces (evenly distributed unlike whole words).

### Technique 2: Cross-layer parameter sharing
- Options: share only FFN, share only attention, share all. DEFAULT = share ALL across layers.
- Prevents params growing with depth.
- Universal Transformer (Dehghani 2018) and DQE (Bai 2019) explored sharing for enc-dec, not pretrain/finetune. DQE reaches equilibrium (input==output emb of a layer). ALBERT: L2 distance and cosine sim of input/output embeddings per layer OSCILLATE not converge -> different solution space. Weight-sharing stabilizes parameters (smoother layer-to-layer transitions); metrics drop vs BERT but don't reach 0 even after 24 layers.
- Ablation: all-shared hurts, less severe for E=128 (-1.5 Avg) than E=768 (-2.5 Avg). Most drop from sharing FFN; sharing attention causes no drop at E=128 (+0.1), slight at E=768 (-0.7). Smaller group size M better but more params. Choose all-shared default.

### Technique 3: SOP (sentence-order prediction) replacing NSP
- BERT NSP: binary, do two segments appear consecutively. Positive = consecutive segments same doc; negative = segments from DIFFERENT documents; 50/50.
- Conjecture: NSP ineffective because too EASY — conflates TOPIC prediction + COHERENCE prediction. Topic prediction is easy and overlaps with MLM. (Negative from different doc => misaligned in both topic AND coherence.)
- SOP: positive = two consecutive segments same doc (like BERT); negative = same two consecutive segments with ORDER SWAPPED. Avoids topic, focuses on coherence/discourse-order.
- Ablation (Table SOP): NSP gets 52.0% on SOP task (~random) => NSP only models topic shift. SOP solves NSP task 78.9% and SOP 86.5%. SOP improves downstream multi-sentence tasks (~+1% SQuAD1.1, +2% SQuAD2.0, +1.7% RACE; Avg +1%).

### Architecture notes
- Backbone: Transformer encoder, GELU. FFN/filter size = 4H. #attention heads = H/64.
- Configs: ALBERT-base L12 H768 E128 12heads 12M; large L24 H1024 E128 18M; xlarge L24 H2048 E128 60M; xxlarge L12 H4096 E128 235M.
- ALBERT-large ~18x fewer params than BERT-large (18M vs 334M).
- ALBERT-xxlarge uses 12 layers (24 gives similar results, more expensive — "no need for models deeper than 12 when all-shared").
- Note: ALBERT-xxlarge actually SLOWER than BERT-large (larger H) despite fewer params; xlarge 2.4x faster than BERT-xlarge; ALBERT-large 1.7x faster than BERT-large.

### Experimental setup (pre-method facts: corpora, tokenizer, optimizer)
- Pretrain on BookCorpus + English Wikipedia ~16GB.
- Input format: "[CLS] x1 [SEP] x2 [SEP]" two segments (segment = multiple sentences). Max length 512, randomly generate shorter sequences with prob 10%.
- Vocab size V=30000, tokenized with SentencePiece (Kudo & Richardson 2018) as in XLNet.
- MLM masking: n-gram masking (SpanBERT, Joshi 2019), length n random with p(n) = (1/n) / sum_{k=1}^N (1/k). Max n-gram length N=3.
- Optimizer: LAMB (You 2019), batch size 4096, learning rate 0.00176. Train 125k steps default. Cloud TPU v3, 64-512 TPUs.

### Other findings
- Embedding size ablation: non-shared larger E slightly better; all-shared E=128 best. Use E=128.
- Additional data (XLNet/RoBERTa data) boosts MLM acc and most downstream (except SQuAD which is Wikipedia-based, out-of-domain hurt).
- Removing dropout: largest models don't overfit even at 1M steps; removing dropout improves MLM acc and downstream (ALBERT-xxlarge: Avg 90.4 -> 90.7). First to show dropout can hurt large Transformers. (Cites batchnorm+dropout harmful interaction Li 2019.)

### Downstream hyperparameters (appendix table) — LR/BSZ/ALBERT-DR/Classifier-DR/TrainSteps/WarmupSteps/MaxSeqLen
- e.g. MNLI 3e-5/128/0/0.1/10000/1000/512; SQuAD2.0 3e-5/48/0/0.1/8144/814/512; RACE 2e-5/32/0.1/0.1/12000/1000/512; SST-2 1e-5/32/0/0.1/20935/1256/512.

## Canonical implementation (google-research/albert modeling.py)
- embedding_lookup: word embedding table shape [vocab_size, embedding_size E].
- embedding_hidden_mapping_in: dense_layer_2d projects E -> hidden_size H (this is the factorization second matrix).
- transformer: with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE): then group_%d / inner_group_%d. AUTO_REUSE => same variables reused across all num_hidden_layers => cross-layer sharing.

## Design decisions -> why
- Factorize E from H: embeddings are context-free lookups, hidden is contextual; decoupling lets H grow without V×H blowup; cheaper, and embedding capacity not the bottleneck.
- E=128: best under all-shared; small enough to scale H.
- Share all layers: biggest parameter saving (params independent of depth), acts as regularizer / stabilizer (oscillating-not-diverging embeddings), allows scaling H massively. FFN-sharing is the costliest in accuracy but all-shared chosen for simplicity + max saving.
- SOP not NSP: NSP too easy (topic, overlaps MLM); SOP forces discourse coherence (same segments, swapped order) — harder, transfers to multi-sentence tasks.
- 12-layer xxlarge: depth past 12 gives no gain once sharing (the shared block is applied repeatedly; more applications saturate).
- LAMB + batch 4096: large-batch optimizer needed for these batch sizes (layer-wise adaptive).
- n-gram masking N=3: predicting spans (whole words/short phrases) is a harder, more useful MLM target than single subwords.
- Remove dropout (final large runs): models underfit (no overf: train loss high), dropout's regularization is counterproductive; removing it raises MLM acc.
```
