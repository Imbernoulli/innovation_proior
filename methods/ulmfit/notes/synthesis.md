# ULMFiT synthesis (verified vs arXiv 1801.06146 source + AWD-LSTM 1708.02182)

## Verified core
- Goal: inductive transfer learning for ANY NLP task, like fine-tuning ImageNet models in CV. Focus: text classification.
- Problem: NLP models trained from scratch. Word-embedding transfer only targets first layer. Hypercolumn approaches (ELMo etc) concat embeddings as FIXED features, still train main model from scratch. LM fine-tuning (Dai & Le 2015) needs millions of in-domain docs.
- Diagnosis: not the IDEA of LM fine-tuning but lack of knowledge of HOW. LMs overfit small datasets and suffer CATASTROPHIC FORGETTING when fine-tuned with classifier. NLP models shallower than CV -> need different fine-tuning.
- Contributions: discriminative fine-tuning, slanted triangular learning rates (STLR), gradual unfreezing.

### Why LM is the ideal source task
- Captures long-term deps, hierarchical relations, sentiment. Near-unlimited data. Adaptable to target idiosyncrasies. Already a component of MT/dialogue. Induces a hypothesis space H useful for many NLP tasks.
- Universal criteria: (1) works across document size/number/label type; (2) single architecture + training process; (3) no custom feature engineering/preprocessing; (4) no additional in-domain docs/labels.

### Base model: AWD-LSTM (Merity et al. 2017)
- Regular 3-layer LSTM, NO attention/shortcut connections. ASGD Weight-Dropped LSTM.
- DropConnect on hidden-to-hidden recurrent weight matrices (same mask across timesteps). Variational dropout elsewhere. Activation regularization (AR), temporal activation reg (TAR), randomized-length BPTT. NT-ASGD optimizer.
- SOTA perplexity: 57.3 PTB, 65.8 WikiText-2.

### Three stages
- (a) General-domain LM pretraining: WikiText-103 (Merity 2016), 28,595 Wikipedia articles, 103M words. Done once.
- (b) Target-task LM fine-tuning: adapt to target distribution. Uses discriminative fine-tuning + STLR. Converges fast, robust even small datasets.
- (c) Target-task classifier fine-tuning: gradual unfreezing + discr + STLR.

### Discriminative fine-tuning
- Different layers capture different info (Yosinski 2014) -> fine-tune to different extents.
- Regular SGD: theta_t = theta_{t-1} - eta * grad_theta J(theta).
- Split theta into {theta^1,...,theta^L}, LRs {eta^1,...,eta^L}. Update per layer: theta_t^l = theta_{t-1}^l - eta^l * grad_{theta^l} J(theta).
- Empirically: choose eta^L (last layer) by tuning last layer only, then eta^{l-1} = eta^l / 2.6 for lower layers.

### Slanted triangular learning rates (STLR)
- Quickly converge to suitable region then refine. Linear increase then linear decay.
- cut = floor(T * cut_frac)
- p = t/cut if t < cut, else 1 - (t-cut)/(cut*(1/cut_frac - 1))
- eta_t = eta_max * (1 + p*(ratio-1)) / ratio
- T = #training iterations (epochs * updates/epoch). cut_frac = fraction increasing LR. ratio = how much smaller lowest LR is than eta_max.
- Defaults: cut_frac=0.1, ratio=32, eta_max=0.01.
- Modifies triangular LR (Smith 2017) with SHORT increase + LONG decay (key). Compared vs aggressive cosine annealing (Loshchilov 2017); one cycle best.

### Classifier fine-tuning details
- Augment LM with TWO additional linear blocks. Each: batch norm + dropout; ReLU for intermediate, softmax at last. ONLY these params learned from scratch. First linear layer takes pooled last hidden states.
- Classifier hidden layer size 50.

### Concat pooling
- Signal often in few words anywhere in doc. Last hidden state alone loses info.
- h_c = [h_T, maxpool(H), meanpool(H)], H = {h_1,...,h_T}, [] concatenation. h_T = last timestep hidden.

### Gradual unfreezing
- Fine-tuning all layers at once risks catastrophic forgetting.
- Unfreeze from LAST layer (least general knowledge). Unfreeze last layer, fine-tune all unfrozen for 1 epoch. Then unfreeze next-lower frozen layer, repeat, until all layers fine-tuned till convergence at last iteration.
- Similar to chain-thaw (Felbo 2017) but ADD a layer at a time to thawed set, rather than training single layer at a time.

### BPT3C (BPTT for Text Classification)
- Divide document into fixed-length batches of size b. At start of each batch, init model with final state of previous batch. Track hidden states for mean/max pooling. Gradients back-propped to batches whose hidden states contributed to final prediction. Variable-length BPTT sequences.

### Bidirectional LM
- Pretrain both forward + backward LM. Fine-tune a classifier per LM independently (BPT3C), AVERAGE the classifier predictions. ~0.5-0.7 boost.

### Hyperparameters (verified)
- AWD-LSTM: embedding size 400, 3 layers, 1150 hidden activations per layer, BPTT batch size 70.
- Dropout: 0.4 to layers, 0.3 to RNN layers, 0.4 to input embedding layers, 0.05 to embedding layers, weight dropout 0.5 to RNN hidden-to-hidden matrix.
- Classifier hidden layer 50.
- Adam beta1=0.7 (not default 0.9), beta2=0.99 (like Dozat 2017).
- Batch size 64. Base LR 0.004 (LM fine-tune), 0.01 (classifier fine-tune). Tune #epochs per task.
- TREC-6: LM fine-tune 15 epochs (no overfit). Classifier ~50 epochs default.
- Special tokens for upper-case words, elongation, repetition.

### Ablation findings (about transfer techniques, diagnostic)
- "Last" (CV standard, fine-tune only last layer) severely underfits in NLP (shallow models).
- Full fine-tuning alone -> catastrophic forgetting: lowest error early (1 epoch on IMDb) then error increases as pretraining knowledge lost.
- Discr + STLR more stable, no catastrophic forgetting.

## Canonical implementation: fast.ai / salesforce awd-lstm-lm. ULMFiT in fastai (text.learner): AWD_LSTM, discriminative LRs via layer groups, slanted triangular / one-cycle, gradual unfreezing via freeze_to.

## Design decisions -> why
- LM as source task: unlimited unlabeled data, captures general language, adaptable; ImageNet analog for NLP.
- AWD-LSTM (vanilla LSTM + dropout, no attention): show transfer works with a generic arch (universality), heavy regularization (DropConnect) prevents LM overfitting.
- Discriminative fine-tuning (eta/2.6 per lower layer): lower layers hold general features (transfer well, shouldn't move much), higher layers task-specific (need bigger updates). 2.6 empirically.
- STLR (short warmup, long decay): warmup finds a good region without destroying pretrained weights; long decay refines. Vs cosine: STLR better on small data.
- Gradual unfreezing (last->first): unfreezing all at once overwrites general low-level features (forgetting); thaw top-down so high-level (task-specific) adapts first while low-level stays protected, then progressively release.
- Concat pooling [h_T, max, mean]: classification cue can be anywhere; last state alone forgets long docs; max captures salient features, mean captures overall.
- Two linear blocks from scratch + batchnorm/dropout: standard CV classifier head; only new params.
- BPT3C: makes BPTT feasible over long documents while keeping pooling.
- Backward LM ensemble: forward LM misses right-context cues; averaging both directions helps.
- Adam beta1=0.7: lower momentum suits the noisy fine-tuning / cyclical LR (Dozat).
```
