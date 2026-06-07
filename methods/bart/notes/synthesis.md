# BART synthesis (verified vs arXiv 1910.13461 source + fairseq bart/model.py)

## Verified core
- BART = denoising autoencoder for pretraining seq2seq models. (1) corrupt text with arbitrary noising function; (2) learn to reconstruct original text.
- Standard Transformer NMT architecture (bidirectional encoder over corrupted text + left-to-right autoregressive decoder).
- Generalizes BERT (bidirectional encoder), GPT (L-to-R decoder), and other schemes. Extreme case (all source info lost) = language model.
- Pretraining loss: negative log-likelihood of original document = cross-entropy between decoder output and original.

### Architecture (verified)
- Standard seq2seq Transformer (Vaswani 2017), but: ReLU -> GeLU; init params from N(0, 0.02) (following GPT).
- Base: 6 encoder + 6 decoder layers. Large: 12 + 12. Hidden 768 (base) / 1024 (large).
- vs BERT: (1) each decoder layer additionally does cross-attention over final encoder hidden layer; (2) BERT uses an extra FFN before word prediction, BART does NOT.
- ~10% more params than equivalently sized BERT.

### Noising transformations (verified, Figure 2; composable)
- Token Masking (BERT-style): random tokens replaced with [MASK].
- Token Deletion: random tokens deleted; model must decide WHICH positions are missing.
- Text Infilling: number of spans sampled, span lengths ~ Poisson(lambda=3); each span replaced with a SINGLE [MASK] token. 0-length spans = insertion of [MASK]. (Inspired by SpanBERT but SpanBERT uses clamped geometric and replaces with sequence of [MASK] of same length.) Teaches model to predict HOW MANY tokens missing from a span.
- Sentence Permutation: document split into sentences by full stops, shuffled randomly.
- Document Rotation: pick token uniformly at random, rotate document to begin with it; trains model to find start of doc.

### Fine-tuning (verified)
- Sequence classification: same input fed to BOTH encoder and decoder; final hidden state of FINAL DECODER TOKEN -> new multi-class linear classifier. Related to BERT's CLS but token added at the END so its representation can attend to the complete decoded input.
- Token classification (e.g. SQuAD answer endpoints): full document into encoder + decoder; top decoder hidden state per word -> classify token. SQuAD: classifiers predict start & end indices.
- Sequence generation: autoregressive decoder fine-tuned directly. encoder input = input seq; decoder generates output autoregressively. (abstractive QA, summarization.)
- MNLI: concat two sentences + appended EOS, to encoder+decoder; representation of EOS token classifies relation.
- Machine translation: replace BART's encoder EMBEDDING layer with a new randomly-initialized source encoder (can use disjoint/foreign vocabulary). Train end-to-end so new encoder maps foreign words to input BART can denoise to English. Uses entire BART (enc+dec) as a pretrained target-side decoder. Two-step training (both backprop cross-entropy from BART output): step 1 freeze most BART, update only new source encoder + BART positional embeddings + self-attention input projection of BART encoder's first layer; step 2 train all params for small #iters.

### Ablation: comparison objectives (replicated within BART framework, base size 6+6, hidden 768, 1M steps on books+Wikipedia)
- Language Model (GPT-like): L-to-R Transformer LM = BART decoder without cross-attention.
- Permuted LM (XLNet-like): sample 1/6 tokens, generate in random order autoregressively. (no relative pos emb / segment attention.)
- Masked LM (BERT-like): replace 15% with [MASK], predict independently.
- Multitask Masked LM (UniLM-like): MLM with additional self-attention masks chosen randomly: 1/6 left-to-right, 1/6 right-to-left, 1/3 unmasked, 1/3 with first 50% tokens unmasked + L-to-R mask for remainder.
- Masked Seq2Seq (MASS-like): mask a span containing 50% of tokens, train seq2seq to predict masked tokens.
- For Permuted/Masked/Multitask LM: use two-stream attention (XLNet) for efficient output likelihoods (diagonal self-attention mask on output to predict L-to-R).
- Two fine-tune options: (1) standard seq2seq (source->encoder, target=decoder out); (2) add source as prefix to target in decoder, loss only on target. Former works better for BART, latter for others.

### Ablation findings (about these objectives — pre-method, diagnostic)
- Performance varies a lot across tasks (LM best on ELI5, worst on SQuAD).
- Token masking crucial: rotation/permutation alone poor. Successful methods use deletion or masking or self-attn masks. Deletion > masking on generation.
- Left-to-right pretraining improves generation (MLM and Permuted LM lack it, worse on generation).
- Bidirectional encoders crucial for SQuAD (L-to-R decoder alone poor; future context needed). BART matches with half the bidirectional layers.
- Pure LM best on ELI5 (output loosely constrained by input).
- Text infilling most consistently strong.

### Large-scale setup (verified)
- Large: 12+12 layers, hidden 1024. Batch 8000, 500000 steps (following RoBERTa). GPT-2 byte-pair encoding. 
- Noise: text infilling + sentence permutation. Mask 30% of tokens in each document, permute all sentences.
- Disable dropout for final 10% of training steps.
- Same data as RoBERTa: 160GB news/books/stories/web.
- Generation fine-tuning: label-smoothed cross-entropy (smoothing 0.1). Generation: beam size 5, remove duplicated trigrams in beam search, tune min-len/max-len/length-penalty on validation.
- MT: WMT16 RO-EN + back-translation data, 6-layer transformer source encoder, beam width 5, length penalty alpha=1.

## Canonical implementation (fairseq fairseq/models/bart/model.py)
- BARTModel(TransformerModel): seq2seq encoder-decoder; BERT-style init; classification_heads dict.
- forward(src_tokens, src_lengths, prev_output_tokens, features_only=False, classification_head_name=None,...).
- Classification: sentence_representation = x[src_tokens.eq(eos), :].view(B, -1, H)[:, -1, :] -> last EOS position of decoder output.
- BARTClassificationHead(input_dim, inner_dim, num_classes, pooler_dropout): dropout -> dense Linear(input,inner) -> tanh -> dropout -> out_proj Linear(inner,num_classes).

## Design decisions -> why
- Seq2seq enc-dec (not BERT-only or GPT-only): decouples input from output, so input can be corrupted by ANY length-changing transformation; bidirectional encoder gives BERT-style understanding, autoregressive decoder gives GPT-style generation. Single model good for both.
- GeLU + N(0,0.02) init: match GPT conventions (smooth activation, calibrated init for deep transformers).
- Decoder cross-attention: needed for enc-dec; lets decoder condition on full encoded corrupted input.
- Drop BERT's pre-prediction FFN: the autoregressive decoder already builds rich features; the extra projection is redundant.
- Text infilling (Poisson lambda=3, single mask per span incl 0-length): single mask hides span LENGTH, forcing the model to predict how many tokens are missing (harder, more global reasoning than fixed-length SpanBERT). 0-length spans teach insertion.
- Sentence permutation + infilling for large model: infilling is the consistent winner; permutation adds global reordering reasoning that larger models can exploit (helped CNN/DM).
- Classification token at END of decoder: so its representation attends over the COMPLETE decoded sequence (a decoder is causal; an end token sees everything).
- MT source encoder swap: BART is a pretrained English denoiser; treat foreign->English as "denoising foreign-encoded input to English"; only need a small new source encoder + monolingual English pretraining (no need to pretrain on every language, unlike XLM/MASS).
- Two-step MT training: protect pretrained BART early (freeze most, adapt only the interface — positional emb + first-layer attention input proj + new encoder), then gently tune all.
- Disable dropout final 10%: let model fully fit data at the end (underfitting regime).
- 30% mask large-scale: heavier corruption for the larger-capacity model.
```
