# RoBERTa synthesis (verified against arXiv 1907.11692 source + fairseq)

## Verified facts (from arXiv source 1907.11692)
- Replication study of BERT pretraining. Finding: BERT was significantly UNDERTRAINED.
- 4 modifications: (1) train longer, bigger batches, more data; (2) remove NSP; (3) train on longer sequences (full-length 512, no short-seq injection); (4) dynamic masking.
- New dataset CC-News collected.

### BERT background recap (Section 2, from paper)
- Input: [CLS] x_1..x_N [SEP] y_1..y_M [EOS], M+N < T.
- Architecture: L layers, A heads, H hidden. (BERT-base: L=12, H=768, A=12, 110M)
- MLM: 15% tokens selected; of those 80% -> [MASK], 10% unchanged, 10% random token. Cross-entropy on masked.
- Original BERT: static mask (done once in preprocessing). Data duplicated 10x so each seq seen with 10 masks over 40 epochs => each mask seen 4 times.
- NSP: binary classification, do two segments follow each other. Positive = consecutive, negative = different docs, 50/50.
- Optimization (BERT): Adam beta1=0.9, beta2=0.999, eps=1e-6, L2 weight decay 0.01. LR warmup over first 10k steps to peak 1e-4 then linear decay. Dropout 0.1 all layers + attention. GELU. S=1M updates, B=256 sequences, max length T=512.
- Data (BERT): BookCorpus + English Wikipedia = 16GB.

### RoBERTa changes (verified)
- Reimplemented in fairseq.
- Adam eps tuned; beta2 = 0.98 for stability with large batches (KEY: beta2=0.98 not 0.999).
- Full-length sequences only; no random short-seq injection; no reduced-length first 90%.
- Mixed precision on DGX-1 (8x32GB V100).
- Data: 5 corpora >160GB total: BookCorpus+Wiki (16GB), CC-News (76GB filtered, 63M articles Sept2016-Feb2019, news-please), OpenWebText (38GB, Reddit >=3 upvotes), Stories (31GB, CommonCrawl Winograd-style).

#### Static vs dynamic masking
- Dynamic masking = generate masking pattern every time a sequence is fed. Comparable or slightly better than static. Use dynamic.

#### Input format / NSP (4 settings compared)
- segment-pair+nsp (original BERT): pair of segments, each multiple sentences, combined < 512, NSP loss.
- sentence-pair+nsp: pair of single sentences; much shorter; increase batch size to match token count; keep NSP.
- full-sentences: packed with full sentences sampled contiguously from one or more docs, <=512; may cross doc boundaries; add extra separator token between docs; REMOVE NSP.
- doc-sentences: like full-sentences but may NOT cross doc boundaries; near end of doc may be shorter, so dynamically increase batch size; REMOVE NSP.
- Findings: single sentences hurt (can't learn long-range deps). Removing NSP matches or slightly improves (contra BERT). doc-sentences slightly better than full-sentences. But doc-sentences => variable batch sizes, so use full-sentences for the rest.

#### Large batches
- BERT: 1M steps @ batch 256. Equivalent compute: 125K steps @ 2K, or 31K steps @ 8K (via gradient accumulation).
- Large batches improve MLM perplexity AND end-task accuracy. Use 8K sequences. (You et al. 2019 went to 32K.)
- NMT precedent: Ott et al. 2018 — large batches improve speed + end-task when LR increased appropriately.

#### Text encoding (BPE)
- BPE (Sennrich 2016): subword units from statistical analysis. Vocab 10K-100K.
- Original BERT: character-level BPE vocab 30K, after heuristic tokenization preprocessing.
- GPT-2 (Radford 2019): byte-level BPE — bytes as base units, vocab 50K, encodes ANY text, no "unknown" tokens, no preprocessing.
- RoBERTa uses byte-level BPE 50K. Adds ~15M (base) / ~20M (large) params. Early experiments: slight differences, byte-BPE slightly worse on some tasks, but universal encoding worth it.

### Aggregation
- RoBERTa = dynamic masking + full-sentences-no-NSP + large batches (8K) + byte-level BPE 50K.
- Large config = BERT-large arch: L=24, H=1024, A=16, 355M params.
- Trained 100K steps over BookCorpus+Wiki (comparable to BERT). 1024 V100 GPUs ~1 day.
- Then +3 datasets (160GB), same 100K steps.
- Then longer: 100K -> 300K -> 500K steps. Best = 500K steps over all 5 datasets.

### Pretraining hyperparameters (Appendix Table)
RoBERTa-large / RoBERTa-base:
- Layers 24/12; Hidden 1024/768; FFN inner 4096/3072; Heads 16/12; head size 64/64.
- Dropout 0.1; Attn dropout 0.1.
- Warmup steps 30k/24k; Peak LR 4e-4/6e-4; Batch 8k/8k; Weight decay 0.01.
- Max steps 500k/500k; LR decay Linear.
- Adam eps 1e-6; beta1 0.9; beta2 0.98; Gradient clipping 0.0.

### Finetuning hyperparameters
- RACE: LR 1e-5, batch 16, wd 0.1, 4 epochs, warmup ratio 0.06.
- SQuAD: LR 1.5e-5, batch 48, wd 0.01, 2 epochs, warmup 0.06.
- GLUE: LR {1e-5,2e-5,3e-5}, batch {16,32}, wd 0.1, 10 epochs, warmup 0.06.
- GLUE single-task dev: median over 5 random inits; early stopping. SQuAD v2.0: add binary answerable classifier, sum classification + span loss.
- RACE: concat each candidate answer with question+passage; encode 4 sequences; pass [CLS] reps through FC to predict; truncate QA pairs > 128, passage so total <= 512.

## Canonical implementation (fairseq fairseq/models/roberta/model.py)
- RobertaModel(FairseqEncoderModel): __init__(args, encoder); apply(init_bert_params); classification_heads = nn.ModuleDict.
- forward(src_tokens, features_only, return_all_hiddens, classification_head_name, **kwargs).
- RobertaEncoder(FairseqEncoder): build_embedding, build_encoder (TransformerSentenceEncoder), build_lm_head.
- RobertaLMHead(embed_dim, output_dim, activation_fn, weight): dense Linear(embed_dim,embed_dim) -> activation (gelu) -> LayerNorm -> project with tied weight + bias. Forward: x = dense(features); x = gelu(x); x = layer_norm(x); x = x @ weight.T + bias.
- RobertaClassificationHead(input_dim, inner_dim, num_classes, activation_fn, pooler_dropout): takes features[:,0,:] (CLS/<s>); dropout -> dense Linear(input_dim,inner_dim) -> tanh -> dropout -> out_proj Linear(inner_dim,num_classes).
- init_bert_params: standard BERT init (normal, std 0.02).
- GPT-2 BPE: byte-level, encoder.json + vocab.bpe, 50K vocab.

## Design decisions -> why
- Remove NSP: BERT claimed NSP helped, but it conflated removing-loss with keeping segment-pair format. When you keep contiguous full-sentence input (long-range context) and just drop the NSP head, downstream is matched/improved. The loss was redundant once the input gives long contiguous context. Single-sentence input is what actually hurts (no long-range deps) — BERT mis-attributed that to NSP.
- Dynamic masking: static mask wastes the model's exposure (same mask 4x); regenerating per-feed gives more varied supervision, crucial when training longer / more data. Negligible cost.
- beta2=0.98: with 8K batches, gradient estimate is far less noisy; default 0.999 averages too long a window and causes instability/slow adaptation of v_hat at large batch; shorter second-moment window (0.98) stabilizes.
- Large batch 8K + higher LR: more accurate gradient per step lets you raise LR (linear/sqrt scaling); fewer, larger steps with better parallelism. Improves both optimization and generalization here.
- Byte-level BPE 50K: universal (no UNK on any unicode/emoji/multilingual), no language-specific preprocessing, single tokenizer for diverse 160GB web text. Slight downstream cost but worth universality + clean pipeline.
- Train longer / more data: BERT undertrained. Held arch fixed, scaled steps 100K->500K and data 16->160GB; monotonic gains, no overfitting => the objective was never the bottleneck, the budget was.
- Gradient clipping 0.0 (off): with stable Adam (beta2=0.98) and warmup, clipping unnecessary.
- LR warmup 6% finetune / linear decay to 0: standard transfer-stability; warmup avoids early large-update divergence from random head.
```
