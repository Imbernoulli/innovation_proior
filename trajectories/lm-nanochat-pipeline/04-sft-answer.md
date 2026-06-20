**Problem (from step 3).** Midtraining installed the format but the assistant is rough: ChatCORE 0.073, GSM8K 0.025, multiple-choice only modestly above random. The model has the protocol; it lacks polish and the right *distribution* of practice. Run a dedicated supervised-finetuning stage on high-quality conversations in exactly the test-time format, with the mixture weighted toward the eval targets.

**Key idea.** Same mechanism as midtraining (masked next-token loss on assistant tokens, same renderer, same MuonAdamW) — the lever is the **mixture and the training refinements**:
- **SmolTalk** (~460K conversations) as the conversational backbone (makes it an assistant, not a quiz-taker);
- **epoch-weighted eval targets:** MMLU ×3 (sharpen single-letter multiple choice), GSM8K ×4 (the hardest, most epoch-hungry tool-use skill gets the most reinforcement);
- **spelling tasks** (Simple-Spelling ~200K, Spelling-Bee ~80K) to fix the BPE-tokenization blind spot ("count the r's in strawberry") by teaching token→character decomposition;
- a small **identity** set (~1000, ×2 epochs) for a consistent persona.
- **Warm-start the optimizer** from the pretrained momentum buffers, but *reset the LRs* to fresh SFT values (pretraining drove them to ~0); a warmup→constant→warmdown schedule; **best-fit BOS-aligned packing** that places the largest conversation that fits entirely and *pads* (never crops) the remainder, keeping every conversation intact and every row starting at `<|bos|>`.

**Why it works.** At this scale the data mixture is the lever: cleaner conversations weighted toward the measured tasks firm up the multiple-choice protocol and reinforce the tool-use motion over more epochs, lifting every report-card row. The warm-started optimizer resumes with accumulated momentum instead of a cold noisy ramp; best-fit-pad packing avoids the malformed truncated targets that cropping would create.

**Change / code.** The SFT mixture and the optimizer warm-start (load momentum, reset LRs):

```python
train_tasks = [
    SmolTalk(split="train"),                                  # 460K rows of general conversations
    CustomJSON(filepath=identity_conversations_filepath),     # 1000 rows of synthetic identity conversations
    CustomJSON(filepath=identity_conversations_filepath),     # 2 epochs of these
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],  # 100K rows/epoch (Multiple Choice)
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],          # 8K rows/epoch (Math + Tool Use)
    SimpleSpelling(size=200000, split="train"),               # spell the word 'apple'
    SpellingBee(size=80000, split="train"),                   # how many 'r' in 'strawberry'?
]   # defaults: mmlu_epochs=3, gsm8k_epochs=4
train_dataset = TaskMixture(train_tasks)
```

```python
# Optimizer warm-start: load pretrained momentum buffers, but RESET the LRs (pretrain warmdown took them to ~0)
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
                                  matrix_lr=args.matrix_lr, weight_decay=0.0)
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, ...)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)            # brings momentum buffers AND stale (~0) LRs
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr                            # restore our fresh SFT LRs
for group in optimizer.param_groups:                         # then run at a fraction of base LR
    group["lr"] = group["lr"] * args.init_lr_frac            # init_lr_frac = 0.8
    group["initial_lr"] = group["lr"]
```

```python
# Best-fit packing: place the largest conversation that fits ENTIRELY; pad (never crop) the remainder.
best_idx, best_len = -1, 0
for i, (conv, _) in enumerate(conv_buffer):
    if len(conv) <= remaining and len(conv) > best_len:
        best_idx, best_len = i, len(conv)
if best_idx >= 0:
    conv, conv_mask = conv_buffer.pop(best_idx)
    row.extend(conv); mask_row.extend(conv_mask)
else:
    row.extend([bos_token] * remaining); mask_row.extend([0] * remaining)  # pad, don't crop
# non-assistant + padding positions -> ignore_index (-1) in the targets
targets[mask_targets == 0] = -1
```
