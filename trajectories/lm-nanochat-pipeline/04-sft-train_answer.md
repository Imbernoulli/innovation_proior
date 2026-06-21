Midtraining got me a model that *can* play every game — it answers in turns, emits single letters, opens a Python block — but the numbers say it plays them roughly: ChatCORE 0.0730, GSM8K barely off the floor at 0.0250, the multiple-choice accuracies above random but not by much. The model has the format; what it lacks is *polish* and the right *distribution* of practice. Midtraining was a broad, coarse adaptation on whatever data was handy, optimized partly just for installing the format. Now I want a dedicated stage whose only job is to make the assistant as good as it can be on the things I am measured on, by training on high-quality conversations in exactly the format the model will face at test time, with the mixture *weighted* toward the eval targets.

I propose **SFT (supervised finetuning) on a curated, eval-weighted conversation mixture**, with a warm-started optimizer and best-fit-pad packing. The mechanism is unchanged from midtraining — masked next-token cross-entropy on the assistant's tokens, the same renderer, the same MuonAdamW — so the lever is the mixture itself, and at this scale the data mixture *is* the lever. The backbone is general conversational ability: I want a large, clean corpus of multi-turn instruction-following conversations so the assistant learns to be helpful, coherent, and on-format across the open-ended prompts a person would actually type. **SmolTalk** (~460K conversations) is exactly this and forms the backbone — it is what makes the model feel like an assistant rather than a quiz-taker.

On top of the backbone I add eval-targeted data, and here I get to *weight* by running several epochs of the most valuable sets. I cannot add an arbitrary amount of any one source without skewing the model, but multiple choice is a learnable protocol that a few epochs sharpen a lot, so I run the **MMLU** auxiliary-train split for about three epochs — enough to firm up the single-letter behavior and the option-reading. Math-with-tools is the hardest and most epoch-hungry skill, because the model must reliably open the tool, write a correct expression, consume the result, and chain to the answer, so I run **GSM8K** for more epochs than MMLU (about four). These exact counts are what a sweep settles; the qualitative point is that GSM8K gets the most reinforcement precisely because it is the weakest and most structured row.

Two more targeted ingredients teach a specific competence the base model is comically bad at because of tokenization. The model sees text as BPE tokens, not characters, so "spell the word 'apple'" or "how many r's are in 'strawberry'" are genuinely hard — the letters live *inside* a token the model cannot introspect. So I synthesize spelling data: a **Simple-Spelling** task (spell a word out letter by letter, ~200K rows) and a **Spelling-Bee** task (count occurrences of a letter, ~80K rows). These barely move the six headline metrics, but they fix an embarrassing and very visible failure mode and teach the model to decompose tokens into characters when asked. And I give the model an **identity**: out of the box it has no idea what it is, so a tiny ~1000-row set of synthetic identity conversations ("who are you", "who made you"), run a couple of epochs since it is small and I want it to stick, gives a consistent persona. So the mixture is SmolTalk as the backbone, a couple epochs of identity, ~3 epochs of MMLU, ~4 epochs of GSM8K, and the two spelling tasks, all rendered through the same conversation renderer with assistant-only masking; the validation set mirrors the same task families (SmolTalk test, MMLU slice, GSM8K slice) in roughly the training proportions, scored by val bpb.

What makes this actually better than just "more midtraining" are three training-loop refinements. First, I **warm-start the optimizer** from the pretrained MuonAdamW momentum buffers rather than starting cold, so the matrix updates resume with their accumulated momentum instead of a noisy ramp. But pretraining's warmdown drove the learning rates to ~0 by the end, so I must *not* inherit those: I load the momentum buffers, then explicitly reset the learning rates to fresh SFT values, or the model would not move at all. The LRs themselves I inherit from the checkpoint metadata (so the recipe scales with depth) and run at a fraction of base. Second, I give the short SFT run a proper schedule — a brief warmup, a constant plateau, and a warmdown over the final fraction (a warmdown ratio around 0.5 works best) — which squeezes a bit more out of the limited steps than a flat rate. Third, the **packing**: SFT conversations vary wildly in length and I need every training row to start at a conversation boundary (a `<|bos|>`) so the model always has clean context, but I do not want to *crop* and discard the tail of a long conversation, because cropping mid-conversation teaches truncated, malformed targets. So I use a best-fit packing dataloader: maintain a buffer of rendered conversations and, for each fixed-length row, greedily place the *largest* conversation that fits *entirely*; when nothing fits, **pad** the remainder with masked ignore-index targets rather than crop. That keeps 100% of every conversation intact and every row BOS-aligned, at the cost of a little padding — cheap, because conversations are short — and the padding positions and all non-assistant positions get the ignore-index, so the gradient is again purely from the assistant's tokens.

The bet is that the same model, given cleaner conversational data weighted toward the eval targets, with a warm-started optimizer and a proper schedule, climbs on every metric: the multiple-choice accuracies firm up, GSM8K improves as the tool-use motion gets reinforced over more epochs, HumanEval ticks up with cleaner coding conversations, and ChatCORE rises accordingly. The honest limit I hold open is that SFT is still purely *imitation* — the model learns to reproduce reference solutions, and on GSM8K the reference is a single chain of tool calls. Imitation can only go so far on a reasoning task with many valid solution paths and a binary payoff (right final number or not); pushing GSM8K further will need an objective that rewards *getting the answer right* over *reproducing one particular derivation*, which is the finale's job.

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
