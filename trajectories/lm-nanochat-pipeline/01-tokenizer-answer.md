**Problem (the foundation).** The model only ever sees integer token ids, and the cost of training scales with how many tokens each document becomes. Character-level vocab makes sequences too long (wasted FLOPs); word-level vocab is unbounded and brittle. I need a vocabulary that spends its slots adaptively — single tokens for common substrings, byte fallback for everything else — so frequent text compresses and nothing is ever out-of-vocabulary.

**Key idea.** A GPT-4-style **byte-level BPE** tokenizer: base alphabet is raw bytes (full coverage), then greedily merge the most frequent adjacent pair until the vocabulary hits **2^15 = 32768** (power of two for tensor cores; small enough that the embedding/lm_head don't bloat a ~100M–1B model). A regex pre-split confines merges to within word/number/punctuation chunks, with one deviation from GPT-4: group numbers at most **two** digits (`\p{N}{1,2}`) rather than three, which is the sweet spot at 32K vocab. Reserve nine special tokens up front — `<|bos|>` plus the conversation/tool channel (`user`, `assistant`, `python`, `output` start/end) — so finetuning never has to resize the vocabulary.

**Why it works.** Data-driven compression: common words collapse to one token (fewer FLOPs per document, longer effective context), rare words degrade gracefully to subword pieces, arbitrary bytes are always representable (no unknowns). Confining merges within regex chunks avoids ugly cross-boundary merges that hurt generalization. The `{1,2}` number grouping avoids wasting scarce 32K slots on rare 3-digit tokens. Reserving specials at train time keeps the embedding table fixed across all later stages.

**Change / code.** Train a 32768-vocab BPE on ~2B characters (Rust backend, GPT-4-style regex), capping each document at 10K chars; cache a token→bytes table so validation loss can be reported as vocab-invariant bits-per-byte. The split pattern and special-token list:

```python
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

```python
# tok_train.py: stream ~2B chars (each doc capped at 10K) through a GPT-4-style BPE
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)  # vocab_size = 32768
# cache token -> number-of-bytes, so val loss can be reported as vocab-invariant bits-per-byte
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0)                              # specials aren't counted
    else:
        token_bytes.append(len(token_str.encode("utf-8"))) # bytes this token decodes to
```
