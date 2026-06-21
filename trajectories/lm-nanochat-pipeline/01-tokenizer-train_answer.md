Before a single transformer weight exists I have to decide what a "token" even is, and on a fixed FLOP budget that decision quietly sets the ceiling on everything downstream. The model never sees characters; it sees a sequence of integer ids, and the cost of training scales with the *number* of those ids each document becomes. The two obvious extremes both fail. A character-level vocabulary of ~256 byte values has a tiny embedding table and never an unknown token, but it shatters every word into many tokens, so a 2048-token context holds only a few hundred words and the model wastes capacity relearning that "t-h-e" is a word — and the long sequences cost FLOPs directly, since attention is quadratic in length and I pay per token regardless. A word-level vocabulary gives short sequences but an unbounded, brittle vocabulary: every rare word, number, and typo is its own type, the embedding table explodes, and anything unseen at train time is a hard unknown at test time. Neither is usable, so the substrate the whole pipeline rides on has to be something that spends its slots *adaptively*.

I propose a GPT-4-style **byte-level byte-pair-encoding (BPE) tokenizer** with a $2^{15} = 32768$ vocabulary, a regex pre-split that confines merges to word/number/punctuation chunks, and nine reserved special tokens for the eventual chat and tool channels. The base of the alphabet is the raw bytes — that guarantees full coverage, any string at all can be encoded, so there is never an out-of-vocabulary failure. BPE then repeatedly scans a large corpus, finds the most frequent adjacent pair of tokens, and merges it into a new token; each merge buys back one token of length on the most common remaining pattern in the data, and the learned ordered merge list *is* the tokenizer. Run it until the vocabulary hits the target size and frequent words collapse to a single token (fewer forward steps to see the same text), rare words degrade gracefully into a few subword pieces, and arbitrary bytes remain always representable. The compression is entirely data-driven, and the FLOP savings are real because fewer tokens per document means fewer forward steps.

The size is a genuine tradeoff, not "bigger is better." A larger vocabulary compresses text into fewer tokens, which is good, but it inflates the two matrices that scale with vocabulary — the input embedding table and the output projection (the lm_head) — and at this tiny model scale those are a meaningful fraction of both the parameters and the final-layer matmul. It also spreads the training signal thinner: each token type is seen less often, so rare tokens are learned worse. The GPT-4 tokenizer uses ~100K tokens, but that is tuned for a model orders of magnitude larger where the embedding tables are a rounding error. For a model in the ~100M–1B range I want the sweet spot, and $2^{15}=32768$ is it: a power of two, so the padded matrices stay friendly to tensor cores, and large enough to compress English well without bloating the embedding/unembedding of a small model.

The split pattern matters more than it looks. Raw BPE will merge across *any* adjacent pair, learning ugly merges that span a word boundary into a following space, or glue digits to letters, which hurts generalization. The GPT-4-style fix is to pre-split the text with a regex into chunks — words, runs of whitespace, punctuation, numbers — and only ever merge *within* a chunk. The pattern handles contractions like "'s", "'ll", "'re", keeps a leading space attached to a word so " the" is one token (matching how words appear mid-sentence), and isolates runs of digits and punctuation. I make one deliberate deviation from GPT-4 for *this* scale. GPT-4 groups numbers in runs of up to three digits (`\p{N}{1,3}`), so each of "100"–"999" can be a single token; at a 32K vocabulary that is wasteful, because three-digit tokens are individually rare, eat scarce slots, and get little training signal while still leaving four-digit numbers fragmented. Grouping at most *two* digits (`\p{N}{1,2}`) spends far fewer slots on numbers and gives the model a cleaner, denser number representation. One digit at a time is too fine — it needlessly lengthens every number — so two is the balance; I verified that two is the sweet spot at 32K, with one digit a bit worse and three worse still.

I also reserve the special tokens *now*, baked into the tokenizer, even though they go unused until finetuning — because adding them later would force a resize of the embedding table and the lm_head. The base model trains on plain documents, each prefixed with a single `<|bos|>` delimiter that marks document boundaries so the model can always look back to the start of the current document. But the eventual goal is an *assistant*, which means rendering structured conversations into the same flat token stream, so I carve out the conversation control tokens up front: `<|user_start|>`/`<|user_end|>` and `<|assistant_start|>`/`<|assistant_end|>`, plus a tool channel — `<|python_start|>`/`<|python_end|>` for when the assistant calls a Python REPL and `<|output_start|>`/`<|output_end|>` for the REPL's reply. These are inert during pretraining; they exist so midtraining and SFT can use them without ever touching the vocabulary. The tokenizer is trained on the vocabulary *minus* the specials, and the specials are assigned the top ids.

For implementation, the heavy BPE training runs in Rust (via the HuggingFace `tokenizers` library, GPT-4-style) over ~2B characters of the pretraining corpus, capping each document at ~10K characters so no single huge document dominates the merge statistics and capping the total at ~2B characters so the tokenizer is quick to build relative to the pretraining run. After saving the merges I precompute one small lookup I will need for evaluation: for each token id, how many UTF-8 *bytes* it decodes to. That lets me report validation loss not as mean cross-entropy per token — which is unfair across tokenizers, since one that uses fewer, longer tokens looks artificially better per token — but as **bits per byte**, cross-entropy divided by bytes, which is invariant to the tokenization and is the smooth number I track for the base model in the next stage. The exactness floor is that encode-then-decode round-trips perfectly, including on numbers, contractions, punctuation, and non-Latin Unicode and emoji; a tokenizer that does not round-trip is broken regardless of its compression.

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
