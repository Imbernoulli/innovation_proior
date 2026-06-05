# Byte Pair Encoding for Subword Segmentation

## Problem

Neural machine translation works over a fixed vocabulary (~30k–50k word types), but
translation is open-vocabulary: names, compounds, loanwords, and inflected forms keep
appearing unseen and collapse to `UNK`. Back-off dictionaries are a leaky, non-end-to-end
patch that assumes 1-to-1 word alignment and cannot transliterate across alphabets.
Goal: a fixed, compact symbol vocabulary that can encode *any* string (no true OOV)
without exploding sequence length, so the network can translate and even *generate*
unseen words from sub-word units.

## Key idea

Adapt byte pair encoding — a compression algorithm that repeatedly replaces the most
frequent adjacent pair of symbols with a new symbol — to word segmentation. Represent
each word as a sequence of characters plus an end-of-word marker; iteratively merge the
most frequent adjacent symbol pair. Each merge adds exactly one symbol, so the number of
merge operations is the single hyperparameter and directly sets the final vocabulary
size (= initial character alphabet + number of merges). Frequent strings (common
morphemes, whole words) get merged into single symbols; rare words bottom out as smaller
pieces down to characters — a granularity gradient that keeps frequent text short and
guarantees any string is representable (the alphabet is always retained). The learned
object is the *ordered list of merges*; test-time segmentation replays it on any word.

## Final algorithm

1. Build a word→frequency table from the training corpus. Represent each word as a tuple
   of characters with `</w>` appended to the last character (so word-final and
   word-internal symbols are distinct and the original tokenization is recoverable).
2. Repeat `num_merges` times: count every adjacent symbol pair (weighted by word
   frequency); take the most frequent pair `(A,B)`; replace every occurrence with `AB`;
   record the merge. Pairs never cross word boundaries.
3. To segment any (possibly unseen) word at test time: split into characters + `</w>`,
   then apply the learned merges in learned order until none applies.
4. Optionally learn merges jointly on the union of source and target vocabularies (joint
   BPE) for consistent cross-lingual segmentation; for differing alphabets, transliterate
   to a common script to learn merges, then map them back.

Feed the resulting symbol sequences to the unchanged embedding → encoder-decoder →
softmax pipeline; the embedding table and output softmax are now sized by the compact
symbol vocabulary.

## Code

```python
import re, collections

def get_stats(vocab):
    """Frequency of each adjacent symbol pair, weighted by word frequency."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """Replace every adjacent occurrence of `pair` with the merged symbol."""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    # whitespace lookarounds ensure we merge this exact pair, not a substring
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        v_out[p.sub(''.join(pair), word)] = v_in[word]
    return v_out

def learn_bpe(word_freqs, num_merges):
    # words as space-separated chars + end-of-word marker, weighted by frequency
    vocab = {' '.join(list(w[:-1]) + [w[-1] + '</w>']): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # most frequent, deterministic tie-break
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges  # ordered list of learned operations

def segment_word(word, ranks):
    """ranks: {pair: learning_order_index}. Apply earliest-learned applicable merge first."""
    symbols = list(word[:-1]) + [word[-1] + '</w>']
    while len(symbols) > 1:
        pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
        cands = [(ranks[p], p) for p in pairs if p in ranks]
        if not cands:
            break
        a, b = min(cands)[1]
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(symbols[i]); i += 1
        symbols = out
    return symbols

def apply_bpe(corpus_lines, merges):
    ranks = {pair: i for i, pair in enumerate(merges)}
    return [' '.join(tok for w in line.split() for tok in segment_word(w, ranks))
            for line in corpus_lines]
```

In production the learner indexes which words contain each pair and updates pair
statistics incrementally after each merge (decrementing the broken pairs, incrementing
the new ones, optionally pruning rare pairs) for `O(corpus + num_merges)`-style cost;
the merges produced are identical to the simple loop above. The segmented text feeds
directly into the existing attention encoder-decoder, whose vocabulary is now the fixed
symbol set.
