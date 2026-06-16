# Byte Pair Encoding for Subword Segmentation

## Problem

Neural machine translation works over a fixed vocabulary (~30k–50k word types), but
translation is open-vocabulary: names, compounds, loanwords, and inflected forms keep
appearing unseen and collapse to `UNK`. Back-off dictionaries are a leaky, non-end-to-end
patch that assumes 1-to-1 word alignment and cannot transliterate across alphabets.
Goal: a fixed, compact symbol vocabulary that can encode unseen words from known
lower-level symbols, with caveats for truly unseen characters and fallbacks for filtered
segments, without exploding sequence length, so the network can translate and even
*generate* unseen words from sub-word units.

## Key idea

Adapt byte pair encoding — a compression algorithm that repeatedly replaces the most
frequent adjacent pair of symbols with a new symbol — to word segmentation. Represent
each word as a sequence of characters plus an end-of-word marker, folded onto the final
character in the version 0.2 implementation; iteratively merge the most frequent
adjacent symbol pair. In the core algorithm, each merge adds exactly one symbol, so the
number of merge operations directly sets the final vocabulary size (= initial character
alphabet + number of merges). The production learner also has a minimum-frequency early
stop, but the merge-count knob is the intended granularity control. Frequent strings
(common morphemes, whole words) get merged into single symbols; rare words bottom out as
smaller pieces down to characters — a granularity gradient that keeps frequent text short
and keeps known characters as the backstop. The learned object is the *ordered list of
merges*; test-time segmentation replays it on any word, then strips the end marker and
marks non-final pieces for detokenization.

## Final algorithm

1. Build a word→frequency table from the training corpus. Represent each word as a tuple
   of characters plus an end marker; the implementation folds `</w>` onto the final
   character so word-final and word-internal symbols are distinct and the original
   tokenization is recoverable.
2. Repeat up to `num_merges` times: count every adjacent symbol pair (weighted by word
   frequency); take a most frequent pair `(A,B)`; in the production learner, break if
   no pair reaches `min_frequency`; replace every occurrence with `AB`; record the merge.
   Pairs never cross word boundaries.
3. To segment any (possibly unseen) word at test time: split into characters + `</w>`,
   apply the learned merges in learned order until none applies, strip the end marker,
   and append a continuation separator such as `@@` to every non-final piece.
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

def learn_bpe(word_freqs, num_merges, min_frequency=2):
    # version 0.2: fold the end-of-word marker onto the final character
    vocab = {' '.join(list(w[:-1]) + [w[-1] + '</w>']): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # most frequent, deterministic tie-break
        if pairs[best] < min_frequency:
            break
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges  # ordered list of learned operations

def encode_word(word, ranks):
    """ranks: {pair: learning_order_index}. Apply earliest-learned applicable merge first."""
    symbols = list(word[:-1]) + [word[-1] + '</w>']
    while len(symbols) > 1:
        cands = [(ranks[pair], i, pair)
                 for i, pair in enumerate(zip(symbols, symbols[1:]))
                 if pair in ranks]
        if not cands:
            break
        _, _, (a, b) = min(cands)
        positions = [i for _, i, pair in cands if pair == (a, b)]
        out, start = [], 0
        for pos in positions:
            if pos < start:
                continue
            out.extend(symbols[start:pos])
            out.append(a + b)
            start = pos + 2
        out.extend(symbols[start:])
        symbols = out
    if symbols[-1] == '</w>':
        symbols = symbols[:-1]
    elif symbols[-1].endswith('</w>'):
        symbols[-1] = symbols[-1][:-4]
    return symbols

def recursive_split(segment, reverse, vocab, separator, final=False):
    """Reverse merges until the printed segment is in vocabulary or cannot split."""
    key = segment + '</w>' if final else segment
    if key not in reverse:
        return [segment]
    left, right = reverse[key]
    if final:
        right = right[:-4]
    out = []
    if left + separator in vocab:
        out.append(left)
    else:
        out.extend(recursive_split(left, reverse, vocab, separator, False))
    right_key = right if final else right + separator
    if right_key in vocab:
        out.append(right)
    else:
        out.extend(recursive_split(right, reverse, vocab, separator, final))
    return out

def check_vocab_and_split(pieces, reverse, vocab, separator):
    """`vocab` contains printed tokens: non-final as `piece@@`, final as `piece`."""
    out = []
    for piece in pieces[:-1]:
        if piece + separator in vocab:
            out.append(piece)
        else:
            out.extend(recursive_split(piece, reverse, vocab, separator, False))
    if pieces[-1] in vocab:
        out.append(pieces[-1])
    else:
        out.extend(recursive_split(pieces[-1], reverse, vocab, separator, True))
    return out

def apply_bpe(corpus_lines, merges, separator='@@', vocab=None):
    ranks = {pair: i for i, pair in enumerate(merges)}
    reverse = {a + b: (a, b) for a, b in merges}
    out = []
    for line in corpus_lines:
        pieces = []
        for word in line.split():
            encoded = encode_word(word, ranks)
            if vocab is not None:
                encoded = check_vocab_and_split(encoded, reverse, vocab, separator)
            pieces.extend(piece + separator for piece in encoded[:-1])
            pieces.append(encoded[-1])
        out.append(' '.join(pieces))
    return out
```

In production the learner indexes which words contain each pair and updates pair
statistics incrementally after each merge (decrementing the broken pairs, incrementing
the new ones, optionally pruning rare pairs and restoring full statistics when needed).
The segmented text feeds directly into the existing attention encoder-decoder, whose
vocabulary is now the fixed symbol set.
