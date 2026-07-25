Neural machine translation usually fixes the vocabulary at 30,000–50,000 word types and maps everything else to a single `UNK` symbol. That is a problem because translation is open-vocabulary: test sentences keep producing names, numbers, compounds, loanwords, and inflected forms that the model has never seen whole. A back-off dictionary can replace some `UNK` outputs, but it assumes a one-to-one source-to-target word correspondence, fails on compounds that fuse several concepts into one word, and cannot transliterate names across different alphabets. Large vocabularies help only a little, because the long tail of rare types is exactly where new words live. Pure character models avoid unknowns entirely, yet they stretch sequences by a factor of five or more, which makes the recurrent encoder-decoder slow and the alignment problem harder. Fixed-length character n-grams shorten sequences but reintroduce unseen units and split frequent and rare strings with the same blunt granularity. Linguistic segmenters split at morpheme boundaries, but they remain conservative and still leave unknown words on the table. What is needed is a fixed, compact symbol set with a single tunable knob that keeps frequent text short, decomposes rare text toward characters, and never loses the ability to represent a new word from the known alphabet.

The method that satisfies these constraints is Byte Pair Encoding, or BPE. It is a bottom-up compression algorithm adapted to word segmentation. Start by representing every word as a sequence of its individual characters plus a special end-of-word marker. Then repeat a simple merge step: count every adjacent pair of symbols in the training vocabulary, weighted by word frequency, pick the most frequent pair, and replace every occurrence of that pair with a single new symbol. Each merge adds exactly one symbol to the vocabulary, so the number of merge operations directly controls the final vocabulary size. Because merges start from characters and build upward, frequent character sequences become single symbols while rare words stay split into smaller pieces. The end-of-word marker, folded onto the final character, keeps word-final units distinct from word-internal ones and makes detokenization possible after decoding.

The learned object is the ordered list of merges, not a table of final symbols. At test time, any word is segmented by replaying the merges in the order they were learned: split the word into characters plus the end marker, then repeatedly apply the earliest learned merge that is currently possible, merging all non-overlapping occurrences of that pair. The result is a sequence of symbols that may be whole frequent words, common morphemes, or character chunks for genuinely rare material. The segmentation layer sits before the unchanged encoder-decoder, so the embedding lookup and output softmax are simply sized by the compact symbol vocabulary. For bilingual settings, the merges can be learned jointly on the union of source and target vocabularies so that the same name or cognate is segmented consistently on both sides, which is especially useful for transliteration. Because the network's accepted symbol set can be smaller than everything the merges produce (rare merged symbols get pruned), encoding also accepts that accepted vocabulary: any produced piece that falls outside it is reversed back through the merge that created it, recursively, until every emitted piece is either accepted or a bare character — the character alphabet is always the floor of what the scheme can produce.

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

In production the learner indexes which words contain each pair and updates pair statistics incrementally after each merge — decrementing the broken pairs, incrementing the new ones, optionally pruning rare pairs and restoring full statistics when needed — rather than rescanning the whole vocabulary on every iteration, but the trade-off this buys is unchanged: a fixed symbol vocabulary, an open-vocabulary fallback all the way down to characters, and a direct knob trading vocabulary size against sequence length, all without touching the underlying translation network.