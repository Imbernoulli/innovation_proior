OK, let me think about this from the actual pain. I have a neural translation model — bidirectional GRU encoder, attention decoder — and the thing that keeps biting is the vocabulary. I cap it at the 30k–50k most frequent words because the output softmax and the embedding table scale with vocabulary size, and that's already the dominant cost. Everything else becomes `UNK`. But translation is open-vocabulary: every test sentence throws names, numbers, compounds, loanwords, inflected forms at me that I either never saw or saw twice. German is brutal here — `Abwasserbehandlungsanlage`, "sewage water treatment plant", is one token to my model, and it's `UNK`. So I produce `UNK` and the translation is broken.

The standard patch is a back-off dictionary: when the model emits `UNK`, look the aligned source word up in a separately trained bilingual dictionary, or just copy it across. And that does help — copying a name like "Obama" from English to German source-to-target is fine. But it assumes things that aren't true. It assumes a 1-to-1 source↔target word correspondence, which collapses for compounds: there's no single English word aligned to `Abwasserbehandlungsanlage`; it's "sewage" + "water" + "treatment" + "plant" fused. And copying only works when the alphabets match. English→Russian, a name has to be *transliterated* — Mirzayeva → Мирзаева — and a dictionary that just copies Latin letters into a Cyrillic sentence is garbage. So the back-off is a leaky, non-end-to-end bolt-on that fails exactly where it's needed most.

Let me step back and ask what's actually true about these hard words. Stare at the rare tokens. A competent human translator can render a word they've never seen, *if* it decomposes into known pieces: names by character copying or transliteration, cognates and loanwords by near-character-level rules (claustrophobia → Klaustrophobie), compounds and inflections by translating the morphemes (Sonne + System). I counted 100 rare German tokens once — the majority were like this: dozens of compounds, a pile of names, some transparent affixations. So the information needed to translate a rare word is *present below the word level*. The word boundary is the wrong place to stop. If I let the network see and emit sub-word units, it can learn "translate this morpheme" and "transliterate this character chunk", and then compose translations for words it never saw whole. And as a bonus, the attention mechanism gets richer: instead of attending to one fixed-length word vector, it can place its weight on different sub-word units at each decoding step — no information bottleneck on a long compound.

So the real question becomes: what is the right set of sub-word units? I want a *fixed, compact* symbol vocabulary that can represent unseen words from known lower-level symbols, with a clear caveat for truly unseen characters and a fallback for filtered-out pieces, without exploding sequence length. Three requirements are pulling against each other.

Try the most extreme open-vocabulary option first: pure characters. Vocabulary is tiny — a few thousand symbols, and any word over the known character set is representable without a whole-word `UNK`. Watch it break, though: sequence length explodes. The German side goes from ~100m word tokens to ~550m character tokens. My encoder-decoder is at least linear in sequence length, and the decoder has to carry information over much longer distances; over-splitting also makes alignment harder. So pure characters satisfy "open vocabulary" and "compact" but ruin "short sequences." Worse, the model now has to *learn* that "S-o-n-n-e" is a unit from scratch, every time, instead of being handed it.

Patch toward shorter: character n-grams. Bigrams, trigrams — fewer tokens per word, but now the vocabulary grows (bigrams ~20k, trigrams ~120k) and, annoyingly, the unigram-only case is the *only* truly open-vocabulary one; bigrams/trigrams reintroduce a handful of unknown symbols on the test set, because some test bigram never occurred in training. And fixed-n is a blunt instrument: it splits "the" into the same granularity as "Abwasserbehandlungsanlage". Frequent strings should stay whole (cheap, no length penalty); only rare strings need to be cracked open. A single n can't do both.

What about the linguistically-motivated segmenters from phrase-based SMT — compound splitting, Morfessor, hyphenation? They split at morpheme/compound boundaries, which is principled. But they're conservative splitters: they only moderately shrink the vocabulary, and — the killer — they still leave unknown words. They don't close the open-vocabulary gap, so I'd *still* need a back-off model. That defeats the whole point.

So I want a segmentation with a tunable granularity: frequent material kept as long units, rare material decomposed down toward characters, with one knob that trades vocabulary size against text length, and a known-character backstop instead of a whole-word `UNK`.

I'm essentially asking for a *good compression* of the text into a fixed symbol table: a code that represents the corpus compactly with a bounded number of symbols. That's a compression problem, and there's a dead-simple compression algorithm with exactly the control I want: byte pair encoding. The original byte version repeatedly finds the most frequent adjacent pair of bytes in the data and replaces every occurrence with a single new, unused byte, recording the merge. Each merge adds exactly one symbol. After `n` merges I've added exactly `n` symbols. The final table size is the initial alphabet plus the number of merges, so the number of merge operations directly sets the final vocabulary size. Because merges are built bottom-up from the alphabet, known characters remain the backstop; if a character itself was never seen, or if a learned larger symbol is filtered out of the network vocabulary, I can fall back by splitting rather than emitting an opaque unknown word.

Now adapt it from bytes to word segmentation. Don't merge bytes of the raw byte stream — merge *characters* (and character sequences) within words. Start by representing each word as a sequence of its characters. Then iterate: count all adjacent symbol pairs across the corpus (weighted by word frequency), find the most frequent pair `(A, B)`, and merge it everywhere into a new symbol `AB`. Each merge produces a symbol standing for a character n-gram. Frequent n-grams — common morphemes, common whole words — get merged early and end up as single symbols; rare words never accumulate enough merges and stay split into smaller pieces, bottoming out at characters. Exactly the granularity gradient I wanted: frequent → whole, rare → decomposed. And I never have to consider pairs that cross word boundaries — words are independent — so I can run the whole thing on the *dictionary* of unique words weighted by frequency, not on the raw token stream. Big efficiency win, and it's the natural unit anyway.

One subtlety I have to get right: word boundaries at test time. If I merge purely on characters, then a symbol like `er` could be a word-final suffix ("lower") or a word-internal chunk, and worse, after applying merges I won't know where one word ended and the next began — I can't reconstruct the original tokenization, which I need to detokenize the translation. So I append a special end-of-word marker to each word before learning. With a separate-marker convention it is a final symbol; with the version 0.2 code convention I fold it onto the last character before applying merges. Either way, word-final `er</w>` is distinct from internal `er`, the marker rides along through merges, and at the end I can split or strip by using the marker. Pair counts are taken inside word representations, so merges do not cross raw word boundaries in the first place.

Let me work through the toy to be sure I believe it. Dictionary `{ "low": 5, "lower": 2, "newest": 6, "widest": 3 }`. Represent each as space-separated characters with the end marker:
`l o w </w>` : 5, `l o w e r </w>` : 2, `n e w e s t </w>` : 6, `w i d e s t </w>` : 3.
Count adjacent pairs weighted by frequency. `e s`, `s t`, and `t </w>` are tied at 9; one valid first merge under the simple listing is `e s` -> `es`, giving `n e w es t </w>` and `w i d es t </w>`. Recount: `es t` and `t </w>` are tied at 9, so a valid next merge is `es t` -> `est`: `n e w est </w>`, `w i d est </w>`. The exact order of equal-frequency ties is not the mathematical point; the rule is always "take a current most frequent adjacent pair, merge all its occurrences, then recount." After a fixed `num_merges`, stop. Frequent suffixes such as `est</w>` and frequent stems such as `low` become single symbols after enough legal merges; rare residue stays closer to characters. The learned object is just the *ordered list of merge operations*.

At test time, segment any word — seen or unseen — by replaying the learned operations in the order they were learned: split the word into characters + end marker, then repeatedly apply the earliest learned merge that is currently possible in this word, merging all non-overlapping occurrences of that pair. Order matters, because later merges may depend on symbols produced by earlier ones. The OOV "lower" with operations learned from {low, lowest, newer, wider} comes out as `low er</w>` under the separate-marker convention — `low` is a known unit and `er</w>` is a known word-final unit, both translatable, even though "lower" itself was never in the learned dictionary. In an implementation that prints continuation markers, the same idea appears as non-final segments marked with something like `@@` and the word-final marker stripped before the text reaches the translation model.

What's the difference from just using a generic compressor like Huffman coding to make a variable-length word code, which people have floated for NMT? The point isn't only compression — it's that BPE's symbols stay *interpretable as sub-word units*. The pieces are morphemes and character chunks the network can attach meaning to and recombine, so it generalizes to new words. A Huffman code over words gives you compact bit-strings with no compositional structure to generalize from.

Now the implementation of the learner. The naive version: each iteration, recompute all pair statistics from scratch over the whole vocabulary, take the argmax, merge it, repeat. That's the minimal version and it's correct:

```python
import re, collections

def get_stats(vocab):
    # count frequency of every adjacent symbol pair, weighted by word frequency
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    # replace every occurrence of the pair (A B) with the merged symbol AB
    v_out = {}
    bigram = re.escape(' '.join(pair))
    # (?<!\S) and (?!\S) require the match to be bounded by whitespace/word edges,
    # so we only merge this exact adjacent pair, not a substring of a larger symbol
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# words as space-separated chars + end-of-word marker, weighted by frequency
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
         'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 10
merges = []
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)   # most frequent adjacent pair
    vocab = merge_vocab(best, vocab)
    merges.append(best)                # the learned operation, in order
```

The recompute-everything loop scans the vocabulary again and again, which is wasteful: one merge only changes the pair counts of the words that contained that pair. So in practice I index, for each pair, the set of words it occurs in, and after a merge I only update the statistics of the affected words — decrement the counts of the pairs that the merge broke, increment the counts of the new pairs the merge created. I can also prune very low-frequency pairs from the active statistics for speed and restore the full statistics when the active maximum gets suspiciously small. The merge decision is still frequency-first; the extra `min_frequency` guard just stops spending merge slots on pairs that occur once and are likely noise.

```python
def get_pair_statistics(sorted_vocab):
    stats = collections.defaultdict(int)
    indices = collections.defaultdict(lambda: collections.defaultdict(int))
    for i, (word, freq) in enumerate(sorted_vocab):
        prev = word[0]
        for sym in word[1:]:
            stats[prev, sym] += freq
            indices[prev, sym][i] += 1   # which words contain this pair, how often
            prev = sym
    return stats, indices

def learn_bpe(word_freqs, num_symbols, min_frequency=2):
    # version 0.2 representation: marker is folded onto the final character
    vocab = [(tuple(w[:-1]) + (w[-1] + '</w>',), f) for w, f in word_freqs.items()]
    vocab = sorted(vocab, key=lambda x: x[1], reverse=True)
    stats, indices = get_pair_statistics(vocab)
    merges = []
    for _ in range(num_symbols):
        if not stats:
            break
        most_freq = max(stats, key=lambda p: (stats[p], p))  # most frequent, deterministic tie-break
        if stats[most_freq] < min_frequency:
            break
        merges.append(most_freq)
        # replace_pair returns only the words that changed; update_pair_statistics
        # subtracts broken neighbors and adds the new neighbors for those words
        changes = replace_pair(most_freq, vocab, indices)
        update_pair_statistics(most_freq, changes, stats, indices)
        stats[most_freq] = 0
    return merges
```

Test-time segmentation replays the merges in learned order and strips the end marker before output. The continuation marker is what lets detokenization know which emitted pieces belong to the same original word:

```python
def encode_word(word, bpe_merges):
    # bpe_merges: dict mapping each learned pair to its rank (learning order)
    symbols = list(word[:-1]) + [word[-1] + '</w>']
    while True:
        # find the adjacent pair present in this word with the lowest (earliest) rank
        candidates = [(bpe_merges[pair], i, pair)
                      for i, pair in enumerate(zip(symbols, symbols[1:]))
                      if pair in bpe_merges]
        if not candidates:
            break
        _, _, (a, b) = min(candidates)  # apply earliest-learned applicable merge first
        positions = [i for _, i, pair in candidates if pair == (a, b)]
        merged, start = [], 0
        for pos in positions:
            if pos < start:             # skip overlaps such as x x x -> xx x
                continue
            merged.extend(symbols[start:pos])
            merged.append(a + b)
            start = pos + 2
        merged.extend(symbols[start:])
        symbols = merged
    if symbols[-1] == '</w>':
        symbols = symbols[:-1]
    elif symbols[-1].endswith('</w>'):
        symbols[-1] = symbols[-1][:-4]
    return symbols

def segment_line(line, merges, separator='@@'):
    ranks = {pair: i for i, pair in enumerate(merges)}
    output = []
    for word in line.split():
        pieces = encode_word(word, ranks)
        output.extend(p + separator for p in pieces[:-1])
        output.append(pieces[-1])
    return ' '.join(output)
```

The only symbols that can be unknown at test time are unknown characters, symbols whose every training occurrence got merged into something larger and then no longer appears as an accepted network symbol, or joint-vocabulary symbols that were learned from the other language side. If a segment is outside the accepted network vocabulary and it was produced by a merge, the implementation can reverse that merge recursively until every emitted piece is in the known symbol vocabulary; truly unseen characters remain the irreducible case.

The bilingual setting creates a final tension. I can learn the merges independently for source and target, or learn one set on the *union* of both vocabularies — joint BPE. Independent is more compact per side and guarantees each unit was seen in that language. But joint forces source and target to segment consistently: the same name splits the same way on both sides, which makes the network's job of learning a sub-word↔sub-word mapping much easier — exactly what transliteration needs. For different alphabets (English/Russian) I'd transliterate Russian to Latin to learn the joint merges, then map the merge operations back to Cyrillic to apply them. So joint BPE for consistency where it matters; independent where compactness matters more.

The vocabulary cap forces `UNK`s and a leaky back-off dictionary; rare words are often translatable from sub-word pieces, so I move below the word level; pure characters are open-vocabulary but too long, fixed-n n-grams and linguistic splitters either reintroduce unknowns or split bluntly; what I want is a fixed compact symbol table with one knob and a known-character backstop, which is exactly a compression code — and byte pair encoding gives it, merging the most frequent adjacent symbol pair `num_merges` times so frequent strings become whole symbols and rare strings decompose toward characters, with an end-of-word marker to keep boundaries reconstructible. Learn the ordered merges on the frequency-weighted word dictionary, replay them to segment any word, mark non-final pieces, and feed the resulting symbol sequences to the unchanged encoder-decoder.
