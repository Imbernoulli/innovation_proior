OK, let me think about this from the actual pain. I have a neural translation model — bidirectional GRU encoder, attention decoder — and the thing that keeps biting is the vocabulary. I cap it at the 30k–50k most frequent words because the output softmax and the embedding table scale with vocabulary size, and that's already the dominant cost. Everything else becomes `UNK`. But translation is open-vocabulary: every test sentence throws names, numbers, compounds, loanwords, inflected forms at me that I either never saw or saw twice. German is brutal here — `Abwasserbehandlungsanlage`, "sewage water treatment plant", is one token to my model, and it's `UNK`. So I produce `UNK` and the translation is broken.

The standard patch is a back-off dictionary: when the model emits `UNK`, look the aligned source word up in a separately trained bilingual dictionary, or just copy it across. And that does help — copying a name like "Obama" from English to German source-to-target is fine. But it assumes things that aren't true. It assumes a 1-to-1 source↔target word correspondence, which collapses for compounds: there's no single English word aligned to `Abwasserbehandlungsanlage`; it's "sewage" + "water" + "treatment" + "plant" fused. And copying only works when the alphabets match. English→Russian, a name has to be *transliterated* — Mirzayeva → Мирзаева — and a dictionary that just copies Latin letters into a Cyrillic sentence is garbage. So the back-off is a leaky, non-end-to-end bolt-on that fails exactly where it's needed most.

Let me step back and ask what's actually true about these hard words. Stare at the rare tokens. A competent human translator can render a word they've never seen, *if* it decomposes into known pieces: names by character copying or transliteration, cognates and loanwords by near-character-level rules (claustrophobia → Klaustrophobie), compounds and inflections by translating the morphemes (Sonne + System). I counted 100 rare German tokens once — the majority were like this: dozens of compounds, a pile of names, some transparent affixations. So the information needed to translate a rare word is *present below the word level*. The word boundary is the wrong place to stop. If I let the network see and emit sub-word units, it can learn "translate this morpheme" and "transliterate this character chunk", and then compose translations for words it never saw whole. And as a bonus, the attention mechanism gets richer: instead of attending to one fixed-length word vector, it can place its weight on different sub-word units at each decoding step — no information bottleneck on a long compound.

So the real question becomes: what is the right set of sub-word units? I want a *fixed, compact* symbol vocabulary that can represent *any* string — zero true OOV symbols — without exploding sequence length. Three requirements pulling against each other.

Try the most extreme open-vocabulary option first: pure characters. Vocabulary is tiny — a few thousand symbols, and literally any word is representable, no `UNK` ever. Watch it break, though: sequence length explodes. The German side goes from ~100m word tokens to ~550m character tokens. My encoder-decoder is at least linear in sequence length, and the decoder has to carry information over much longer distances; over-splitting also makes alignment harder. So pure characters satisfy "open vocabulary" and "compact" but ruin "short sequences." Worse, the model now has to *learn* that "S-o-n-n-e" is a unit from scratch, every time, instead of being handed it.

Patch toward shorter: character n-grams. Bigrams, trigrams — fewer tokens per word, but now the vocabulary grows (bigrams ~20k, trigrams ~120k) and, annoyingly, the unigram-only case is the *only* truly open-vocabulary one; bigrams/trigrams reintroduce a handful of unknown symbols on the test set, because some test bigram never occurred in training. And fixed-n is a blunt instrument: it splits "the" into the same granularity as "Abwasserbehandlungsanlage". Frequent strings should stay whole (cheap, no length penalty); only rare strings need to be cracked open. A single n can't do both.

What about the linguistically-motivated segmenters from phrase-based SMT — compound splitting, Morfessor, hyphenation? They split at morpheme/compound boundaries, which is principled. But they're conservative splitters: they only moderately shrink the vocabulary, and — the killer — they still leave unknown words. They don't close the open-vocabulary gap, so I'd *still* need a back-off model. That defeats the whole point.

So I want a segmentation with a tunable granularity: frequent material kept as long units, rare material decomposed down toward characters, with one knob that trades vocabulary size against text length, and a hard guarantee that any string is representable. The knob and the guarantee are the crux.

Here's the reframe that cracks it. I'm essentially asking for a *good compression* of the text into a fixed symbol table: a code that represents the corpus compactly with a bounded number of symbols. That's a compression problem, and there's a dead-simple compression algorithm that has exactly the knob I want: byte pair encoding. The original byte version: repeatedly find the most frequent adjacent pair of bytes in the data and replace every occurrence with a single new, unused byte, recording the merge. Each merge adds exactly one symbol. So after `n` merges I've added exactly `n` symbols. The final table size = initial alphabet + number of merges. *That's the knob* — number of merge operations is the single hyperparameter, and it directly sets the final vocabulary size. And because merges are built bottom-up from the alphabet, the alphabet is always retained, so any string is still representable. Open-vocabulary guarantee for free.

Now adapt it from bytes to word segmentation. Don't merge bytes of the raw byte stream — merge *characters* (and character sequences) within words. Start by representing each word as a sequence of its characters. Then iterate: count all adjacent symbol pairs across the corpus (weighted by word frequency), find the most frequent pair `(A, B)`, and merge it everywhere into a new symbol `AB`. Each merge produces a symbol standing for a character n-gram. Frequent n-grams — common morphemes, common whole words — get merged early and end up as single symbols; rare words never accumulate enough merges and stay split into smaller pieces, bottoming out at characters. Exactly the granularity gradient I wanted: frequent → whole, rare → decomposed. And I never have to consider pairs that cross word boundaries — words are independent — so I can run the whole thing on the *dictionary* of unique words weighted by frequency, not on the raw token stream. Big efficiency win, and it's the natural unit anyway.

One subtlety I have to get right: word boundaries at test time. If I merge purely on characters, then a symbol like `er` could be a word-final suffix ("lower") or a word-internal chunk, and worse, after applying merges I won't know where one word ended and the next began — I can't reconstruct the original tokenization, which I need to detokenize the translation. So I append a special end-of-word marker to each word before learning — write it as `</w>` on the last character. Now `er</w>` (word-final "er") is a distinct symbol from internal `er`, the marker rides along through merges, and at the end I can split the symbol stream back into words by looking for the marker. It also stops merges from accidentally gluing the end of one word to the start of the next, since the marker sits between them.

Let me actually run the toy to be sure I believe it. Dictionary `{ "low": 5, "lower": 2, "newest": 6, "widest": 3 }`. Represent each as space-separated characters with the end marker:
`l o w </w>` : 5, `l o w e r </w>` : 2, `n e w e s t </w>` : 6, `w i d e s t </w>` : 3.
Count adjacent pairs weighted by frequency. `e s` appears in "newest" (6) and "widest" (3) → 9, the max. Merge `e s` → `es`: now `n e w es t </w>` and `w i d es t </w>`. Recount: `es t` appears 6+3 = 9 → merge `es t` → `est`: `n e w est </w>`, `w i d est </w>`. And so on — `est </w>`, then `l o` (5+2=7), `lo w` (7), … The frequent suffix `est</w>` becomes a single symbol after a couple of merges; "low" gets glued together; a rare residue like a foreign character just stays a character. After a fixed `num_merges`, stop. The learned object is just the *ordered list of merge operations*.

At test time, segment any word — seen or unseen — by replaying the learned operations in the order they were learned: split the word into characters + end marker, then apply each merge rule in turn, merging that pair wherever it occurs. Order matters, because later merges may depend on symbols produced by earlier ones. The OOV "lower" with operations learned from {low, lowest, newer, wider} would come out as `low er</w>` — `low` is a known unit and `er</w>` is a known unit, both translatable, even though "lower" itself was never in the learned dictionary. That is the open-vocabulary property made concrete: the network's vocabulary is fixed and finite, yet it can encode, translate, and *productively generate* new words from sub-word units.

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

The recompute-everything loop is `O(num_merges × corpus)`, which is wasteful: one merge only changes the pair counts of the words that contained that pair. So in practice I index, for each pair, the set of words it occurs in, and after a merge I only update the statistics of the affected words — decrement the counts of the pairs that the merge broke, increment the counts of the new pairs the merge created. I can also prune very low-frequency pairs from the active statistics for speed and only restore them if needed. Same merges, much faster:

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

def learn_bpe(word_freqs, num_symbols):
    # word -> tuple of chars with end-of-word marker on the last char
    vocab = [(tuple(w[:-1]) + (w[-1] + '</w>',), f) for w, f in word_freqs.items()]
    stats, indices = get_pair_statistics(vocab)
    merges = []
    for _ in range(num_symbols):
        if not stats:
            break
        most_freq = max(stats, key=lambda p: (stats[p], p))  # tie-break for determinism
        merges.append(most_freq)
        # replace pair in every word that contains it, and incrementally fix stats
        update_pair_statistics(most_freq, indices, vocab, stats)  # +/- only affected pairs
        stats[most_freq] = 0
    return merges
```

Test-time segmentation replays the merges in learned order:

```python
def segment_word(word, bpe_merges):
    # bpe_merges: dict mapping each learned pair to its rank (learning order)
    symbols = list(word[:-1]) + [word[-1] + '</w>']
    while True:
        # find the adjacent pair present in this word with the lowest (earliest) rank
        pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
        candidates = [(bpe_merges[p], p) for p in pairs if p in bpe_merges]
        if not candidates:
            break
        _, (a, b) = min(candidates)     # apply earliest-learned applicable merge first
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b); i += 2
            else:
                merged.append(symbols[i]); i += 1
        symbols = merged
    return symbols
```

The only symbols that can be unknown at test time are unknown characters, or symbols whose every training occurrence got merged into something larger — vanishingly rare in practice, and recoverable by un-doing specific merges if it ever bit me.

One last design choice on the bilingual side. I can learn the merges independently for source and target, or learn one set on the *union* of both vocabularies — joint BPE. Independent is more compact per side and guarantees each unit was seen in that language. But joint forces source and target to segment consistently: the same name splits the same way on both sides, which makes the network's job of learning a sub-word↔sub-word mapping much easier — exactly what transliteration needs. For different alphabets (English/Russian) I'd transliterate Russian to Latin to learn the joint merges, then map the merge operations back to Cyrillic to apply them. So joint BPE for consistency where it matters; independent where compactness matters more.

To recap the chain: the vocabulary cap forces `UNK`s and a leaky back-off dictionary; but rare words are translatable from sub-word pieces, so I move below the word level; pure characters are open-vocabulary but too long, fixed-n n-grams and linguistic splitters either reintroduce unknowns or split bluntly; what I want is a fixed compact symbol table with one knob and a representability guarantee, which is exactly a compression code — and byte pair encoding gives it, merging the most frequent adjacent symbol pair `num_merges` times so frequent strings become whole symbols and rare strings decompose toward characters, with an end-of-word marker to keep boundaries reconstructible. Learn the ordered merges on the frequency-weighted word dictionary, replay them to segment any string, and feed the resulting symbol sequences to the unchanged encoder-decoder.
