## Research question

Neural machine translation models operate over a fixed, closed vocabulary —
typically the 30,000–50,000 most frequent word types — and represent every other
word with a single `UNK` symbol. Translation is an open-vocabulary problem:
test text contains names, numbers, compounds, loanwords, and inflected
forms that never appeared (or appeared too rarely) in training. This is especially
visible for morphologically productive languages (German compounding and Russian
inflection), where a single lemma spawns dozens of surface forms and new words are
formed compositionally at will. Can a neural translation system translate and even
*generate* words it has never seen by working below the word level, while keeping the
network vocabulary small enough that the output softmax and embedding tables stay
affordable?

The softmax over the target vocabulary is the dominant cost in training and decoding,
and sequence length drives the cost of the recurrent encoder/decoder, so any sub-word
scheme trades vocabulary size against text length.

## Background

**Why rare words are translatable below the word level.** Many word classes are
transparent to a competent translator even when the whole word is novel: named
entities can be copied or transliterated character-by-character (Barack Obama →
Барак Обама); cognates and loanwords differ in regular, near-character-level ways
(claustrophobia → Klaustrophobie); morphologically complex words can be translated
morpheme-by-morpheme (solar system → Sonnensystem = Sonne + System). A diagnostic
count of 100 rare tokens in German training data finds the majority are translatable
through smaller units: 56 compounds, 21 names, 6 loanwords with common origin,
5 transparent affixations, 1 number, and 1 computer language identifier. This
diagnostic motivates segmenting rare words into sub-word units: the information needed
to translate them is present at the sub-word level.

**The encoder-decoder with attention.** The translation model is an RNN
encoder-decoder with attention. A bidirectional GRU encoder reads the source
`x = (x_1,…,x_m)` into annotation vectors `h_j`; an RNN decoder predicts the target
`y = (y_1,…,y_n)`, each `y_i` from a hidden state `s_i`, the previous word `y_{i-1}`,
and a context vector `c_i = Σ_j α_{ij} h_j`, where the alignment weights `α_{ij}`
come from a small feedforward alignment model trained jointly. The vocabulary enters
in two places: the input embedding lookup and the output softmax. A variable-length
sub-word representation lets attention place its weight on different sub-word units at
each decoding step, instead of attending to a single fixed-length word vector.

## Baselines

**Word-level NMT with `UNK` (WUnk).** Limit the vocabulary to the top ~30k–50k types;
every other word becomes `UNK`. Core idea: keep the softmax tractable.

**Word-level NMT with a back-off dictionary (WDict; Jean et al. 2015; Luong et al.
2015).** Same closed vocabulary, but at test time `UNK` outputs are replaced via a
separately trained bilingual dictionary / alignment back-off (e.g. fast-align), often
copying the aligned source word. Core idea: handle OOVs outside the network with a
1-to-1 source↔target word correspondence.

**Large-vocabulary tricks (Jean et al. 2015).** Push the vocabulary up (e.g. to
500k) with importance-sampling approximations to the softmax. Core idea: cover more
words directly.

**Character n-gram segmentation.** Represent words as contiguous character n-gram
chunks (unigrams, bigrams, trigrams) that do not cross word boundaries, optionally
leaving a shortlist of the k most frequent words unsegmented. Core idea: trade
sequence length against vocabulary size by choosing n. The unigram (pure character)
representation is open-vocabulary; it has the smallest vocabulary (~3k) and the longest
sequences (~550m tokens vs ~100m words). Bigrams/trigrams shorten sequences and use a
small number of additional symbols.

**SMT morphological segmenters (compound splitting — Koehn & Knight 2003; Morfessor —
Creutz & Lagus 2002; rule-based hyphenation — Liang 1983).** Linguistically motivated
segmenters from phrase-based SMT. Core idea: split words at morpheme/compound
boundaries.

## Evaluation settings

WMT 2015 shared translation task, English→German (4.2M sentence pairs, ~100M tokens)
and English→Russian (2.6M pairs, ~50M tokens). Data tokenized and truecased with
Moses scripts. Development set newstest2013; test sets newstest2014 and newstest2015.
Network: bidirectional GRU encoder-decoder with attention (Groundhog), hidden size
1000, embedding size 620, output softmax/shortlist of τ=30000 words, Adadelta,
minibatch 80, gradient clipping, beam search (beam 12) with length-normalized
probabilities. Metrics: BLEU (mteval-v13a.pl) and chrF3 (character n-gram F3); for the
specific claim about rare/unseen words, unigram F1 (harmonic mean of clipped unigram
precision and recall), reported separately for all words, rare words (not in the top
50k of training), and OOVs (absent from training). Corpus statistics — number of
tokens, number of types, number of `UNK` symbols on the dev set — characterize each
segmentation.

## Code framework

The segmentation layer sits *before* the translation network: it learns a fixed set
of operations on the training text, then rewrites both training and test text into
symbol sequences that feed the existing embedding → encoder-decoder → softmax
pipeline unchanged. The harness below has the frequency counting and the encoder-decoder
in place; the segmentation learner and the test-time segmenter are the empty slots.

```python
import re, collections

def get_word_frequencies(corpus_lines):
    """Whitespace-tokenize the training text and count word frequencies."""
    freqs = collections.Counter()
    for line in corpus_lines:
        for word in line.split():
            freqs[word] += 1
    return freqs

def learn_segmentation(word_freqs, num_operations):
    # TODO: from a frequency table over words, learn a fixed-size set of
    # operations that builds a compact symbol vocabulary from lower-level units.
    pass

def segment_word(word, learned_operations):
    # TODO: rewrite a single (possibly unseen) word into a sequence of symbols
    # using the learned operations.
    pass

def apply_segmentation(corpus_lines, learned_operations):
    out = []
    for line in corpus_lines:
        pieces = []
        for word in line.split():
            pieces.extend(segment_word(word, learned_operations))
        out.append(' '.join(pieces))
    return out

# --- existing translation stack (unchanged) ---
class AttentionEncoderDecoder:
    """Bidirectional GRU encoder + attention decoder over a fixed symbol set.
    Embedding lookup and output softmax are sized by len(symbol_vocab)."""
    def train_step(self, src_ids, tgt_ids):
        pass
    def translate(self, src_ids, beam_size=12):
        pass

def build_symbol_vocab(segmented_corpus, min_count):
    vocab = collections.Counter(tok for line in segmented_corpus for tok in line.split())
    return {s for s, c in vocab.items() if c >= min_count}
```
