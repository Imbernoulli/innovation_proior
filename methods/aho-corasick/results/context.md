## Research question

Given a finite set of keywords (and phrases) and an arbitrary text string, locate
*all* occurrences of *any* of the keywords as substrings of the text — including
overlapping occurrences — in a single left-to-right pass over the text.

The deliverable is a single self-contained C++17 program that reads from stdin
and writes to stdout. Its input is an integer `k`, followed by `k`
whitespace-delimited keywords, followed by the text string. It must print one
line per match as `start_index keyword`, using 0-indexed starts, including
overlapping matches, sorted by `(start_index, keyword)`.

The setting is bibliographic search. At a large industrial research library,
machine-readable citation tapes accumulate into a corpus of hundreds of thousands
of citations and on the order of 10^7 characters. A bibliographer issues a query
that is a Boolean function of many keywords and phrases — "find all titles
containing both `ion` and `bombardment`" — possibly with embedding constraints
(so `ions` may count as a match for `ion` while `motion` must not). The query may
carry dozens of keywords. The straightforward first implementation — take each
keyword in turn and scan it against the whole corpus — costs on the order of
(number of keywords) × (length of text).

## Background

The problem sits between two well-developed bodies of technique.

Single-keyword string matching had just been put on a linear footing. The
Knuth–Morris–Pratt algorithm (Knuth, Morris, and Pratt; circulated as a 1974
Stanford technical report, published 1977) finds a single pattern of length m in
a text of length n in O(m+n) time. Its central object is a precomputed *failure
table*. Aligning the pattern against the text and scanning left to right, when a
mismatch occurs at pattern position j after having matched `pattern[1..j-1]`, the
pattern is slid right by an amount read from the table rather than restarting one
position over. The table entry `f[j]` is the length of the longest proper prefix
of `pattern[1..j-1]` that is also a suffix of it (the longest proper *border*);
sliding by that much keeps the already-matched prefix aligned, so the text
pointer never backs up. KMP also defines an optimized variant `next[j]`, which
short-circuits a failure transition that would only re-test a character already
known to mismatch. The table itself is computed in O(m) by running the matching
idea of the pattern against itself.

The trie (Knuth, *The Art of Computer Programming*) is the natural structure for
a *set* of strings. Inserting each keyword as a root-to-leaf path, with shared
prefixes sharing nodes, yields a tree in which reading one input symbol advances
along one edge — already a partial automaton in which a root-to-node path spells
out a prefix of some keyword. A node reached by a complete keyword can be tagged
with that keyword.

Finite automata for pattern matching were also well understood in principle.
Regular-expression-to-automaton constructions (Kleene; McNaughton–Yamada;
Rabin–Scott; Thompson's regular-search construction; Brzozowski's derivatives)
show that any regular set — including `Σ*(y1 | y2 | … | yk)` for a set of
keywords — can be recognized by a deterministic finite automaton making exactly
one transition per input symbol. Building an NFA and determinizing it by the
subset construction can produce on the order of 2^r states for an expression of
length r, and the construction is accompanied by state minimization.

A few partial steps toward the multi-keyword case existed: special-purpose
hardware/microprogrammed finite-state search machines for full-text retrieval
(Bullen–Millen), finite-state lexical analyzers (Johnson et al.), and an
unpublished suggestion (attributed to Hopcroft and Karp) for finding the first
occurrence of any of a set of keywords.

## Baselines

**Repeated single-pattern scan (the straightforward method).** Take each keyword
in turn; slide it down the entire text (with naive matching, or even with KMP per
keyword). Reports all occurrences correctly. Cost is at best proportional to
(number of keywords) × (length of text); with KMP per keyword it is
O(Σ|y_i| + k·n) for k keywords.

**Single-pattern KMP (Knuth–Morris–Pratt).** O(m+n) for one keyword via the
failure table; the text pointer never backs up. Run k times it costs
O(Σ|y_i| + k·n).

**Trie / prefix-tree search.** Walking the text through a trie of the keywords
matches all keywords that *start* at a given text position simultaneously, in
time proportional to the depth reached. It finds the keywords anchored at the
current start; when the walk falls off the trie (no edge for the next symbol),
one restarts the walk at the next text position.

**NFA-for-regex then determinize.** Build an automaton for `Σ*(y1|…|yk)` and run
it: one transition per symbol, O(n) search. The determinized automaton can grow
to ~2^r states.

## Evaluation settings

The natural yardstick is the bibliographic-search workload that motivates the
problem: a cumulative citation index built from machine-readable tapes (the
cumulated data behind a fortnightly internal citation bulletin), on the order of
150,000 citations and ~10^7 characters after a few years of accumulation. Queries
are Boolean functions of keywords and phrases, with embedding options
(full / left / right / none) governing whether a keyword may match inside a
larger word. A representative comparison runs the same query set under the old
straightforward matcher and under a candidate machine on the same hardware (a
Honeywell 6070), with query prescriptions of varying keyword counts (e.g. 15 vs
24 keywords), measuring CPU time per search. The quantity of interest is how
search cost scales with the number of keywords. A controlled stress case is the
degenerate keyword set {a, a², a³, …, a^k} against text aⁿ, where every keyword
ends at (almost) every position — the case that exercises overlapping matches and
output volume, against which any matcher must print the same large answer.

## Code framework

The scaffold fixes the input and output shape for a single-file C++17 program.
The missing work is to compute every required match and leave it in `matches`
for sorted printing to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int k;
    if (!(cin >> k)) return 0;

    vector<string> keywords(k);
    for (int i = 0; i < k; ++i) cin >> keywords[i];

    string text;
    cin >> text;

    vector<pair<int, string>> matches;

    // TODO:

    sort(matches.begin(), matches.end());
    for (const auto& [start, keyword] : matches) {
        cout << start << ' ' << keyword << '\n';
    }
    return 0;
}
```
