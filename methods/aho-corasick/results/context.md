## Research question

Given a finite set of keywords (and phrases) and an arbitrary text string, locate
*all* occurrences of *any* of the keywords as substrings of the text — including
overlapping occurrences — in a single left-to-right pass, with a running time
that does not grow with the number of keywords.

The setting that makes this urgent is bibliographic search. At a large industrial
research library, machine-readable citation tapes accumulate into a corpus of
hundreds of thousands of citations and on the order of 10^7 characters. A
bibliographer issues a query that is a Boolean function of many keywords and
phrases — "find all titles containing both `ion` and `bombardment`" — possibly
with embedding constraints (so `ions` may count as a match for `ion` while
`motion` must not). The query may carry dozens of keywords. The natural first
implementation — take each keyword in turn and scan it against the whole corpus —
costs on the order of (number of keywords) × (length of text). With many keywords
this is ruinously slow: a single search can consume an entire machine-time
budget before it finishes. What is needed is a method whose per-search cost is
essentially the cost of reading the text once, no matter how many keywords are in
the query.

## Background

The problem sits between two well-developed bodies of technique that, at the
time, were rarely brought together.

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
idea of the pattern against itself. The reusable insight is precisely this:
after matching a prefix and then failing, the longest prefix you can keep is a
function only of *how far you had matched*, computable ahead of time.

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
one transition per input symbol. The catch is construction cost: building an NFA
and determinizing it by the subset construction can produce on the order of 2^r
states for an expression of length r, and state minimization is fiddly to
program. This complexity is the reason automata were "frequently shunned by
programmers" for everyday matching tasks, despite their one-transition-per-symbol
elegance. Diagnostically, then, the field had a fast theoretical object (the DFA)
that was painful to build, and a fast practical object (KMP) that handled only
one pattern.

A few partial steps toward the multi-keyword case existed: special-purpose
hardware/microprogrammed finite-state search machines for full-text retrieval
(Bullen–Millen), finite-state lexical analyzers (Johnson et al.), and an
unpublished suggestion (attributed to Hopcroft and Karp) for finding the first
occurrence of any of a set of keywords. None packaged a simple, cheaply
constructed machine that reports *all* occurrences of *all* keywords in one pass.

## Baselines

**Repeated single-pattern scan (the straightforward method).** Take each keyword
in turn; slide it down the entire text (with naive matching, or even with KMP per
keyword). Reports all occurrences correctly. Cost is at best proportional to
(number of keywords) × (length of text); with KMP per keyword it is
O(Σ|y_i| + k·n) for k keywords. The keyword count multiplies the text length —
exactly the term that makes large bibliographic queries infeasible. This is the
method whose cost prompted the search for something better.

**Single-pattern KMP (Knuth–Morris–Pratt).** O(m+n) for one keyword via the
failure table; the text pointer never backs up. The gap: it matches one pattern.
Run k times it regains the k·n factor. But its failure-function idea — keep the
longest still-viable matched prefix on a mismatch — is the seed that a
multi-pattern method must generalize from a line (the single pattern) to a tree
(all patterns at once).

**Trie / prefix-tree search.** Walking the text through a trie of the keywords
matches all keywords that *start* at a given text position simultaneously, in
time proportional to the depth reached. The gap: it only finds keywords anchored
at the current start; when the walk falls off the trie (no edge for the next
symbol), it has no notion of how far to back off to continue matching keywords
that started later. It lacks the failure mechanism, so naively one restarts the
walk at every text position — again a per-position re-scan.

**NFA-for-regex then determinize.** Build an automaton for `Σ*(y1|…|yk)` and run
it: one transition per symbol, O(n) search. The gap: the determinized automaton
can blow up to ~2^r states and the construction plus minimization is complex to
implement — the very reason such automata were avoided for routine matching.

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

The pieces that already exist: a way to read the text symbol by symbol, a tree
(trie) structure for storing a set of strings, and the general shape of a
finite-state driver (a current state, a per-symbol transition, an occasional
emit). The slots the method must fill are the construction of the machine from
the keyword set and the per-symbol transition rule that lets a single pass report
every match.

```python
from collections import deque


class MultiKeywordMatcher:
    def __init__(self):
        # goto graph: list of dicts, transitions[state][symbol] -> state.
        self.goto = [{}]          # state 0 is the root
        self.output = [set()]     # keywords associated with a state
        # TODO: the auxiliary structure that lets a mismatch continue the scan
        #       without re-reading text (generalize the single-pattern failure
        #       table from a line to the whole tree of keywords).

    def add_keyword(self, word):
        # Insert word as a root-to-node path in the goto graph (trie insert),
        # tag the terminal state with the keyword.
        state = 0
        for ch in word:
            nxt = self.goto[state].get(ch)
            if nxt is None:
                nxt = len(self.goto)
                self.goto.append({})
                self.output.append(set())
                self.goto[state][ch] = nxt
            state = nxt
        self.output[state].add(word)

    def build(self):
        # TODO: compute the per-state continuation links from the goto graph,
        #       and arrange that every keyword endable at a state is reported
        #       there. Process states in an order that makes this well-defined.
        pass

    def search(self, text):
        # TODO: drive the machine over the text one symbol at a time, using the
        #       continuation links on a mismatch, emitting (position, keyword)
        #       for every match. Exactly one symbol consumed per step.
        pass
```
