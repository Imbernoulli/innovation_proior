# Synthesis — Aho–Corasick

## Pain point (pre-method, from §1/§5/§7 + self-account)
Locate ALL occurrences of ANY of a finite set K of keywords in a text x, in one
pass. Bibliographic search at Bell Labs: ~150k citations, ~10^7 chars; queries
are Boolean functions over many keywords. Straightforward method: for each
keyword, scan it down the whole text → cost ∝ (#keywords) × |text|. Scales badly;
a bibliographer hit the $600 limit. Goal: cost roughly INDEPENDENT of #keywords.

## Tools on the table (load-bearing ancestors)
- KMP (Knuth–Morris–Pratt 1977, ref [13] = the 1974 Stanford TR): single keyword
  in O(m+n). The text pointer never backs up; on a mismatch at pattern position
  j, slide using next[j]/f[j] = the longest proper prefix of pattern[1..j-1]
  that is also a suffix. f[j] = longest proper border. This is the failure
  function. KMP also gives next[] (the optimized f', skipping redundant
  re-tests). Key reusable idea: a precomputed table that says, after matching a
  prefix and then failing, what is the longest already-matched prefix we can
  keep.
- The trie (Knuth TAOCP vol 1/3, refs [11],[12]): keywords share prefixes;
  walking one symbol per node spells out a keyword along a root-to-node path.
  Reading a symbol advances along an edge — already a partial automaton.
- Finite automata / DFA from regular expressions (Kleene, McNaughton–Yamada,
  Rabin–Scott, Thompson): in principle build an NFA for (.* )(y1|...|yk) and
  determinize. But subset construction can blow up to 2^r states and is fiddly
  to program; that complexity is exactly why programmers "shunned" automata (§1).

## The method (the bridge)
1. goto g (Algorithm 2): build a trie of K. enter(y) walks/extends from root;
   output(end-of-y) = {y}. Then add a self-loop at root on every symbol with no
   trie edge, so g(0, a) != fail for all a. This makes the search consume exactly
   one symbol per cycle (never stalls at the root).
2. failure f (Algorithm 3): generalize KMP's f to the trie. Lemma 1: f(s)=t iff
   v (string of t) is the LONGEST proper suffix of u (string of s) that is also
   a prefix of some keyword. Compute by BFS in increasing depth. For state s =
   g(r, a): set state=f(r); follow f until g(state,a) != fail; f(s)=g(state,a).
   The root self-loop guarantees termination. (Inner loop mirrors Algorithm 1.)
3. output (Algorithms 2+3): when f(s)=s', merge output(s') into output(s). So a
   node reports not only the keyword ending exactly there but every keyword that
   is a suffix of the node's string (e.g. node "she" also reports "he"). These
   are the "output links" — follow suffix/failure links to collect all matches.
4. search (Algorithm 1): state=0; for each symbol a: while g(state,a)=fail do
   state=f(state); state=g(state,a); emit output(state) with position.
5. DFA variant (Algorithm 4): precompute delta(s,a) = result of all failure
   transitions then the goto, so search makes exactly one transition per symbol.
   delta(0,a)=g(0,a); delta(r,a)=g(r,a) if !=fail else delta(f(r),a). BFS order.

## Correctness (lemmas, all proved in §4 — to be lived out in reasoning)
- Lemma 1 (f characterization): induction on |u|=depth. For u=a1..aj with s the
  state, r the state of a1..a_{j-1}: the chain r,f(r),f(f(r)),... gives, at each
  v_i, the longest proper suffix of the previous that is a prefix of some
  keyword; the first one that has a goto on aj yields f(s), and v_n a_j is the
  longest proper suffix of u that is a prefix of some keyword. QED.
- Lemma 2 (output): output(s) = {keywords that are suffixes of u}. Induction on
  depth using the merge step + Lemma 1.
- Lemma 3 (invariant): after the j-th cycle, Algorithm 1 is in the state
  represented by the LONGEST suffix of a1..aj that is a prefix of some keyword.
  Hence at each position it reports exactly the keywords ending there.
- Theorem 1: Algorithms 2,3 produce valid functions (Lemmas 2,3).

## Complexity (§5)
- Theorem 2: Algorithm 1 makes < 2n state transitions on text of length n.
  Proof: each cycle = zero+ failure transitions then exactly 1 goto; from depth
  d at most d failure transitions; depth increases by exactly 1 per goto and
  drops by >=1 per failure; total failure transitions < total goto transitions
  = n. So < 2n. (Amortized: depth is the potential.) [Footnote: for ONE keyword,
  KMP shows O(log d) failure transitions per cycle max — Fibonacci-string bound.]
- Theorem 3: Algorithm 2 (goto) is O(sum of keyword lengths).
- Theorem 4: Algorithm 3 (failure) is O(sum of keyword lengths) — same amortized
  argument bounds total state<-f(state) executions; output merge in O(1) per edge
  with linked lists since output(s) and output(f(s)) are disjoint at merge time.
- Output printing: proportional to total length of keywords actually reported;
  in worst case (K={a,a^2,...,a^k}, text=a^n) any algorithm must print them all.

## Design decisions -> why
- Root self-loop (g(0,a)!=fail for all a): so the search never has to special-case
  "fell off the trie at the root"; guarantees one symbol consumed per cycle and
  guarantees the failure-link walk terminates. Without it, the inner while could
  loop at the root forever / need an extra guard.
- failure links not full DFA first: the goto+failure machine is cheap to build
  (O(sum lengths)) and store (f is a 1-D array; g stored sparsely). A full DFA
  (Algorithm 4) is 2× faster at most in transitions but bigger in memory and in
  practice the machine sits in state 0 most of the time, so the saving "would
  virtually never be achieved." Hence failure links are the default; DFA is an
  option.
- output links (merge on failure): a node's string can have multiple keywords as
  suffixes; you must report all. Merging outputs along failure links at build
  time means O(1) reporting per match instead of re-walking suffixes each step.
- next/f' optimization (§3, generalizing KMP's next): f'(i)=f'(f(i)) when the
  failure target offers no new viable goto symbol, avoiding a provably useless
  failure transition. Optional; the DFA subsumes it.
- sparse vs dense goto storage (§5): 2-D array = O(1) lookup but |S|×|alphabet|
  memory; linear lists per state = small memory, cost ∝ #nonfail edges; direct
  tables for hot states (esp. state 0). A reasonable compromise stated in §5.

## Canonical implementation (code/)
code/aho_corasick.py — goto as list of dicts, output as list of sets, fail as
list; add_keyword (Alg 2), build (Alg 3, BFS), search (Alg 1), to_dfa (Alg 4).
Verified: on {he,she,his,hers} over "ushers" -> she@1, he@2, hers@2 (matches the
paper's Figure 2 walk). Fuzzed against brute force: 2000 search trials + 500 DFA
trials OK. Structure mirrors cp-algorithms.com canonical reference.
