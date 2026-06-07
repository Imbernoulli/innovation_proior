# Synthesis — Knuth–Morris–Pratt

## Sources read (this run)
- **PRIMARY**: Knuth, Morris, Pratt, "Fast Pattern Matching in Strings", SIAM J. Comput. 6(2):323–350, 1977 — read IN FULL (refs/kmp-1977-siam.pdf → src/kmp-1977.txt). §1 informal (sliding pattern, `next` table), §2 program + correctness invariant + `f[j]`/`next[j]` preprocessing, §3 efficiency refinements + compile-to-code, §4 extensions (all matches, trie/Aho–Corasick), §5 theory (periods, golden-ratio log_φ m bound, Fibonacci-string worst case), §6 palindromes (palstars recognizable in O(n)), **§7 Historical remarks (THE self-account)**, §8 Boyer–Moore postscript.
- **SELF-ACCOUNT**: §7 of the paper itself (first-person genesis). + Knuth, Web of Stories interview #92 "The Knuth-Morris-Pratt algorithm" — full verbatim transcript in refs/webofstories-knuth-92-transcript.txt. Added to SELF_ACCOUNT_SOURCES.md.
- **ANTECEDENT (Cook's theorem / construction)**: Glück, "Simulation of Two-Way Pushdown Automata Revisited", EPTCS 129 (2013), pp.250–258 (refs/cook-2dpda-revisited-1309.5142.pdf) — read pp.1–5: gives the precise mechanism of Cook's construction (surface configs c=(p,i,A); terminators; linear-time via memoizing terminators T[c]). Cook 1972 original ("Linear time simulation of deterministic two-way pushdown automata", IP71, pp.75–80) NOT freely downloadable — GAP noted; Glück is a faithful, citable reconstruction.
- **CANONICAL CODE**: Wikipedia kmp_search / kmp_table (0-indexed, next-style T with T[k]<0). Saved + ran in code/kmp_canonical.py; reproduces paper example.

## The discovery path (from §7 + interview — the BACKBONE)
Two independent routes converged:
1. **Morris (summer 1969, Berkeley):** implementing a text editor on the CDC 6400; the buffering for "backing up" the input file was complicated, so he sought a method that never backs up the text. Used finite-automata ideas; got the algorithm but his write-up didn't make the O(m+n) bound clear; other implementors didn't understand it and "fixed" it into a shambles.
2. **Knuth (early 1970):** learned of **Cook's theorem** — any language a 2-way deterministic pushdown automaton (2DPDA) recognizes (in ANY time, even exponential) can be recognized on a RAM in **O(n)**. D. Chester had shown "strings beginning with an even palindrome" is 2DPDA-recognizable. Knuth couldn't see how to recognize palstars in less than ~n² on a normal computer. So he "laboriously went through all the steps of Cook's construction as applied to Chester's automaton," to **"distill off"** what made it fast. After hours at a blackboard: "ah ha, of course, that's the trick." He abstracted the mechanism (for the palindrome case), generalized it to "find the longest prefix of one given string that occurs in another" — i.e. string matching. **"This was the first time in Knuth's experience that automata theory had taught him how to solve a real programming problem better than he could solve it before."** He showed Pratt, who improved the data structure to be alphabet-independent; Pratt described it to Morris, who recognized his own algorithm and learned of the O(m+n) bound. Morris & Pratt: tech report (Berkeley, 1970); the joint 1977 paper.

THE TRACE FOLLOWS KNUTH'S ROUTE (richer, more "derivation-like"), with the naive-matching pain (the "you already know it's an I, not a D, so skip ahead" intuition from his interview) as the opening, Cook's theorem as the seed that says "linear is possible," and the mechanical distillation of the automaton → the failure function.

## Why Cook's construction yields the failure function (the load-bearing link)
- A 2DPDA recognizing "begins with even palindrome / palstar" works by: at text position, push the left half, then read forward comparing against the popped stack (reversal); on mismatch, back up. Naively exponential because it re-explores.
- Cook: each **surface configuration** c=(p,i,A) (state, head position, top stack symbol) in a terminating run has a **unique terminator** = the configuration reached when the stack first drops below c's height. **Memoize terminators** (table T[c]); then each config is processed once → O(n). The sharing of terminators is *why* it's linear (Glück).
- "Distilling" this for the specific palindrome/matching automaton: the only state that matters when you've matched a prefix and hit a mismatch is *how long a prefix you've matched* (= position j in the pattern). The terminator-table collapses to a table indexed by j: "given a mismatch after matching j−1 pattern chars, where do you resume?" = **next[j] / the failure function**. The exponential backtracking the automaton would do is precomputed once, off the pattern alone. The text head **never moves backward** — exactly Morris's "don't back up the file" goal, and exactly what the linear 2DPDA simulation guarantees.

## The algorithm (final, from §1–§3)
- Slide pattern over text; maintain text pointer k, pattern pointer j. On match advance both. On mismatch at pattern[j], set j := next[j] (slide pattern, DON'T move k).
- `next[j]`: largest i<j with pattern[1..i−1]=pattern[j−i+1..j−1] AND pattern[i]≠pattern[j] (the ≠ refinement avoids a re-comparison that's bound to fail). The weaker `f[j]` (just the longest proper border) also gives linear time but loses the log_φ m per-step bound.
- Preprocessing computes next[] by **matching the pattern against itself** — same algorithm, text=pattern. O(m).
- f[j] = longest proper border of pattern[1..j−1]; next[j]=f[j] if pattern[j]≠pattern[f[j]] else next[f[j]].
- Correctness invariant (§2): with p=k−j, text[p+i]=pattern[i] for 1≤i<j, and no full match starts left of p.
- Total: text pointer k advances ≤ n times, j:=next[j] performed ≤ n times total (amortized; j increases by 1 per outer step, stays ≥0, so decreases ≤ n times). O(m+n), text never backs up, O(m) memory.

## Design-decision → why
- **Don't move the text pointer back** → Morris's buffering pain; Cook's linear simulation is precisely a no-backup scan. The whole point.
- **A precomputed table indexed by pattern position j** (not by text) → all the automaton's would-be backtracking depends only on the matched-prefix length, which is j; it's a function of the *pattern alone*, so precompute once.
- **next[j] uses the ≠ refinement** (pattern[i]≠pattern[j]) rather than the plain border f[j] → if pattern[i]=pattern[j], resuming at i just re-compares the same mismatching text char; skip straight to next[f[j]]. Needed for the per-character O(log_φ m) bound (else pattern a^m gives m redundant comparisons at one text char — §5). Plain f[j] still gives overall linear time though.
- **Preprocess by matching pattern against itself** → next[]'s defining border condition is identical in form to the matching invariant; reuse the same loop.
- **T[0] = −1 sentinel (0-indexed canonical code)** → encodes "mismatch at the very first pattern char, slide pattern one and advance text" cleanly without a special branch; corresponds to next[1]=0 in the paper.
- **Amortized linearity** → text pointer monotonically advances; the inner j:=next[j] loop, summed over the whole scan, runs ≤ (number of times j was incremented) = ≤ n. Same argument for preprocessing with m.

## In-frame plan
- Open from the naive scanner's pain: O(nm), and the buffering nuisance of backing up the text file. Narrator = inventor (Knuth's route).
- The seed: Cook's theorem (a real, citable prior result by S.A. Cook) — a 2DPDA, however slow, simulates on a RAM in O(n). Stare at the palstar automaton; can't beat n² by hand; so go *through* Cook's construction to distill why it's linear.
- Derive: surface configs, unique terminators, memoize → the table; collapse to "indexed by matched-prefix length j" → failure function; the ≠ refinement; self-matching preprocessing; the invariant; the amortized count; the log_φ m / Fibonacci worst-case as a noticed sharpening. Land on runnable code = code/kmp_canonical.py.
- NOT in-frame: don't cite the 1977 paper; Cook 1972 / Morris 1969 prior-art citations are fine in context.md.
