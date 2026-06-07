# Synthesis — Quicksort, as Hoare found it

## The pain point (real, from the self-account)
Moscow State University, 1959-60. British Council exchange student; National Physical Laboratory
has offered him a job on machine translation of Russian -> English. Translation needs to look up
each word of a Russian sentence in a Russian-English dictionary. The dictionary lives on a long
magnetic tape, in alphabetical order, and reading it end-to-end takes minutes. Random access on
tape is brutal (rewind). So: if you sort the words of the sentence into the dictionary's order
first, you can look them all up in ONE forward pass of the tape. The sub-problem he actually has to
solve is: sort a short list of words held in fast (random-access) memory, fast.

## The path (in Hoare's own beats)
1. He knows Mercury Autocode; figures he can just program a sort. (He claims not to know the
   existing sorting literature.)
2. First thought: the obvious method (bubble sort / insertion sort) — rejected as "obviously rather
   slow" (quadratic).
3. Second thought, "just as quickly": Quicksort. The partition idea.
4. He writes the PARTITION in Mercury Autocode immediately — that part is easy.
5. WALL: he cannot write the bookkeeping for the *list of still-unsorted segments*. The
   administration of the recursion is too complicated in Mercury Autocode (no recursion, no
   built-in stack). He gives up on programming it.
6. Resolution: reading the ALGOL 60 report, he discovers recursive procedures (Brighton course,
   Easter 1961). The recursion administers the segment list "behind the scenes" via a stack. Now he
   can express it. He writes the procedure, names it Quicksort, publishes Algorithm 64.

## The key insight (derive it, don't state it)
Why is the obvious sort slow? In a comparison-exchange sort like bubble/insertion, an element can
only move past its *immediate* neighbor per step; a misplaced element crawls one position at a time,
so it takes ~N passes, ~N^2 work. The waste: comparisons of pairs that are already roughly in order
teach you almost nothing.

The leverage: a single comparison against a well-chosen *reference* element tells you which HALF of
the array an item belongs in. Pick one key value, the "bound" (pivot). Compare every other item to
it ONCE. Items with smaller key go left, larger go right. Now you have a dividing line, known
position, with the guarantee: everything left <= bound <= everything right. The two sides can be
sorted INDEPENDENTLY — sorting one never disturbs the other, because no element ever needs to cross
the line again. That independence is the whole engine: it converts "sort N" into "partition N (one
linear pass) + sort two pieces", and recursing gives ~N log N when the pieces are balanced.

## Partition mechanics (from the 1962 paper, exact)
- Two pointers. Lower starts at lowest address, moves up; upper starts at highest, moves down.
- Lower pointer moves up while key <= bound; stops at first key > bound.
- Upper pointer moves down while key >= bound; stops at first key < bound.
- The two stopped items are on the wrong sides -> exchange them, step both inward, repeat.
- Continue until the pointers cross. When they cross, suppress the exchange, draw the dividing line
  between them. Done in ONE pass (~N comparisons), in place (only exchanges, no extra array).

## The termination subtlety (from the 1962 paper — a real wall he flags)
"An awkward situation": if the bound is the greatest/least key in the segment, or all keys equal,
the dividing line could land OUTSIDE the segment, so a sub-segment equals the whole segment ->
infinite loop. Fix: guarantee at least one item is placed in its final position each partition. The
item that supplied the bound is, after the final exchange, put at the dividing point; so the sum of
the two resulting segment sizes is always one less than the original. Each partition strictly
shrinks the total -> termination guaranteed.

## The recursion wall + the nest (the heart of the Hoare story)
He needs to remember the not-yet-sorted segments. Without recursion in Mercury Autocode he must
manage this list himself. Two design facts from the paper:
- The list is LIFO: a "nest" (pushdown store / stack). Postpone a segment -> push (lo,hi); when a
  segment finishes (1 or 0 items, or small-sort), pop the most recently postponed one.
- Bounding the nest: if you always postpone the LARGER of the two segments and continue on the
  smaller, the nest never holds more than log2(N) entries — you can size it in advance. (This is
  the worst-case-stack-depth fix.)
- The footnote: the ALGOL recursive version is "deceptively simple, since the use of recursion
  means that the administration of the nest does not have to be explicitly described." i.e. the
  recursion stack IS the nest. That is exactly what he could not hand-code before ALGOL.

## Practical refinements (Part Two of the 1962 paper; in-frame as "things I'd want")
- Small segments (< 3-4 items): sort by a special-purpose routine, not by recursing.
- Sentinels at both ends: lets you drop the pointer-range test from the inner comparison loop.
- Choose the bound randomly (or median of a small sample) to keep the randomness assumption valid /
  avoid the already-sorted worst case. (Algorithm 63 literally uses random(M,N).)
- Cyclic exchange / partition-without-exchange (copy form) to save instructions on certain machines.

## Cost (Part One of the 1962 paper — derive it)
Average comparisons satisfy T_N = (2/N) sum_{1..N} T_r + aN + b + c, leading to ~2N ln N = 2N log2 N
* ... actually expected comparisons ~ 2N log_e N ~ 1.4 * (N log2 N). The information-theoretic floor
is log2(N!) ~ N log2 N comparisons; Quicksort's average exceeds it by factor 2 log_e 2 ~ 1.4.
Median-of-sample pivoting can shrink that factor. Worst case (bad pivots every time) is N^2, the
reason pivot choice matters.

## In-frame discipline
- Narrator IS Hoare in his Moscow dorm room, 1959-60, then at the Brighton ALGOL course 1961.
- Never cite the Turing lecture / oral history / the 1962 paper as artifacts.
- The "bubble sort first, rejected as slow" beat, the "wrote partition but couldn't do the segment
  list" wall, and the "ALGOL recursion rescued it" resolution are the spine.
- End on real, runnable partition + recursion code (both the recursive form and the explicit-nest
  form, since the nest is the part he actually struggled with).

## Code framework correspondence
Pre-method scaffold: a bare in-memory sort harness — `def sort(a): # TODO the method` plus a
comparison/exchange primitive — nothing about partition/pivot/nest (those are the contribution).
Final code fills: hoare_partition(), quicksort_recursive(), quicksort_explicit_stack() (the nest).
