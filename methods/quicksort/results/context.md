# Context: sorting a list of words in random-access store, fast

## Research question

A machine-translation program reads a Russian sentence and must look up every word in a
Russian-English dictionary. The dictionary is held on a long magnetic tape, in alphabetical order;
reading it from one end to the other takes minutes, and tape has no usable random access — to jump
backward you rewind. So the only economical way to look up a whole sentence is in a single forward
pass of the tape, which requires the words of the sentence to already be in the dictionary's
alphabetical order before the lookup begins.

That reduces to a concrete, self-contained sub-problem: given a short list of items (words, each
with a sort key) sitting in the computer's fast, randomly-addressable store, rearrange them into
ascending key order, as fast as possible and without demanding much extra store. The store is small
and precious, so a method that needs a second array as big as the first is a real cost. Speed
matters because this sort runs once per sentence, inside the translation loop. The goal: a sort
whose running time grows much more slowly than the number of items squared, that works in place, and
that is simple enough to actually program on the machine at hand.

## Background

The machine has a small random-access working store and a much larger but far slower backing store
(magnetic tape or disc). The cost model that matters is: comparisons of keys, and movements of items
in the store. A "sort" rearranges items so their keys ascend; we count the comparisons and the data
movement, and we care about how both scale with N, the number of items.

The naive sorts all share one structural weakness. They are built out of comparing *adjacent* (or
near-adjacent) items and swapping the ones found out of order. Because each elementary step only
ever moves an item past its immediate neighbor, an item that belongs far from where it starts can
only migrate one position per pass. An item N positions out of place needs on the order of N steps
to get home, and there are N items, so the work piles up to order N-squared comparisons and
movements. The deeper waste is informational: comparing two items that are already in roughly the
right order relative to each other tells you almost nothing new, yet these methods spend most of
their comparisons doing exactly that.

There is also a hard fact about how cheap sorting could ever be. To distinguish which of the N!
possible orderings the input is in, using only binary (yes/no) key comparisons, you need at least
log2(N!) comparisons, which for large N is about N log2 N. So N log2 N comparisons is a floor no
comparison sort can beat; anything of order N-squared is far above it. The open practical question
is whether a method can get near that floor while still working in place in the small store.

A relevant programming fact of the time: the available machine language (Mercury Autocode and its
kin) has no notion of a procedure that calls itself, and no automatically managed stack. A routine
has fixed storage; if an algorithm needs to keep an open-ended list of "things still to be done" and
process them in last-started-first-finished order, the programmer has to build and administer that
list by hand, in a fixed block of store, getting every index right themselves.

## Baselines

**Insertion sort.** Walk left to right; to place the next item, slide it leftward past every
already-placed item with a larger key until it sits in order. Simple, in place, and genuinely fast
when the data is already nearly sorted. But each insertion can shift order-N items, and there are N
insertions, so on random data it is order N-squared comparisons and moves. The misplaced item still
only crawls one slot at a time.

**Bubble sort.** Repeatedly pass over the list comparing each adjacent pair and swapping if out of
order; after each pass the largest unplaced item has "bubbled" to its end. In place, trivial to
write — and the canonical embodiment of the adjacent-swap weakness: order N passes, order N work
each, order N-squared total, with an item moving at most one position per comparison.

**Shellsort (Shell, 1959, "A high-speed sorting procedure", CACM 2(7):30-32).** A clever escape
from the adjacent-only trap: do insertion-sort-like passes but comparing items a large *gap* apart,
then repeat with smaller and smaller gaps down to gap 1. The early wide-gap passes let a far-out-of-
place item leap a long distance in one move, so by the time the gap is 1 the array is nearly sorted
and the final ordinary insertion pass is cheap. Empirically much faster than the quadratic sorts and
in place; its exact running time depends on the gap sequence and is subquadratic but hard to pin
down. It is the method to beat: faster than insertion/bubble, in place, but still doing many passes
over the whole array and still built on near-neighbor comparison and exchange.

**Merge sort.** Split the list in two, sort each half, then merge the two sorted halves by repeatedly
taking the smaller front item. Merging is linear and the recursion is balanced, so it achieves order
N log N comparisons — at the floor, asymptotically. Its weakness is space and movement: the merge
step is not in place; the standard form needs a second area the size of the data to merge into, and
it streams data rather than rearranging in situ. In a small store where you want to fill the whole
store with data to be sorted, having to reserve a second copy is a serious cost.

The gap each leaves open: insertion and bubble are simple and in place but quadratic; Shellsort is in
place and fast in practice but still pass-over-the-whole-array and not near the N log N floor with a
clean analysis; merge sort hits the floor but pays an extra full array of store and moves data out
of place. Nothing on the table is simultaneously near-N-log-N, in place (no second array), and with
a tight inner loop.

## Evaluation settings

The natural yardstick is sorting random data in the random-access store and counting key comparisons
and item exchanges/movements as functions of N, plus measured wall-clock time on a real machine. A
concrete setting of the era: a computer with a small fast working store (hundreds of words) and a
slow block-addressed backing store (disc/drum/tape), with per-instruction inner-loop times in the
fraction-of-a-millisecond range and backing-store block access tens of milliseconds. Items are
multi-word records with a sort key (sometimes a multi-word key). The standing reference method to
compare against is a merge sorting method hand-coded for the same machine, run on random inputs of a
few hundred to a few thousand items, timed automatically by the computer.

## Code framework

The pieces that already exist: items live contiguously in an array `a` in the random-access store; we
can read and compare their keys and exchange two of them in place. A sort is a routine that
rearranges `a` into ascending order using only these primitives, ideally touching no store beyond `a`
itself.

```python
def key(item):
    # the sort key of an item (for words, the spelling)
    return item

def exchange(a, p, q):
    # swap two items in place — the only data-movement primitive we rely on
    a[p], a[q] = a[q], a[p]

def sort(a):
    # Rearrange a[0..len(a)-1] into ascending key order, in place,
    # fast (we want far better than N^2) and without a second array.
    # TODO: the method.
    pass

# An auxiliary fixed block of store we are allowed to manage by hand, if the
# method needs one. On this machine the language gives us no automatic stack,
# so any such block is ours to administer.
work_store = []  # TODO: only if the method needs it
```
