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
matters because this sort runs once per sentence, inside the translation loop.

## Background

The machine has a small random-access working store and a much larger but far slower backing store
(magnetic tape or disc). The cost model that matters is: comparisons of keys, and movements of items
in the store. A "sort" rearranges items so their keys ascend; we count the comparisons and the data
movement, and we care about how both scale with N, the number of items.

There is a hard fact about how cheap sorting could ever be. To distinguish which of the N!
possible orderings the input is in, using only binary (yes/no) key comparisons, you need at least
log2(N!) comparisons, which for large N is about N log2 N. So N log2 N comparisons is a floor no
comparison sort can beat.

A relevant programming fact of the time: the available machine language (Mercury Autocode and its
kin) has no notion of a procedure that calls itself, and no automatically managed stack. A routine
has fixed storage; if an algorithm needs to keep an open-ended list of "things still to be done" and
process them in last-started-first-finished order, the programmer has to build and administer that
list by hand, in a fixed block of store, getting every index right themselves.

## Baselines

**Insertion sort.** Walk left to right; to place the next item, slide it leftward past every
already-placed item with a larger key until it sits in order. Simple and in place, each insertion
shifts items one slot at a time, giving order N-squared comparisons and moves on random data.

**Bubble sort.** Repeatedly pass over the list comparing each adjacent pair and swapping if out of
order; after each pass the largest unplaced item has "bubbled" to its end. In place and trivial to
write; order N passes, order N work each, order N-squared total.

**Shellsort (Shell, 1959, "A high-speed sorting procedure", CACM 2(7):30-32).** A refinement of
insertion sort: do insertion-sort-like passes but comparing items a large *gap* apart, then repeat
with smaller and smaller gaps down to gap 1. The early wide-gap passes let a far-out-of-place item
move a long distance in one step, so by the time the gap is 1 the array is nearly sorted. Empirically
much faster than the quadratic sorts and in place; its exact running time depends on the gap sequence
and is subquadratic.

**Merge sort.** Split the list in two, sort each half, then merge the two sorted halves by repeatedly
taking the smaller front item. Merging is linear and the recursion is balanced, achieving order
N log N comparisons. The merge step uses a second area the size of the data to merge into and streams
data rather than rearranging in situ.

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

The deliverable is a single self-contained C++17 program. It reads `N` followed by `N` integer keys
from standard input, rearranges the keys into ascending order in place, and writes the sorted keys to
standard output, space-separated on one line. Any additional bookkeeping store must be managed
explicitly inside the program.

```cpp
#include <bits/stdc++.h>
using namespace std;

static void quicksort(vector<long long>& a) {
    // TODO: implement
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    long long n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (long long k = 0; k < n; ++k) cin >> a[k];
    quicksort(a);
    for (long long k = 0; k < n; ++k) {
        cout << a[k];
        if (k + 1 < n) cout << ' ';
    }
    cout << '\n';
    return 0;
}
```
