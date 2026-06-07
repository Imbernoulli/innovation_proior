# Quicksort

## Problem

Sort N items, held contiguously in a computer's fast random-access store, into ascending key order —
quickly (far better than the N² of the naive sorts) and **in place** (no second array the size of
the data). The motivating use: putting the words of a sentence into alphabetical order so they can be
looked up in one forward pass of a dictionary held on slow magnetic tape.

## Key idea

Sort by *partitioning around a bound* instead of by swapping neighbours:

1. **Partition.** Choose one item's key as the bound. With a single pass, rearrange the segment so
   that every item with key <= bound is below a dividing line and every item with key >= bound is
   above it. One comparison against the bound sends each item to its correct *side* in one move, the
   leap that a neighbour-swap sort cannot make. Do it in place with two pointers scanning inward from
   the ends: the lower pointer stops at a key greater than the bound, the upper pointer stops at a key
   less than the bound, and the two stopped items are exchanged until the pointers cross.
2. **Recurse.** The two resulting sub-segments are independent, so sort each by the same method.
   The partition returns the two recursive ranges `lo..j` and `i..hi`, not a single pivot index.

Balanced splits give ≈ N log N: the work per level is ≈ N (one partition pass over all items at that
level) and there are ≈ log₂ N levels. A random (or median-of-sample) bound keeps the split balanced
on average and prevents already-sorted input from being a systematic worst case.

**Cost.** With a random bound, the expected comparison recurrence has the form
`C_N = (1/N) * sum_{r=1..N} (C_{r-1} + C_{N-r}) + aN + O(1)`. For the usual comparison count this is
`2N ln N + O(N)`, or about `1.39 * N log2 N`. The information-theoretic floor for a comparison sort
is `log2(N!) ~= N log2 N`, so the average is above the floor by the factor `2 ln 2 ~= 1.4`. Worst
case, if the bound is consistently extreme, is `N^2`.

**Termination.** Equality is deliberately passed over by both scans, so equal keys do not trap the
pointers. When the pointers cross, the dividing line is between `j` and `i`. If the item that supplied
the bound still lies inside one recursive side, it is exchanged to that side's edge and excluded by
moving `i` or `j`; in equality edge cases the remaining pieces are already singletons. No recursive
call receives the same whole segment back.

**The recursion / the nest.** The collection of partitioned-but-unsorted segments forms a
last-in-first-out list. With language recursion the procedure calling itself on a sub-segment keeps
this list automatically (the chain of suspended calls *is* the list). Without recursion you maintain
an explicit "nest" (a pushdown stack of segment bounds); always postponing the *larger* segment and
recursing into the *smaller* caps the nest depth at log₂ N.

## Algorithm

```
partition(a, lo, hi):
    f := random position in lo..hi
    bound := key(a[f])
    i := lo; j := hi
    scan i upward while key(a[i]) <= bound
    scan j downward while key(a[j]) >= bound
    while i < j:
        exchange a[i] and a[j]
        step i upward and j downward
        resume the two scans
    if the bound-supplying item lies inside the upper recursive side:
        exchange it with the first item of that side; move i upward
    else if it lies inside the lower recursive side:
        exchange it with the last item of that side; move j downward
    return i, j

quicksort(a, lo, hi):
    if lo < hi:
        i, j := partition(a, lo, hi)
        quicksort(a, lo, j)
        quicksort(a, i, hi)
```

## Code

```python
import random

def key(item):
    return item                       # sort key (for words, the spelling)

def exchange(a, p, q):
    a[p], a[q] = a[q], a[p]

def partition(a, lo, hi):
    f = random.randint(lo, hi)        # an actual item, chosen randomly for average balance
    bound = key(a[f])
    i = lo
    j = hi
    while True:
        while i < hi and key(a[i]) <= bound:
            i += 1
        while j > lo and key(a[j]) >= bound:
            j -= 1
        if i < j:
            exchange(a, i, j)
            i += 1
            j -= 1
            continue
        if i < f:                     # bound item lies in the upper recursive side
            exchange(a, i, f)
            i += 1
        elif f < j:                   # bound item lies in the lower recursive side
            exchange(a, f, j)
            j -= 1
        return i, j

def sort(a, lo=0, hi=None):
    if hi is None:
        hi = len(a) - 1
    if lo < hi:                       # a segment of 0 or 1 items is already sorted
        i, j = partition(a, lo, hi)
        sort(a, lo, j)
        sort(a, i, hi)
    return a
```

Explicit-nest form, for a machine without language recursion (the part that is hard to hand-code):

```python
def sort_with_explicit_nest(a):
    lo, hi = 0, len(a) - 1
    nest = []                         # pushdown list of postponed segments

    def segment_size(segment):
        first, last = segment
        return max(0, last - first + 1)

    while True:
        while lo < hi:
            i, j = partition(a, lo, hi)
            left, right = (lo, j), (i, hi)
            left_size, right_size = segment_size(left), segment_size(right)
            # continue on the smaller, postpone the larger -> nest depth <= log2(N)
            if left_size < right_size:
                if right_size > 1:
                    nest.append(right)
                lo, hi = left
            else:
                if left_size > 1:
                    nest.append(left)
                lo, hi = right
        if not nest:
            break
        lo, hi = nest.pop()           # resume the most-recently-postponed segment
    return a
```

## Practical refinements

- **Small segments**: below a cutoff of a few items, sort by a special-purpose routine (e.g. insertion
  sort) rather than recursing.
- **Sentinels**: place impossibly-large/small keys at the ends so the inner comparison loop can drop
  the pointer-range test.
- **Better pivots**: take the median of a small random sample to push the
  average comparison count nearer the floor and further from the N² worst case.
