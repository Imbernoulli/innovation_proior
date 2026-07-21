Two features of this problem decide the shape of the solution before I write any
algorithm. First, the queries are all handed to me up front — the offline
setting — so I am free to answer them in any order I like as long as I scatter
the results back to their input positions. That freedom is the whole lever.
Second, values run over `[-10^9, 10^9]` and can be negative, so I cannot index a
count array by a raw value; I will coordinate-compress `a` to ranks in `[0, V)`
with `V <= n`. And with up to `2*10^5` answer lines I want to accumulate the
output in one string and flush once rather than issue that many `cout` calls.
Those two housekeeping facts — compress the values, batch the output — are fixed
regardless of the algorithm.

The definitional method is: for each query, walk `a[l..r]` and count distinct
values with a `seen` set. Unarguably correct, and I will keep it as a
brute-force oracle on small inputs. But a single query can span the whole array,
so each is `O(n)` worst case and the batch is `O(q*n)`: at the limits `2*10^5 *
2*10^5 = 4*10^10` element touches, tens of seconds against a 2-second budget.
Out for the full constraints.

Can I answer each query independently and sublinearly, the way prefix sums give
range sums in `O(1)`? No — and the reason is what points me at the real method.
Distinct-count is not subtractive across a split, because a value can appear on
both sides. On `a = [1, 2, 1]`, the prefix `[0,1]` has distinct count 2 and the
prefix `[0,2]` also has 2, yet `2 - 2 = 0` is not the count of the suffix `[2,2]`
(which is 1). A single scalar per prefix cannot untangle the double-counting.
(There is an online `O((n+q) log n)` sweep-by-`r` with a Fenwick tree marking
last occurrences; it works but needs a second structure and buys nothing in
wall-clock over what follows, so I only fall back to it if the simpler tool
misses the limit.)

The move is to stop rebuilding from scratch between queries. Keep a single
window `[L, R]` and a table `cnt[value]` = occurrences of that value inside the
window. Extending the window right is `O(1)`: increment `cnt[a[R+1]]`, and if it
went `0 -> 1` the running distinct total rises by one. Shrinking is the mirror:
decrement, and if it hit `0` the total falls by one. So moving an endpoint by one
costs `O(1)`, and answering a query is just "march `L` and `R` to its endpoints,
read the total." The entire cost is now the total distance the two pointers
travel across all queries — and in input order that is unbounded, since
adversarial queries ping-pong the window across the whole array. So the game
reduces to one question: in what order should I visit the queries so that total
pointer travel is small?

Split the index axis `[0, n)` into blocks of width `B` and sort the queries by
`(block of l, then r)`. Now cost the two pointers so I can pick `B` rather than
guess it:

- *Left pointer.* Within one `l`-block every query has `l` inside a width-`B`
  window, so each consecutive `L`-move is at most `B`: `O(q*B)` in total.
- *Right pointer.* Within one block the queries are sorted by `r`, so `R` only
  marches forward, covering at most `O(n)` per block. With `n/B` blocks (and one
  `O(n)` snap-back per block boundary) that is `O(n^2 / B)` in total.

Total travel `O(q*B + n^2/B)` is minimized by balancing the terms: `q*B =
n^2/B`, so `B = n / sqrt(q)`, and both terms collapse to `O(n*sqrt(q))`. The
whole batch is `O((n+q) sqrt(q))` pointer moves. At `n = q = 2*10^5`, `sqrt(q) ~=
450`, so ~`9*10^7` moves — a fraction of a second, comfortably inside 2s. This
block-sort key is the whole reason the method is fast; the `O(1)` add/remove is
the easy half.

One refinement halves the right-pointer constant. With plain `(block(l), r)`
order, `R` sweeps left-to-right through a block and then snaps back to the block
start for the next one. Sort `r` ascending in even-numbered blocks and descending
in odd-numbered ones (boustrophedon): `R` sweeps right through one block, and the
next begins where it ended and sweeps left, no reset. Same asymptotics, roughly
half the travel — worth it at full scale.

Data layout falls out of this. Compress the values (sort, unique, `lower_bound`)
so `cnt` is a plain `vector<int>` of size `V <= n`, oblivious to sign and
magnitude. Carry `(l, r, idx)` per query so I can scatter answers back to input
order. The window is inclusive `[L, R]`, started *empty* as `curL = 0, curR =
-1`, so the first "extend right" lands on index 0.

The delicate part is the order of the four while-loops that march the pointers.
My first cut:

```
while (curL < Q.l) remove(curL++);  // shrink left
while (curR < Q.r) add(++curR);     // grow right
while (curL > Q.l) add(--curL);     // grow left
while (curR > Q.r) remove(curR--);  // shrink right
```

with `add(pos)` doing `if (cnt[a[pos]]++ == 0) distinct++;` and `remove(pos)`
doing `if (--cnt[a[pos]] == 0) distinct--;`. The differential tester against the
rescan oracle catches a mismatch, and the minimal reproduction is telling: `a =
[1, 2]` (compressed `[0, 1]`), queries `(1,1)` then `(0,0)` — after sorting they
can land adjacent like this. On query `(1,1)`, starting `L=0, R=-1`, the very
first loop is `while (curL < 1) remove(0)` — but index 0 was never added, so
`cnt[0]` goes `0 -> -1`. The window invariant "`cnt` reflects exactly the
positions in `[curL, curR]`" is violated the instant a `remove` touches a
position outside the window, and everything downstream is corrupt.

The fix is forced: grow before you shrink. Do both `add`-loops (extend `R` right,
extend `L` left — these only enlarge the window) first, then both `remove`-loops.
That guarantees every `remove` acts on a position currently inside the window:

```
while (curR < Q.r) add(++curR);     // grow right
while (curL > Q.l) add(--curL);     // grow left
while (curR > Q.r) remove(curR--);  // shrink right
while (curL < Q.l) remove(curL++);  // shrink left
```

Now query `(1,1)` grows right to `{0,1}` then shrink-left drops index 0 for
answer 1, and no `remove` can touch an unadded position; the full differential
run comes back clean.

Compression is what makes the edge cases boring rather than dangerous:
negatives and zeros (`[-1,0,-1,5]` maps to `[0,1,0,2]`) and the `+-10^9` extremes
both collapse to `[0,V)`, so the count array is `O(n)` and never sees a negative
index; counts are bounded by `n` and the distinct total fits a `long long`
trivially. `n=1`, all-equal (every answer 1), all-distinct (`r-l+1`),
single-element ranges `l==r`, and repeated identical queries all fall out of the
same grow/shrink loops — a duplicate query is four no-op loops reading the same
total. The nastiest input for the count table is a size-1 or size-2 alphabet,
where a value's count repeatedly hits and leaves zero; the differential run
matches the oracle there too.

On a worst-case build — size-2 alphabet, mix of narrow and near-full ranges at
`n = q = 2*10^5` — wall time is ~0.1-0.16 s and peak memory ~10 MB, well inside
2 s / 256 MB. The block size is `max(1, n/sqrt(q))` so it never degenerates to 0
when `q` is tiny relative to `n`.

A Fenwick sweep-by-`r` or a persistent segment tree would reach the same
wall-clock with strictly more machinery and more bug surface, so Mo's algorithm
is the right fit at these limits. The full submission is in the answer.
