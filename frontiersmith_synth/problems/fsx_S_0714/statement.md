# Grid Librarian: Reordering Errands to Stop Re-Shelving

## Problem

A branch library ran a fire-safety reshuffle: every reading table and every
book slot got a brand-new numeric code, and nobody kept the map back to the
old catalogue. All that survives is the errand log from the last audit.

The audit consists of **N errands**. Two kinds are mixed together
(unlabelled -- you must tell them apart from their shape):

- a **consultation** errand reads several shelf slots (to compare their
  current contents) and then writes one private result slot;
- a **clearance** errand is a quick one-off check: it reads a single
  dedicated slot and writes that same slot (ticking it off). Some
  consultations cannot start until a handful of their own clearances are
  done first.

You are given every such **dependence**: `u v` means errand `u` must be
completed strictly before errand `v`.

You must output a full ordering of the N errands -- a permutation that
respects every dependence -- to minimize how many times the librarian has to
walk to a distant shelf.

The librarian works with a **cart of capacity K**. To touch a slot (read or
write), if it's already on the cart, no walk is needed (a *hit*), and it
becomes the most-recently-used item on the cart. Otherwise it's a *walk* (a
*miss*): the item is fetched and placed on the cart, evicting the
**least-recently-used** item first if the cart already holds K items. Within
one errand, its read slots are touched in the exact order listed, then its
write slot.

The audit doesn't say so, but the consultations actually belong to several
distinct reading-list collections, each drawing on its own small pool of
slots. The reshuffle erased every numeric hint of which consultation belongs
to which collection; the only trace left is which slots get touched
*together* by the same errand.

## Input (stdin)

```
N M K
id_1 k_1 a_1 ... a_k_1 w_1
...
id_N k_N a_1 ... a_k_N w_N
u_1 v_1
...
u_M v_M
```
`id_i` are the N errand ids (some permutation of a size-N id set, not
necessarily 0..N-1 in the order given). `k_i` is how many slots errand
`id_i` reads (`a_1..a_k_i`, in the order they are touched); `w_i` is the
slot it writes. Each of the M dependence lines `u v` means errand `u` must
precede errand `v` in your output.

## Output (stdout)

N whitespace-separated integers: a permutation of the N errand ids, in your
chosen execution order.

## Feasibility

- Must be exactly N tokens, each an integer, forming a permutation of the
  given errand-id set (no missing/duplicate/foreign id, no non-finite
  token).
- For every dependence `u v`: `u` must appear strictly before `v`.
Any violation scores 0.

## Objective & Scoring

Simulate the cart (capacity K, LRU eviction, starting empty) over the
address trace produced by your order, and count the total misses `F`. The
checker also builds its own cache-blind baseline order (ascending-id
topological sort) and counts its misses `B`. Score:

```
Ratio = min(1, B / F)
```
(clamped and rescaled internally; a 10x-lower miss count than the baseline
already saturates the score). Fewer misses is better -- this is a
minimization objective.

## Constraints

1 ≤ N ≤ ~450, 0 ≤ M < N, 1 ≤ k_i ≤ 5. Time limit 5s, memory 512MB.

## Worked Example (illustrative shape only, not a real test)

Suppose N=4, M=0, K=2, and the four errands (ids 0..3) are:
```
0 3 0 1 2 4
1 3 1 0 3 5
2 3 2 0 3 6
3 3 3 1 2 7
```
Take the order `0 1 2 3`. The access trace is:
`0,1,2,4, 1,0,3,5, 2,0,3,6, 3,1,2,7` (16 touches). Simulating an LRU cart of
size 2 from empty: the first two touches (0, then 1) miss (cart now `[0,1]`);
the third (2) misses and evicts 0 (cart `[1,2]`); and so on. Working through
all 16 touches gives 15 misses out of 16 (only touch #13, a repeat of slot
3, hits). This happens to be exactly the checker's own cache-blind baseline
order here (M=0, ties broken by ascending id), so `B=F=15` and `Ratio ≈
0.1` -- a smarter order that keeps related slots together scores higher.
