What separates this from an ordinary maximum-subarray is the deletion: from any contiguous non-empty window I may drop at most one element, and what remains must still be non-empty — so a length-`1` window keeps its single element, it can never be emptied. That non-empty-remainder clause is what keeps the answer defined on all-negative input: the best I can do there is the least-negative single element, never the empty sum. This is not the "empty subarray allowed, so the floor is `0`" problem; a non-empty subarray is mandatory.

Scale first, because it fixes the types before I touch the algorithm. With `n <= 10^5` and `|a[i]| <= 10^9`, a window sum can reach `10^5 * 10^9 = 10^14`, two orders of magnitude past the ~`2.1*10^9` reach of 32-bit. Every accumulator that holds a running sum has to be `long long`; an `int` here is a silent wrong answer on the large tests, not a crash that would warn me.

The cheap idea is to lean on Kadane, which I already know cold: find the best no-deletion window, then delete its most negative element if that helps. But Kadane picks its window under the no-deletion rule, and a deletion changes which window is best — so I don't trust that the optimal window to delete from is the one Kadane stops on. It breaks concretely on `[5, 6, -3, 3, -5]`. Kadane's running max is `11` on `[5, 6]` — the `-3` is exactly why it stops there — and that window's most negative element is the positive `5`, so deleting nothing helps and the greedy reports `11`. But `[5, 6, -3, 3]` with the `-3` deleted gives `5 + 6 + 3 = 14`. The deleted element is the very one that made Kadane cut the window short, so the greedy never even considers the right window. The failure is structural, not something a tweak fixes, and a small random search finds many more of the same shape (e.g. `[3, 6, 6, -6, 5, 1]`, greedy `15` vs correct `21`). Greedy is out.

So I carry the state the scan actually depends on — how many deletions I have spent inside the current window — as two Kadane-style quantities, both "best subarray *ending exactly at* `i`": `noDel` with no element deleted, `oneDel` with exactly one. `noDel` is ordinary Kadane, `max(a[i], noDel_{i-1} + a[i])`: start fresh at `i` or extend. For `oneDel` there are two ways to end at `i` having spent one deletion. Either the deleted element is `a[i]` itself — take a no-deletion window ending at `i-1`, append `a[i]`, and immediately delete it, leaving exactly `noDel_{i-1}` — or the deletion happened earlier and `a[i]` is kept, extending `oneDel_{i-1} + a[i]`. So `oneDel_i = max(noDel_{i-1}, oneDel_{i-1} + a[i])`. Both transitions read the *previous* generation of the pair.

**Stress-testing the greedy before committing.** Hand-waving "delete the worst element of the best window feels right" is exactly how a wrong solution gets shipped. So I attack it with concrete instances. I want a case where the *best window to delete from* is a different, larger window than the one plain Kadane stops on. Intuitively that happens when a single bad element makes plain Kadane cut the window short — but with a deletion that bad element is free, so it would have been profitable to keep extending through it.

I seed both at a negative sentinel standing for "no subarray yet". At `i = 0`, `noDel` becomes `a[0]` and `oneDel` stays at the sentinel — correct, because deleting the lone element would empty the window. `oneDel` only becomes a real value from `i = 1`, which is exactly when `noDel_{i-1}` refers to a genuinely non-empty remainder. The answer is the running max over both states at every `i`.

Let me confirm the recurrence by hand on the example `a = [5, 6, -3, 3, -5]`, expected answer `14`. Initialize `noDel = oneDel = NEG`, `best = NEG`.
- i=0 (5): `newNoDel = max(5, NEG+5) = 5`; `newOneDel = max(NEG, NEG+5) = NEG`. State `(noDel, oneDel) = (5, NEG)`, `best = 5`.
- i=1 (6): `newOneDel = max(noDel=5, oneDel+6=NEG) = 5` (delete the `6`, keep the `5`); `newNoDel = max(6, 5+6=11) = 11`. State `(11, 5)`, `best = 11`.
- i=2 (-3): `newOneDel = max(noDel=11, oneDel + (-3) = 5-3=2) = 11` (window `[5,6,-3]` delete `-3` -> `11`); `newNoDel = max(-3, 11-3=8) = 8`. State `(8, 11)`, `best = 11`.
- i=3 (3): `newOneDel = max(noDel=8, oneDel + 3 = 11+3=14) = 14` (extend the one-deletion window `[5,6,-3 (deleted)]` by `+3`); `newNoDel = max(3, 8+3=11) = 11`. State `(11, 14)`, `best = 14`.
- i=4 (-5): `newOneDel = max(noDel=11, oneDel + (-5) = 14-5=9) = 11`; `newNoDel = max(-5, 11-5=6) = 6`. State `(6, 11)`, `best = max(14, 11) = 14`.

Tracing `[5, 6, -3, 3, -5]` to confirm the recurrence lands on `14`, starting `(noDel, oneDel) = (NEG, NEG)`:
- `i=0` (5): `noDel = 5`, `oneDel = NEG`, best `5`.
- `i=1` (6): `oneDel = max(5, NEG) = 5` (delete the `6`, keep `5`), `noDel = max(6, 11) = 11`, best `11`.
- `i=2` (-3): `oneDel = max(11, 5-3) = 11`, `noDel = max(-3, 8) = 8`, best `11`.
- `i=3` (3): `oneDel = max(8, 11+3) = 14`, `noDel = max(3, 11) = 11`, best `14`.
- `i=4` (-5): `oneDel = max(11, 14-5) = 11`, `noDel = max(-5, 6) = 6`, best stays `14`.

The winning path is precisely the window `[5, 6, -3, 3]` with the `-3` deleted — the one the greedy structurally could not reach.

Buggy trace. Start `noDel = oneDel = best = NEG`.
- i=0 (10): `noDel = max(10, NEG+10) = 10`; `oneDel = max(noDel=10, NEG+10) = 10`; `best = 10`. Already wrong: `oneDel = 10` claims a one-deletion subarray ending at index 0 with score `10`, but the only such "subarray" would be `[10]` with its element deleted — empty, illegal. The bug let `oneDel` read the just-updated `noDel` (`10`) as if it were `noDel_{i-1}`.
- i=1 (-100): `noDel = max(-100, 10-100) = -90`; `oneDel = max(noDel=-90, oneDel + (-100) = 10-100 = -90) = -90`; `best = 10`.
- i=2 (10): `noDel = max(10, -90+10) = 10`; `oneDel = max(noDel=10, oneDel+10 = -90+10 = -80) = 10`; `best = 10`.

The one place this dies in transcription is update order. `oneDel_i` reads `noDel_{i-1}`, so if I write the loop body as `noDel = max(a[i], noDel + a[i]); oneDel = max(noDel, oneDel + a[i]);`, the second line reads the just-overwritten `noDel_i`, letting `oneDel` "delete `a[i]`" from a window that already restarted fresh at `a[i]` — an empty remainder dressed up as a legal one. That naive order returns `10` on `[10, -100, 10]` (true answer `20`: delete the `-100`), because at `i = 0` it already reads `noDel = 10` into `oneDel`. The fix is to compute both new values from the old pair into temporaries and only then assign:

Re-trace `[10, -100, 10]`. Start `(noDel, oneDel) = (NEG, NEG)`.
- i=0 (10): `newOneDel = max(NEG, NEG) = NEG`; `newNoDel = max(10, NEG) = 10`. State `(10, NEG)`, `best = 10`.
- i=1 (-100): `newOneDel = max(noDel=10, oneDel + (-100) = NEG) = 10`; `newNoDel = max(-100, 10-100) = -90`. State `(-90, 10)`, `best = 10`.
- i=2 (10): `newOneDel = max(noDel=-90, oneDel + 10 = 10+10 = 20) = 20`; `newNoDel = max(10, -90+10) = 10`. State `(10, 20)`, `best = 20`.

```
long long newOneDel = max(noDel, oneDel + a[i]);  // old noDel, old oneDel
long long newNoDel  = max(a[i], noDel + a[i]);
noDel = newNoDel; oneDel = newOneDel;
```

Re-tracing `[10, -100, 10]` with temporaries: `i=0` gives `(10, NEG)`; `i=1` gives `oneDel = max(10, NEG) = 10`, `noDel = -90`; `i=2` gives `oneDel = max(-90, 10+10) = 20`, `noDel = 10`, best `20`. Correct.

The corners fall out of the recurrence and the seed. `n = 1` with `a = [-7]` gives `noDel = -7`, `oneDel = NEG`, answer `-7` — not `0`, since the window can't be emptied. All-negative like `[-3, -1, -4]` peaks at the least-negative single element `-1`, since deleting one element of an all-negative window still leaves a negative sum. Overflow is covered by `long long` on every sum; the sentinel `LLONG_MIN/4` is only ever read inside a `max`, and adding `a[i]` to it stays far from underflowing `LLONG_MIN`, so it never wraps into a spuriously large value.

Hand traces only cover the cases I thought of, so I differential-tested against a from-scratch brute force — enumerate every subarray, score it with no deletion and, for length `>= 2`, with each interior element deleted, take the global max. Across the explicit edge battery plus 600+ randomized cases over biased distributions (all-negative, all-positive, alternating, near-zero, full `10^9` magnitudes), zero mismatches. On `n = 10^5` with random `10^9`-magnitude values the DP finishes in well under 10 ms and a few MB, so the `O(n)` / `O(1)` cost has real headroom under the 1 s / 256 MB budget.

I ship the two-state DP: the loop above, both states seeded at `LLONG_MIN/4`, folding `noDel` and `oneDel` into a running `best`, printing one integer. The full self-contained program is in the answer.
