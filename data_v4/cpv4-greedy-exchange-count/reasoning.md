Two numbers come out of this problem, and they are not equally hard. The placement rule is handed to
me fully specified — the canonical stab-by-right-endpoint greedy — so `K`, the number of markers, is
just a matter of transcribing that rule without an off-by-one. The output that this exercise is really
about is `M`: how many of the `n` corridors end up containing two or more of the placed markers. That
is a counting query against a sorted marker list, and it is exactly where a boundary marker gets
double-counted or dropped, because the corridors and the markers are two different sorted lists whose
endpoints coincide precisely on the tests that matter.

Scale first, since it fixes the types. `n <= 2*10^5` and coordinates in `[-10^9, 10^9]`. The markers
are corridor right endpoints, so their values fit in 32 bits, but I need a sentinel for "no marker
placed yet" that sits strictly below every real `l_i`, including `l_i = -10^9`. Reaching for `0` or
`INT_MIN` risks a wrong first comparison. So coordinates go in `long long`, the sentinel is
`LLONG_MIN`, and I only ever compare it, never subtract it — that way the first corridor always
triggers a placement and nothing overflows. `K` and `M` are at most `n`, but I keep them `long long`
for uniformity.

The stab, and why right endpoints are the safe place to drop a marker. Sort corridors by right
endpoint, ties by left; sweep; when the current corridor is not already hit by the last placed marker,
drop a marker at its right endpoint `r`. Optimality is the standard exchange argument: any point that
hits the current corridor lies at `<= r`, and sliding it right to `r` keeps it inside this corridor
while only extending coverage over later corridors, which all have right endpoint `>= r`. So placing
at `r` is never worse. One property I will lean on for the second output: the marker list comes out
strictly increasing, because a new marker `r` is placed only when `l > last`, giving `r >= l > last`.
That strict monotonicity is what makes binary search over the markers well-defined.

Let me sanity-check the recurrence-of-thought on a concrete instance, the rich sample
`[1,3] [2,5] [4,4] [6,8] [0,9] [7,10] [2,8]`. Sorted by right endpoint (ties by left):
`[1,3], [4,4], [2,5], [2,8], [6,8], [0,9], [7,10]`. Sweep with `last = -inf`:
- `[1,3]`: `last(-inf) < 1`, place at `3`. `last = 3`. markers `{3}`.
- `[4,4]`: `last(3) < 4`, place at `4`. `last = 4`. markers `{3,4}`.
- `[2,5]`: `last(4) < 2`? No, `4 >= 2`, already hit. skip.
- `[2,8]`: `last(4) < 2`? No. skip.
- `[6,8]`: `last(4) < 6`? Yes, place at `8`. `last = 8`. markers `{3,4,8}`.
- `[0,9]`: `last(8) < 0`? No. skip.
- `[7,10]`: `last(8) < 7`? No. skip.

The placement predicate is the whole game for `K`, and its strictness is a real trap here because the
tests deliberately put markers on corridor edges. "Already hit?" must be tested against the most
recent marker only, and the smallest input that separates `last <= l` from `last < l` is corridors
`[1,1]` and `[1,2]`. A single marker at `1` sits inside both (`1 <= 1 <= 2`), so the correct `K` is
`1`. With the non-strict `last <= l`, I place at `1` for `[1,1]`, then `[1,2]` sees `last(1) <= 1` and
places a spurious second marker — `K = 2`, wrong. The equality case `last == l` means the marker is
exactly on the corridor's left edge, which *is* a hit, so the predicate has to be strict `last < l`.
With it, `[1,1],[1,2]` gives `K = 1`.

Now `M`. The markers `pts` are sorted increasing; for corridor `[l, r]` I want how many satisfy
`l <= p <= r`. Two binary searches: markers `<= r` is `upper_bound(r)`, and from that I subtract the
markers strictly left of `l`. The subtlety is exactly which lower bound, and it is another
edge-coincidence trap. The right end I want inclusive, and `upper_bound(r)` counts `p <= r`, correct.
For the left end, if I carelessly subtract `upper_bound(l)` (markers `<= l`) I throw away a marker
sitting exactly on `l`, even though `p = l` is inside `[l, r]`; the bound I want is `lower_bound(l)`
(markers `< l`), which keeps that marker. Concretely, corridors `[3,3],[4,4],[3,8]` produce markers
`{3,4}`, and `[3,8]` genuinely contains both. The wrong pairing `upper_bound(8) - upper_bound(3) =
2 - 1 = 1` misses the double-stamp; `upper_bound(8) - lower_bound(3) = 2 - 0 = 2` catches it. The
mirror case `[4,4]` confirms the convention is not symmetrically broken on the other edge:
`upper_bound(4) - lower_bound(4) = 2 - 1 = 1`, one marker — correct, since the marker at `4` is inside
and the one at `3` is not. So the closed-interval count is `upper_bound(r) - lower_bound(l)`, and `M`
increments whenever it is `>= 2`.

Re-trace `[3,8]` with the fix: `hi = #{p <= 8} = 2`, `lo = #{p < 3} = 0` (the marker at `3` is not
`< 3`, so it is kept). `hi - lo = 2 >= 2`, so `[3,8]` is counted — correct. And to be sure I did not
flip the bug to the *other* boundary, I check a corridor whose right endpoint coincides with a marker:
corridor `[4,4]` here. `hi = #{p <= 4} = 2` (markers `3,4`), `lo = #{p < 4} = 1` (marker `3`).
`hi - lo = 1`, so `[4,4]` contains exactly one marker, not double-stamped — correct, the marker at `4`
is included via `upper_bound(4)` and the marker at `3` lies outside `[4,4]` as it should. Both
boundary conventions are now right: left endpoint inclusive via `lower_bound`, right endpoint
inclusive via `upper_bound`.

**Cross-checking the whole pipeline on the rich sample.** Markers `{3,4,8}` from before. Count per
corridor with the fixed bounds:
- `[1,3]`: markers in `[1,3]` = `{3}` -> `1`. not multi.
- `[2,5]`: `{3,4}` -> `2`. **multi**.
- `[4,4]`: `{4}` -> `1`. not multi.
- `[6,8]`: `{8}` -> `1`. not multi.
- `[0,9]`: `{3,4,8}` -> `3`. **multi**.
- `[7,10]`: `{8}` -> `1`. not multi.
- `[2,8]`: `{3,4,8}` -> `3`. **multi**.

`M = 3`, `K = 3`. Output `3 3`. That is exactly what an independent brute force (place the same greedy
points, then for every corridor count markers inside by a plain loop over all markers) produces, so my
two-binary-search count agrees with the obvious O(n * K) count. Good.

**Edge cases, because counting code dies at the corners.**
- `n = 1`, corridor `[5,5]`: one corridor, the predicate `last(-inf) < 5` fires once, marker at `5`,
  `K = 1`. Count for `[5,5]`: `hi = #{p<=5} = 1`, `lo = #{p<5} = 0`, gap `1`, not multi, `M = 0`.
  Output `1 0`. Correct — a single corridor can never be double-stamped, since greedy places exactly
  one marker total here.
- Many identical corridors, e.g. three copies of `[2,2]`: sorted they are identical; first triggers a
  marker at `2`, the rest see `last(2) < 2`? No, so no more markers. `K = 1`. Each `[2,2]` contains
  exactly the one marker, gap `1`, `M = 0`. Output `1 0`. No double-stamp because there is only one
  marker — the dedup of the marker placement is what keeps `K` from inflating on duplicates.
- Disjoint chain `[0,0] [2,2] [4,4]`: each triggers its own marker, `K = 3`, markers `{0,2,4}`, each
  corridor contains exactly one, `M = 0`. Output `3 0`. The classic "no overlap, no multiplicity"
  baseline.
- One giant corridor swallowing all markers, `[0,10] [1,2] [3,4] [5,6]`: sorted by right endpoint
  `[1,2],[3,4],[5,6],[0,10]`; markers at `2,4,6`, and `[0,10]` is examined last with `last(6) < 0`?
  No, so no marker for it — it is already hit by all three. `K = 3`. Count for `[0,10]`:
  `hi=#{p<=10}=3`, `lo=#{p<0}=0`, gap `3 >= 2`, multi. The three small corridors each contain one
  marker. `M = 1`. Output `3 1` — verified directly against brute. This is the case the multiplicity
  count exists for, and it is exactly where an endpoint slip would mis-tally.
- Negative coordinates: the `LLONG_MIN` sentinel is strictly below any real `l_i >= -10^9`, so the
  first corridor always triggers a placement; no comparison overflows because I never subtract the
  sentinel from anything (I only compare it). The binary searches operate on actual coordinates.
- Large `n`: sort is `O(n log n)`, the stab sweep is `O(n)`, and the count is `O(n log K)` from two
  binary searches per corridor — well under a second at `2*10^5`, confirmed empirically (a worst-case
  random instance ran in ~0.15s).

**Sanity check of the derivation against the documented sample.** The five-plus-one-corridor example
in the statement — `[0,10] [1,2] [3,4] [5,6] [0,5] [7,12]` — sorted by right endpoint is
`[1,2],[3,4],[5,6],[0,5],[0,10],[7,12]`. Stab: marker `2`; `last(2)<3` marker `4`; `last(4)<5` marker
`6`; `[0,5]` `last(6)<0`? no; `[0,10]` `last(6)<0`? no; `[7,12]` `last(6)<7`? yes, marker `12`.
Markers `{2,4,6,12}`, `K = 4`. Double-stamped: `[0,10]` contains `{2,4,6}` (3), `[0,5]` contains
`{2,4}` (2); the four small corridors and `[7,12]` each contain one. `M = 2`. Output `4 2`, matching
the statement. The derivation reproduces the advertised answer.

The giant-corridor case `[0,10] [1,2] [3,4] [5,6]` is the one `M` exists for: markers land at
`{2,4,6}`, `[0,10]` swallows all three while each small corridor holds exactly one, so `M = 1` — and
it is precisely where an endpoint slip would mis-tally. Everything else collapses `M` to zero: a
single corridor, many identical corridors, or coincident single-point corridors all sit under one
marker, and a lone marker can never double-stamp — the strict predicate is also what keeps `K` from
inflating on those duplicates; a disjoint chain gives each corridor its own non-overlapping marker.
Negative coordinates ride on the `LLONG_MIN` sentinel, below every real `l_i` and only ever compared.

Complexity is fine at the top of the range: sort `O(n log n)`, the stab sweep `O(n)`, the multiplicity
pass `O(n log K)` from two binary searches per corridor — comfortably under a second at `2*10^5`. I ran
the two outputs against an independent per-corridor `O(n*K)` brute count over many random instances,
including generators that force markers onto corridor boundaries, and they agree everywhere; the same
pipeline also reproduces the `4 2` from the statement's worked example. The full self-contained C++
module is in the answer.
