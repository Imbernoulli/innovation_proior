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
