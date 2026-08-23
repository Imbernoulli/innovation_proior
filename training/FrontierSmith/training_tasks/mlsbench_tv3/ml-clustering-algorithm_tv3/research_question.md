Real feature matrices carry passengers: acquisition glitches, boundary
cases, rare writing styles of a digit, points sitting in the gap between two
interleaved arcs. A clusterer that grants every such straggler full
citizenship lets a handful of rows drag centroids, bridge genuine groups,
and shave the geometric margin that silhouette measures. This variant asks
for an algorithm built around an explicit notion of "belongs nowhere".

The output contract is untouched — every row must still receive an integer
label, and the same scores on the same inputs decide the result — so the
craft lies in a two-tier design: identify low-support points, fit the
cluster cores without letting them vote, then hand each straggler the
assignment that does the least damage. Silhouette is the score under
pressure: attaching stragglers should erode it gracefully rather than
catastrophically, while label agreement holds because the cores were
estimated cleanly.

Boundaries for the variant:
- The outlier judgment must come from a support statistic computed on the
  input (local density, neighbor counts, reachability), not from discarding
  a fixed fraction blindly.
- Core fitting and straggler attachment must be separable stages you can
  reason about, with the same attachment rule for every input.
- Thresholds derive from the data's own scale; nothing may be keyed to a
  recognized dataset identity.

Defend, at the end, a mechanism: which points your method refused to let
shape the clusters, and how that refusal shows up as preserved geometry and
intact label agreement in the reported numbers.
