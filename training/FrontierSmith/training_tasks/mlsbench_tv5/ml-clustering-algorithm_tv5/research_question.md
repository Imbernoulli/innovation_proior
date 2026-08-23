Take the geometry seriously: this variant puts silhouette — the only one of
the three scores computable without ground truth — in the driver's seat. A
clustering a practitioner would trust in the wild has to look right from
inside the feature space itself: tight groups, wide margins, boundaries
tracing low-density valleys. Chasing label agreement alone can produce
partitions that are extrinsically correct yet geometrically mushy, and such
partitions inspire no confidence wherever no labels exist to check them.

The objective ordering is explicit. First, drive the intrinsic score as
high as the data allows on every input, including the high-dimensional one
where compactness is hardest to earn. Second, protect a floor under ARI and
NMI: a geometry-first method forfeits its point if it manufactures compact
nonsense by shearing true classes apart or dissolving them into a couple of
giant super-clusters. Any silhouette gain purchased with a larger loss of
label agreement is a regression under this brief.

Ground rules:
- Compactness must be earned by the representation and the assignment rule
  (metric shaping, an embedding fitted inside the method, margin-seeking
  refinement), never by trivially shrinking the cluster count to its
  minimum.
- Guard rails against extrinsic collapse belong inside the algorithm — for
  instance, refusing merges or reassignments beyond a structural budget
  derived from the data itself.
- The same settings apply everywhere; tuning compactness per dataset is out
  of scope.

What must be defended is a partition whose geometric quality would convince
an unlabeled observer, with reported label agreement close enough to prove
the geometry was not bought with structural vandalism.
