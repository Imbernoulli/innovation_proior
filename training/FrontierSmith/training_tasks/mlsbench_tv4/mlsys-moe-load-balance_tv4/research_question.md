Placement decisions in this system happen at two very different price
levels: traffic that crosses a node boundary pays interconnect cost, and a
node carrying more than its share of tokens throttles every GPU inside it
no matter how neatly those GPUs split the work among themselves. Rank the
objectives to match those prices. Inter-node structure comes first:
distribute expert-group traffic so that node-level loads are flat, and keep
each expert's replicas confined to a single node so the locality measure
stays at its ceiling. Only after those two commitments are locked does the
intra-node replica-to-GPU assignment matter — and by then the group-to-node
stage has already determined how much GPU-level flatness is achievable at
all.

The stress profile is where this ordering earns its keep: thirty-two groups
over sixteen nodes means each node receives exactly two groups, so
group-to-node packing becomes a genuine combinatorial choice rather than an
afterthought — pairing two heavy groups on one node is a mistake no
downstream stage can repair. The study should treat that first stage as the
primary algorithm: what pairing rule flattens node loads under long-tailed
group weights; how replication interacts with the pairing (a hot expert
relieved by in-node copies changes its node's total not at all); and what
the achievable node-balance frontier looks like when locality is pinned at
one.

Scoring is the standard four-term combination; the re-aim is which terms
lead. The claim to defend: a hierarchy-first method scores visibly better
on node balance and locality than a GPU-first method of equal effort while
conceding little on per-GPU balance — evidence that the top of the
hierarchy is where this placement problem is actually decided.
