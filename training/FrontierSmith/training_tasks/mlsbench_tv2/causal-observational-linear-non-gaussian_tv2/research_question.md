Full identifiability is the promise of the linear non-Gaussian model, and a total causal
order is the form that promise takes: once the variables are correctly sequenced, every
remaining question is a regression, and no arrowhead can point backwards. This variant
commits to that route. The method must produce, explicitly, a single topological order over
all variables — thirty, fifty, or one hundred of them — and derive the reported directed
graph from that order, rather than assembling a graph out of independent per-pair
orientation verdicts.

The order is where the non-Gaussian signal must be spent. Pairwise asymmetry statistics,
residual-independence contrasts, iterative root-finding in the DirectLiNGAM style, or any
aggregation of local evidence into a global permutation are all admissible, but the
aggregation itself is the contribution under test: local orientation cues contradict each
other in finite samples, and a global order is precisely a mechanism for resolving those
contradictions consistently. One configuration must survive all three noise families —
super-Gaussian, skewed, and bounded — and the settings span a factor of three in node
count, so both the cost of the ordering procedure and its noise sensitivity have to scale.

Judgment comes from the usual directed-edge metrics, read with an ordering lens. Order
mistakes are the expensive ones: a variable placed too early converts all of its incoming
edges into reversed reports, damaging precision and SHD in bulk, whereas edge-selection
mistakes along a correct order cost one edge at a time. The claim to defend is that
resolving direction globally — sequence first, select edges second — yields a better
precision, recall, and SHD profile at these scales than deciding each pair on its own
evidence.
