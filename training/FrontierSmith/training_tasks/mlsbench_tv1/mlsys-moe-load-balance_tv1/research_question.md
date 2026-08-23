Give a placement algorithm twice as many physical slots as logical experts and
most strategies look alike: replicate the hot set aggressively and the per-GPU
maxima flatten out on their own. This variant studies the opposite regime —
the replication budget only barely exceeds the expert count, traffic follows a
long tail in which a handful of experts dominate whole layers, and the machine
hierarchy is deep enough that a bad decision at the node level cannot be
papered over by shuffling replicas within a node. Under scarcity, replication
stops being the answer and *rationing* becomes the question: which few experts
deserve the spare slots, and where must everything else sit so that the
imbalance that remains lands where it costs least?

Because the task score multiplies per-configuration results, the binding
constraint is the hardest deployment, not the friendly ones: a scheme tuned
for the mild real-model profiles that buckles when the node count doubles and
the skew sharpens is dominated by its weakest column. The evaluation also
prices the two classic escapes explicitly. Scattering replicas across nodes to
chase per-GPU flatness is charged through the locality term, which decays with
every additional node an expert's replicas touch, weighted by that expert's
traffic. Solving placement with careful sequential Python is charged through
measured wall time, which must stay controlled as the topology grows to
sixteen nodes and more than a hundred GPUs — the arithmetic should therefore
be expressible as batched tensor operations across all layers at once.

Structural invariants are unchanged: exact slot counts per GPU, at least one
replica per expert, counts consistent with the returned mappings. The
deliverable is the decision rule, not just its numbers: characterize the
traffic statistic that determines where scarce replicas go, and demonstrate
that the same rule degrades gracefully from the comfortable configurations to
the pathological long-tail one.
