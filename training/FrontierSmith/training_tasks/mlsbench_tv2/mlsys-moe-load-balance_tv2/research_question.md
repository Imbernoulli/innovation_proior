In expert-parallel decoding the step time of a layer is set by whichever GPU
finishes last; nothing else about the load distribution shows up in the
latency. This variant therefore fixes attention on a single number per
layer: the token count of the hottest device. Placement is to be read as
minimax scheduling — every stage decision (which groups share a node, which
experts earn replicas, which replicas share a GPU) is good exactly insofar
as it lowers the worst device's queue, first at GPU granularity and then
again at node granularity, where the same straggler logic applies to the
network-attached group of devices as a unit.

The reported balance ratios are mean-over-max, which makes them straggler
gauges in disguise: shaving the peak raises them even when the average load
is untouched, while any amount of tidying below the peak moves nothing.
That asymmetry should steer the algorithm toward peak-directed machinery —
longest-processing-time orderings, repair passes that specifically unload
the argmax device, replication decisions that ask "does one more copy of
this expert cut the current maximum?" rather than "is this expert popular?".
Locality and wall time still appear in the score, so the peak cannot be
bought by scattering an expert's replicas across the fabric, nor by an
assignment loop whose own latency dwarfs the imbalance it removes.

The deliverable is a straggler audit alongside the method: for each
deployment profile, identify which stage leaves the residual peak behind
and quantify how far the final maximum sits above the ideal even-split
load. The claim to defend is that attacking the maximum directly beats
optimizing averages everywhere the suite can tell the two apart.
