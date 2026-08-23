A rebalancer that runs inside a serving control loop is itself a tenant of
the system it optimizes: whatever wall time the placement computation takes
is taken from decode capacity, at every rebalancing interval, on every
deployment. This variant makes that cost the organizing constraint. Set an
explicit latency envelope for the entire placement call — comfortably under
the tens of milliseconds a loop-based reference burns on a medium profile,
ideally single-digit milliseconds even on the hundred-and-twenty-eight-GPU
topology — and then pursue the best balance and locality obtainable without
ever leaving that envelope.

The envelope dictates the implementation vocabulary. Per-item Python
iteration over hundreds of replica slots is off the table; the method has
to live in vectorized tensor arithmetic that handles every layer of the
model simultaneously, with data-dependent control flow reduced to sorts,
top-k selections, and scatters whose cost is predictable and grows tamely
with topology. Inside that vocabulary the design question becomes sharp:
which increments of load-awareness pay for themselves? A single global
sort? One round of load-ordered matching? A repair pass? The variant asks
for these increments to be ranked by balance gained per millisecond spent —
and the median-of-twenty timing the harness reports will expose any
regression immediately.

All four scored terms stay live: a degenerate instant placement that
ignores load forfeits the balance terms, and shortcutting the hierarchy by
scattering replicas forfeits locality. The claim to defend is therefore a
frontier claim: the submitted method sits at the fast edge of the runtime
range across all profiles while giving up only a bounded, quantified margin
of balance relative to the slow reference — a margin that should shrink,
not grow, as the topology scales up.
