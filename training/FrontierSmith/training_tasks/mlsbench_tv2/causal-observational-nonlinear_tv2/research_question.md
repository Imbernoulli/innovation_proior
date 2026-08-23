Directed scoring makes the undirected adjacency layer load-bearing: a spurious
adjacency is charged once against precision and then again through SHD when its
arrow is inevitably wrong, whereas a cautiously omitted edge costs recall only
once. The three evaluation settings sharpen this asymmetry from different
sides. Mixed nonlinearities defeat any single parametric dependence test; the
150-sample Laplace setting hands an uncalibrated test enough spurious
correlations to bury the true graph; and twenty-node graphs multiply every
per-pair false-positive rate by 190 candidate pairs.

Skeleton construction therefore becomes an exercise in calibrated hypothesis
testing. Dependence must be detected with statistics that carry no
functional-form assumption — rank measures, distance-covariance flavors,
mutual-information estimates — and every acceptance threshold must come from
an explicit null, meaning permutation of the data at hand rather than a
constant that once worked at n = 2000. Multiplicity across candidate pairs has to be
priced in, since a per-pair test that is honest in isolation still floods a
twenty-node skeleton. Spurious adjacencies that arise from shared causes
rather than direct links must be screened by conditional checks before any
orientation is attempted; orientation itself may stay routine, because here it
inherits whatever the skeleton feeds it.

The claim to defend on the unchanged scoreboard — directed F1, precision,
recall, SHD, on the same three settings — is that a high-precision adjacency
layer with honest, data-derived thresholds converts into better directed
scores than a liberal skeleton patched up afterwards, and that the same
calibration recipe, rerun on 150 or 2000 samples, adapts its own strictness
without any hand-set switch between regimes.
