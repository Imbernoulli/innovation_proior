An overconfident classifier does not spread its test predictions evenly over
the confidence range: the neural network's softmax and the forest's vote
ratios pile their mass into the last few bins of any reliability diagram. Because ECE weights each bin by
occupancy, those crowded high-confidence bins dominate the reported number
almost by themselves; and because log-loss is unbounded on confident errors,
the same region is where NLL is won or lost. The sparse middle of the
diagram, however photogenic, is nearly irrelevant to what is scored.

This variant therefore aims the entire correction at the high-confidence
region. The design question is surgical: estimate how reliable the classifier
really is precisely where it claims near-certainty, and revise those claims —
while disturbing the middle and low ranges as little as possible, since
evidence there is thin and meddling mostly adds variance. Two complications
give the problem substance. The top of the range is where the calibration
split has the most predictions but the fewest errors, so the quantity being
estimated is a small failure rate that demands careful uncertainty handling.
And cutting overconfidence too aggressively spills Brier and NLL losses onto
the many near-certain predictions that were in fact correct.

What to demonstrate on the unchanged metrics: a correction concentrated above
a high-confidence threshold, with the region below left essentially
untouched, captures most of the ECE reduction available to any method while
improving — not trading against — NLL on confident mistakes. As a starting
point you get a single linear remap of the above-threshold region onto the
tail's measured accuracy, plus a per-bin tail table that nothing consults
yet; graduating from one global slope to a genuinely bin-resolved,
uncertainty-aware repair is the intended work.
