An embedding can lie in two directions. It can invent neighbors — placing
points side by side that were far apart in the original space — or it can tear
the manifold, sending true neighbors to opposite ends of the map. The two
error modes are measured separately here: trustworthiness catches invented
neighbors, continuity catches torn ones. Attraction-heavy neighbor-embedding
recipes tend to buy one at the price of the other, while purely linear
projections make the opposite trade. Each dataset's score averages both
fidelities together with the 7-NN class accuracy, and the datasets combine
multiplicatively, so a method that shines on the raw 784-dimensional image
sets but breaks on the 50-dimensional SVD-compressed text set simply inherits
the text set's number.

The re-aimed problem: produce 2-D maps whose local errors are symmetric —
false-neighbor and lost-neighbor rates simultaneously low — from one fixed
algorithm and configuration across all three data geometries, inside the
pipeline's hard per-dataset CPU budget. Whatever adaptation happens must come
from statistics the method computes on the data it is handed (neighborhood
scale, spectrum decay, distance concentration), because pixel intensities and
TF-IDF-then-SVD features arrive with very different scales and intrinsic
dimension, and no per-dataset switches or tuning are available at fit time.

The scaffold starts from a linear spectral initialization (PCA) followed by a
budget-guarded neighborhood-smoothing loop, plus a geometry probe whose value
is currently almost unused; treat those as the levers, not the answer. The
question to settle is whether a budgeted, self-tuning embedding can hold both
fidelity directions at once where the reference methods each pick a side —
judged by the same three numbers per dataset as always, with the weakest
dataset setting the tone of the final score.
