The scaffold hands me one editable slot and the cheapest baseline imaginable, and the honest first move is to put in the cheapest classical idea that already encodes something true about anomalies — anomalies are *few* and *different* — so that its measured strengths and failures set the agenda for everything above it. The most direct operationalization of "different" is distance: score a point by how far it sits from its neighbors, far-from-everything is anomalous. That tradition is distribution-free and rankable, exactly what the harness wants since it reads AUROC off a continuous score. But it has a structural disease I can name before I run anything: it compares every point to a *single global distance scale*. Picture a loose, low-density cluster off to one side and a tight, dense cluster on the other, with a stray sitting just off the dense cluster's edge. To the eye that stray is a glaring anomaly — conspicuously detached relative to how tightly its neighbors pack — yet its absolute distance to the dense cluster can be smaller than the ordinary nearest-neighbor spacing *inside* the loose cluster, so a global distance score ranks ordinary loose-cluster members at or above the genuine stray. A single scale cannot say "anomalous for the company it keeps." I want the cheapest detector that fixes exactly this, because that fix is the first real idea on the ladder.

I propose the Local Outlier Factor (Breunig et al. 2000), the classical detector whose entire content is the locality fix to distance-based detection. The verdict it returns is *local* and *graded*: not "is this point far from the bulk" but "is this point's neighborhood much emptier than the neighborhoods of the points around it," expressed as a real number I can rank rather than a yes/no — because the whole trouble was that the stray and an ordinary loose point are indistinguishable on any single global threshold, and the only way to separate them is to measure *how much* each deviates from its own local context.

The construction is forced once I demand locality. I need a per-point density that uses no distribution and no global volume threshold, and the distances to a point's `MinPts` nearest neighbors are the cheapest distribution-free density proxy available: close neighbors mean a dense region, far neighbors a sparse one. But the raw average distance is jittery — a point that happens to fall very near one neighbor would look absurdly dense — and since I am about to build a *ratio* out of these densities, noise in the denominator swings every score. The stabilization is the reachability distance: when measuring how far `p` is "reachable" from a neighbor `o`, never let the distance drop below `o`'s own k-distance (the radius of `o`'s neighborhood),
$$\text{reach-dist}(p,o) = \max\{\,k\text{-distance}(o),\ d(p,o)\,\}.$$
Every point inside `o`'s neighborhood then gets the same reach-distance to `o`, namely `k-distance(o)`, a smoothed density-of-`o` quantity, so the local fluctuations collapse. The local reachability density is the inverse of the average reach-distance over `p`'s neighbors,
$$\text{lrd}(p) = \frac{1}{\operatorname{mean}_{o}\ \text{reach-dist}(p,o)},$$
inverse so that larger means denser, and averaged rather than summed so it is comparable across points whose tie-inclusive neighborhoods differ in size. But `lrd` is still *absolute* — large in the dense cluster, small in the loose one — so on its own it is k-NN distance in disguise and inherits the same global-scale disease. The locality has to come from a *comparison*, so I take the ratio of a point's neighbors' densities to its own and average it,
$$\text{LOF}(p) = \operatorname{mean}_{o}\ \frac{\text{lrd}(o)}{\text{lrd}(p)}.$$
The absolute scale cancels. Deep inside any cluster, dense or loose, a point and its neighbors have roughly equal `lrd`, so the ratio is $\approx 1$; the stray sits in a sparse pocket (small `lrd`) surrounded by dense neighbors (large `lrd`), so its ratio is clearly above 1. The ratio untangles exactly the two cases the global ranking confused, and it is graded, so I can rank by it: $\text{LOF} \approx 1$ means "as dense as your neighbors" (inlier), $\text{LOF} > 1$ means "emptier than your neighbors" (outlier, larger = stronger). The one knob is `MinPts` (the neighbor count, `n_neighbors` in code); because LOF is non-monotonic in it — too small and the density estimate is pure single-sample jitter, too large and the cluster's own density gradient leaks in — I take the classical low-tens default, `n_neighbors = 20`.

One task-specific detail decides whether this lands the reference numbers, because the scaffold has a wrinkle textbook LOF never sees. The harness fits a `StandardScaler` before handing me `X`, so my features arrive zero-mean and unit-variance. LOF is built entirely on a neighbor graph, and that graph is decided by the metric geometry — by feature scaling — *before* any reachability is computed. A pure z-score scaling and a [0,1] min-max scaling produce *different* neighbor graphs, hence different reach-distances, hence different scores; LOF is invariant to a global distance rescaling but emphatically *not* to a per-feature one that changes which points are neighbors. The reference LOF numbers on these exact datasets come from the ADBench protocol, whose data generator min-max normalizes each feature to [0,1] before fitting. So I re-normalize internally: inside `fit` I fit a `MinMaxScaler` on the training rows, transform them to [0,1], and fit LOF on that; inside `decision_function` I apply the *same* fitted scaler to the query rows before scoring. This is not a new algorithm — it is the same $\max\{k\text{-distance}, d\}$ reachability-ratio detector — it just restores the feature geometry LOF was calibrated under, which the StandardScaler had overwritten. PyOD's wrapper already handles the sign convention (it stores $-\text{LOF}$ internally and exposes a higher-is-anomalous decision function), so I negate nothing myself.

I am deliberately not reaching for anything stronger here, and it is worth being explicit about LOF's boundaries, because they are what the next rungs must repair. It is a neighbor method, so it pays a k-NN search, and on these standardized tabular features distances begin to concentrate as dimensionality climbs (Satellite has 36 features), washing out the very density contrast it lives on. Locality cuts both ways: LOF is *blind to global outliers that sit inside a locally-coherent group* — a point in a tight clump of other anomalies has $\text{LOF} \approx 1$ even when the whole clump is a glaring global outlier, because each member shields the others. And the score is purely geometric in the neighbor graph; it carries no notion of per-feature tail extremeness, so a row that is moderately unusual on many features at once does not necessarily surface as a sparse neighborhood. These are not bugs to patch within LOF — they are the boundaries that motivate moving to a model with a *global* notion of rarity at the next rung.

```python
class CustomAnomalyDetector:
    """Local Outlier Factor anomaly detector (ADBench protocol).

    Applies MinMax normalization internally to match the preprocessing
    used by ADBench (data_generator.py: MinMaxScaler().fit(X_train)).
    LOF is density-based and extremely sensitive to feature scaling,
    so this is required to reproduce the Table D4 numbers.
    """

    def __init__(self):
        from pyod.models.lof import LOF

        # PyOD defaults (matches ADBench with no hyperparameter tuning):
        # n_neighbors=20, algorithm='auto', metric='minkowski', p=2,
        # contamination=0.1.
        self.model = LOF()
        self._scaler = None

    def fit(self, X):
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler()
        Xs = self._scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def decision_function(self, X):
        Xs = self._scaler.transform(X)
        return self.model.decision_function(Xs)
```
