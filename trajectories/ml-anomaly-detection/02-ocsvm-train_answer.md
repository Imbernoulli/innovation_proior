LOF ranked these datasets at chance: AUROC sat at 0.547 on Cardio, 0.550 on Satellite, 0.531 on Shuttle — barely off the 0.5 coin-flip — with only Thyroid (0.706) clearly above the floor, and the per-seed spread was tiny, so this was not variance but a *systematic* miss. The diagnosis is sharp. Locality was a real fix to global-scale distance detection, but on these datasets the anomalies are not "sparse pockets between clusters of differing density" — they are global tail events, and LOF's local neighbor-graph view is structurally blind to those. A neighbor method that asks "is my neighborhood emptier than my neighbors'?" finds nothing when the answer everywhere is "about the same," which is what standardized, distance-concentrated tabular features give. I need to abandon the neighbor graph and model *what region the normal mass occupies*, so that a point outside it scores high regardless of whether its local neighborhood happens to be coherent.

I propose the One-Class SVM (Schölkopf et al. 2001): estimate the *support* of the data rather than its density. The full density-estimation reflex — estimate $p$ everywhere, keep $\{x : p(x) \ge t\}$ — solves a larger problem than I need and is exactly what dies in high dimension. I do not need the height of the distribution everywhere; I need a decision *region*, and one knob that means "allow this fraction of training points to fall outside," because with no labels a contamination prior is the only honest thing I can set. The precise way to say "small region containing most of the mass" is the minimum-volume set, but input-space volume over a rich high-dimensional class is not a computation I can run. The support-vector machinery is the lever LOF never had: the supervised SVM's $\tfrac12\|w\|^2$ term does not measure input-space volume — it controls the *flatness* of a decision function in a kernel-induced feature space, and the kernel trick lets a flat function there draw a nonlinear boundary in input space. That is what I borrow. The obstacle is that an ordinary SVM separates two labeled classes and I have one class and no negatives.

With one class I need a fixed reference to push against, and the only canonical feature-space point I get for free is the origin. So I check the geometry rather than assume it. With the Gaussian (RBF) kernel $k(x,y) = \exp(-\gamma\|x-y\|^2)$, every diagonal entry $k(x,x) = 1$, so all mapped points have unit norm and live on the unit sphere in feature space; and every pairwise $k(x_i,x_j) > 0$, so all pairwise angles are acute — the data sit together on a patch of the sphere, cleanly separable from the origin. The maximum-margin hyperplane to the origin then cuts off a spherical cap containing the data, and crucially that cap is a *global* region: a point is anomalous because it falls outside the support of the whole cloud, not because its local neighborhood is sparse. The margin problem writes down directly — push most mapped points onto the far side of a hyperplane $\langle w, \Phi(x_i)\rangle \ge \rho$, allow slacks for the failures, reward moving $\rho$ off the origin, and keep $\|w\|$ small:
$$\min_{w,\rho,\xi}\ \tfrac12\|w\|^2 + \frac{1}{\nu\ell}\sum_i \xi_i - \rho \quad\text{s.t.}\quad \langle w,\Phi(x_i)\rangle \ge \rho - \xi_i,\ \ \xi_i \ge 0.$$
The $-\rho$ term rewards pushing the hyperplane away from the origin, and the odd $1/(\nu\ell)$ slack scaling is not arbitrary — the KKT conditions justify it. Forming the Lagrangian with multipliers $\alpha_i \ge 0$ on the margin constraints and $\beta_i \ge 0$ on the slacks, stationarity gives $w = \sum_i \alpha_i \Phi(x_i)$, $\alpha_i = 1/(\nu\ell) - \beta_i$ (so $0 \le \alpha_i \le 1/(\nu\ell)$), and $\sum_i \alpha_i = 1$. Substituting back collapses everything to the dual,
$$\min_\alpha\ \tfrac12 \sum_{i,j} \alpha_i \alpha_j\, k(x_i,x_j) \quad\text{s.t.}\quad 0 \le \alpha_i \le \frac{1}{\nu\ell},\ \ \sum_i \alpha_i = 1,$$
with decision rule $f(x) = \sum_i \alpha_i\, k(x_i,x) - \rho$, positive on the data side, where $\rho$ is read off any non-bound support vector (where $0 < \alpha_i < 1/(\nu\ell)$ forces the margin tight and the slack zero, so $\rho = \sum_j \alpha_j k(x_j, x_i)$).

The counting argument is what makes $\nu$ the one meaningful knob, and it is the interpretable contamination control LOF's `n_neighbors` never had. If a training point lands outside the region, its slack is positive, complementary slackness forces $\beta_i = 0$, hence $\alpha_i = 1/(\nu\ell)$ — every outlier is pinned at the top of the box. Since the $\alpha_i$ sum to 1 and are nonnegative, at most $\nu\ell$ of them can sit at that ceiling, so the training outlier fraction is at most $\nu$. Symmetrically, each nonzero $\alpha_i$ contributes at most $1/(\nu\ell)$, so making the sum equal 1 needs at least $\nu\ell$ nonzero coefficients — the support-vector fraction is at least $\nu$. Asymptotically the two bounds meet: $\nu$ is simultaneously an upper bound on the outlier fraction and a lower bound on the support-vector fraction. The strange $1/(\nu\ell)$ has turned a penalty into a count. The endpoints check out — $\nu \to 0$ drives the ceiling to infinity, recovering a hard-margin support estimator; $\nu = 1$ forces every $\alpha_i = 1/\ell$ and the expansion becomes a thresholded Parzen-window estimate — and the default $\nu = 0.5$ sits in the middle, assuming no contamination rate it cannot know while emitting a continuous signed-distance score that AUROC reads regardless of where the threshold lands.

One more detail collides with the scaffold the same way LOF's did. The RBF exponent is an input-space distance $\gamma\|x-y\|^2$, so the learned region depends entirely on feature scaling — a large-scale feature dominates the squared distance and the kernel sees only that axis. PyOD's default `gamma='auto'` is $1/n_\text{features}$, a fixed bandwidth that only makes sense when features are comparably scaled, and the reference OCSVM numbers come from the ADBench protocol's [0,1] min-max geometry. The harness instead pre-applies a StandardScaler, where heavy-tailed standardized features can have very different spreads than [0,1]. So this rung re-normalizes internally, exactly as LOF did: fit a `MinMaxScaler` on the training rows, fit the one-class SVM on the [0,1] features, and apply the same fitted scaler at score time. The algorithm is unchanged — the same box-plus-equality dual with the RBF cap — only the geometry the $1/n_\text{features}$ bandwidth and the reference numbers were calibrated under is restored. For the sign, the libsvm-backed solver returns $\sum_i \alpha_i k(x_i,x) - \rho$, positive for inliers, and PyOD's wrapper already inverts it to higher = more anomalous, so I take its `decision_function` as-is. Concretely the rung is a PyOD `OCSVM()` at defaults (`kernel='rbf'`, `nu=0.5`, `gamma='auto'`) wrapped in a re-applied `MinMaxScaler`.

```python
class CustomAnomalyDetector:
    """One-Class SVM anomaly detector (ADBench protocol).

    Applies MinMax normalization internally to match ADBench's
    preprocessing. Uses PyOD defaults: kernel='rbf', nu=0.5,
    gamma='auto' (= 1/n_features).
    """

    def __init__(self):
        from pyod.models.ocsvm import OCSVM

        # PyOD default: kernel='rbf', nu=0.5, gamma='auto'.
        self.model = OCSVM()
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
