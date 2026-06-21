The identity default does nothing, so the floor is whatever miscalibration the four frozen classifiers already carry: the forest piles its votes toward 0 and 1, the SVM margin runs through some S-shape, the GBM pushes scores around in its additive-logistic way, and in every case the reliability plot bows off the diagonal. What every one of these plots also shows, robustly, is that the points still march upward — higher score, higher empirical positive rate. The classifier ranks correctly; only the vertical axis is wrong. So I want a map from the score it gives me to the probability it should have given me, learned from the held-out calibration split, and for this first rung I want the map that commits to the *least* about the distortion's shape. The two options already on the table both have a flaw I can name: a fitted sigmoid is exactly one rigid shape and leaves residual miscalibration whenever the true curve bends some other way; equal-count binning is shape-free but its boundaries fall wherever equal counts land rather than where the score's meaning changes, and the bin count is a hyperparameter I cannot cross-validate on the small binary splits.

I propose **isotonic regression** — fit the best *monotone non-decreasing* function from score to label, the one prior the diagnostic plots robustly support, with no committed shape and no bin count to tune. Concretely, sort the calibration pairs by score, $s_1 \le s_2 \le \dots \le s_n$, carrying the 0/1 targets $g_i$, and find fitted values $\hat g_i$ that are non-decreasing and close to the labels in weighted squared error:

$$\min \sum_i w_i (g_i - \hat g_i)^2 \quad \text{s.t.} \quad \hat g_1 \le \hat g_2 \le \dots \le \hat g_n.$$

The choice of squared error is not arbitrary — it is the choice that makes the answer come out as an honest probability. Drop the constraint and ask what single constant $c$ minimizes $\sum_i w_i (g_i - c)^2$; differentiating gives $c = (\sum_i w_i g_i)/(\sum_i w_i)$, the weighted mean of the targets, which for 0/1 labels *is* the empirical fraction of positives, a number in $[0,1]$. So wherever the monotonicity constraint forces the fit to be constant on a group of examples, the value it takes there is exactly that group's empirical positive rate — the calibrated probability I wanted. (Cross-entropy has the same constant-minimizer, since both are Bregman losses for the mean, so the order-restricted fit is identical under either; I solve the squared-error form because it has the cleanest quadratic structure.)

What makes the solution computable is that the optimum is forced once I know the right partition into constant blocks. On any maximal run where $\hat g$ is constant, the constraint is active; between blocks it strictly increases, so the value on a block free at both ends must be the unconstrained minimizer over that block — the weighted mean. The whole problem therefore reduces to finding the blocks, and the **pool-adjacent-violators** algorithm finds them in $O(n)$ after the sort: walk left to right with each point its own block valued $g_i$; whenever an adjacent pair violates monotonicity ($v_L > v_R$), the weighted-squared-error projection onto $z_L \le z_R$ lands on the boundary $z_L = z_R$ at the pooled mean $(w_L v_L + w_R v_R)/(w_L + w_R)$, so pool the two into one block; pooling can leave the new block below its left neighbor, so back-merge leftward until the blocks are ordered, then resume. The crucial property for calibration is that the resulting intervals *are* a binning, but one chosen by the data — coarse where the ranking is poor (many violations pooled away), fine where it is good (blocks stay small). That repairs equal-count binning's arbitrariness without a cross-validated bin count, and it is more flexible than the sigmoid because it never assumed a shape.

To turn the fitted intervals into a function on *new* scores, I aggregate tied calibration scores first (one threshold, target the weighted average of the tied labels), store the non-redundant thresholds, and interpolate linearly between adjacent fitted values for a continuous non-decreasing map; a test score outside the calibration range carries no information, so I clamp the *input* to the fitted domain (`out_of_bounds="clip"`) and bound the fitted values to $[0,1]$ (`y_min=0`, `y_max=1`) so the output is always a probability. The multiclass case is where the surface forces a choice. The monotonicity trick is one-dimensional — "higher score, higher probability" only makes sense along a single axis — and a non-parametric fit on the $(k-1)$-simplex needs far more data than I have as $k$ grows. So I refuse the joint object and reduce to one dimension per class: for each class $c$, define the binary problem "is the true label $c$?" against the classifier's score for class $c$, run the isotonic fit, and keep the list. These per-class outputs each estimate $P(c\mid x)$ directly, so I reconcile them by flooring at $10^{-15}$ (to avoid a $0/0$ all-zero row) and renormalizing so each row sums to one, which is exactly what the harness's row-sum assertion requires.

I should be honest about the cost, because I expect it to bite. The isotonic map is non-parametric — as many effective degrees of freedom as there are blocks — so it needs *more* calibration data not to overfit, and it is piecewise-constant, returning the same probability to every test point in a block, with that block value estimated from a handful of calibration examples. That is exactly the behavior binned ECE rewards (the average within a bin can look excellent) and the proper scores punish (NLL pays for every example where the flat block value is wrong). So I expect a split: strong ECE, especially on the small binary tasks, but relatively weak NLL — most visibly where the calibration split is small and the curve steep. If that split is what I see, the next rung's job writes itself: spend fewer degrees of freedom, buy back data efficiency with a parametric family, and accept a little shape rigidity to stop the proper scores from bleeding.

```python
# EDITABLE region of custom_calibration.py (lines 45-102) - step 1: isotonic regression
class CalibrationMethod(BaseEstimator):
    """Isotonic Regression calibration.

    Fits a non-parametric, monotonically non-decreasing function
    from uncalibrated probabilities to calibrated ones.
    """

    def __init__(self):
        self.is_binary = None
        self.calibrators_ = None

    def fit(self, probs, labels):
        from sklearn.isotonic import IsotonicRegression as IR

        if probs.ndim == 1:
            self.is_binary = True
            iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs, labels)
            self.calibrators_ = [iso]
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.calibrators_ = []
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)       # one-against-all per class
                iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(probs[:, c], binary_labels)
                self.calibrators_.append(iso)
        return self

    def predict_proba(self, probs):
        if self.is_binary:
            calibrated = self.calibrators_[0].predict(probs)
            return np.clip(calibrated, 0, 1)
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                calibrated[:, c] = self.calibrators_[c].predict(probs[:, c])   # r_c(x)
            calibrated = np.clip(calibrated, 1e-15, None)         # avoid 0/0 all-zero row
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)     # renormalize
            return calibrated
```
