The problem is to learn whether an intervention changed an outcome when the treated group's post-treatment counterfactual is missing. We observe a treated group and an untreated comparison group both before and after the change, so the challenge is to use the available cells to reconstruct the path the treated group would have followed without treatment. A simple post-period comparison between groups is unreliable because the two groups may differ in stable ways that have nothing to do with the policy. A simple before-and-after comparison within the treated group is also unreliable because the outcome may be moving over time for reasons unrelated to the intervention. Each naive comparison removes one source of confounding while leaving the other intact.

The way forward is to stop using the comparison group as a substitute for the treated group's level and instead use it as a substitute for the treated group's movement. The treated group's own pre-period observation anchors its level, so the fixed gap between groups can be differenced away. The comparison group's observed pre-to-post change then provides an estimate of the common time trend. Subtracting that comparison change from the treated group's observed change isolates the treated group's timing-specific departure from its projected untreated path. This contrast is the core of difference-in-differences.

Difference-in-differences estimates the treatment effect by comparing how the treated group changes over time with how a comparison group changes over the same interval. Formally, with treated indicator G equal to one for the treated group and zero otherwise, and Post equal to one after treatment and zero before, the estimand is the treated change minus the comparison change: tau equals E[Y | G=1, Post=1] minus E[Y | G=1, Post=0] minus E[Y | G=0, Post=1] minus E[Y | G=0, Post=0]. The first bracket removes the treated group's fixed level and captures both the treatment effect and the common time movement. The second bracket estimates the common time movement from the untreated group. Their difference leaves the treated group's post-treatment departure from its counterfactual trend.

In regression form the same contrast is obtained from a saturated two-period model: Y_it equals alpha plus gamma G_i plus lambda Post_t plus tau times the interaction of G_i and Post_t plus an error. The group fixed effect gamma absorbs stable differences between groups, the time fixed effect lambda absorbs shocks common to both groups, and the interaction coefficient tau is the difference-in-differences estimate. Equivalently, with only two waves per unit one can collapse to a first-difference regression: Delta Y_i equals a constant plus c times Treated_i plus an error, where Delta Y_i is the individual post-minus-pre change. The coefficient c in that regression is identical to the interaction coefficient in the saturated regression. If treatment intensity varies, the binary treated indicator can be replaced by an exposure measure, with untreated units coded at zero exposure, while keeping the same logic.

Identification relies on the parallel-trends assumption: the treated group's untreated trend must equal the comparison group's untreated trend over the window. The groups are allowed to differ in levels, but they must not be on diverging paths that coincide with treatment timing. The design is threatened by anticipation, spillovers, composition changes, or a treated-group-specific shock that the comparison group does not experience. Event-study plots and alternative comparison groups can help assess the credibility of the trend assumption, but they cannot directly observe the missing counterfactual.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Simulated panel: 200 units, 2 periods, treatment after period 0
np.random.seed(0)
n = 200
df = pd.DataFrame({
    "unit": np.repeat(np.arange(n), 2),
    "post": np.tile([0, 1], n),
    "treated": np.repeat(np.random.binomial(1, 0.5, n), 2),
})

# Parallel untreated trend plus a treatment effect of +2
df["y"] = (
    1.0
    + 3.0 * df["treated"]
    + 0.5 * df["post"]
    + 2.0 * df["treated"] * df["post"]
    + np.random.normal(0, 0.5, len(df))
)

# Saturated two-way fixed-effects regression
sat = smf.ols("y ~ treated + post + treated:post", data=df).fit()
print("Saturated regression:")
print(sat.summary().tables[1])

# Equivalent first-difference regression
diff = df.pivot(index="unit", columns="post", values="y").reset_index()
diff.columns = ["unit", "y_pre", "y_post"]
meta = df.loc[df["post"] == 0, ["unit", "treated"]].reset_index(drop=True)
diff = diff.merge(meta, on="unit")
diff["delta_y"] = diff["y_post"] - diff["y_pre"]
fd = smf.ols("delta_y ~ treated", data=diff).fit()
print("\nFirst-difference regression:")
print(fd.summary().tables[1])

# Manual four-cell estimate
means = df.groupby(["treated", "post"])["y"].mean()
did_manual = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
print(f"\nManual DID estimate: {did_manual:.3f}")
```
