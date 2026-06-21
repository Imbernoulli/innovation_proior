We want to put a uniform high dose on a tumour while sparing whatever sits next to it, and we want to do it with a magnetically scanned proton pencil beam. What makes protons worth the trouble is the depth–dose curve: a proton coasts through the entrance region depositing relatively little, then near the end of its range dumps almost all of its energy in a sharp spike — the Bragg peak — and past that spike the dose falls to essentially nothing. The depth of the spike is set by the proton energy, so one energy paints a thin high-dose shell at one depth and nothing beyond it. That distal nothing is the prize: unlike a photon ray, which exits the far side dosing everything downstream, a proton ray can be made to stop at the far edge of the tumour and spare tissue behind it as no photon arrangement can. The classical way to cover a thick target is to fire several graded energies at the same spot and add their peaks at successive depths; if the relative weights are chosen right, the superposition is a flat plateau — the spread-out Bragg peak — and the plateau is *linear* in the per-peak weights. With magnetic scanning we lift this from one pencil to a full 3-D grid of spots, each spot a (lateral position, energy) pencil that deposits its own fixed pattern (a lateral Gaussian times a Bragg curve in depth) scaled by its own freely settable weight.

The conventional way to use those knobs is timid. Single-field-uniform-dose and classical SOBP planning make each field deliver a uniform dose to the target on its own, then sum a few fields from different angles. That is safe, but it throws away most of the freedom: if every field must be self-flat, one field's deliberate hot patch can never fill another field's deliberate cold patch, which is exactly what is needed to carve dose around a concave organ tucked into the target. The photon world already crossed this bridge with intensity modulation — let each field be deliberately inhomogeneous and require only that the *sum* be uniform on the target and low on the OARs — but the photon machinery has no notion of a Bragg peak in depth, no depth modulation, and no handle on the range fragility protons introduce, so transplanting it unchanged yields nominal-optimal but unstable plans. The intermediate lower-dimensional schemes (modulate only proximal weights, or only across fields) are nearly as good as full modulation when there are many fields, but they fail to hold both target homogeneity and OAR sparing as the field count drops. And the photon reflex of expanding the target by a static geometric margin does not rescue protons at all. So the gap is concrete: we need full 3-D modulation of every individual Bragg spot, and we need it to survive the uncertainty in where the distal edge actually lands.

I propose intensity-modulated proton therapy with robust optimization (IMPT). The design variable is the weight vector $w$ over all spots, one entry per (lateral position, energy) pencil. Because each proton deposits independently and superposition holds, the dose in voxel $i$ is the weighted sum of the per-spot patterns, and stacking over voxels gives the linear model
$$D = P w,$$
where $P$ is the dose-influence matrix, $P_{ij}$ the dose to voxel $i$ per unit weight of spot $j$. $P$ is computed once from the beam model and the patient CT, and then *every* question about a plan is linear algebra on $w$. This is exactly the photon beamlet scaffold; the only thing different about protons is what lives inside $P$ — Bragg peaks in depth instead of exponential rays — and, as it turns out, how fragile that $P$ is. A weight is a number of protons, so $w \ge 0$ componentwise is a hard bound, not a wish; without it the optimiser would sculpt dose with negative weights and hand back an undeliverable plan.

The system is wildly underdetermined and the goals conflict, so rather than solve a linear system we minimise a penalty built voxel by voxel and let a bound-constrained optimiser find $w$. On the target every voxel wants the prescription dose $d_{\text{ref}}$, and both directions are bad — cold is a coverage failure, hot is an overdose — so the natural smooth symmetric penalty is the squared deviation $(D_i - d_{\text{ref}})^2$. On an OAR the penalty must be one-sided: dosing it *below* tolerance is good, only exceeding a maximum $d_{\max}$ hurts, so we penalise the overshoot alone, $\max(D_i - d_{\max}, 0)^2$. The positive-part operator kills the term wherever the OAR is already under tolerance; penalising a symmetric deviation from $d_{\max}$ would perversely push the OAR dose *up* toward tolerance and fight our own sparing. The same logic gives a one-sided floor on the target, $\max(d_{\min} - D_i, 0)^2$, penalising only underdosing. Different structures matter differently and a large structure should not dominate just by counting more voxels, so each structure carries a penalty $p_s$ and is normalised by its voxel count $|V_s|$:
$$f(w) = \sum_s \frac{p_s}{|V_s|} \sum_{i \in V_s} g_s(D_i)^2,$$
with $g_s$ being the deviation on a target, the over-dose ramp on an OAR, the under-dose ramp on a target floor. Squared penalties are the right choice over absolute value or a hard dose-volume-histogram criterion precisely because we optimise with gradients over thousands of variables: a sum of squares is differentiable everywhere with a clean gradient, the absolute value has a kink at zero, and a DVH criterion is genuinely non-convex and non-differentiable — better bolted on later as a constraint than used to *drive* the optimisation. The squared ramps are convex in $D$, and since $D$ is linear in $w$ the whole program is convex in $w$ with a simple bound — exactly what a gradient-based bound-constrained solver eats for breakfast.

The gradient is free by the chain rule. Since $f$ depends on $w$ only through $D = P w$,
$$\nabla_w f = P^{\mathsf T}\,\frac{\partial f}{\partial D},$$
where $\partial f / \partial D$ is a per-voxel delta read straight off each penalty: $2(D_i - d_{\text{ref}})$ on a target, $2\max(D_i - d_{\max}, 0)$ on an OAR (zero wherever it is under tolerance), and for the floor, since the penalty is $\max(d_{\min} - D_i, 0)^2$, the derivative is $2\min(D_i - d_{\min}, 0)$ — negative while the voxel is cold, so the gradient points in the direction that raises a cold target voxel's dose. Each delta is scaled by $p_s/|V_s|$, scattered back to its voxels, and one multiply by $P^{\mathsf T}$ back-projects the voxel-space gradient into spot-weight space: same matrix $P$ forward for dose, transpose backward for gradient, both cheap sparse products. Handed $f$ and $\nabla_w f$, a bound-constrained solver started from positive $w$ gives the nominal IMPT plan — strictly more capable than single-field-uniform-dose, and the only scheme that keeps both homogeneity and sparing as the field count is cut.

And then comes the wall. We picked up the photon scaffold whole, including its tacit assumption that small geometric errors smear the dose a little but do not change its structure. For photons that holds — shift the patient a couple of millimetres and every ray still deposits along its slightly moved path; the dose translates. For protons it is poison, and the culprit is the distal edge we were so pleased about. The cleanest way to spare a serial OAR sitting right behind the target — the spinal cord, say — is to stop the Bragg peaks *just* upstream of it and use the steep fall-off as a knife edge. The squared penalties love this and the optimiser drives straight to it, and the nominal plan looks gorgeous. But that edge lands at the proton range, and the range is uncertain: the dominant error is the CT-Hounsfield-to-stopping-power conversion, good only to about 3–3.5% (stress-tested at ±5%), growing with water-equivalent depth to several millimetres, on top of setup and anatomical change. A small overshoot slides the high-dose region into the cord — the exact catastrophe we were avoiding — and a small undershoot opens a cold slab at the distal target edge. The feature is the bug, and it is *worse* for the fully modulated multi-field plan than for the timid single-field one: when fields are inhomogeneous the optimiser arranges their hot and cold patches to cancel, but only in the nominal geometry; a range or setup error hits each field differently and the cancellation comes apart. A static margin cannot fix this because, unlike photons, a proton shift does not translate the dose, it *reshapes* it — the peaks move in depth, the distal edge migrates — so covering a geometrically expanded volume in the nominal scenario guarantees nothing under error.

The cure lives in the same redundancy that allowed the fragile plan in the first place: the problem is so underdetermined that many weight vectors give the same nominal dose, and among them are robust ones — plans that meet the target nominally *and* hold up under error — if only we optimise for robustness instead of picking fragility by accident. So we stop optimising the nominal dose and optimise across a *set* of error scenarios. Enumerate plausible errors — range over/undershoot of a few percent, setup shifts of a few millimetres per axis, plus the nominal — index them $k = 1\dots K$; each has its own geometry, hence its own influence matrix $P^{(k)}$ and its own dose $D^{(k)} = P^{(k)} w$ for the *one* plan $w$ we deliver. What scalar to minimise over this family depends on what we trust. If the errors are random with probabilities $\pi_k$, the principled choice is the expected objective $\sum_k \pi_k f(w; D^{(k)})$, whose gradient is the probability-weighted sum $\sum_k \pi_k (P^{(k)})^{\mathsf T} (\partial f/\partial D^{(k)})$ — each scenario back-projected through its own matrix, then averaged; this is smooth and less conservative, and to make the average good the optimiser naturally backstops the distal target from a second beam rather than betting on one knife edge. Pushed to its limit, the probabilistic view optimises the objective on the expected dose and adds an integral variance term $w^{\mathsf T}\Omega w$, accumulated from the scenario distribution, with gradient $2\Omega w$ — penalising variance is penalising fragility stated in dose. If instead we will not trust a probability model and demand safety for the whole bounded error set, we minimise the worst case, $\min_w \max_k f(w; D^{(k)})$, which guarantees the bound at the cost of caring only about the single worst scenario — the right conservatism for a serial OAR that must be protected under *any* plausible error. Worst case admits several readings: voxel-wise (VWWC) builds one synthetic worst dose by taking, per voxel, the minimum over scenarios in the target and the maximum in an OAR, then penalises that — cheap, one evaluation, but a Frankenstein dose no single real error produces, so over-pessimistic, with the gradient of each voxel flowing back only through the $P^{(k)}$ of the scenario that won the min/max there; composite worst case (COWC) evaluates the whole objective separately per scenario, $f_k = \sum_s f_s(D^{(k)})$, and minimises $\max_k f_k$ over *realisable* doses, which is more physical but non-smooth, so the hard max is replaced by a soft one, $\frac{1}{\tau}\log\sum_k \exp(\tau f_k)$ or a large-$p$ $p$-norm, turning the gradient into a blend weighted toward the worst scenario; and objective-wise worst case (OWC) takes the worst scenario per objective term, slightly more conservative than COWC and less synthetic than VWWC. The engine underneath never changes: penalised-least-squares structure objectives, a per-voxel delta back-projected through an influence matrix transpose, $f$ and $\nabla_w f$ handed to a bound-constrained solver with $w \ge 0$. Robustness only adds a *list* of influence matrices, the per-scenario doses, a combine rule that produces one value and one gradient, and back-projection of each scenario's contribution through its own $P^{(k)}$. The robust optimiser is the nominal one wrapped in a scenario loop and a combine rule — costlier per iteration, but the plan stays good when the range comes in a few millimetres off.

```python
import numpy as np
from scipy.optimize import minimize

# ---- dose objectives: unweighted value + unweighted dose-gradient ----
class DoseObjective:
    def __init__(self, penalty=1.0, robustness="none"):
        self.penalty = penalty
        self.robustness = robustness

class SquaredDeviation(DoseObjective):   # target: (D - dref)^2
    def __init__(self, dref, penalty=1.0, robustness="none"):
        super().__init__(penalty, robustness)
        self.dref = dref

    def value(self, d):
        dev = d - self.dref
        return (dev @ dev) / d.size

    def dose_grad(self, d):
        return 2.0 * (d - self.dref) / d.size

class SquaredOverdosing(DoseObjective):  # OAR: max(D - dmax, 0)^2
    def __init__(self, dmax, penalty=1.0, robustness="none"):
        super().__init__(penalty, robustness)
        self.dmax = dmax

    def value(self, d):
        over = np.maximum(d - self.dmax, 0.0)
        return (over @ over) / d.size

    def dose_grad(self, d):
        over = np.maximum(d - self.dmax, 0.0)
        return 2.0 * over / d.size

class SquaredUnderdosing(DoseObjective): # target floor: max(dmin - D, 0)^2
    def __init__(self, dmin, penalty=1.0, robustness="none"):
        super().__init__(penalty, robustness)
        self.dmin = dmin

    def value(self, d):
        under = np.minimum(d - self.dmin, 0.0)
        return (under @ under) / d.size

    def dose_grad(self, d):
        under = np.minimum(d - self.dmin, 0.0)
        return 2.0 * under / d.size       # negative while the voxel is cold

# ---- projection: forward dose and transpose back-projection ----
class DoseProjection:
    def __init__(self, physical_dose, scenario_prob=None,
                 physical_dose_exp=None, omega=None):
        self.P = physical_dose
        self.scenario_prob = scenario_prob
        self.P_exp = physical_dose_exp
        self.omega = omega

    def dose(self, scen, w):
        return self.P[scen] @ w

    def expected_dose(self, w):
        return self.P_exp @ w

    def back_project(self, scen, dose_grad):
        return self.P[scen].T @ dose_grad

    def back_project_prob(self, dose_grad_exp, omega_grad):
        return self.P_exp.T @ dose_grad_exp + 2.0 * omega_grad

def max_value_and_weights(values, approx="logsumexp", tau=10.0):
    values = np.asarray(values, dtype=float)
    if approx == "none":
        weights = np.zeros_like(values)
        weights[np.argmax(values)] = 1.0
        return float(values.max()), weights
    if approx == "logsumexp":
        m = values.max()
        ex = np.exp(tau * (values - m))
        weights = ex / ex.sum()
        return float(m + np.log(ex.sum()) / tau), weights
    if approx == "pnorm":
        p = max(values.size, 2)
        norm = np.linalg.norm(values, ord=p)
        if norm == 0:
            return 0.0, np.zeros_like(values)
        weights = values ** (p - 1) / (norm ** (p - 1))
        return float(norm), weights
    raise ValueError(approx)

def objective_and_gradient(w, projection, structures, scenarios,
                           max_approx="logsumexp", tau=10.0):
    doses = {k: projection.dose(k, w) for k in scenarios}
    dose_grads = {k: np.zeros_like(doses[k]) for k in scenarios}
    f = 0.0

    cowc_values = np.zeros(len(scenarios))
    cowc_grads = {k: np.zeros_like(doses[k]) for k in scenarios}
    dose_grad_exp = None
    omega_grad = np.zeros_like(w)

    for st in structures:
        idx = st["idx"]
        role = st["role"]
        objectives = st["objectives"]

        for obj_ix, obj in enumerate(objectives):
            rob = obj.robustness

            if rob == "none":
                k = scenarios[0]
                d = doses[k][idx]
                f += obj.penalty * obj.value(d)
                dose_grads[k][idx] += obj.penalty * obj.dose_grad(d)

            elif rob == "STOCH":
                for local, k in enumerate(scenarios):
                    d = doses[k][idx]
                    scale = projection.scenario_prob[local] * obj.penalty
                    f += scale * obj.value(d)
                    dose_grads[k][idx] += scale * obj.dose_grad(d)

            elif rob == "PROB":
                d_exp = projection.expected_dose(w)
                if dose_grad_exp is None:
                    dose_grad_exp = np.zeros_like(d_exp)
                f += obj.penalty * obj.value(d_exp[idx])
                dose_grad_exp[idx] += obj.penalty * obj.dose_grad(d_exp[idx])
                key = st.get("omega_key")
                if obj_ix == 0 and projection.omega is not None and key is not None:
                    Omega = projection.omega[key]
                    omega_w = Omega @ w
                    scale = obj.penalty / idx.size
                    f += scale * float(w @ omega_w)
                    omega_grad += scale * omega_w

            elif rob in ("VWWC", "VWWC_INV"):
                stack = np.column_stack([doses[k][idx] for k in scenarios])
                use_max = (role in ("OAR", "EXTERNAL") and rob == "VWWC") or \
                          (role == "TARGET" and rob == "VWWC_INV")
                wc_dose = stack.max(axis=1) if use_max else stack.min(axis=1)
                winners = stack.argmax(axis=1) if use_max else stack.argmin(axis=1)
                f += obj.penalty * obj.value(wc_dose)
                delta = obj.penalty * obj.dose_grad(wc_dose)
                for local, k in enumerate(scenarios):
                    m = winners == local
                    dose_grads[k][idx[m]] += delta[m]

            elif rob == "COWC":
                for local, k in enumerate(scenarios):
                    d = doses[k][idx]
                    cowc_values[local] += obj.penalty * obj.value(d)
                    cowc_grads[k][idx] += obj.penalty * obj.dose_grad(d)

            elif rob == "OWC":
                values = []
                local_grads = []
                for k in scenarios:
                    d = doses[k][idx]
                    values.append(obj.penalty * obj.value(d))
                    local_grads.append(obj.penalty * obj.dose_grad(d))
                f_owc, weights = max_value_and_weights(values, max_approx, tau)
                f += f_owc
                for local, k in enumerate(scenarios):
                    dose_grads[k][idx] += weights[local] * local_grads[local]

            else:
                raise ValueError(rob)

    if np.any(cowc_values):
        f_cowc, weights = max_value_and_weights(cowc_values, max_approx, tau)
        f += f_cowc
        for local, k in enumerate(scenarios):
            dose_grads[k] += weights[local] * cowc_grads[k]

    weight_grad = sum(projection.back_project(k, dose_grads[k]) for k in scenarios)
    if dose_grad_exp is not None:
        weight_grad += projection.back_project_prob(dose_grad_exp, omega_grad)
    return f, weight_grad

def plan(P_scen, structures, n_spots, scenario_prob=None):
    w0 = np.ones(n_spots)
    bounds = [(0.0, None)] * n_spots
    projection = DoseProjection(P_scen, scenario_prob=scenario_prob)
    scenarios = list(range(len(P_scen)))
    res = minimize(lambda x: objective_and_gradient(x, projection, structures, scenarios),
                   w0, jac=True, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x
```
