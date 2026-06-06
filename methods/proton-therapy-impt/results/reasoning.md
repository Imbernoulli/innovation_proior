OK, let me think this through from scratch. I have a proton beam and a tumour, and I want to put a uniform high dose on the tumour while sparing whatever sits next to it. The thing that makes protons special is the depth–dose curve: a proton doesn't bleed dose all the way through like a photon. It coasts through the entrance region depositing relatively little, then near the end of its range it dumps almost everything in a sharp spike — the Bragg peak — and past that spike the dose falls to essentially nothing. The depth of the spike is set by the proton's energy: more energy, deeper peak. So one energy paints a thin high-dose shell at one depth and nothing beyond it.

That distal nothing is the whole prize. A photon ray going into a patient also comes out the other side, dosing everything downstream of the tumour. A proton ray can be made to stop *at* the far edge of the tumour and deposit nothing past it. If I can cover the target and exploit that clean distal edge, I spare tissue behind the tumour in a way no photon arrangement can.

But one pristine peak is far too narrow in depth to cover a tumour that's, say, 6 cm thick. So the classical move: take several proton energies, fire them at the same spot, and add their peaks at graded depths. A low-energy peak near the proximal edge, a slightly higher one a bit deeper, and so on to the distal edge. If I pick the relative *weights* of those peaks right, the superposition of the spikes is a flat plateau of uniform dose spanning the tumour depth — the spread-out Bragg peak. The plateau is *linear* in the per-peak weights. Each pristine peak deposits a fixed spatial pattern, and I scale it by a weight (physically, the number of protons I send at that energy). Shaping the depth dose is nothing but *choosing weights on a fixed set of dose patterns*.

Now lift that from one pencil to a whole field. With magnetic scanning I can steer a thin proton pencil beam laterally anywhere in the field and switch its energy to set the depth. So I don't have just a stack of energies at one lateral position — I have a 3-D grid of "spots", each spot being one (lateral position, energy) pencil that deposits its own fixed dose pattern (a lateral Gaussian times a Bragg curve in depth), and each scaled by its own weight. The intensity of every individual spot is an independent knob.

The conventional way to use those knobs is timid. Spread-out-Bragg-peak planning, single-field-uniform-dose: I make *each field* deliver a uniform dose to the target on its own, then sum a few fields from different angles. Every field is self-flat. That's safe and it's what passive scattering basically forces on you. But stare at it — I'm throwing away most of the freedom. If each field has to be uniform by itself, I can't use one field's deliberate hot patch to fill another field's deliberate cold patch, which is exactly what I'd need to carve dose around a concave organ tucked into the tumour. The photon people already crossed this bridge: instead of demanding each field be uniform, let each field be *deliberately inhomogeneous*, modulate the fluence across the field, and only require that the *sum* over fields be uniform on the target and low on the organs. That's intensity modulation. The question is whether I can do the proton version — and not just laterally like a photon, but in depth too, because protons let me modulate where in depth each spot dumps its dose. Full 3-D modulation of every individual Bragg spot.

So the design variable is the weight vector over all spots, w, with one entry per (lateral position, energy) pencil. Let me set up the dose model. Spot j deposits a fixed spatial pattern; call the dose it gives to voxel i, per unit weight, d_{ij}. Because each proton deposits independently (to first order the beam is a linear system — superposition holds), the dose in voxel i from the whole plan is just the weighted sum over spots:

  D_i = Σ_j d_{ij} w_j.

Stack that over voxels and it's a matrix–vector product, D = P w, where P is the dose-influence matrix: P_{ij} = dose to voxel i per unit weight of spot j. P is fixed — I compute it once from the beam model and the patient CT — and then *every* question about a plan is linear algebra on w. This is exactly the photon inverse-planning scaffold (beamlets, an influence matrix, weights), and I'm reusing it wholesale; the only thing that's different about protons is what's *inside* P — Bragg peaks in depth instead of exponential rays — and, as I'll find out the hard way, how fragile that P is.

Now, what do I want from w? "Uniform D_pres on the target, low on the OARs." There are thousands of spots and the system is wildly underdetermined and the goals conflict, so I can't just solve a linear system — I pose it as minimising a penalty and let an optimiser find w. What penalty? Let me think about what "wrong" means voxel by voxel and just sum it up.

On the target, every voxel wants to sit at the prescription dose D_pres. Both directions are bad: too cold is a coverage failure, too hot is an overdose. The natural, smooth, symmetric penalty is the squared deviation, (D_i − D_pres)². So for a target voxel I penalise (D_i − D_pres)².

On an OAR it's not symmetric. Dosing an OAR *less* than its tolerance is fine — good, even. Only *exceeding* a max dose D_max is bad. If I penalised the squared deviation from D_max I'd be pushing the OAR dose *up* toward D_max wherever it's below, which is insane — I'd be fighting my own sparing. So the OAR penalty must be one-sided: penalise the overshoot only, max(D_i − D_max, 0)². The positive-part operator kills the penalty whenever the OAR is already under tolerance. Same logic the other way for a target minimum-dose floor: penalise max(D_min − D_i, 0)² — only the underdosing hurts. I can write the one-sided versions as a ramp H_+(r) = r H(r), which is r when r>0 and 0 otherwise, and square that ramp.

Different structures matter differently, and a structure with more voxels shouldn't dominate just by counting more terms, so I attach a per-structure penalty p_s and normalise by the number of voxels |V_s| in that structure. The whole objective is

  f(w) = Σ_s (p_s / |V_s|) Σ_{i∈V_s} [penalty]_i²,

with [penalty] being (D_i − D_pres) on a target, the ramp max(D_i − D_max,0) on an OAR, the ramp max(D_min − D_i,0) on a target floor. The p_s are the dials the planner turns to trade target coverage against OAR sparing.

Why squared and not, say, absolute value or a hard dose-volume constraint? Because I'm going to optimise with gradients over thousands of variables, and I want something smooth and, in the dose, convex. A sum of squares is differentiable everywhere and has a clean gradient; the absolute value has a kink at zero; a dose-volume-histogram criterion ("no more than X% of the OAR above Y") is genuinely non-convex and not differentiable, so it makes a lousy thing to *drive* the optimisation — better to bolt it on later as a constraint if I need it, and let smooth quadratics do the heavy lifting. Quadratic penalties it is.

One more piece of physics constrains w: a weight is a number of protons. I can't deliver a negative number of protons. So w ≥ 0, componentwise. That's a bound constraint, not something I can wish away — without it the optimiser would happily use negative weights to sculpt the dose and hand me an undeliverable plan.

So the problem is: minimise f(w) = Σ_s (p_s/|V_s|) Σ_{i∈V_s} g_s(D_i)², with D = P w and w ≥ 0. It's a bound-constrained nonlinear least-squares problem; convex in D, and since D is linear in w, convex in w too (the squared ramps are convex). Good — convex with a simple bound, that's a problem a gradient-based bound-constrained solver eats for breakfast. I'll need the gradient.

Chain rule. The objective depends on w only through the dose D = P w. So

  ∂f/∂w = Pᵀ (∂f/∂D).

The inner factor ∂f/∂D is a per-voxel "delta" that I read straight off each penalty: for a target voxel, ∂/∂D (D − D_pres)² = 2(D − D_pres); for an OAR overdose term, 2·max(D − D_max, 0), so the delta is zero wherever the OAR is under tolerance; for a target floor, the penalty is max(D_min − D,0)², so the derivative is −2·max(D_min − D,0), equivalently 2·min(D − D_min,0). That negative sign matters: if a target voxel is cold, the gradient points in the direction that raises its dose. Multiply each by its p_s/|V_s|, scatter the deltas back to their voxels, and I have a full-length per-voxel delta vector. Then one multiply by Pᵀ back-projects that voxel-space gradient into spot-weight space. Same matrix P forward (dose) and transpose backward (gradient). Both are sparse matrix–vector products — cheap, and I precomputed P once. Hand f and ∂f/∂w to a bound-constrained optimiser, start from some positive w, iterate. That's the whole nominal IMPT optimiser.

Let me sanity-check that it does the right thing before I get clever. With enough fields and enough spots, this will find a w that makes the target beautifully uniform and pushes the OARs down — the squared target term forces homogeneity, the one-sided OAR terms carve dose away wherever an OAR pokes above tolerance. And with many fields, full 3-D spot modulation versus cruder schemes barely differ; they all converge to good plans because the problem is over-supplied with degrees of freedom. The full-3-D modulation only really earns its keep as I *reduce* the number of fields — then only modulating every individual spot in depth keeps both target homogeneity and OAR sparing, where the restricted schemes start to fail. Fine. So as a nominal planner this works, and it's strictly more capable than single-field-uniform-dose.

And now I hit the wall.

I picked up the photon scaffold whole, including its tacit assumption that the dose is more or less the dose — that small geometric errors smear things a little but don't change the structure. For photons that's roughly true: shift the patient a couple of millimetres and every ray still deposits dose all along its now-slightly-moved path; the dose distribution translates, it doesn't deform. For protons that assumption is poison, and the reason is the very distal edge I was so pleased about.

Watch what my optimiser does when an OAR — say the spinal cord — sits right behind the target. The cleanest possible way to spare the cord is to point a field so its Bragg peaks stop *just* upstream of the cord, using the steep distal fall-off as a knife edge: full dose in the last slice of tumour, near-zero a couple of millimetres later in the cord. My squared penalties love this — it nails target coverage and keeps the cord cold, so the optimiser drives straight to it. The *nominal* plan looks gorgeous.

But where exactly does that distal edge land? It lands at the proton range, and I don't know the range exactly. The dominant error is that I get the range from the patient's CT, converting Hounsfield units to proton stopping powers, and that conversion is only good to a few percent — call it 3 to 3.5% of the range, and people stress-test plans at ±5%. The error grows with the water-equivalent depth, so a deep target can be off by several millimetres. Add daily setup error and anatomical change. Now my knife edge, which I parked two millimetres in front of the cord, is in the wrong place. A small overshoot and the high-dose region slides *into* the cord — an overdose of a serial organ, the exact catastrophe I was trying to avoid. A small undershoot and a cold slab opens at the distal target edge — a coverage hole. The nominal plan was optimal and the delivered plan can be a disaster, and the more the optimiser exploited the sharp gradient for sparing, the worse the blow-up. Steep longitudinal gradients buy me sensitivity to range error; steep lateral gradients buy me sensitivity to setup error. The feature is the bug.

And it's *worse* for the fully modulated multi-field plan than for the timid single-field-uniform-dose one, which stings because full modulation was the whole point. When every field is allowed to be inhomogeneous, the optimiser arranges each field's hot and cold patches so they cancel — but they cancel *only in the nominal geometry*. A range or setup error hits different fields differently (a field coming from the left moves its peaks differently than one from below), so the carefully arranged cancellation comes apart and the residual hot/cold patches show through. Single-field-uniform-dose, where each field is independently flat on the target, has nothing to come apart — no field is leaning on another to fill its hole. So the more aggressively I modulate, the more fragile I am.

Reflex: just add a margin. Photon planning expands the clinical target to a planning target volume by a geometric margin and plans to cover the bigger volume. So expand the target, cover the expansion, done? No — and the reason is the same deformation that bit me a moment ago. A margin works for photons because their dose is shift-invariant: cover a slightly bigger volume and a small shift still leaves you covered. For protons a shift doesn't translate the dose, it *reshapes* it — the peaks move in depth, the distal edge migrates, the whole high-dose region changes form. Covering a geometrically expanded volume in the nominal scenario does not guarantee coverage of the real target in a shifted/range-errored scenario, because the shifted dose isn't the nominal dose translated. A static margin is the wrong tool. I have to make the optimiser reason about the errors themselves.

The problem is massively underdetermined — many different weight vectors give the *same* nominal dose. That redundancy is exactly why the optimiser was free to pick a fragile solution (knife-edge behind the cord). But it also means robust solutions exist in that same null-space-rich set — plans that achieve the target nominally *and* don't fall apart under error — if only I optimise for robustness instead of letting it pick fragility by accident. The redundancy is both the danger and the cure.

So: stop optimising the nominal dose. Build a *set* of error scenarios and optimise across them. Enumerate plausible errors — range overshoot and undershoot of a few percent, setup shifts of a few millimetres along each axis, plus the nominal — index them k = 1…K. For each scenario the geometry is different, so the dose-influence matrix is different: P^{(k)}, the influence matrix recomputed (or shifted) under error k. The dose under scenario k is D^{(k)} = P^{(k)} w. Same w — I deliver one plan — but it produces K different doses. The question is what scalar to minimise over this family.

If I think of the errors as random with a probability distribution — scenario k happens with probability π_k — then the principled thing is to minimise the *expected* objective:

  f_exp(w) = Σ_k π_k f(w; D^{(k)}).

This is just my old objective averaged over scenarios. Its gradient is the probability-weighted sum of the per-scenario gradients, ∂f_exp/∂w = Σ_k π_k (P^{(k)})ᵀ (∂f/∂D^{(k)}) — each scenario back-projects through its own matrix, then I average. Smooth, easy, less conservative. The optimiser, to make the *average* good, naturally redistributes dose so no single likely error wrecks the plan — it'll prefer, for instance, to cover the distal target region from a second beam direction so that a range error in the first beam is backstopped, rather than betting everything on one knife edge. If I push the probabilistic view to its limit I can optimise the objective on the expected dose and add an integral variance term, wᵀΩw, where Ω is accumulated from the scenario distribution. Its gradient is 2Ωw, scaled by the same structure penalty normalization, so it directly punishes weight configurations whose dose swings a lot when the error varies. Penalising variance is penalising fragility, stated in dose.

If instead I don't trust a probability model and just say "the error lies somewhere in this bounded set and I must be safe for *all* of it," then I minimise the *worst* scenario:

  f_wc(w) = max_k f(w; D^{(k)}),
  i.e. solve  min_w max_k f(w; D^{(k)}).

A min–max. This guarantees the bound — whatever error actually occurs, the realised objective is no worse than the value I drove down — at the cost of being pessimistic (it cares only about the single worst case, however unlikely). For a serial OAR that I must protect under *any* plausible error, this is the right conservatism.

Worst case can mean several different things, and they differ in how pessimistic and how physical they are.

The bluntest is voxel-wise. For each voxel, scan across the scenarios and keep the dose that is worst *for that voxel's role*: in the target take the *minimum* over scenarios (cold is the target's enemy), in an OAR take the *maximum* over scenarios (hot is the OAR's enemy). That builds a single synthetic worst-case dose vector D_wc voxel by voxel, and I just feed D_wc into my ordinary penalty f. Cheap — one penalty evaluation. Its gradient is subtle but mechanical: each voxel's delta only flows back to the *scenario that realised the worst value at that voxel*, so I compute the deltas on D_wc and scatter each voxel's delta through the P^{(k)} of whichever scenario k won the min/max at that voxel. The catch: the per-voxel worst doses come from *different* scenarios at neighbouring voxels, so D_wc is a Frankenstein dose that no single real error ever produces — it can be over-pessimistic, because it asks the plan to survive a combination of worst cases that can't physically co-occur.

The more physical version is composite worst case. Instead of taking the worst per voxel, take the worst per *whole scenario*: evaluate the entire objective separately in each scenario, f_k = Σ_s f_s(D^{(k)}), and minimise the maximum over scenarios, max_k f_k. Each f_k is a real, internally consistent dose from one real error, so I'm not asking the plan to beat an impossible combination — just the worst single realisable error. The price is that "max over scenarios" of a scalar is non-smooth: the gradient is the gradient of whichever scenario is currently worst, and that argmax can flip discontinuously from iteration to iteration, which can stall a gradient method at the kinks. I can use the hard argmax if the optimiser tolerates it, but a smoother path is to replace max by log-sum-exp, (1/τ) log Σ_k exp(τ f_k), which approaches max_k f_k as τ grows, or by a large-p p-norm of the f_k. Then the gradient becomes a weighted blend of the per-scenario gradients, with most of the weight on the worst scenario. Objective-wise worst case takes the worst scenario per objective term rather than per whole plan, then applies the same hard or smoothed maximum. That is slightly more conservative than COWC and less synthetic than voxel-wise worst case.

Either way — expected value, expected-dose-plus-variance, voxel-wise worst case, composite worst case, objective-wise worst case — the *engine* is the one I already built. I still have penalised-least-squares structure objectives. I still get the gradient by back-projecting a per-voxel delta through an influence matrix with a transpose. I still hand f and ∂f/∂w to a bound-constrained solver with w ≥ 0. The only thing the robustness adds is: keep a *list* of influence matrices, one per scenario; compute the per-scenario doses; combine them into one value and one gradient according to the chosen rule; back-project each scenario's contribution through its own P^{(k)}. The robust optimiser is the nominal optimiser wrapped in a scenario loop and a combine rule. The cost is real — several scenarios evaluated every iteration, several matrix products instead of one — but that's the price of a plan that's still good when the range comes in a few millimetres off.

Let me write it. I need the code to keep the same separation: dose objectives compute an unweighted value and an unweighted dose-gradient; the wrapper multiplies by the structure penalty, chooses the robustness mode, accumulates scenario dose-gradients, and only then back-projects them through the scenario influence matrices. I'll write it in NumPy/SciPy, but keep the factorization aligned with that structure.

```python
import numpy as np
from scipy.optimize import minimize

# ----- dose objectives: unweighted value + unweighted dose-gradient -----
class DoseObjective:
    def __init__(self, penalty=1.0, robustness="none"):
        self.penalty = penalty
        self.robustness = robustness

class SquaredDeviation(DoseObjective):   # target: (D - dref)^2, both directions
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

# ----- projection: forward dose and transpose back-projection -----
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

The chain, start to finish: a proton spot deposits a fixed Bragg-peaked dose pattern that scales linearly with its weight, so the whole dose is D = P w with P a precomputed influence matrix — the same beamlet scaffold photons use, but with Bragg peaks inside P. I want D uniform on the target and low on the OARs, so I minimise a sum of voxel penalties — squared deviation on targets, one-sided squared over/under-dose on OARs and target floors — with w ≥ 0 because a weight is a proton count, and I get the gradient for free by back-projecting the per-voxel deltas through Pᵀ. That nominal optimiser is strictly stronger than making each field uniform, and it's the only thing that holds up as I cut the field count. But protons' sharp distal edge, which I leaned on for sparing, lands at an uncertain range — a few percent, several millimetres — so the nominal-optimal plan deforms catastrophically under range and setup error, and a static margin can't fix it because the dose reshapes rather than translates. The cure is in the same redundancy that let the optimiser pick a fragile plan: build a set of error scenarios, give each its own influence matrix, and optimise across them — the expected objective if I trust a probability model, the worst-case (smoothed) objective if I must be safe over a whole uncertainty set — back-projecting each scenario through its own P. The robust optimiser is the nominal one wrapped in a scenario loop and a combine rule, and it spends the underdetermined freedom on plans that stay good when the range comes in wrong.
