# Context: inverse planning for scanned proton beams

## Research question

A proton beam does not behave like a photon beam. Photons attenuate roughly exponentially: after a short build-up the dose falls off monotonically with depth, so every photon ray deposits dose all along its path, including well past the tumour and out the far side. Protons instead deposit little dose along their entrance track and then dump almost all of their energy in a narrow peak — the Bragg peak — at a depth fixed by their initial energy, with a near-zero exit dose beyond it. A single proton energy therefore paints a thin high-dose shell at one depth.

The clinical promise is enormous: because there is essentially no dose past the peak, protons can in principle spare tissue distal to the target far better than any photon technique. The open problem is how to turn a set of pristine Bragg peaks — each narrow, each at one depth, each delivered by a magnetically steered pencil beam that can be aimed laterally and re-energised for depth — into a *uniform* high dose over a three-dimensional tumour volume while keeping the dose to neighbouring organs at risk (OARs) low, and how to *choose* the intensity of every one of the thousands of individual spots automatically. A solution must (i) model how spot intensities combine into a 3-D dose, (ii) pose "uniform on target, low on OARs" as something an optimiser can minimise, and (iii) cope with the fact that the very feature that makes protons attractive — the sharp distal edge — also makes the delivered dose acutely sensitive to small errors in where that edge actually lands.

## Background

**The Bragg peak and the spread-out Bragg peak (SOBP).** A monoenergetic proton beam slows continuously in tissue; its rate of energy loss rises as it slows, producing a sharp maximum of dose deposition (the Bragg peak) near the end of range, then an abrupt distal fall-off. The depth of the peak is set by the beam energy. A single pristine peak is far too narrow in depth to cover a tumour, so the classical fix is to superpose several pristine peaks of graded energies with chosen relative weights so that their sum is a flat plateau of uniform dose across the target depth — the spread-out Bragg peak. The SOBP is literally a weighted superposition of pristine Bragg curves at different energies; the weights are picked so the plateau is flat. This already contains the germ of the whole idea: dose is *linear* in the per-peak weights, and shaping a dose distribution is choosing those weights.

**Beam delivery.** Two delivery paradigms exist. Passive scattering broadens and shapes a single SOBP with scatterers, range modulators, apertures and compensators; every field delivers a uniform dose block to the target. Active pencil-beam scanning instead steers a thin proton pencil beam magnetically across the field (lateral position) and switches its energy (depth), depositing dose spot by spot. Scanning makes the intensity of each individual Bragg spot an independent, freely settable degree of freedom — which is the enabling technology for modulating intensity in depth, not just laterally.

**Inverse planning for photons.** For photons, the inverse-planning idea (Brahme; Bortfeld; Webb) was already established: discretise each beam into many small beamlets ("bixels"), precompute a dose-deposition (dose-influence) matrix giving the dose each unit-weight beamlet deposits in each voxel, then solve for the beamlet weights that best meet dose objectives. Because the underlying dose is linear in the beamlet fluences, the dose to a voxel is a matrix-vector product of the influence matrix with the weight vector, and the planning problem becomes a large numerical optimisation — typically minimising a weighted sum of quadratic, voxel-based penalties (squared deviation from the prescribed target dose; one-sided squared overdose penalties on OARs) subject to non-negative weights. This is intensity-modulated radiation therapy (IMRT). The same scaffolding — influence matrix, voxel-based quadratic penalties, gradient-based numerical optimisation — is what a scanned proton plan would be built on.

**Dose-volume objectives.** Clinical goals are stated as the prescription dose to the target plus dose-volume limits on OARs (e.g. no more than a given volume above a given dose). For numerical optimisation these are reduced to per-voxel surrogates: a squared deviation from prescription on the target, one-sided squared penalties for exceeding a maximum on an OAR or falling below a minimum on the target, each structure carrying a penalty weight that encodes its clinical priority. Dose-volume-histogram constraints proper are non-convex and are usually added on top as constraints rather than as the smooth driving objective.

**Why proton plans are fragile — range and setup uncertainty.** The proton range in a patient is not known exactly. The dominant source is the conversion of CT Hounsfield units to proton stopping powers, which carries an uncertainty on the order of a few percent of the range (commonly quoted around 3–3.5%, and stress-tested at ±5%); anatomical change, weight change and daily setup also shift the geometry. A range error proportional to the water-equivalent depth slides the whole distal edge forward or backward by several millimetres. Because that distal edge is where the dose gradient is steepest, a small range error converts directly into a large dose error just beyond it. It is a documented pathology that an optimiser, left to its own devices, will deliberately tuck a Bragg peak's distal edge immediately upstream of a serial OAR (such as the spinal cord) to exploit the steep fall-off for sparing — and that this is exactly the configuration that a small overshoot turns into an OAR overdose, or a small undershoot turns into a target cold spot. Steep longitudinal gradients make plans sensitive to range errors; steep lateral gradients make them sensitive to setup errors. A static geometric target margin, which suffices for photons because their dose is nearly shift-invariant, does not rescue protons: a shift *changes the shape* of the deposited dose (peaks move in depth), so a margin-expanded volume does not guarantee coverage under error.

**Multi-field modulation makes this sharper.** When several scanned fields are combined and each field is allowed to be deliberately non-uniform, their individual hot and cold patches are arranged to cancel only in the nominal scenario. Any misalignment between fields — from a range or setup error that affects fields differently — breaks that cancellation, so a fully modulated multi-field plan can be *more* fragile than one in which each field is independently uniform on the target. The redundancy of solutions (many weight vectors give the same nominal dose) is what lets the optimiser pick a fragile member of that set in the first place — the same nominal dose is reachable by many differently-fragile weight vectors.

**Where the nominal scaffold stalls under uncertainty.** The photon inverse-planning machinery optimises a single dose distribution computed from one influence matrix — the nominal geometry. It has no place to put the fact that the range and setup are uncertain: the optimiser sees one dose and drives it to optimality, with no knowledge that the delivered dose may differ. Folding the documented range/setup fragility into the planning objective is the open difficulty, and any attempt to do so has to contend with the cost of repeatedly recomputing dose under shifted geometry.

## Baselines

**Single-field uniform dose (SFUD) / classical SOBP planning.** Each field independently delivers a uniform dose to the target by weighting its own Bragg spots so that field's contribution alone is flat across the tumour; the fields are then summed. Core idea: superpose graded-energy Bragg peaks per field to flatten depth, shape laterally with the scan pattern. Gap: because every field must be self-uniform, the technique cannot exploit inter-field modulation to carve dose around a concave OAR, and it leaves the largest gains (sparing with few fields, conformality around complex shapes) on the table. It is, however, intrinsically more robust than fully modulated planning, since no field relies on another to fill a hole.

**Photon IMRT inverse planning (Bortfeld; Webb).** Beamlet weights of exponentially attenuating photon beams are optimised against voxel-based quadratic objectives using a precomputed dose-influence matrix, with non-negative weights. Core idea and math: dose = (influence matrix)·(weights); minimise Σ over structures of penalty·(squared deviation / one-sided squared over/under-dose). Gap for protons: it assumes beams that deposit dose all along their path and whose dose is nearly invariant to small shifts; it has no notion of a Bragg peak in depth, no depth modulation, and — critically — no mechanism for the range fragility that protons introduce, so transplanting it unchanged yields nominal-optimal but unstable proton plans.

**Lower-dimensional proton intensity modulation.** Intermediate schemes modulate intensity within a field in fewer than three dimensions — e.g. 2-D modulation of the proximal weights while leaving the depth-stack fixed, or modulating only across fields. Core idea: a restricted set of the spot weights is free. Gap: with many fields these restricted schemes are nearly as good as full modulation, but as the number of fields is reduced only full 3-D modulation of every individual Bragg spot preserves both target homogeneity and OAR sparing — so the restricted schemes cannot deliver the conformality that few-field, complex-geometry cases demand.

**Static-margin (PTV-based) proton planning.** Expand the clinical target by a geometric margin, then plan to cover the expanded volume in the nominal scenario. Core idea: borrow the photon CTV→PTV margin recipe. Gap: protons' dose is not shift-invariant, so a geometric margin does not translate into dose robustness; coverage under a range or setup error is not guaranteed, leaving the margin recipe inadequate for protons.

## Evaluation settings

The natural yardstick is a planning CT of a real anatomy with the target volume(s) and OARs contoured, a small set of beam directions (gantry angles), and a prescription dose to the target plus dose limits on each OAR. Plans are judged on dose-volume histograms of target and OARs — target homogeneity and coverage, OAR mean/max dose and volume-above-threshold — and on conformity, as a function of the number of fields used. Robustness is assessed by recomputing the planned dose under a battery of error scenarios (range over/undershoot of a few percent; isocentre setup shifts of a few millimetres along each axis) and inspecting the spread of the resulting DVHs — the worst-case and band of curves across scenarios — rather than the nominal curve alone. Representative sites are those where a serial OAR sits immediately behind the target (skull-base and paraspinal tumours near the brainstem or spinal cord, head-and-neck), where the distal-edge fragility bites hardest. Optimisation cost (time to solve, memory, per-iteration work) is itself a reported quantity.

## Code framework

The primitives are those already standard in beamlet-based inverse planning: a precomputed linear dose-influence operator, a vector of non-negative beam weights, a smooth scalar objective built from per-structure voxel penalties, and a bound-constrained numerical optimiser. The empty slots are the per-structure penalty objective, the gradient back-projected through the influence operator, and the assembly of one scalar value and one weight-gradient for the solver.

```python
import numpy as np
from scipy.optimize import minimize

# P           : sparse dose-influence matrix (dose per unit spot weight, per voxel).
# structures  : list of dicts with voxel indices, role, and objective objects.

class DoseObjective:
    def __init__(self, penalty=1.0):
        self.penalty = penalty

    def value(self, dose):
        raise NotImplementedError          # TODO

    def dose_grad(self, dose):
        raise NotImplementedError          # TODO

class DoseProjection:
    def __init__(self, P):
        self.P = P

    def dose(self, w):
        return self.P @ w

    def back_project(self, dose_grad):
        return self.P.T @ dose_grad

def objective_and_gradient(w, projection, structures):
    # TODO: evaluate structure objectives on the dose and
    #       return (objective_value, weight_gradient).
    pass

def plan(P, structures, n_spots):
    w0 = np.ones(n_spots)
    bounds = [(0.0, None)] * n_spots
    projection = DoseProjection(P)
    res = minimize(lambda x: objective_and_gradient(x, projection, structures),
                   w0, jac=True, bounds=bounds, method="L-BFGS-B")
    return res.x
```
