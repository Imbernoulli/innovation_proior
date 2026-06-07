Let me start from the thing I actually can't do, because that's where the leverage is. I want to look inside a sealed cargo container and find a lump of uranium or plutonium hidden in it — maybe wrapped in lead. The whole difficulty is that the object I'm hunting is the object that defeats my probe. X-rays die in a few centimetres of steel; a shielded actinide core is just a black hole to them. I could fire a high-energy beam through it, but then I need an accelerator, I'm dosing the cargo, and I have a one-sided geometry that a closed box doesn't give me. So the ordinary radiographic toolbox is out before I start. I need a probe that genuinely goes *through* metres of dense matter, that I don't have to generate, and that says something specifically about high atomic number, because that's what distinguishes a uranium brick from a pallet of canned food.

What goes through metres of steel for free? Cosmic-ray muons. They rain down everywhere — the secondaries of cosmic-ray cascades, `pi -> mu nu`, heavy charged leptons with a couple of GeV of energy on average. A few-GeV muon loses energy only by ionization, about `2 MeV` per `g cm^-2`, so it sails through metres of rock or iron where a photon wouldn't survive a hand's breadth. And the flux is this gentle steady drizzle — roughly one muon through a fingernail-sized patch per minute, about `10^4 m^-2 min^-1`, call it 160 Hz over a square metre. That number is a gift twice over. It's low enough that I can catch and reconstruct muons *one at a time* — no pileup, every muon is its own little experiment. And it's high enough that over minutes-to-hours I accumulate real statistics over a container. So the probe exists, it's penetrating, it's free, and it's single-particle. Now: what does a muon *tell* me as it crosses the box?

Two channels. One, it loses energy and can range out — so I could count how many muons survive in each direction and make an absorption radiograph. That's exactly what George did in '55 measuring rock over a tunnel, and what Alvarez did under the Pyramid of Chephren, looking for hidden chambers by the deficit in the transmitted flux. But think about what that signal is. It's a count of survivors along each line of sight — a line integral of areal density, nothing more. No depth: I can't tell *where* along the line the dense thing sits, so no 3D. And it's weakly `Z`-sensitive — a thick slab of water can attenuate as much as a thin slab of lead, so a low-`Z` mass impersonates a high-`Z` one. And to see a statistically significant *deficit* in survivors I need a huge exposure or a mountain of overburden. For finding a small shielded core in a truck, absorption is the wrong channel. So set it aside, but remember the geometry it gave me: muons coming in from a spread of angles, crossing the volume, detectable on the far side.

The second channel is scattering. As the muon crosses material it doesn't just lose energy — it gets kicked, many times, by elastic Coulomb scattering off nuclei. Rutherford: the differential cross-section goes like `Z^2 / sin^4(theta/2)`, so the kicks are mostly tiny-angle and there are a *lot* of them, and crucially the rate scales as `Z^2`. Many independent small kicks — by the central limit theorem the net projected deflection is Gaussian, zero mean, with some width. Let me get that width, because that's the whole game. The standard result for the central ~98% of the projected scattering angle is the Highland form,

    theta_0 = (13.6 MeV / (beta c p)) * z * sqrt(L / X0) * [1 + 0.038 * ln(L z^2 / (X0 beta^2))]

with `p`, `beta c` the muon's momentum and velocity, `z=1` its charge, `L` the thickness, and `X0` the radiation length of the material. For a relativistic muon `beta ≈ 1` and the log is a slow correction I can fold into the constant; people use `theta_0 ≈ (15 MeV / p) sqrt(L / X0)`. Stare at the `Z` dependence. It's all hiding in `X0`, the radiation length — the characteristic depth for electromagnetic interactions, and `X0` *plummets* with atomic number. As a depth it's about 36 cm in water, 11 cm in concrete, 1.8 cm in iron, 0.56 cm in lead, 0.32 cm in uranium. So `1/X0`, the thing under the square root, jumps by two orders of magnitude from water to uranium. Per centimetre, high-`Z` material scatters muons *enormously* more. Let me just put numbers on it for a 3 GeV muon through 10 cm: water about 2.6 mrad RMS, iron about 12, lead about 21, uranium about 28. That's an order of magnitude of contrast between ordinary cargo and a high-`Z` core — exactly the discriminant I was missing in the absorption channel.

And here's why this channel is *practical* and absorption isn't. The signal is an angle of a few-to-tens of milliradians. I can measure that from *position*: put a tracking station above the box and one below it, with enough position planes in each station to fit a line in two projected views, and I get the incoming straight-line track and the outgoing straight-line track for every muon. The bend between them is the scatter. Millimetre positions over a metre of lever arm give me milliradian angles for free — no magnetic spectrometer, no precise energy measurement. Scattering is about as `Z`-sensitive as energy loss, but I can *read it off geometry*. So the measurement is: for each muon, two tracks, in and out. That's my raw datum.

Now I need to turn "this muon bent by angle `theta`" into "there is dense material *here*." Let me first pin down what physical quantity I'm even trying to reconstruct, because the angle alone isn't it — the angle depends on momentum and thickness too, and I want a property of the *material*. Square the Highland width: `theta_0^2 = (15/p)^2 (1/X0) L`. The thickness comes out linearly, the material enters as `(1/X0)`, and the momentum as `1/p^2`. Fix a nominal momentum `p_0` (I can't measure each muon's momentum from tracking alone, so I'll pretend they're all at `p_0 ≈ 3 GeV` and pay for it in extra variance later) and define

    lambda := (15 / p_0)^2 * (1 / X0) = theta_0^2 / L.

This `lambda` is the *scattering density*: the mean-square projected scatter per unit length, for a nominal muon, in this material. It's a clean material fingerprint — it's `theta_0^2/L` so the thickness is normalized out, and because it's `~1/X0` it inherits the strong `Z` separation. That's the field I want to image: a 3D map of `lambda(x,y,z)`, low for air and water, medium for iron and concrete, high for lead, tungsten, uranium. So my target is a voxelized `lambda_hat`.

Good. Reconstructing a spatial field from many rays crossing it from many angles is just tomography, and I know that machinery. Discretize the volume into voxels, write each ray's signal as a sum over the voxels it crosses weighted by path length, invert. Let me try to borrow it directly. In ordinary X-ray CT the ray signal is a deterministic line integral, `s_i = sum_j w_ij f_j` with `w_ij` the path length of ray `i` through voxel `j`, and I solve `s = W f` by back-projection or ART or ML-EM. So the obvious move is: make `f_j = lambda_j` and write `s_i = sum_j L_ij lambda_j`. But wait — what *is* `s_i` here? In CT it's a measured number, an attenuation integral. Here the muon's measured projected scatter `theta_i` is a single noisy draw from a Gaussian whose *variance* is `lambda L`. One muon through a slab gives me one sample of a zero-mean random variable, not a line integral. I can't set `theta_i = sum_j L_ij lambda_j`, because `E[theta_i] = 0`. The thing that's linear in `lambda` is the *variance*, not the signal:

    Var(theta_i) = sum_j L_ij lambda_j,

since the muon's total scatter is the sum of independent per-voxel scatters and variances add. So the deterministic raysum is the right skeleton but on the wrong moment — `s = W f` becomes `v = L lambda` where `v` is the per-ray *variance*. That's a real complication: I don't observe `v_i`, I observe one `theta_i` whose variance is `v_i`. I'll have to come back and do this properly with a likelihood. But the full tomographic inversion — many rays, every voxel on each path sharing the signal, a big system to solve — is heavy, and at this point I don't even have a working proof-of-principle. Let me find the *cheapest possible* thing that localizes the scattering, get a picture out, and only then earn the right to the full inversion.

So back up and ask the laziest question. I have, per muon, an entry track and an exit track, and a bend between them. Where did the bend happen? In reality it happened gradually, smeared over the whole path — but if there's a small dense object somewhere in the volume, *most* of the bending happened right there, where the muon crossed it. So let me make the boldest simplification I can: pretend *all* of the muon's scattering happened at a single point. One kink, straight legs on either side. If that's true, the kink is at the intersection of the extended incoming and outgoing tracks. Find that point, declare "the muon scattered *here*," and dump its scattering signal into the voxel that contains it. That's it. No system to solve, no iteration — one point per muon, accumulate.

Let me make sure the geometry is sane. In a 2D slice the two extended lines must cross, so "the point" is just their crossing. In 3D the entry and exit lines are generally *skew* — they don't meet — so "intersection" isn't defined. But "where do they come closest" always is. The point of closest approach: along each line find the pair of nearest points, take their midpoint. That's the single-scatter vertex estimate. Let me derive it so I can code it. Line one: `r_1(t) = p_in + t v_in`. Line two: `r_2(s) = p_out + s v_out`. Minimize the squared gap `|r_1(t) - r_2(s)|^2`. Set the gradient to zero in `t` and `s`. Let `w_0 = p_in - p_out`, and `a = v_in·v_in`, `b = v_in·v_out`, `c = v_out·v_out`, `d = v_in·w_0`, `e = v_out·w_0`. The two stationarity conditions are that the connecting vector `r_1 - r_2` is perpendicular to *both* directions:

    (r_1 - r_2)·v_in = 0  ->  a t - b s + d = 0,
    (r_1 - r_2)·v_out = 0  ->  b t - c s + e = 0.

Solve the 2×2: from these, `t = (b e - c d)/(a c - b^2)` and `s = (a e - b d)/(a c - b^2)`. The denominator `a c - b^2` is `|v_in|^2|v_out|^2 - (v_in·v_out)^2`, which is zero exactly when the directions are parallel — i.e. no kink, no defined vertex, and I just skip that muon. Otherwise the PoCA point is the midpoint `(r_1(t) + r_2(s))/2`. Clean. This is the same ray-tracing trick used to localize nuclear scattering vertices, repurposed.

Now what do I deposit there? `lambda` is a mean-square *projected* scatter per unit length — a variance density — and the unbiased estimate of a zero-mean projected Gaussian's variance from a single sample is just that sample squared. In a 2D view the signal is `s = (theta_out - theta_in)^2`. In 3D I have two independent projected views, so the matching one-projection variance sample is the average

    s = 1/2 * [ (Delta_theta_x)^2 + (Delta_theta_y)^2 ].

I deposit `s` into the PoCA voxel. The denominator should count every muon whose estimated straight path crosses a voxel, because a muon whose PoCA point is elsewhere is still a zero contribution for the voxels it crossed. After `M` muons, each voxel `j` holds a sum of projected-variance samples assigned to it and a path count `I_j`; the scattering-density estimate is

    lambda_hat(j) = ( sum of assigned s in voxel j ) / ( I_j * L ).

The mean square projected signal is the variance estimate; dividing by `L` turns variance-per-crossing into variance-per-unit-length, which is `lambda`. This is exactly the single-slab estimator `lambda_hat = (1/L)(1/M) sum theta_i^2` I'd write for one homogeneous block, except PoCA has *routed* the nonzero part of each muon's contribution into one voxel and left zeros along the rest of its path. So the whole algorithm is: loop over muons, compute the estimated path voxels, compute the PoCA point and the projected-angle signal, add the signal to the PoCA voxel, increment the path counters, and divide. Linear in the number of muons, no inversion. Let me write it.

```python
def line_voxels(p0, p1, grid):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / np.min(grid.size))) * 3)
    hits, seen = [], set()
    for alpha in np.linspace(0.0, 1.0, n):
        j = grid.index(p0 + alpha * (p1 - p0))
        if j is not None and j not in seen:
            seen.add(j); hits.append(j)
    return hits

def closest_approach_point(p_in, v_in, p_out, v_out):
    v_in = v_in/np.linalg.norm(v_in); v_out = v_out/np.linalg.norm(v_out)
    w0 = p_in - p_out
    a, b, c = v_in@v_in, v_in@v_out, v_out@v_out
    d, e = v_in@w0, v_out@w0
    denom = a*c - b*b
    if abs(denom) < 1e-9:           # parallel: no kink to localize
        return None
    t = (b*e - c*d)/denom; s = (a*e - b*d)/denom
    return 0.5*((p_in + t*v_in) + (p_out + s*v_out))

def projected_angles(v):
    v = v/np.linalg.norm(v)
    vz = max(abs(v[2]), 1e-12)
    return np.array([np.arctan2(v[0], vz), np.arctan2(v[1], vz)])

def projected_scattering_signal(v_in, v_out):
    dtheta = projected_angles(v_out) - projected_angles(v_in)
    return 0.5 * float(dtheta @ dtheta)              # one projected variance sample

def reconstruct_poca(muons, grid):
    S = np.zeros(tuple(grid.n)); I = np.zeros(tuple(grid.n))
    for p_in, v_in, p_out, v_out in muons:
        p_in, v_in = np.asarray(p_in, float), np.asarray(v_in, float)
        p_out, v_out = np.asarray(p_out, float), np.asarray(v_out, float)
        path = line_voxels(p_in, p_out, grid)
        for j in path:
            I[j] += 1                                # zeros count along the path
        pt = closest_approach_point(p_in, v_in, p_out, v_out)
        if pt is None:
            continue
        j = grid.index(pt)
        if j is None:
            continue
        if I[j] == 0:
            I[j] += 1
        S[j] += projected_scattering_signal(v_in, v_out)
    lam = np.zeros_like(S); nz = I > 0
    lam[nz] = S[nz]/(I[nz]*grid.L)              # mean projected theta^2 per unit length
    return lam
```

When is this good and when does it lie to me? The single-scatter assumption is the whole bet. If the volume holds *one small, isolated* dense object in otherwise thin material, then yes — almost all the bending really does happen in that object, the PoCA point really does land in it, and the picture is sharp. That's the regime where PoCA shines and it's exactly the proof-of-principle scenario: a dense block in air. But the assumption is a lie whenever scattering is *distributed* — two dense objects along one muon's path, or a thick extended mass. Then the muon picks up real kicks in several places, the in/out lines cross at some geometric compromise point that needn't be inside *any* of the real scatterers, and I deposit the signal in the wrong voxel. The reconstruction smears and grows ghost density in the gaps. PoCA also throws away information: it uses only the angle, not the lateral displacement, and it can't tell *where along the path* a given amount of scattering happened beyond that one crossing point. And the fixed-`p_0` assumption means a slow muon and a fast muon with the same material give different `theta`, adding spread I'm not modelling. So PoCA is the cheap first cut — fast, non-iterative, great for isolated high-`Z` lumps, the thing that gets a real image on the wall to prove the concept — and it degrades exactly where the single-scatter story breaks.

To do better I have to stop pretending the scatter happened at one point and go back to the honest tomographic statement I derived and then ran from: the per-ray variance of a projected angle is `Var(theta_i) = sum_j L_ij lambda_j`, a raysum over *every* voxel the muon's (straight-line-approximated) path crosses, weighted by path length. Now I won't route the whole signal into one voxel — I'll let every voxel on the path carry its share, and let *many* muons crossing overlapping sets of voxels from many angles jointly pin down which voxels are actually dense. That's real tomography. But I still have the moment problem: I observe one projected `theta_i`, not its variance `v_i`. The two projected detector views can be treated as two independent samples with the same path-length row. So I need a likelihood. Model each projected scattering angle as zero-mean Gaussian with variance `v_i = sum_j L_ij lambda_j`:

    P(theta_i | lambda) = (1/sqrt(2 pi v_i)) exp( - theta_i^2 / (2 v_i) ),   v_i = sum_j L_ij lambda_j.

Rays are independent, so the data likelihood is the product. Take `-2 log` and drop constants: for each ray the contribution is `ln(v_i) + theta_i^2 / v_i`. So the reconstruction is

    lambda_hat = argmin_lambda  sum_i [ ln(v_i) + theta_i^2 / v_i ],   v_i = (L lambda)_i,   subject to lambda_j >= lambda_air.

Notice this is *not* least squares — the unknown enters as the *variance*, so the cost has the `ln(v)` term that penalizes blowing the variance up to fit, balanced against `theta^2/v` that rewards fitting the observed scatter. It's the negative log-likelihood of a Gaussian with unknown variance, which is the correct object given that the muon's scatter *is* a variance measurement. The non-negativity constraint matters: a negative scattering density is physically meaningless, so I floor every voxel at `lambda_air`, the (tiny) scattering density of air, which also regularizes the under-determined cells. This is the upgrade that resolves what PoCA can't: distributed scattering and overlapping objects, because the signal is shared along the path and cross-constrained by many rays rather than collapsed to one point.

```python
def reconstruct_mls(signals, Lmat, lambda_air, lam0=None):
    Lmat = np.asarray(Lmat, float)
    s2 = np.asarray(signals, float)**2
    def nll(lam):
        v = np.maximum(Lmat @ lam, 1e-12)
        return np.sum(np.log(v) + s2/v)          # -2 logL of projected Gaussians
    x0 = np.full(Lmat.shape[1], lambda_air) if lam0 is None else lam0
    return minimize(nll, x0, method="L-BFGS-B",
                    bounds=[(lambda_air, None)]*Lmat.shape[1]).x
```

I can squeeze out one more piece of information I've been discarding entirely: the lateral *displacement*. A muon doesn't just exit at a kinked angle — it also exits shifted sideways from where its incoming track would have put it. Angle alone is *degenerate along the ray*. A thin dense layer near the top of the path and the same layer near the bottom produce the *same* total scattering angle — `Var(theta) = sum_j L_j lambda_j` doesn't care where on the path the scattering sits. So angle-only reconstruction genuinely cannot tell top from bottom of a column; that's a real ambiguity, and it's where MLS leaves artifacts. Displacement breaks it, because *where* the kick happens changes how much lateral offset it accumulates by the exit. Let me get the joint statistics. For a single slab of thickness `L`, scattering and displacement are jointly Gaussian, zero mean, with

    Var(Delta_theta) = L lambda,   Var(Delta_x) = (L^3/3) lambda,   Cov(Delta_theta, Delta_x) = (L^2/2) lambda.

The displacement variance carries `L^3` and the cross term `L^2` — those higher powers of path length are exactly the *position-along-path* information the bare angle lacks. (As a sanity check the correlation is `(L^2/2)/sqrt(L · L^3/3) = (1/2)/(1/sqrt 3) = sqrt(3)/2 ≈ 0.866`, independent of material — angle and displacement are strongly but not perfectly correlated, so displacement really does add an independent coordinate.) Now propagate this through a stack of cells, because a real ray crosses many. Layer `j` contributes a kick `Delta_theta_j` and a local shift `Delta_x_j`, but a kick in an *upstream* layer also gets amplified into displacement by the lever arm of everything *downstream* of it. Track that. Let `T_j` be the total path length downstream of cell `j` along the ray. Then the total angle is `Delta_theta = sum_j Delta_theta_j`, and the total displacement is `Delta_x = sum_j ( Delta_x_j + T_j Delta_theta_j )` — each layer's local shift plus its kick swung out over the remaining distance `T_j`. Take variances; cross-layer terms vanish because layers scatter independently and everything is zero-mean, so only `j = k` survives. For displacement:

    Var(Delta_x) = sum_j E[ (Delta_x_j + T_j Delta_theta_j)^2 ]
                 = sum_j [ Var(Delta_x_j) + 2 T_j Cov(Delta_theta_j, Delta_x_j) + T_j^2 Var(Delta_theta_j) ]
                 = sum_j [ (L_j^3/3) + T_j L_j^2 + T_j^2 L_j ] lambda_j,

substituting the single-slab moments. And the cross term:

    Cov(Delta_theta, Delta_x) = sum_j E[ Delta_theta_j (Delta_x_j + T_j Delta_theta_j) ]
                              = sum_j [ Cov(Delta_theta_j, Delta_x_j) + T_j Var(Delta_theta_j) ]
                              = sum_j [ (L_j^2/2) + L_j T_j ] lambda_j,

and the angle variance is the old `Var(Delta_theta) = sum_j L_j lambda_j`. So each ray now has three weight vectors, all *linear in lambda*:

    W_theta(j)  = L_j,
    W_thetax(j) = L_j^2/2 + L_j T_j,
    W_x(j)      = L_j^3/3 + T_j L_j^2 + T_j^2 L_j,

and the ray's 2×2 covariance is `Sigma_i = [[W_theta·lambda, W_thetax·lambda], [W_thetax·lambda, W_x·lambda]]`. The data per ray is the vector `d_i = [Delta_theta_i; Delta_x_i]`, where `Delta_theta_i = theta_out - theta_in` in one projected view and the displacement is obtained by projecting the incoming track forward to the exit plane to get the un-scattered position `x_proj`, then `Delta_x_i = (x_out - x_proj) cos(theta_avg)` to measure it perpendicular to the mean path. The likelihood is now a 2D zero-mean Gaussian, `P(d_i|lambda) = 1/(2 pi |Sigma_i|^{1/2}) exp(-1/2 d_i^T Sigma_i^{-1} d_i)`, and the reconstruction is the same shape as before with the determinant and quadratic form replacing the scalar pair:

    lambda_hat = argmin_lambda  sum_i [ ln|Sigma_i| + d_i^T Sigma_i^{-1} d_i ],   subject to lambda_j >= lambda_air.

The structure is identical to MLS — minimize a sum of `ln(variance) + data·variance^{-1}·data`, floored at air — just with the scalar variance promoted to a 2×2 covariance that now encodes, through the `L^3` and `L^2` moments and the downstream lever arms `T_j`, *where along each ray* the scattering happened. That's what lifts the top-versus-bottom degeneracy that angle-only reconstruction can't.

So the chain is: muons are free, penetrating, single-particle, and they scatter far more in high-`Z` material, with the projected scatter's variance set by the Highland width `(15/p)sqrt(L/X0)`, so a per-unit-length scattering density `lambda = (15/p_0)^2/X0` is a strong, `Z`-sharp material fingerprint I want to map in 3D. Measure each muon's in- and out-track with tracking stations above and below. The cheapest reconstruction pretends all the scattering happened at one point, locates it as the closest-approach point of the two tracks, deposits one projected-variance sample there, and divides by the path count and voxel length — fast, non-iterative, and sharp for isolated high-`Z` lumps, but it smears wherever scattering is distributed because the single-scatter story fails. The honest fix is to model each projected scatter as a zero-mean Gaussian whose variance is the path-length raysum of voxel scattering densities and maximize the likelihood over all voxels — turning many overlapping rays into a real tomographic inversion — and then to fold in the lateral displacement, whose `L^3/3` and `L^2/2` moments and downstream lever arms encode where along each path the scattering sat, lifting the along-ray degeneracy that the angle alone leaves behind.
