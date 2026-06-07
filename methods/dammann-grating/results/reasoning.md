OK, let me think this through from scratch. I have one beam, and I want a symmetric set of copies of it — equally bright, sitting on a neat little grid around the axis — coming out of a single thin piece of glass. No moving parts, no stack of half-mirrors. The constraints are what make this hard: equal intensities, many spots, and almost no light thrown away, all from one passive surface I can actually fabricate.

Let me start from what a thin transparency does to a beam, because that's the only physics I get to use. A plane wave hits a transparency with complex amplitude transmittance t(x,y); in the far field I see the Fraunhofer pattern, which is just the Fourier transform of t. That's the whole game. So if I want spots — discrete bright points — I want a transform that is a set of deltas. And the way to make a Fourier transform into a comb of deltas is to make t *periodic*. A periodic transmittance of period d has a far field supported only at sinθ_m = mλ/d: the diffraction orders, evenly spaced in angle. Good — that already gives me a regular array of spots for free, and the spacing is set entirely by the period d.

But the *positions* of the spots are not my problem; their *brightnesses* are. So what controls how bright order m is? Expand one period of t in a Fourier series: t(x) = Σ_m c_m e^{i2πmx/d}. The amplitude that lands in order m is exactly the coefficient c_m, and the intensity of that spot is |c_m|². So if I want a one-dimensional fan-out with equal spots at −L, …, 0, …, L, I need |c_0| = |c_1| = … = |c_L|, and I want the remaining coefficients small. The problem is now algebra on the coefficients of a periodic function. Nice — that's a much smaller object than "an optical system."

Now, what kind of t do I dare use? My first instinct is the dumb one: an amplitude grating, modulate |t| between 0 and 1, a bar pattern, and just dial the bar widths until the Fourier coefficients come out equal. But wait — this throws light away. Wherever |t| < 1 I'm absorbing or blocking, so most of the incident power never makes it into any order at all, and on top of that the absorbed-mask spectra are dominated by the zero order. If efficiency matters even a little, an absorbing element is dead on arrival. I want every photon that comes in to leave in *some* order; I just want to choose *which* orders.

That instantly forces |t(x)| = 1 everywhere. A pure phase element: t(x) = e^{iφ(x)}. It absorbs nothing — by Parseval the total power Σ|c_m|² is conserved and just redistributed among the orders. This is the real lever. With phase only, I'm not fighting to keep light; I'm only steering it. So the question sharpens: choose a phase profile φ(x) whose Fourier coefficients have equal magnitude on the target orders.

What φ(x)? The natural continuous try is a sinusoidal phase, φ(x) = a·sin(2πx/d). I know this one — its order amplitudes are Bessel functions, c_m ∝ J_m(a). But that's a disaster for *equal* orders: the J_m for fixed a are all different and I only have one knob, a, so I cannot independently set |c_0| = |c_1| = …. The order weights ride a fixed special-function curve; I have no freedom to flatten them. I'd need a φ with many independent degrees of freedom in its *shape*.

And here's the other pressure, the one from the bench rather than the math: I have to *build* this thing. A continuously-varying φ(x) means a continuously-varying surface relief — a gray-scale etch, hard to make faithfully, and errors in depth scramble all the coefficients. The most robust thing I can fabricate is a profile with exactly *two* depths: a single etch step, regions deep and regions shallow. So let me restrict φ to take only two values. Which two? If φ ∈ {0, π}, then t = e^{iφ} ∈ {+1, −1} — two real values, maximal contrast on the unit circle, as far apart as two phases can be. Any other pair {0, Δφ} with Δφ ≠ π still keeps |t| = 1 but the two values sit closer together on the circle, which weakens how much power I can move out of the zero order. 0 and π it is. And the etch depth that produces a π phase step in a medium of index n is a half-wave of optical path, d_etch = λ/(2(n−1)) — one number, one step. (Concretely: GaAs in the infrared, n ≈ 3.26 at 12 µm, gives about 2.65 µm of relief; n ≈ 3.30 at 5 µm gives about 1.09 µm. The *pattern* doesn't care about λ — only this depth does.)

So now t(x) is a periodic function that takes the values +1 and −1, and the question is purely: where do I let it flip sign? Let me normalize the period to 1 and work on [−1/2, 1/2]. The places where t jumps between +1 and −1 are my only design variables — call them transition points. Between consecutive transitions t is constant at +1 or −1, alternating. So the whole infinite-dimensional "choose φ(x)" has collapsed to "choose a finite list of flip positions." That feels right; that's the kind of object I can optimize.

Before I start placing flips arbitrarily, let me buy myself some structure. I want the output array *symmetric* about the axis (equal spots at +m and −m), and I'd like the coefficients to be *real* so different orders don't fight each other with random phases. Both of those fall out if I make the period *even*: t(x) = t(−x). For an even real function the Fourier coefficients are real and symmetric, c_{−m} = c_m. So I'll force the profile to be even — pick the flip positions on the half-period [0, 1/2] and mirror them. Now I only need transition points 0 < x_1 < x_2 < … < x_K < 1/2 on the half-period, with t = +1 on [0, x_1), −1 on [x_1, x_2), +1 on [x_2, x_3), and so on — sign (−1)^j on the j-th interval — and the whole period built by reflection.

Now actually compute the coefficients, because this is where I find out whether the idea has teeth. Because t is real and even, its m-th coefficient is

  a_m = ∫_{−1/2}^{1/2} t(x) cos(2πmx) dx = 2 ∫_0^{1/2} t(x) cos(2πmx) dx.

On [0, 1/2], t is piecewise constant at (−1)^j on each interval [x_j, x_{j+1}] (with x_0 = 0, x_{K+1} = 1/2). So for m ≥ 1,

  a_m = 2 Σ_{j=0}^{K} (−1)^j ∫_{x_j}^{x_{j+1}} cos(2πmx) dx
      = 2 Σ_{j=0}^{K} (−1)^j [sin(2πm x_{j+1}) − sin(2πm x_j)] / (2πm)
      = (1/(πm)) Σ_{j=0}^{K} (−1)^j [sin(2πm x_{j+1}) − sin(2πm x_j)].

Look at the boundary terms first: the j=0 piece has sin(2πm·0) = 0, and the very last term has sin(2πm·1/2) = sin(πm) = 0. Both endpoints vanish — good, they were never going to be design knobs. Now collect what multiplies each *interior* transition x_j (1 ≤ j ≤ K). The point x_j appears as the upper limit of interval j−1 (sign (−1)^{j-1}, contributing +sin(2πm x_j) with that sign) and as the lower limit of interval j (sign (−1)^j, contributing −sin(2πm x_j) with that sign). Add them: (−1)^{j-1} sin − (−1)^j (−sin)… let me be careful. From interval j−1: term (−1)^{j-1}·(+sin(2πm x_j)). From interval j: term (−1)^j·(−sin(2πm x_j)) = (−1)^{j-1} sin(2πm x_j). They're equal — each interior transition contributes 2(−1)^{j-1} sin(2πm x_j). So

  a_m = (1/(πm)) Σ_{j=1}^{K} 2(−1)^{j-1} sin(2πm x_j) = (2/(πm)) Σ_{j=1}^{K} (−1)^{j-1} sin(2πm x_j).

There it is — the order amplitudes are a closed-form sum of sines of the transition points, with alternating signs. That's exactly the structure I hoped for: each diffraction order is a simple trigonometric function of where I put the flips, and the flips are shared across all the equations. The sign flips (−1)^{j-1} are the fingerprint of a binary ±1 profile.

The zeroth order is special because cos drops to 1: a_0 = 2 ∫_0^{1/2} t dx = 2 Σ_j (−1)^j (x_{j+1} − x_j). So a_0 is just the *signed length balance* of the period — how much of the half-period is +1 minus how much is −1. That's the knob for the central spot.

Let me sanity-check the signs before I trust it. Take two arbitrary transitions, say x = {0.1, 0.2}. The zeroth term is a_0 = 2[(0.1) − (0.1) + (0.3)] = 0.6. For m = 1 the sine sum is (2/π)(sin(0.2π) − sin(0.4π)), which is negative because the second transition contributes with the opposite sign and a larger sine. That is exactly what the interval picture says should happen: the middle negative band is wider in Fourier phase than the first positive band. No missing endpoint term, no extra factor of two.

Now the design. I have K interior transitions — K real numbers — so I can impose on the order of K independent conditions. If I target the nonnegative orders m = 0, …, L, the cleanest statement is

  |a_0| = |a_1| = … = |a_L|,

and among the configurations that satisfy it, take the one that puts the most power in those target orders. Counting degrees of freedom: with one transition I get one condition beyond a_0, which equalizes a_0 and a_1 and therefore gives a symmetric trio of spots. With two transitions I can equalize a_0, a_1, and a_2, so the symmetric fan-out has five spots. In this notation K transitions naturally aim at 2K+1 equal orders, m = −K, …, 0, …, K, although the nonlinear equations still have to cooperate.

Can I solve the equal-magnitude system in closed form? Look at the smallest case to see. One transition x_1: a_0 = 4x_1 − 1 (from 2[x_1 − (1/2 − x_1)]), and a_1 = (2/π) sin(2πx_1) (one term in the sum). Set the intensities equal, a_0² = a_1²:

  (4x_1 − 1)² = ( (2/π) sin(2πx_1) )².

This is transcendental — a polynomial in x_1 set equal to a sine — so there's no algebraic root; I solve it numerically. If F(x_1) = (4x_1 − 1)^2 − [(2/π) sin(2πx_1)]^2, then F(0.30) < 0 and F(0.39) > 0, so a root sits between them. A bisection gives x_1 ≈ 0.36763. Plug back: a_0 = 4(0.36763) − 1 ≈ 0.4705, a_1 = (2/π) sin(2π·0.36763) ≈ 0.4705 — equal, as demanded. The intensities are I_0 = I_1 ≈ 0.2214 each. Efficiency — the power that actually lands in my three target orders — is a_0² + 2a_1² ≈ 0.6642, about 66%. The remaining light leaks into the other orders; evenness makes +m and −m equal, but it does not make all even unwanted orders vanish. So a single flip already gives three equal beams at two-thirds efficiency. That two-thirds isn't a failure to optimize — it's the binary 0/π constraint biting: with only two phase levels there's a ceiling on how much I can corral into a chosen equal set, and that ceiling sits below what a many-level blaze could reach.

For more spots I add transitions and the equal-magnitude conditions stop having a tidy bracket — the sum has several sine terms and the system is genuinely multidimensional and nonconvex, with many local solutions. So I don't try to be clever; I optimize. Define a cost that is primarily the *non-uniformity* of the target orders, (I_max − I_min)/(I_max + I_min), which is zero exactly when they're all equal, and add a small penalty for low captured power so that two perfectly uniform layouts prefer the brighter one. Then run a derivative-free search (Nelder–Mead) from many random starting layouts and keep the best, because a single start lands in a bad local minimum. For two transitions aiming at five equal orders, this settles to a layout with the five target orders uniform to numerical zero and about 77% efficiency in them — three-quarters of the light, and better than the three-order case because two transitions give the spectrum more room to flatten and suppress leakage.

Let me make sure I haven't smuggled in anything the math doesn't support. The transitions must stay ordered and inside (0, 1/2); I clip and sort them inside the cost so the optimizer can't cross or escape them. The efficiency I report is exactly Σ over the target orders of |a_m|² (with the ±m counted twice), nothing normalized away. And the whole construction is wavelength-independent in pattern: x_j are pure numbers on the normalized period; only the physical period d (which sets spot spacing through sinθ_m = mλ/d) and the half-wave etch depth λ/(2(n−1)) carry the wavelength. Same mask, any λ, just re-scale the depth.

So the recipe that's emerged, end to end: far field = Fourier coefficients of one period; demand equal magnitudes on a symmetric target set, which would be lossy for an amplitude mask, so go to a pure phase element to conserve power; pick two phase levels 0 and π for one-etch fabrication and maximal contrast; make the period even so the coefficients are real and the fan-out is symmetric; that leaves only the sign-flip positions as variables; the order amplitudes are then the closed-form sum a_m = (2/(πm)) Σ_j (−1)^{j-1} sin(2πm x_j) with a_0 the signed length balance; equalize their magnitudes and maximize the captured power by solving for the transition points numerically. Let me write it.

```python
import numpy as np
from scipy.optimize import brentq, minimize

def binary_profile(x, transitions, start=+1.0):
    """Render one even +/-1 period from sign-flip positions on the half-period."""
    transitions = np.sort(np.asarray(transitions, float))
    edges = np.concatenate(([0.0], transitions, [0.5]))
    idx = np.searchsorted(edges, np.abs(x), side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return start * (-1.0) ** idx

def order_amplitudes(transitions, M):
    """a_0..a_M for an even +/-1 profile with transitions in (0, 1/2)."""
    t = np.sort(np.asarray(transitions, float))
    a = np.zeros(M + 1)
    edges = np.concatenate(([0.0], t, [0.5]))
    a[0] = 2.0 * np.sum(((-1.0) ** np.arange(len(edges) - 1)) * np.diff(edges))
    if M >= 1 and len(t):
        m = np.arange(1, M + 1)
        sign = (-1.0) ** np.arange(len(t))
        S = (sign[None, :] * np.sin(2 * np.pi * np.outer(m, t))).sum(axis=1)
        a[1:] = (2.0 / (np.pi * m)) * S
    return a

def design_cost(transitions, num_nonnegative, efficiency_weight=0.05):
    """Equalize m=0..num_nonnegative-1 and prefer higher captured power."""
    t = np.sort(np.clip(transitions, 1e-4, 0.5 - 1e-4))
    I = order_amplitudes(t, num_nonnegative - 1) ** 2
    target = I[:num_nonnegative]
    eta = I[0] + 2.0 * np.sum(I[1:num_nonnegative])
    uniformity = (target.max() - target.min()) / (target.max() + target.min() + 1e-12)
    return uniformity + efficiency_weight * (1.0 - eta)

def design(num_nonnegative, num_transitions=None, restarts=400, seed=0):
    """Multistart search for a 2*num_nonnegative-1 order fan-out."""
    K = num_nonnegative - 1 if num_transitions is None else num_transitions
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        t0 = np.sort(rng.uniform(0.01, 0.49, K))
        r = minimize(design_cost, t0, args=(num_nonnegative,), method='Nelder-Mead',
                     options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
        if best is None or r.fun < best.fun:
            best = r
    t = np.sort(np.clip(best.x, 1e-4, 0.5 - 1e-4))
    I = order_amplitudes(t, num_nonnegative - 1) ** 2
    eta = I[0] + 2.0 * np.sum(I[1:num_nonnegative])
    target = I[:num_nonnegative]
    uniformity = (target.max() - target.min()) / (target.max() + target.min() + 1e-12)
    return t, eta, uniformity, I

for q in (2, 3):                                          # q=2 -> 1x3, q=3 -> 1x5
    t, eta, uniformity, I = design(q, restarts=400)
    print(f"{2*q-1} equal orders: transitions={np.round(t, 4)}  "
          f"eff={eta*100:.2f}%  non-unif={uniformity*100:.3f}%")

f = lambda t1: (4 * t1 - 1) ** 2 - (2 / np.pi * np.sin(2 * np.pi * t1)) ** 2
t1 = brentq(f, 0.30, 0.39)
a0, a1 = 4 * t1 - 1, 2 / np.pi * np.sin(2 * np.pi * t1)
print(f"[1x3] t1={t1:.5f}  I0=I1={a0**2:.5f}  eff={(a0**2 + 2*a1**2)*100:.3f}%")
```

The causal chain, once more so I trust it: periodic phase profile → far field is its Fourier series, so order m carries |c_m|²; demanding equal |c_m| on a symmetric target set rules out a lossy amplitude mask and forces a pure phase element; a binary 0/π profile is the one I can etch in a single step and gives real, symmetric coefficients when the period is made even; that leaves only the sign-flip positions, and the coefficients come out as the closed-form sine sums above; equalizing their magnitudes while keeping the most power gives a small nonlinear system, solved in closed form for the three-order case (t₁≈0.368 for the positive-DC branch, ~66%) and by multistart search for more spots (five orders, ~77%) — a single thin transparent grating that splits one beam into an array of equally bright ones.
