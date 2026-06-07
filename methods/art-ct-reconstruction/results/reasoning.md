OK, let me think this through from scratch. I have an object I can't see inside — a protein under an electron beam, a slice of a head under X-rays — and all I get to measure are projections. A ray of radiation goes through the object, and the detector tells me how much got absorbed. If I take the logarithm of the transmitted-over-incident intensity, Beer's law turns the product of attenuations into a sum, and what I'm left holding is the *line integral* of the object's density along that ray: `p = ∫ f(x,y) dℓ` over the ray's path. I can do this for many rays, from many directions. So my data is a pile of line integrals of an unknown 2-D function, and I want the function back. That's inverting the Radon transform.

There's a clean analytic way to do this, and I should be honest about why I'm not just going to use it. The projection-slice theorem says the 1-D Fourier transform of a projection at angle θ is a radial slice, at angle θ, of the 2-D Fourier transform of `f`. So if I have projections at enough angles, I fill up 2-D Fourier space slice by slice and inverse-transform. In practice that becomes filtered backprojection: ramp-filter each projection with `|ω|` to undo the `1/r` oversampling of low frequencies, then smear each filtered projection back along its rays and sum. It's fast, it's `O(M N log N)`, it's exact in the continuous limit. But look at what it assumes: dense, uniformly-sampled, *straight* rays over a *full* angular range, and a clean sinogram. My situation is the opposite. In electron microscopy the stage only tilts ±60° or so, and the dose has to stay low — I have few views and they're noisy and the angular coverage is incomplete. The ramp filter, being a high-pass, will scream the noise back at me, and the missing angles will throw streaks across the image. And FBP is a fixed linear operator — there's nowhere to tell it that density can't be negative, or that the object is zero outside a known disk. I *know* those things and I'd like to use them. For diffracting sources — ultrasound, microwaves — it's worse: the rays bend, so the straight-line slice relationship that FBP rests on simply isn't true. So FBP is the right tool when data are plentiful and rays are straight, and I'm in neither regime. I need something else.

Let me go fully discrete and see what the problem actually is as a computation. Lay a square grid of `N` cells over the unknown image and say `f` is constant — call it `f_j` — in cell `j`. A ray isn't a zero-width line anymore; it's a fat strip, and its measured ray-sum `p_i` is just the sum over the cells it crosses of (cell value) × (how much of that cell the ray covers). Write that weight `w_ij` — most naturally the *length* of ray `i`'s intersection with cell `j` (or the area, for a finite-width strip). Then each ray gives me one equation

  `Σ_j w_ij f_j = p_i`,

and stacking all `M` rays, `A f = p` with `A = [w_ij]`. Fine — it's a linear system, solve it. Except look at the size. A 256×256 image is `N ≈ 65000` unknowns, and I'll have a comparable number of rays, so `A` is something like 65000×65000. I cannot invert that, I can't even store it densely. And it's worse than just big: with measurement noise, or whenever `M ≠ N`, there's no exact inverse — I'd be pushed into least squares, `(AᵀA)⁻¹Aᵀp`, which at this scale is just as hopeless. So direct linear algebra is dead on arrival. The matrix is enormous.

But it's enormous *and* almost entirely zero. Each ray only crosses `O(√N)` cells — a handful out of sixty-five thousand — so each row of `A` has a handful of nonzeros and the rest vanish. That sparsity has to be the lever. I want a method that never forms `A⁻¹`, never even forms `A` densely, that only ever touches one ray's handful of weights at a time and computes them on the fly. Iterative, row-at-a-time.

So let me stop thinking of `f` as an image and think of it as a single point in `N`-dimensional space, `(f_1, …, f_N)`. What is one equation `Σ_j w_ij f_j = p_i` in that space? It's `a_i · f = p_i`, where `a_i = (w_i1, …, w_iN)` is the i-th row — a linear constraint, so it pins `f` to a *hyperplane*. The hyperplane is perpendicular to `a_i` (move along `a_i` and the dot product changes; move within the plane and it doesn't), sitting at signed distance `p_i / ‖a_i‖` from the origin. Each of my `M` rays is one such hyperplane. If there were a unique solution it would be the single point where all the hyperplanes cross. So reconstruction is: find a point on the intersection of `M` hyperplanes, with `M` and `N` both in the tens of thousands. Stated that way, I don't need to *solve* anything globally — I just need to get onto all the hyperplanes at once.

Here's the move that suggests itself. I can't satisfy all the equations simultaneously without inverting the matrix. But I can satisfy *one* equation trivially: given any current guess, I can snap it onto a single hyperplane. And "snap onto" has an obvious best meaning — go to the *nearest* point of the hyperplane, the orthogonal projection. Then cycle: snap onto hyperplane 1, then onto hyperplane 2, then 3, …, and when I run out, start over at 1. Bounce between the constraints. This is exactly Kaczmarz's old method of projections (1937) for linear systems — and it's perfect for me because each snap only needs the one row `a_i`, nothing else, and never touches the matrix as a whole.

Let me actually derive the projection, because the precise update — and especially its normalization — is the whole thing. I have a point `x` and I want the closest `z` on the hyperplane `a_i · z = p_i`. Minimize `‖x − z‖²` subject to `a_i · z = p_i`. Lagrange: `∇(‖x−z‖²) = μ ∇(a_i·z)`, i.e. `−2(x − z) ... ` — let me just parameterize. The constraint moves me only along the normal direction `a_i`, so write `z = x + t a_i` and find `t` so `z` lands on the plane. The closest point is reached moving perpendicular to the plane, and the plane's normal *is* `a_i`, so that ansatz is exactly right. Plug in: `a_i · (x + t a_i) = p_i`, so `a_i·x + t ‖a_i‖² = p_i`, giving

  `t = (p_i − a_i·x) / ‖a_i‖².`

Therefore the projection is

  `z = x + (p_i − a_i·x)/‖a_i‖² · a_i.`

So the update I'll iterate, ray by ray, is

  `f ← f + (p_i − a_i·f)/‖a_i‖² · a_i.`

Let me read it. `a_i·f` is the *computed* ray-sum for ray `i` from my current image — what my current guess predicts ray `i` should measure. `p_i` is what it *actually* measured. Their difference, the residual `p_i − a_i·f`, is how wrong I am along this one ray. I spread that residual back onto exactly the cells the ray touches (multiply by `a_i`), and the `‖a_i‖²` in the denominator is not decoration — it's precisely what makes this the *orthogonal* projection, the closest point, the geometrically optimal single-equation step. Drop it and I'd overshoot or undershoot the plane. Keep it and after this step `a_i·f = p_i` exactly: this ray is now perfectly satisfied. The normalization is the difference between "move toward the plane" and "land on the plane."

Does this converge? For a consistent, well-posed system, yes — Kaczmarz showed cyclic projections drive the iterate to the common intersection. But I have to think about the two ways my real problem is *not* that clean, because they change what I should do.

First, underdetermined. In the limited-tilt-range, few-view regime I have `M < N` — fewer rays than pixels — so the hyperplanes don't pin down a single point; their intersection is a whole flat of solutions, infinitely many images all consistent with my data. Which one does the cyclic projection give me? Tanabe's result (1971) settles it: starting from `x_0`, the iteration converges to the solution *closest to `x_0`*. So if I start from the zero image, I converge to the *minimum-norm* solution — the smallest image consistent with the data. That's a sensible default; among all images that fit, take the least-energy one and let constraints sharpen it. Good — so `x_0 = 0` isn't arbitrary, it's choosing my prior.

Second, and this is the one that bites: even with noiseless data the discrete system is mildly *inconsistent*. My weights `w_ij` are an approximation — a pixel is assumed constant, the strip width is fudged — so the computed ray-sums never perfectly equal the true line integrals, and the `M` hyperplanes don't all pass through one point. Add real measurement noise and it's worse. Now the cyclic projection can't converge to a point at all; it settles into a limit cycle, oscillating around the cloud of near-intersections. And here's the visible damage: when I *exactly* satisfy ray `i`'s equation — full projection, land right on the plane — I lay down a hard stripe of correction along that ray. Then ray `i+1`, crossing many of the same cells at a different angle, snaps onto *its* plane and overwrites part of what ray `i` just set. The two corrections fight. Cycle through thousands of rays from all directions and these competing per-ray stripes pile up into a characteristic salt-and-pepper noise smeared across the whole image. The RMSE can look like it's dropping while the picture looks awful — fast convergence in norm, ugly convergence in the eye. The trouble is precisely that I'm satisfying each contradictory equation *too* well.

So I shouldn't take the full step. If I only move a fraction of the way to the plane — introduce a relaxation parameter `λ < 1`,

  `f ← f + λ (p_i − a_i·f)/‖a_i‖² · a_i,`

then no single ray gets to fully impose itself; its stripe is softened, and across many rays the contradictions *average* instead of fighting. With `λ = 1` I'm back to pure projection, fast and noisy. With `λ` small the salt-and-pepper calms down and the image becomes intelligible, at the cost of needing more sweeps to settle — I'm trading some of Kaczmarz's rapid convergence for a picture I can actually read. Often `λ` is even shrunk as iterations go on, large early to move fast, small late to settle quietly. So underrelaxation is the price of admission for real, noisy data, not pseudo-data.

There's a second knob hiding in the *order* I visit the rays, and it's not cosmetic — it controls the convergence rate. Go back to the geometry. If two hyperplanes are perpendicular, projecting onto one then the other lands me exactly on their intersection in a single pair of steps. If they're nearly parallel — small angle between them — the iterate zig-zags down a long shallow wedge, taking forever to reach the corner. Now, adjacent rays within a view, or rays from two views a degree apart in angle, are *nearly parallel* — their hyperplanes have a tiny angle. So cycling rays in their natural sequential order is the worst case: I keep projecting between near-parallel planes and crawl. The fix is to deliberately interleave: take rays from views that are far apart in angle, so successive hyperplanes are as close to orthogonal as I can manage. Hounsfield used exactly this — successive projections separated by a large angle (something like 73.8°) — first as a heuristic about decorrelating the information in neighboring views, later justified as genuinely speeding the convergence of the cyclic projection. So the ray order is a free lever on convergence and I should use it.

And because the whole thing lives in image space and I update by adding a correction image, I can fold in priors for free. After each projection I can clip negative pixels to zero — density is nonnegative, that's physics — and zero out anything outside the known support disk. Each of those is itself a projection (onto the nonnegative orthant, onto the support set), so interleaving them with the hyperplane projections is consistent with the whole framework and only helps, both regularizing and speeding things up. That's something FBP could never do.

Let me write down what I have. Build the sparse `A` by tracing each parallel-beam ray across the grid and recording intersection lengths. Then, starting from `f = 0`, sweep over the rays in a wide-angle-interleaved order, and for each ray do the relaxed Kaczmarz update, clipping after each step. That's ART — the algebraic reconstruction technique, Kaczmarz's projection method wearing a tomography hat. (In its very first incarnation the weights were even taken binary — `w_ij = 1` if the center of cell `j` falls in ray `i`'s strip, else 0 — so `‖a_i‖²` is just the *count* `N_i` of cells the ray hits, and the correction `(p_i − q_i)/N_i` is spread equally over them. Cheap to decide at run time, but a cruder forward model means a more inconsistent system and more salt-and-pepper; using true intersection lengths, with the denominator the ray length `L_i`, is strictly a better model.)

Now I'm bothered by the salt-and-pepper at a deeper level, because relaxation only *masks* it — I want to know *why* it's there and remove the cause. Stare at the mechanism again. The noise is born from the *sequential* updates: ray `i` sets some cells, ray `i+1` immediately disturbs them, and for a cell that several rays in the *same view* all cross, the "correction" it receives depends on the accident of which ray touched it last. The per-pixel correction within a view is *ambiguous* — ill-defined — and that ambiguity is exactly the salt-and-pepper. So the cure shouldn't be to make each ray's step smaller; it should be to make each pixel's correction *well-defined*.

What if, within one projection (one view), I don't apply the corrections ray by ray at all, but compute them all against the *same* current image, hold them aside, and then apply them together? Then no ray inside the view disturbs another's input; every cell in the view gets *one* correction, defined uniquely as the combination of all the rays that cross it. Now there's no last-ray-wins ambiguity. This is a simultaneous update — but, crucially, simultaneous *within a view*, not simultaneous across the entire data set. (Plain SIRT does the latter: it averages corrections over *all* equations at once before applying anything, which kills the per-ray fighting but throws away the directional, view-by-view structure and crawls to convergence over many full passes.) Per-view simultaneity keeps the fast, directional character of ART — I still march view by view, getting a smooth update after each scan direction — while removing the within-view ambiguity that makes the noise.

Let me build the per-view update carefully, because the normalization is the substance. Inside view `V`, with the *frozen* current image `g`, each ray `i ∈ V` has residual `p_i − q_i` where `q_i = Σ_k a_ik g_k` is the computed ray-sum. I want to back-distribute these onto the pixels. Two normalizations, and they play different roles. First, per ray: the natural average correction along ray `i` is the residual divided by the ray's total weight `Σ_k a_ik`, which is the ray's length `L_i` through the region — that makes the correction an intensity, independent of how long the ray happens to be. Then I splat it onto pixel `j` with weight `a_ij`. So pixel `j` accumulates `Σ_{i∈V} a_ij (p_i − q_i)/Σ_k a_ik` from all the rays in the view that cross it. But now a pixel crossed by many rays in this view would get a bigger pile than one crossed by few — purely an artifact of ray density, not of the data. So I normalize a *second* time, by the total weight pixel `j` received in this view, `Σ_{i∈V} a_ij` — the column sum over the view. That makes each pixel's correction independent of how many rays happened to graze it: uniform treatment. So the update is

  `g_j ← g_j + λ · [ Σ_{i∈V} a_ij (p_i − q_i)/(Σ_k a_ik) ] / ( Σ_{i∈V} a_ij ).`

Row sums normalize the residual into a per-ray intensity; column sums normalize the back-projection into a per-pixel correction. That double weighting — by ray length and by pixel coverage — is what makes the per-view update smooth and uniform, and it's the heart of the simultaneous scheme. I cycle this over views, refreshing `g` after each view's full correction.

A couple of refinements drop out of caring about the forward model. The salt-and-pepper ultimately traces to inconsistency, and inconsistency traces to a crude `A`. The pixel basis — `f` constant in each cell — makes `f` discontinuous and approximates the ray integral badly. The simplest improvement is bilinear elements: pyramid-shaped basis functions over 2×2-pixel supports, so the represented `f` is continuous, the ray integrals are better approximated, the system is *less* inconsistent, and I can get away with *fewer* rays per view (no need to overdetermine ~4× just to average out a bad model). With bilinear elements the `g_j` are sample values on the lattice, and I approximate each ray integral by sampling equidistant points along the ray (step `Δs` about half the lattice spacing) and bilinearly interpolating — the net weight `a_ij = Σ_m d_ijm Δs` is the accumulated contribution of sample point `m` to pixel `j`. I have to be careful at the two ends of a ray where it enters and exits the circular region: adjust the first and last weights so that `Σ_j a_ij` still equals the true physical ray length `L_i`, or the normalization above is off.

And one more, purely about which corrections to trust. A ray's contribution near where it just clips the boundary of the circle is based on a sliver of pixel and is less reliable than its contribution through the middle. So I can weight the back-distribution *along* the ray — emphasize mid-ray corrections, de-emphasize the entry/exit ends — with a longitudinal Hamming window: `w_ij = Σ_m h_im d_ijm Δs` with `h_im` the window value at sample `m`. Heuristic, but it suppresses the boundary noise that the partial-pixel ends would otherwise inject. With bilinear elements, per-view simultaneous updates, and the longitudinal window, the per-view scheme produces a clean image in essentially a single pass — no relaxation needed to make it intelligible — which is the whole point: get ART's speed and SIRT's smoothness at once. This refined per-view algorithm is SART.

Let me now make this real in code, and keep it honest to the math: build the sparse parallel-beam `A` of intersection lengths; run the relaxed cyclic Kaczmarz sweeps with a wide-angle ray order and nonnegativity clipping (that's ART); and run the per-view simultaneous update with the row-sum/column-sum weighting (that's SART); and check both on a small phantom.

```python
import numpy as np
from scipy.sparse import csr_matrix

def build_system_matrix(n, angles, n_rays=None, det_spacing=1.0):
    # parallel-beam forward model: A[i,j] = length of ray i inside pixel j.
    # trace each ray by oversampling points along it and splatting the step
    # length ds onto whichever pixel each sample lands in -> intersection length.
    if n_rays is None:
        n_rays = n
    rows, cols, vals = [], [], []
    s = (np.arange(n_rays) - (n_rays - 1) / 2.0) * det_spacing  # ray offsets
    half = n / 2.0
    n_samp = 4 * n
    t = (np.arange(n_samp) - (n_samp - 1) / 2.0) * (np.sqrt(2.0) * n / n_samp)
    ds = np.sqrt(2.0) * n / n_samp
    i = 0
    for theta in angles:
        c, sn = np.cos(theta), np.sin(theta)
        for off in s:                                  # one ray = one hyperplane
            x = off * (-sn) + t * c
            y = off * (c) + t * sn
            px = np.floor(x + half).astype(int)
            py = np.floor(y + half).astype(int)
            inside = (px >= 0) & (px < n) & (py >= 0) & (py < n)
            flat = py[inside] * n + px[inside]
            if flat.size:
                uniq, cnt = np.unique(flat, return_counts=True)
                rows += [i] * uniq.size
                cols += uniq.tolist()
                vals += (cnt * ds).tolist()            # accumulate length per pixel
            i += 1
    A = csr_matrix((vals, (rows, cols)), shape=(len(angles) * n_rays, n * n))
    ray_view = np.repeat(np.arange(len(angles)), n_rays)
    return A, ray_view

def art(A, b, n_sweeps=8, lam=0.2, x0=None, nonneg=True, order=None):
    # cyclic Kaczmarz: for each ray, project the estimate onto its hyperplane.
    m, ncol = A.shape
    x = np.zeros(ncol) if x0 is None else x0.astype(float).copy()
    A = A.tocsr()
    row_norm2 = np.asarray(A.multiply(A).sum(axis=1)).ravel()   # ||a_i||^2
    idx = np.arange(m) if order is None else np.asarray(order)  # wide-angle order
    for _ in range(n_sweeps):
        for i in idx:
            if row_norm2[i] == 0.0:
                continue
            ai = A.getrow(i)
            resid = b[i] - ai.dot(x)[0]                # p_i - a_i . x
            # x <- x + lam (p_i - a_i.x)/||a_i||^2 a_i  : the relaxed projection
            x = x + lam * (resid / row_norm2[i]) * ai.toarray().ravel()
            if nonneg:                                 # clip = projection onto x>=0
                np.clip(x, 0.0, None, out=x)
    return x

def sart(A, b, ray_view, n_iters=8, lam=1.0, x0=None, nonneg=True):
    # per-view simultaneous update with row-sum (ray length) and column-sum weighting.
    m, ncol = A.shape
    x = np.zeros(ncol) if x0 is None else x0.astype(float).copy()
    A = A.tocsr()
    row_sum = np.asarray(A.sum(axis=1)).ravel()                 # L_i = sum_k a_ik
    row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
    for _ in range(n_iters):
        for v in np.unique(ray_view):
            sel = np.where(ray_view == v)[0]
            Av = A[sel]
            resid = b[sel] - Av.dot(x)                          # p_i - a_i . x
            weighted = resid / row_sum[sel]                     # / ray length L_i
            num = Av.T.dot(weighted)                            # back-project: sum_i a_ij(.)
            col = np.asarray(Av.sum(axis=0)).ravel()            # sum_i a_ij over view
            col = np.where(col == 0.0, 1.0, col)
            x = x + lam * num / col                             # / per-pixel coverage
            if nonneg:
                np.clip(x, 0.0, None, out=x)
    return x

if __name__ == "__main__":
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]; cx = cy = (n - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.where(r < n * 0.42, 0.3, 0.0)
    img += np.where((xx - n*0.40)**2 + (yy - n*0.55)**2 < (n*0.12)**2, 0.6, 0.0)
    img += np.where((xx - n*0.62)**2 + (yy - n*0.42)**2 < (n*0.08)**2, 0.5, 0.0)
    angles = np.linspace(0, np.pi, 30, endpoint=False)
    A, ray_view = build_system_matrix(n, angles, n_rays=n)
    b = A.dot(img.ravel())
    nv = len(angles)                                            # wide-angle ray order
    perm = np.argsort([(k * (nv // 2 + 1)) % nv for k in range(nv)])
    order = np.concatenate([np.where(ray_view == v)[0] for v in perm])
    x_art = art(A, b, n_sweeps=8, lam=0.2, order=order).reshape(n, n)
    x_sart = sart(A, b, ray_view, n_iters=8, lam=1.0).reshape(n, n)
    rmse = lambda u: np.sqrt(np.mean((u - img) ** 2))
    print("ART RMSE:", round(rmse(x_art), 4), " SART RMSE:", round(rmse(x_sart), 4))
```

The chain that got me here: a detector only gives line integrals, so reconstruction is inverting the Radon transform; the analytic inverse (filtered backprojection) needs dense, straight, full-range, low-noise data, which the few-view, noisy, limited-tilt regime doesn't have, and it can't take priors; so discretize to `A f = p`, which is too huge and too sparse for direct inversion; recognize each equation as a hyperplane and reach for Kaczmarz's cyclic orthogonal projection, where the `‖a_i‖²` normalization is exactly what makes each step land on its plane; temper the steps with relaxation `λ<1` because the inconsistent, noisy system makes full projections fight into salt-and-pepper; order the rays wide-apart in angle so near-parallel hyperplanes don't stall convergence; clip to nonnegativity and support for free; then trace the salt-and-pepper to within-view ambiguity and kill it by going per-view simultaneous, with residuals normalized by ray length (row sums) and back-projection normalized by pixel coverage (column sums), sharpened by bilinear elements and a longitudinal window — landing on a method that has ART's speed and SIRT's smoothness in essentially one pass.
