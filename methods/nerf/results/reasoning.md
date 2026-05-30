OK, let me think this through from scratch. I have a handful of photographs of some static scene — a lego bulldozer, a ship, a fern on a table — each with a known camera pose and intrinsics, and I want to render the scene from viewpoints I never photographed. Photorealistically. From maybe a hundred images, sometimes a few dozen.

What do I actually have to produce? A function that, given a virtual camera, gives me a pixel color for every ray through that camera. So really I need some representation of the scene that I can *render* — turn into a 2D image — and I need to be able to fit that representation to my photos. The only supervision I have is "this ray, through this pixel of this real photo, should come out this color." Nothing else. No 3D scans, no depth, no meshes. So whatever rendering procedure maps my representation to a pixel has to be differentiable, end to end, so I can push the pixel error back into the representation by gradient descent.

Let me look at what people reach for and where each one breaks, because the failure modes are going to tell me what the representation has to be.

The mesh route: represent the scene as a triangle mesh with some per-vertex or per-face appearance, render it with a differentiable rasterizer or pathtracer, compare to the photos, backprop to the vertices. The trouble is twofold. You have to *initialize* with a template mesh of the right topology — and for an arbitrary real scene I have no idea what topology it is. And even given that, optimizing geometry by image reprojection is a mess: the loss landscape is full of local minima, badly conditioned, a vertex moving a hair can pop a triangle across an edge and the gradient is garbage. So meshes give me a compact representation but a hostile optimization. Cross it off.

The discrete-volume route is much friendlier to optimize. Put the scene on a voxel grid — an RGB and an opacity at every cell — and render by compositing along rays. This is differentiable, it handles fuzzy and complex geometry, no template needed. The methods that do this well are genuinely impressive. But the cost is brutal: a grid is O(n^3) in memory and compute, and to render fine detail at high resolution you need a fine grid. One forward-facing method I know stores over fifteen gigabytes for a *single* scene. That's not a representation, that's a hard drive. The discreteness is the enemy: resolution is baked into the data structure.

So I want the friendliness of volumes without paying for a grid. What's a *continuous* container that's also compact? There's a line of work that stores a function directly in the weights of a fully-connected network — map a 3D coordinate to a signed distance, or to an occupancy probability. The network is tiny (a few megabytes) and it's continuous: you can query any point, no grid, no resolution ceiling. That's exactly the compactness and continuity I want.

But those methods have a wall of their own. To train them from 2D images you have to render them, and the way they render is to find, along each ray, the *one* surface point — the zero-crossing of the SDF, or where occupancy flips — and color that point. One opaque surface intersection per ray, one color. That throws away everything volumetric. You can't represent a soft edge, a wisp, a thin structure, partial transparency, or the way a surface's color bleeds. And empirically these come out oversmoothed, simple shapes only. The single-surface assumption is the bottleneck.

Stare at that for a second. The discrete-volume people are happy precisely because they *don't* collapse to a single surface — they composite a whole distribution of opacity along the ray. The neural-implicit people are happy because they're continuous and compact. I want both. So: keep the continuous MLP container, but instead of asking it for a surface, ask it for a *volume* — a density and a color at every point in space — and render it the way the volume people do, by compositing along the ray. Don't intersect; integrate.

So the representation is a function of a 3D point x = (x,y,z) returning a local color c and a local density σ. An MLP, F: x → (c, σ). Continuous, compact, and — crucially — if I render it by volumetric compositing the whole pipeline stays differentiable, so I can fit it from images alone. That's the seed. Now I have to actually turn "density and color along a ray" into "a pixel," and do it differentiably.

How does light actually accumulate along a ray through a cloud of emitting, absorbing particles? Let me build it physically rather than guess. Think of space filled with tiny particles. The density σ(x) is the differential probability that the ray hits — terminates at — a particle in an infinitesimal length around x. So in a slab [t, t+dt] along the ray r(t) = o + t·d, the probability that a particle blocks the ray there is σ(r(t)) dt.

Now define the transmittance T(t): the probability that the ray makes it from the start t_n all the way to t *without* having hit anything. Consider going one more step dt. The chance of surviving that extra step is (1 − σ(r(t)) dt). So T(t+dt) = T(t)·(1 − σ dt), which gives dT/dt = −σ(r(t))·T(t). That's a simple linear ODE; integrating, T(t) = exp(−∫_{t_n}^{t} σ(r(s)) ds). Good — transmittance is the exponential of the negative accumulated density. Makes sense: the more stuff between t_n and t, the less light gets through, and it falls off exponentially.

Now the color reaching the eye. A particle at t emits color c(r(t), d). For that emitted light to reach the eye it has to (a) have a particle there, probability σ(r(t)) dt, and (b) survive the trip back from t to the eye through everything in front of it, which is exactly T(t). So the contribution of the slab at t is T(t)·σ(r(t))·c(r(t),d) dt, and the total expected color of the ray is

  C(r) = ∫_{t_n}^{t_f} T(t) σ(r(t)) c(r(t), d) dt,   T(t) = exp(−∫_{t_n}^{t} σ(r(s)) ds).

That's the rendering equation for this representation. It's an integral of (transmittance)×(density)×(color) over the ray between near and far bounds t_n, t_f. And notice it's differentiable in σ and c everywhere — exactly what I need to fit the MLP. Rendering a full image is just evaluating this integral for one ray per pixel.

I can't evaluate a continuous integral with a network in the loop, so I need a quadrature. Let me derive the discrete form carefully, because the exact constants matter and I want to *see* it reduce to something I already know how to compute. Partition [t_n, t_f] into N intervals with sample points t_1 < t_2 < … and widths δ_i = t_{i+1} − t_i. Assume σ and c are constant on each interval, σ_i and c_i. Then split the integral into a sum over intervals: C = Σ_i ∫_{interval i} T(t) σ_i c_i dt.

Take one interval and parameterize it by s ∈ [0, δ_i] from its start. The transmittance at offset s into interval i is T at the *start* of the interval times the decay within the interval. The decay within a constant-σ_i interval over distance s is exp(−σ_i s). Call the transmittance up to the start of interval i just T_i. So inside the interval, T(s) = T_i · exp(−σ_i s). Then

  contribution of interval i = ∫_0^{δ_i} T_i exp(−σ_i s) σ_i c_i ds = T_i c_i σ_i ∫_0^{δ_i} exp(−σ_i s) ds.

That last integral is ∫_0^{δ_i} exp(−σ_i s) ds = (1/σ_i)(1 − exp(−σ_i δ_i)). The σ_i cancels, beautifully, leaving

  contribution of interval i = T_i (1 − exp(−σ_i δ_i)) c_i.

And what's T_i, the transmittance up to the start of interval i? It's exp of the negative accumulated density of all the intervals before i: T_i = exp(−Σ_{j<i} σ_j δ_j). So the whole thing is

  Ĉ(r) = Σ_{i=1}^N T_i (1 − exp(−σ_i δ_i)) c_i,   T_i = exp(−Σ_{j=1}^{i−1} σ_j δ_j).

Now look at what this is. Define α_i = 1 − exp(−σ_i δ_i). Then exp(−σ_i δ_i) = 1 − α_i, so T_i = exp(−Σ_{j<i} σ_j δ_j) = Π_{j<i} exp(−σ_j δ_j) = Π_{j<i} (1 − α_j). And the render is Ĉ = Σ_i (Π_{j<i}(1−α_j)) α_i c_i. That's *exactly* alpha compositing — front-to-back blending with per-sample opacity α_i, where the accumulated transmittance is the running product of (1 − α) of everything in front. The thing the discrete-volume people already do every day. The continuous physics, discretized, lands precisely on the operation graphics already has a name and an implementation for. And α_i = 1 − exp(−σ_i δ_i) is the bridge: it converts a density (a coefficient with units of inverse length) into an opacity by accounting for how far the ray travels through it. A thick slab (large δ) or a dense one (large σ) is more opaque; a thin or sparse one barely occludes. It's monotone, it's in [0,1), and it's differentiable. Everything I need.

The per-sample weight in this sum, w_i = T_i α_i = T_i(1 − exp(−σ_i δ_i)), is how much the i-th sample's color counts toward the pixel. Its sum over the ray, Σ_i w_i, is the total opacity the ray accumulates — useful later, and it's also the natural place to handle a known background: if the ray accumulates total opacity less than one, the leftover (1 − Σ w_i) is "see-through," so for a white-background dataset I just add (1 − Σ w_i) of white to the composited color.

Now, *where* do I put the sample points t_i? The obvious thing is a fixed evenly-spaced set. But think about what that does over training: the MLP only ever gets queried at the same discrete locations along every ray, every iteration. That quietly turns my continuous representation back into a discrete one — I've reinvented the voxel grid through the back door, just with the grid living at fixed depths. The whole point was continuity. So instead: keep N evenly-spaced *bins*, but within each bin draw the sample uniformly at random,

  t_i ~ U[ t_n + (i−1)/N (t_f − t_n),  t_n + i/N (t_f − t_n) ].

Stratified sampling. Each render uses a discrete set of samples, so the quadrature is honest, but across iterations the MLP gets evaluated at *continuous* positions covering the whole interval — so it's forced to be a continuous function of depth, not a lookup table at fixed depths. And δ_i is just the gap to the next sample. Good. (One implementation nuance: the per-sample distance δ_i should be measured in true world units, so if the ray direction isn't a unit vector I scale δ_i by ‖d‖. And the final far interval has no "next sample," so I give it a large δ to absorb whatever remains.)

So I have a continuous MLP field, a physically-grounded differentiable volume render, and a sampling scheme that keeps it continuous. Let me just wire it up: per ray, draw stratified samples, query F at each for (c_i, σ_i), composite with α_i = 1−exp(−σ_i δ_i), get a pixel, take squared error against the true pixel, backprop. Optimize one network per scene. The loss across many views should force the network to put high density and the right color exactly where the real geometry is, because that's the only configuration that explains all the photos at once.

Let me think about that "explains all the photos at once" a bit harder, because it's where a subtle design choice hides. Suppose I let the MLP output a density that depends on viewing direction as well as position. Then nothing stops it from cheating: it could put opaque stuff at different depths depending on which camera is looking, fitting each photo independently and learning no coherent geometry at all. Geometry is a property of the *world*, not of who's looking. So density must be a function of position only: σ = σ(x). That constraint is what forces multiview consistency — the same solid has to be in the same place for every camera, and the only way to satisfy all the views is to get the geometry right.

But color is a different story. Real surfaces are not Lambertian. A specular highlight on the ship's hull, the glint off water, a varnished surface — the color you see genuinely depends on the viewing direction, even at a fixed physical point. If I force color to depend on position only, the network can't reproduce specularities; it'll average them into a dull diffuse blob or, worse, smear fake geometry to fake the highlight. So color should be a function of both position and direction: c = c(x, d). I'll express direction as a unit 3-vector rather than two angles, easier to feed in.

So the architecture has to *route* these differently: σ comes out before direction is ever introduced, c comes out after. Concretely: run x through a stack of fully-connected ReLU layers — call it eight layers of width 256 — producing σ (rectified through a ReLU so density is nonnegative) and also a feature vector. Only *then* concatenate the viewing direction onto that feature vector and pass it through one more small layer (width 128) to produce the RGB color (squashed through a sigmoid into [0,1]). This way σ structurally cannot see d, and c can. One detail I'll borrow from the SDF-network architecture: with eight layers deep, the input coordinate gets "forgotten" by the middle, so I'll add a skip connection that concatenates the input back in at the fifth layer.

Let me build a first version and look at it. … And it's blurry. Disappointingly blurry. The silhouettes are roughly right, the coarse shape is there, but all the fine texture and sharp geometry is washed out — it looks like someone ran a heavy blur over the truth. The network has clearly learned the *low-frequency* content and refuses to learn the high-frequency content. Why?

This rings a bell. There's an analysis of plain ReLU networks showing through Fourier analysis that they're biased toward low-frequency functions — the low-frequency components of a target get fit first and fast, and high-frequency detail is intrinsically hard for the network to express, needing finely tuned weights. My input is a raw 3D coordinate (and a 3D direction), and I'm asking the MLP to be a function that varies *very* rapidly in space — the color and density can flip over a tiny spatial distance at a sharp edge or a fine texture. That's a high-frequency function of x. And a plain MLP, fed the raw coordinate, just won't go there; it produces the smoothest function consistent with the data. The smoothness I'm seeing isn't a capacity problem — the network is a universal approximator, it *could* in principle fit it — it's a bias problem in what optimization actually reaches.

So the fix isn't a bigger network, it's changing what I feed it. The same analysis points the way: if you map the inputs through high-frequency functions *before* the network, it can fit high-frequency targets far more easily. Why does that work? Intuitively: a plain MLP is a smooth, roughly distance-respecting function of its input — nearby inputs map to nearby outputs. If "nearby in 3D" forces "nearby in color," I can never get a sharp edge. But if I first lift each coordinate into a space where two physically-close points can be *far apart*, then a smooth function of *that* lifted coordinate can still produce a sharp change in 3D. I want a lifting that pulls nearby points apart at many scales.

Sinusoids of geometrically increasing frequency do exactly that. Map a scalar coordinate p to

  γ(p) = ( sin(2^0 π p), cos(2^0 π p), sin(2^1 π p), cos(2^1 π p), …, sin(2^{L−1} π p), cos(2^{L−1} π p) ).

L frequency bands, doubling each time, each as a (sin, cos) pair. The high-frequency bands, sin(2^{L−1} π p), change sign over a spatial distance of order 2^{−L} — so two points that distance apart, indistinguishable to a low-frequency representation, land far apart in this encoding, and the downstream MLP can assign them very different colors with a perfectly smooth function of the encoded vector. It's a Fourier feature map: I'm handing the network a high-frequency basis so it doesn't have to manufacture high frequencies out of a low-frequency-biased architecture. I apply γ separately to each of the three components of x (normalized to [−1,1]) and to each of the three components of the direction d (already in [−1,1] by construction). So the MLP is now F = F' ∘ γ — an un-learned encoding followed by the regular network.

How many bands L? More bands = higher representable frequency, but also more input dimensions and, past a point, nothing to gain. The useful ceiling is set by the highest frequency actually present in my images: once 2^L exceeds the finest detail in the data (the images are around a thousand pixels across, so detail up to roughly 2^10), extra bands encode frequencies the data can't even pin down, so they don't help and may hurt. So something like L = 10 for position. For the viewing direction I expect appearance to vary much more *smoothly* with angle than texture varies with position — specular lobes are broad — so I need far fewer bands there; L = 4 for direction. (I'd want to check: too few bands, say 5 for position, and detail comes back muted; going from 10 to 15 buys nothing — consistent with the ceiling argument.)

With the encoding in, the renders get their sharpness back — fine texture and crisp geometry appear. Good. So the two real ingredients beyond "MLP + volume render" are: density-from-position / color-from-(position,direction), and the positional encoding to defeat the low-frequency bias.

Now efficiency. My render draws N samples per ray and queries the MLP at every one. But most of those samples are wasted: along a typical ray, the vast majority of the interval [t_n, t_f] is empty air or sits *behind* the first opaque surface, contributing essentially nothing to the integral — their weight T_i α_i is ≈ 0. I'm spending almost all my network queries on regions that don't matter, and to resolve a thin surface well I'd need a very fine uniform sampling everywhere, which is exactly the O(resolution) cost I was trying to escape. I want to put samples where the *content* is.

But I don't know where the content is until I've evaluated the field — chicken and egg. The way out: do it in two passes with two networks, a "coarse" one and a "fine" one. First, sample a modest set of N_c points by stratified sampling and evaluate the coarse network everywhere, composite as usual. Now I have something I didn't have before: the per-sample weights w_i = T_i(1 − exp(−σ_i δ_i)) from the coarse pass. Those weights *are* a map of where the ray's color is actually coming from — they spike wherever there's visible, unoccluded content.

So let me turn the coarse weights into a sampling distribution. Rewrite the coarse composited color as Ĉ_c = Σ_i w_i c_i; the w_i are nonnegative and tell me each sample's importance. Normalize them, ŵ_i = w_i / Σ_j w_j, and read that off as a piecewise-constant probability density along the ray. Then draw a second set of N_f points from this PDF by inverse-transform sampling — build the CDF, draw uniforms, invert — so the new samples concentrate exactly where the coarse pass said the content is. Evaluate the *fine* network at the union of all N_c + N_f samples and composite that for the final color. The fine network now spends its samples on the parts of the ray that matter.

A subtlety on what this is and isn't. This looks like importance sampling, but I'm not using it as a Monte-Carlo estimator where each sample is an independent unbiased estimate of the whole integral. I'm using the drawn locations as a *non-uniform discretization* of the whole integration domain — I still composite all of them with their δ-spacing as a quadrature, I've just placed the quadrature points cleverly. That keeps the render deterministic-quadrature in spirit and avoids the variance you'd get treating each sample as a standalone integral estimate.

For this to work the coarse network's weights have to actually be meaningful, which means the coarse network has to be trained too. So I supervise *both* renders against the true pixel:

  L = Σ_{r in batch} [ ‖Ĉ_c(r) − C(r)‖² + ‖Ĉ_f(r) − C(r)‖² ].

The final image comes from the fine render, but I keep the coarse term so the coarse weight distribution stays a good guide for where to sample. Plain squared error on RGB, summed over a batch of rays. Optimize with Adam, learning rate decaying from 5e-4 down to 5e-5 over training; a batch is a few thousand rays drawn from all pixels of all images. One network pair per scene.

One more thing for real, outward-shot scenes rather than objects in a box. For a synthetic object I just scale the scene into a cube and set near/far bounds around it. But a real forward-facing capture — a fern on a table, shot roughly toward it — has content from right in front of the camera out to effectively infinity. The integral wants finite bounds t_n, t_f, and "infinity" breaks both the bounds and the uniform-in-depth sampling (you'd waste everything near the camera and undersample the far stuff, or never reach the back). I want to remap the ray so that the unbounded far range becomes a bounded interval, and so that I sample more densely up close (where disparity changes fast) than far away. There's a coordinate system that does exactly this for free: normalized device coordinates, the cube the graphics pipeline projects a view frustum into. It preserves straight lines, and — the key property — it makes the depth axis linear in *disparity* (inverse depth) rather than in metric depth, so the point at infinity maps to a finite coordinate.

Let me derive the ray remap, because I want to sample linearly inside NDC and have it correspond to something sensible in the real scene. Start from the standard perspective projection matrix (camera looking down −z, near/far planes n, f, frustum half-extents r, t at the near plane):

  M = [[ n/r, 0, 0, 0 ], [ 0, n/t, 0, 0 ], [ 0, 0, −(f+n)/(f−n), −2fn/(f−n) ], [ 0, 0, −1, 0 ]].

Project a point (x,y,z,1): left-multiply by M, then divide by the fourth homogeneous coordinate, which comes out to −z. So a point maps to ( (n/r)·x/(−z),  (n/t)·y/(−z),  (f+n)/(f−n) − (2fn/(f−n))·(1/(−z)) ). Write that compactly as (a_x x/z, a_y y/z, a_z + b_z/z) with a_x = −n/r, a_y = −n/t, a_z = (f+n)/(f−n), b_z = 2fn/(f−n).

Now I want, for a ray o + t·d in camera space, an origin o' and direction d' in NDC such that the projection of the ray equals the NDC ray o' + t'·d' for some reparameterized t' — i.e. the two trace the same set of points, possibly at different rates. Component-wise the requirement is

  ( a_x (o_x + t d_x)/(o_z + t d_z),  a_y (o_y + t d_y)/(o_z + t d_z),  a_z + b_z/(o_z + t d_z) ) = o' + t' d'.

To pin the one extra degree of freedom, demand t = 0 ↦ t' = 0. Plugging t = 0 in directly gives o' = (a_x o_x/o_z, a_y o_y/o_z, a_z + b_z/o_z) — which is just the projection of the original origin, o' = π(o). Now subtract o' from the general-t expression to isolate t' d'. Take the x-component:

  a_x (o_x + t d_x)/(o_z + t d_z) − a_x o_x/o_z = a_x · [ o_z(o_x + t d_x) − o_x(o_z + t d_z) ] / [ (o_z + t d_z) o_z ] = a_x · [ t(o_z d_x − o_x d_z) ] / [ (o_z + t d_z) o_z ] = a_x · ( t d_z/(o_z + t d_z) ) · ( d_x/d_z − o_x/o_z ).

The y-component is identical with x→y. The z-component: b_z/(o_z + t d_z) − b_z/o_z = b_z·(o_z − (o_z + t d_z))/((o_z+t d_z)o_z) = −b_z·( t d_z/(o_z+t d_z) )·(1/o_z). Every component carries the common factor t d_z/(o_z + t d_z). So set

  t' = t d_z/(o_z + t d_z) = 1 − o_z/(o_z + t d_z),

and read off the direction

  d' = ( a_x (d_x/d_z − o_x/o_z),  a_y (d_y/d_z − o_y/o_z),  −b_z/o_z ).

Check the limits: t=0 gives t'=0 as demanded, and t→∞ gives t'→1. So the entire infinite far range of the real ray is compressed into t' ∈ [0,1) — exactly the bounded, disparity-linear interval I wanted, with t'=1 the point at infinity. For a forward-facing capture I take the far plane to infinity (cheap, since NDC already uses inverse depth), and in that limit a_z = (f+n)/(f−n) → 1 and b_z = 2fn/(f−n) → 2n. Writing the frustum extents via the pinhole focal length (a_x = −f_cam/(W/2), a_y = −f_cam/(H/2)):

  o' = ( −f_cam/(W/2) · o_x/o_z,  −f_cam/(H/2) · o_y/o_z,  1 + 2n/o_z ),
  d' = ( −f_cam/(W/2)(d_x/d_z − o_x/o_z),  −f_cam/(H/2)(d_y/d_z − o_y/o_z),  −2n/o_z ).

And one bookkeeping step: before converting, shift the origin to the ray's intersection with the near plane z = −n, i.e. o ← o + t_n d with t_n = −(n + o_z)/d_z. Then in NDC I can just sample t' linearly from 0 to 1, and that linear sweep is a linear sweep in disparity from the near plane out to infinity in the real scene — which is exactly the sampling density I argued for. (For real scenes I also found it helps to add a little zero-mean Gaussian noise to the raw σ before the ReLU during training, as mild regularization on the density.)

Let me put the whole thing down as code, since each piece now corresponds to a derivation step.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- positional encoding: lift each coord into a high-frequency Fourier basis
#     so a low-frequency-biased MLP can still express sharp spatial variation ---
class Embedder:
    def __init__(self, num_freqs, include_input=True):
        self.fns = []
        self.out_dim = 0
        if include_input:                         # keep the raw coordinate too
            self.fns.append(lambda x: x); self.out_dim += 3
        freqs = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)   # 2^0 .. 2^{L-1}
        for f in freqs:
            for p in (torch.sin, torch.cos):      # the (sin, cos) pair per band
                self.fns.append(lambda x, p=p, f=f: p(x * f * torch.pi))
                self.out_dim += 3
    def __call__(self, x):
        return torch.cat([fn(x) for fn in self.fns], -1)

# --- the field: sigma from x only (multiview consistency),
#     color from (x, d) (view-dependent appearance) ---
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, in_x=63, in_d=27, skips=(4,)):
        super().__init__()
        self.skips = skips
        self.pts = nn.ModuleList(
            [nn.Linear(in_x, W)] +
            [nn.Linear(W + in_x, W) if i in skips else nn.Linear(W, W)
             for i in range(D - 1)])          # skip: reinject gamma(x) at layer 5
        self.feature = nn.Linear(W, W)
        self.alpha = nn.Linear(W, 1)          # sigma, before any direction is seen
        self.view = nn.Linear(in_d + W, W // 2)
        self.rgb = nn.Linear(W // 2, 3)

    def forward(self, x_enc, d_enc):
        h = x_enc
        for i, l in enumerate(self.pts):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([x_enc, h], -1)
        sigma = self.alpha(h)                 # density depends on position only
        feat = self.feature(h)
        h = F.relu(self.view(torch.cat([feat, d_enc], -1)))   # now bring in direction
        rgb = self.rgb(h)
        return rgb, sigma                     # raw; squashed in the renderer

# --- the volume-rendering quadrature: continuous integral -> alpha compositing ---
def raw2outputs(rgb_raw, sigma_raw, z_vals, rays_d, raw_noise_std=0., white_bkgd=False):
    dists = z_vals[..., 1:] - z_vals[..., :-1]                       # delta_i
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)         # true world length
    rgb = torch.sigmoid(rgb_raw)                                     # color into [0,1]
    noise = torch.randn_like(sigma_raw[..., 0]) * raw_noise_std if raw_noise_std > 0 else 0.
    alpha = 1. - torch.exp(-F.relu(sigma_raw[..., 0] + noise) * dists)   # 1 - exp(-sigma*delta)
    # T_i = prod_{j<i}(1 - alpha_j)  ==  exclusive cumulative product
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1. - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * T                                              # w_i = T_i * alpha_i
    rgb_map = torch.sum(weights[..., None] * rgb, -2)               # sum_i w_i c_i
    acc = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc[..., None])                   # leftover transmittance -> white
    return rgb_map, weights

# --- inverse-transform sampling of the coarse weight PDF (hierarchical, fine pass) ---
def sample_pdf(bins, weights, N, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)            # normalized w_hat_i
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    u = (torch.linspace(0., 1., N) if det else torch.rand(*cdf.shape[:-1], N))
    u = u.expand(*cdf.shape[:-1], N).contiguous()
    inds = torch.searchsorted(cdf, u, right=True)                   # invert the CDF
    below = (inds - 1).clamp(min=0); above = inds.clamp(max=cdf.shape[-1] - 1)
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(*cdf.shape[:-1], N, cdf.shape[-1]),
                         -1, torch.stack([below, above], -1))
    bins_g = torch.gather(bins.unsqueeze(-2).expand(*bins.shape[:-1], N, bins.shape[-1]),
                          -1, torch.stack([below, above], -1))
    denom = (cdf_g[..., 1] - cdf_g[..., 0]); denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    return bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

# --- one ray batch: stratified coarse pass, then weight-guided fine pass ---
def render_rays(rays_o, rays_d, viewdirs, near, far, embed_x, embed_d,
                coarse, fine, N_c=64, N_f=128, perturb=True, white_bkgd=False):
    def query(net, pts, vd):
        x_enc = embed_x(pts.reshape(-1, 3))
        d_enc = embed_d(vd[:, None].expand(pts.shape).reshape(-1, 3))
        rgb, sigma = net(x_enc, d_enc)
        return rgb.reshape(*pts.shape[:-1], 3), sigma.reshape(*pts.shape[:-1], 1)

    t = torch.linspace(0., 1., N_c)
    z_vals = (near * (1. - t) + far * t).expand(rays_o.shape[0], N_c)   # coarse depths
    if perturb:                                                          # stratified: jitter in each bin
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    rgb, sigma = query(coarse, pts, viewdirs)
    rgb_c, w = raw2outputs(rgb, sigma, z_vals, rays_d, white_bkgd=white_bkgd)

    # hierarchical: resample where the coarse weights say content is
    z_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_fine = sample_pdf(z_mid, w[..., 1:-1], N_f, det=not perturb).detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_fine], -1), -1)         # union of coarse+fine
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    rgb, sigma = query(fine, pts, viewdirs)
    rgb_f, _ = raw2outputs(rgb, sigma, z_vals, rays_d, white_bkgd=white_bkgd)
    return rgb_c, rgb_f          # both are supervised so coarse weights stay meaningful

# --- training: squared error on BOTH coarse and fine renders ---
def train_step(batch_rays_o, batch_rays_d, batch_viewdirs, near, far,
               target_rgb, embed_x, embed_d, coarse, fine, opt, white_bkgd=False):
    rgb_c, rgb_f = render_rays(batch_rays_o, batch_rays_d, batch_viewdirs,
                               near, far, embed_x, embed_d, coarse, fine,
                               white_bkgd=white_bkgd)
    loss = F.mse_loss(rgb_c, target_rgb) + F.mse_loss(rgb_f, target_rgb)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```

Tracing the whole causal chain back: I wanted novel views, fit from images alone, continuous and compact and able to render fine detail with specularities. Meshes optimize badly and need a template; discrete volumes are easy to optimize but their resolution is shackled to an O(n^3) grid; continuous neural-implicit shapes are compact but collapse each ray to one surface. So I keep the continuous MLP container but make it a *volume* — density and color at every point — and render it with the classical emission–absorption integral, which discretizes exactly into differentiable alpha compositing with α = 1−exp(−σδ); stratified sampling keeps the representation continuous rather than a fixed grid. Making density depend on position alone forces multiview-consistent geometry while letting color depend on direction captures specular appearance. The raw MLP came out blurry because plain networks are biased to low frequencies, so I lift each input coordinate through a Fourier encoding γ to hand the network the high frequencies it can't manufacture. And since uniform sampling wastes almost every query on empty space, a coarse pass produces a weight distribution that a fine pass importance-samples, both supervised by the same squared pixel error — with an NDC remap to bound the unbounded real scenes.
