# Continuous scene representation for novel view synthesis — the ground before the method

## Research question

Given a sparse-to-moderate set of RGB photographs of a static scene, each with a known
camera pose and intrinsics, synthesize photorealistic images of the scene from *new*
viewpoints that were never captured. This is the novel-view-synthesis problem, and it has
been open for decades. The question is how to build a scene representation that can be
optimized from 2D image supervision alone and rendered from arbitrary camera positions.

## Background

**Classical volume rendering (the emission–absorption optical model).** The physical model
behind rendering participating media (fog, smoke, clouds) treats a volume as a field of
tiny absorbing/emitting particles. Kajiya and Von Herzen (1984) introduced ray tracing of
volume densities to graphics; Max (1995) wrote the canonical review of the optical models
and their numerical evaluation. The model assigns each point a density (absorption
coefficient) σ(x) and an emitted color c(x). For a ray r(t) = o + t·d, define the
transmittance T(t) — the fraction of light that survives from the start of the ray to t
without being absorbed. Over an infinitesimal step, a fraction σ(t)·dt of the remaining
light is absorbed, giving dT/dt = −σ(t)·T(t), whose solution is T(t) = exp(−∫ σ ds). The
radiance reaching the eye is the integral, over the ray, of (emitted color) × (probability
a particle sits there, σ dt) × (probability the light survives to the eye, T(t)). To
evaluate this integral on a computer it is discretized: assuming σ and c are piecewise
constant on short intervals, each interval contributes a term, and the accumulation reduces
to the **alpha-compositing** operation (Porter and Duff, 1984) already standard in
graphics — front-to-back blending with per-sample opacities. This optical model and its
discrete alpha-compositing form are pre-existing tools.

**Coordinate-based MLPs as function representations.** A separate line of work encodes a
function directly in the weights of a fully-connected network that maps a low-dimensional
coordinate to a value: images (Stanley, 2007, compositional pattern-producing networks),
textured materials and BTFs (Henzler et al., 2020; Oechsle et al., 2019; Rainer et al.,
2019/2020), and indirect-illumination functions (Ren et al., 2013). For 3D shape, networks
map (x,y,z) to a signed distance (Park et al., 2019, DeepSDF) or an occupancy probability
(Mescheder et al., 2019, occupancy networks; Genova et al., 2020; Jiang et al., 2020). The
appeal is that such a representation is continuous and tiny. DeepSDF also introduced a
practical architecture detail — a skip connection that re-injects the input coordinate into
a middle layer of the MLP.

**Spectral bias of neural networks.** Rahaman et al. (2018) showed, via Fourier analysis,
that deep ReLU networks have a learning bias toward low-frequency functions: low-frequency
components of a target are fit first and fastest, and expressing high-frequency variation
requires finely tuned parameters and is intrinsically harder. They further observed that
mapping the inputs through high-frequency functions before the network makes it markedly
easier to fit data with high-frequency content. This is a pre-existing diagnostic fact
about MLPs: a network fed raw low-dimensional coordinates tends to produce oversmoothed
outputs.

## Baselines

**Differentiable mesh / surface optimization.** Mesh-based scene representations with
diffuse (Waechter et al., 2014) or view-dependent (Buehler et al., 2001; Debevec et al.,
1996; Wood et al., 2000) appearance, optimized with differentiable rasterizers (Chen et
al., 2019; Liu et al., 2019, soft rasterizer; Loper and Black, 2014, OpenDR) or
differentiable pathtracers (Li et al., 2018, redner; Mitsuba 2). Core idea: render the mesh,
compare to images, backprop to vertices/appearance.

**Discrete volumetric, directly colored or CNN-predicted.** Early methods colored voxel
grids directly from images (Seitz and Dyer, 1999, voxel coloring; Kutulakos and Seitz,
2000, space carving). Modern methods train deep networks to predict a sampled volumetric
representation from input views and render by alpha-compositing or learned compositing
(Flynn et al., 2019; Kar et al., 2017; Penner and Zhang, 2017; Zhou et al., 2018; Mildenhall
et al., 2019, multiplane images / local light field fusion). Local Light Field Fusion (LLFF)
predicts a frustum-sampled RGBA multiplane image per input view with a 3D CNN, then blends
nearby MPIs into a novel view. Core idea: discrete RGBA volume + alpha-compositing.

**Voxel grid + CNN hybrids.** DeepVoxels (Sitzmann et al., 2019) optimizes a learned 3D
feature grid plus a CNN that compensates for low-resolution discretization artifacts;
Neural Volumes (Lombardi et al., 2019) optimizes a 3D CNN that predicts a 128^3 RGBA voxel
grid plus a warp field, rendered by marching rays through the warped grid. Core idea: store
the scene in an explicit grid, lean on a CNN to hide discretization.

**Continuous neural implicit, 2D-supervised.** Differentiable Volumetric Rendering
(Niemeyer et al., 2019) represents surfaces as a 3D occupancy field, finds the ray–surface
intersection numerically, and uses implicit differentiation to get an exact gradient at the
intersection, which feeds a neural texture field for color. Scene Representation Networks
(Sitzmann et al., 2019, SRN) map each (x,y,z) to a feature vector and train an RNN to march
along the ray and decide where the surface is, then decode the final feature into one color.
Core idea: continuous MLP scene + differentiable ray-marching, supervised by 2D images.

## Evaluation settings

The natural yardsticks are per-scene novel-view synthesis benchmarks. Synthetic object
datasets: the DeepVoxels set of four Lambertian objects with simple geometry, rendered at
512×512 from views on the upper hemisphere (hundreds of input views, ~1000 test views); and
pathtraced datasets of objects with complex geometry and non-Lambertian materials, rendered
at 800×800 from views on a hemisphere or full sphere (≈100 input, ≈200 test). Real scenes:
forward-facing handheld captures (e.g. cellphone, 20–62 images per scene, ~1008×756),
holding out a fraction of views for test; camera poses and bounds estimated with
structure-from-motion (Schönberger and Frahm, 2016, COLMAP). Quantitative image-quality
metrics are PSNR and SSIM (higher is better) and the LPIPS perceptual metric (Zhang et al.,
2018, lower is better). The protocol fits a separate model per scene (except dataset-trained
predictors like LLFF). Optimization uses Adam (Kingma and Ba, 2015).

## Code framework

The primitives that already exist: camera geometry to turn pixels into rays, a generic
fully-connected network module, an Adam optimizer, a mean-squared image loss, and the
classical alpha-compositing accumulation. The pieces the representation-and-rendering method
will occupy are left as stubs.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- existing: camera geometry -> rays -----------------------------------
def get_rays(H, W, K, c2w):
    """Pixel grid -> per-pixel ray origins o and directions d in world space."""
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    dirs = torch.stack([(i - K[0][2]) / K[0][0],
                        -(j - K[1][2]) / K[1][1],
                        -torch.ones_like(i)], -1).float()
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

# --- TODO: how do we turn a low-D coordinate into network input? ---------
def encode_input(p):
    # TODO: the input representation we will design
    pass

# --- TODO: the scene representation ---------------------------------------
class SceneField(nn.Module):
    """A function from a continuous coordinate to local appearance/opacity.
    Architecture and exactly what it consumes/produces are to be designed."""
    def __init__(self):
        super().__init__()
        # TODO: the network we will design
        pass
    def forward(self, x):
        # TODO
        pass

# --- TODO: pick where along each ray to evaluate the field ----------------
def sample_along_ray(rays_o, rays_d, near, far, N):
    # TODO: the sampling strategy we will design
    pass

# --- existing primitive: classical alpha-compositing accumulation ---------
def composite(colors, opacities):
    """Front-to-back alpha-composite per-sample (color, opacity) into a pixel.
    Standard graphics accumulation; how opacity is *obtained* is TODO."""
    # TODO: fold the field's outputs into (colors, opacities) and accumulate
    pass

# --- existing: a fully-connected scaffold, MSE loss, Adam -----------------
def render(rays_o, rays_d, near, far, field):
    pts = sample_along_ray(rays_o, rays_d, near, far, N=...)   # TODO
    raw = field(encode_input(pts))                              # TODO
    pixel = composite(*raw)                                     # TODO
    return pixel

def train_step(field, rays_o, rays_d, near, far, target_rgb, opt):
    pred = render(rays_o, rays_d, near, far, field)
    loss = F.mse_loss(pred, target_rgb)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```
