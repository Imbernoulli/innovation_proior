# Context: real-time, high-quality novel-view synthesis from photos (circa 2020-2023)

## Research question

Given a set of photographs of a static scene, with cameras calibrated by Structure-from-Motion
(SfM), recover a 3D scene representation that can render *novel* views — viewpoints not in the
input set — with high visual quality, fast optimization, and fast rendering, for complete,
unbounded scenes (large outdoor environments and full indoor rooms, with background at infinity).
The only supervision available is the input photos: the scene must be recovered by comparing
rendered images to the captured ones. There is no ground-truth geometry, no depth, no normals —
SfM hands over only calibrated cameras and a *sparse* point cloud as a by-product.

## Background

**The image-formation model that everything shares.** Continuous radiance-field rendering
computes the color of a pixel by integrating emission and absorption along the camera ray.
With per-sample density `sigma_i`, color `c_i`, and inter-sample spacing `delta_i`, the
quadrature form is

```
C = sum_{i=1}^N T_i * (1 - exp(-sigma_i * delta_i)) * c_i ,   T_i = exp(-sum_{j<i} sigma_j delta_j),
```

which, writing `alpha_i = 1 - exp(-sigma_i delta_i)` and `T_i = prod_{j<i}(1 - alpha_j)`,
becomes ordinary front-to-back alpha compositing

```
C = sum_{i=1}^N c_i * alpha_i * prod_{j=1}^{i-1} (1 - alpha_j).
```

This is the central fact of the field: a *point-based* renderer that blends ordered overlapping
primitives — color `c_i`, opacity `alpha_i` — computes color by *exactly the same* compositing
sum (e.g. Kopanas et al. 2021; Yifan et al. 2019, where `alpha_i` is a 2D footprint times a
learned per-point opacity). The image-formation model is identical on both sides; what differs
is the rendering *algorithm* and the *representation* behind it. Both are differentiable in
`c_i` and `alpha_i`, so either can be optimized by backpropagating a photometric loss.

**The continuous family.** Neural Radiance Fields (Mildenhall et al. 2020) store the scene as
an MLP mapping position + direction to density and color, query it at many samples per ray, and
composite. The continuity helps optimization — gradients flow smoothly and the field can grow or
erase geometry by adjusting density anywhere. The current top of the quality table, Mip-NeRF 360
(Barron et al. 2022), handles unbounded scenes with anti-aliased, multi-scale sampling and a
scene contraction, reaching outstanding quality on held-out views.

**The fast continuous methods.** A line of work accelerates the continuous field by storing
interpolatable features in a spatial data structure — sparse voxel grids (Plenoxels,
Fridovich-Keil & Yu et al. 2022) or multi-resolution hash grids (InstantNGP, Müller et al. 2022)
— shrinking or removing the MLP. Plenoxels forgoes the network entirely and stores spherical
harmonic (SH) coefficients on a grid; InstantNGP keeps a tiny MLP fed by hashed features. Both
train in minutes and render at interactive rates.

**Spherical harmonics for view-dependent color.** Real surfaces are not Lambertian: their color
changes with viewing direction (specular highlights, glints). SH form an orthonormal basis for
functions on the sphere, so a few low-order coefficients per color channel compactly encode
view-dependent appearance; degree 3 (16 coefficients per channel) is a common, cheap choice
adopted across the fast methods above. A fixed RGB triple cannot represent these effects.

**The explicit / point-based family.** Points and meshes are explicit and map directly onto
fast GPU rasterization, which is why they dominate classical graphics. Plain point-sample
rendering rasterizes an unstructured set of points but suffers holes, aliasing, and is strictly
discontinuous. High-quality point rendering fixes this by *splatting*: giving each point a finite
2D footprint — an elliptical disc or surfel — larger than a pixel, and blending the overlapping
footprints. The mathematical machinery for splatting an elliptical kernel to screen and
anti-aliasing it cleanly is EWA splatting (Zwicker et al. 2001): a kernel with a world-space
2nd-moment (variance) matrix `V`, an affine map `Phi(x)=Mx+c` that takes its variance to
`M V M^T`, a *local affine approximation* of the (non-affine) perspective projection with Jacobian
`J` (the first-order Taylor term of the projective map), the resulting ray-space variance
`V_screen = J W V W^T J^T` for viewing transform `W`, and the reduction of a 3D variance matrix
to its 2D screen footprint by *skipping the third row and column* of the 3×3 matrix. The blending
itself uses the front-to-back compositing above, justified by the first-order Taylor expansion
`exp(-x) ≈ 1 - x` of the attenuation.

Differentiable point-based renderers that augment points with neural features and a CNN achieve
fast or even real-time view synthesis (Aliev et al. 2020; Rückert et al. 2022; Kopanas et al.
2022). Prior 2D-splatting renderers (Kopanas et al. 2021; Yifan et al. 2019) assume each
primitive is a small *planar* disc *with a normal*. Pulsar (Lassner & Zollhöfer 2021) achieves
fast *sphere* rasterization by sorting primitives once per frame instead of per pixel.

## Baselines

These are the prior methods a new approach would be measured against and reacts to.

**NeRF / Mip-NeRF 360 (Mildenhall et al. 2020; Barron et al. 2022).** Represent the scene as a
continuous volumetric field (an MLP, with positional encoding and importance sampling; Mip-NeRF
360 adds multi-scale anti-aliased "mip" cones and a scene contraction for unbounded scenes).
Render by marching each ray and compositing many samples with the quadrature above. *Core math:*
`C = sum T_i alpha_i c_i` over sampled points; the field is queried per sample.

**Plenoxels / InstantNGP (Fridovich-Keil & Yu et al. 2022; Müller et al. 2022).** Accelerate the
continuous field by interpolating features stored in a structured spatial data structure — a
sparse voxel grid holding density and SH coefficients (Plenoxels, no network), or a
multi-resolution hash grid feeding a tiny MLP (InstantNGP). *Core idea:* replace expensive MLP
queries with cheap grid lookups + interpolation, then composite along the ray as before.

**Differentiable point / surfel splatting (Yifan et al. 2019; Kopanas et al. 2021).** Represent
the scene as an unstructured set of points, each a small *planar* elliptical disc with a normal,
and render by projecting to 2D footprints and alpha-compositing in depth order. *Core math:*
`C = sum_i c_i alpha_i prod_{j<i}(1-alpha_j)`, with `alpha_i` a 2D Gaussian footprint times a
learned opacity.

**Pulsar (Lassner & Zollhöfer 2021).** Fast sphere-based neural rendering that sorts primitives
once per image (not per pixel) for speed. *Core idea:* tile the screen, sort spheres globally,
rasterize in parallel.

**MVS-based neural point rendering (e.g. Aliev et al. 2020; Rückert et al. 2022; Kopanas et al.
2022).** Augment an MVS point cloud with neural features and render with a CNN.

## Evaluation settings

The natural yardsticks already in use:

- **Mip-NeRF 360 dataset** (Barron et al. 2022): nine real scenes — four outdoor (bicycle,
  flowers, garden, stump, treehill) and indoor (room, counter, kitchen, bonsai) — covering both
  bounded indoor rooms and large unbounded outdoor environments with background at infinity. The
  standard protocol holds out **every 8th image** for testing and trains on the rest.
- **Tanks & Temples** (Knapitsch et al. 2017) scenes (e.g. Truck, Train) and **Deep Blending**
  (Hedman et al. 2018) scenes (e.g. Dr Johnson, Playroom), with varied, less-structured capture
  styles.
- **Synthetic Blender** scenes (Mildenhall et al. 2020): bounded objects with exact cameras and
  an exhaustive set of views; a white background is used for compatibility.
- **Metrics:** PSNR (higher better), SSIM (higher better), LPIPS (lower better), on held-out
  views; alongside training time, rendering frame rate, and memory.
- Input is the calibrated cameras plus the sparse SfM point cloud (Schönberger & Frahm 2016);
  no MVS, depth, or normals are provided.

## Code framework

The optimizer plugs into a standard per-scene novel-view-synthesis training harness: a data
pipeline that yields one calibrated camera and its ground-truth image at a time, a differentiable
*renderer* that turns the current scene representation into an image for that camera, a
photometric loss comparing rendered to ground-truth, an Adam optimizer over the representation's
parameters, and a loop that renders, computes the loss, backpropagates, and steps. What is *not*
settled is the scene representation itself and the renderer that turns it into pixels. The
substrate below is only the generic machinery that already exists: one empty scene slot plus a
neutral per-iteration bookkeeping hook for whatever representation is eventually designed.

```python
import torch


# ---- photometric loss used across the field ----
def l1_loss(rendered, gt):
    return (rendered - gt).abs().mean()

def ssim(rendered, gt):
    # standard windowed SSIM (Wang et al. 2004); D-SSIM term = 1 - ssim
    ...

def photometric_loss(rendered, gt, lam=0.2):
    return (1.0 - lam) * l1_loss(rendered, gt) + lam * (1.0 - ssim(rendered, gt))


class SceneRepresentation(torch.nn.Module):
    """The scene we optimize and the renderer that turns it into an image.
    Both the parameterization of the scene and how a camera renders it are
    exactly what we will design."""

    def __init__(self, init_points, init_colors):
        super().__init__()
        # initialized from the sparse SfM point cloud (positions + colors)
        # TODO: the parameters that define the scene we will design.
        pass

    def render(self, camera):
        # produce an image for this camera, differentiably, from the parameters.
        # TODO: the renderer we will design.
        pass


def representation_bookkeeping(scene, step, stats):
    """A neutral per-step hook for representation-specific bookkeeping."""
    # TODO: whatever bookkeeping the representation we design needs, if any.
    pass


def train(scene, cameras, images, num_steps=30000, lr=1e-3):
    opt = torch.optim.Adam(scene.parameters(), lr=lr)
    for step in range(num_steps):
        cam, gt = sample_view(cameras, images)     # one calibrated camera + its photo
        rendered = scene.render(cam)               # forward through the renderer
        loss = photometric_loss(rendered, gt)      # compare to the captured image
        loss.backward()                            # gradients to every parameter
        opt.step()
        opt.zero_grad(set_to_none=True)
        representation_bookkeeping(scene, step, stats=None)
```

The loop, the loss, and the optimizer are fixed; `SceneRepresentation` (its parameters and its
`render`) and the neutral bookkeeping hook are the slots the method fills.
