# Anti-aliased neural volumetric rendering across scales

## Research question

A continuous neural volumetric scene representation can be recovered from posed RGB
photographs by casting one ray per pixel, querying a coordinate network for density and
color at point samples along the ray, and compositing them into a pixel color. The
question is how to make a coordinate-network volumetric representation reason about the
*scale* at which each pixel observes the scene, so that a single recovered model renders
correctly across a range of camera distances and resolutions.

## Background

**The point-sampled volumetric pipeline.** A scene is encoded in the weights of an MLP that
maps a 3D position (and a view direction) to a volume density σ and an emitted RGB color.
A pixel is rendered by casting a ray r(t) = o + t·d from the camera center through the pixel
center, choosing sorted sample distances {t_k} between near and far bounds, evaluating the
MLP at each r(t_k), and numerically integrating the emission-absorption volume rendering
integral (Max 1995) by alpha-compositing:
C = Σ_k T_k (1 − exp(−σ_k δ_k)) c_k, with T_k = exp(−Σ_{k'<k} σ_{k'} δ_{k'}) and δ_k =
t_{k+1} − t_k. The whole pipeline is differentiable, so the MLP is fit by minimizing squared
pixel error under gradient descent.

**Positional encoding (Fourier features).** A bare MLP fed raw 3D coordinates can only learn
low-frequency functions — it is spectrally biased. Mapping the input through a sinusoidal
positional encoding γ(x) = [sin(x), cos(x), …, sin(2^{L−1}x), cos(2^{L−1}x)] (Rahimi &
Recht 2007; Tancik et al. 2020 analyze it as a Fourier-feature kernel) lets the network
represent high-frequency detail. L sets the bandwidth, and it is a hyperparameter tuned per
scene.

**Hierarchical sampling with two networks.** To avoid wasting samples on empty space, the
single-scale pipeline runs two MLPs. A "coarse" network is evaluated at stratified samples;
its compositing weights w_k = T_k(1 − exp(−σ_k δ_k)) define a piecewise-constant PDF along
the ray; a "fine" network is then evaluated at additional samples drawn from that PDF by
inverse-transform sampling, plus the coarse samples. Two sets of weights, two networks.

**Sampling, aliasing, and prefiltering in graphics.** A pixel is not a point — it is a
little square footprint, and its true color is the *integral* of incoming radiance over the
cone of directions subtended by that footprint. Sampling a continuous signal below its
Nyquist rate produces aliasing: frequencies above the sampling rate fold down into spurious
low-frequency patterns (jaggies, moiré, shimmer). Graphics has two classical approaches.
*Supersampling* casts many rays per pixel to raise the sampling rate toward Nyquist; its cost
scales linearly with the supersampling rate. *Prefiltering* instead lowpasses the scene
content so the Nyquist rate needed to render it without aliasing drops; the canonical instance
is the mipmap (Williams 1983), a precomputed pyramid of a texture at successive downsampling
scales, from which the level matching the pixel's projected footprint is selected at render
time. Equivalently, one can think of tracing a *cone* per pixel instead of an infinitesimal
ray (Amanatides 1984), and looking up scene content at the scale set by the cone's footprint
where it hits geometry. Prefiltering presupposes the geometry is known so the pyramid can be
precomputed.

**Multiscale benchmark construction.** The standard synthetic benchmark places every camera at
the same distance with the same focal length and resolution. A multiresolution variant can be
constructed from it by box-downsampling each image by 2×, 4×, 8× and adjusting camera
intrinsics accordingly, combining all four scales into one dataset (by projective geometry,
equivalent to moving the camera back by those factors).

## Baselines

- **Point-sampled neural radiance field (the immediate predecessor).** MLP from encoded
  3D position + direction to (σ, c); render by alpha-compositing point samples along one ray
  per pixel; positional encoding γ with tuned degree L; two networks (coarse + fine) for
  hierarchical sampling; optimize squared pixel error with Adam. Core idea gives
  photorealistic results on single-scale benchmarks.

- **Discrete voxel / multiscale graphics structures (mipmaps, sparse-voxel octrees).** Store
  scene content on a grid or tree and prefilter into a pyramid; select the level matching the
  cone footprint at render time. Requires the geometry to be known so the pyramid can be
  precomputed; memory grows as O(n³).

- **Sparse-voxel-octree multiscale neural implicit surfaces (Takikawa et al. 2021).** A
  multiscale octree of features for continuous neural *implicit surfaces*, fitting a signed
  distance function to a given mesh. Models surfaces, not a volumetric radiance field
  recovered from images.

- **Brute-force supersampled point-sampled NeRF.** Cast multiple jittered rays per pixel and
  average their rendered colors.

## Evaluation settings

- **Datasets.** The single-scale synthetic Blender dataset (8 objects: chair, drums, ficus,
  hotdog, lego, materials, mic, ship), each with images rendered from cameras at a fixed
  distance. A multiscale variant constructed from it by box-downsampling each image by 2×,
  4×, 8× and adjusting camera intrinsics accordingly, combining all four scales into one
  dataset.
- **Metrics.** PSNR (↑), SSIM (↑), LPIPS (↓). A summary "average" error: the geometric mean
  of MSE = 10^(−PSNR/10), √(1−SSIM), and LPIPS. Also wall-clock training/eval time and
  parameter count.
- **Protocol.** Known camera poses and intrinsics; recover a model per scene from the
  training views; evaluate on held-out views. For multiscale training, scale each pixel's
  loss by the area of its footprint in the original image (so the few low-res pixels are not
  drowned out by the many high-res pixels). Train with Adam, batch 4096, learning rate
  annealed logarithmically from 5·10⁻⁴ to 5·10⁻⁶ over 1M iterations.

## Code framework

Available pieces: a posed-image dataset and ray generator; a sinusoidal positional
encoding; an MLP backbone mapping encoded position (+ direction) to (σ, c); the
volume-rendering alpha-compositor; stratified and inverse-transform PDF sampling along rays;
an Adam training loop on squared pixel error. One piece is left unimplemented below.

```python
import torch, torch.nn as nn

def generate_rays(camera):
    # origins o, directions d from camera intrinsics/extrinsics
    raise NotImplementedError

def sample_along_ray(origin, direction, near, far, n, weights=None):
    # stratified if weights is None, else inverse-transform sample from a PDF
    raise NotImplementedError  # returns sorted distances t

def featurize(origin, direction, t, radius):
    # TODO: produce the MLP input feature for this query along the ray.
    raise NotImplementedError

class SceneMLP(nn.Module):
    def __init__(self, in_dim, W=256, D=8):
        super().__init__()
        # MLP: encoded feature (+ encoded direction) -> (density, rgb)
        ...
    def forward(self, feat, dir_feat):
        raise NotImplementedError  # returns (sigma, rgb)

def volume_render(sigma, rgb, t):
    delta = t[..., 1:] - t[..., :-1]
    alpha = 1.0 - torch.exp(-sigma * delta)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    w = alpha * T
    color = (w[..., None] * rgb).sum(dim=-2)
    return color, w

def render_pixel(model, origin, direction, radius, near, far):
    # coarse pass -> weights -> fine pass; combine.
    raise NotImplementedError

def train_step(model, batch, opt):
    # squared error between rendered and observed pixel colors.
    raise NotImplementedError
```
