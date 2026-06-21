# Context: Estimating Dense Per-Pixel Motion Between Two Frames

## Research question

Given two consecutive RGB frames $I_1, I_2$, estimate a dense displacement field — for every pixel in $I_1$, the 2D vector $(f^1, f^2)$ pointing to its corresponding location in $I_2$. This is optical flow, and it has stayed unsolved for forty years because the failure modes are exactly the cases that matter: fast-moving objects, occlusions, motion blur, and large textureless regions where the matching signal is ambiguous.

The concrete problem a new method must solve well:

- **Large displacements.** Two frames of a fast scene can move objects by hundreds of pixels. The matching problem is then a search over a huge space — each source pixel could plausibly correspond to thousands of target pixels.
- **Small fast-moving objects.** A bird, a flying limb, a thin pole: these occupy few pixels yet move far. They are the first thing every coarse-resolution method loses.
- **Subpixel accuracy near motion boundaries.** Where two motions meet, the flow field is discontinuous; smoothing across the boundary or warping across it corrupts the estimate.
- **Generalization.** Ground-truth flow is essentially unobtainable for real video, so methods are trained on synthetic data (FlyingChairs, FlyingThings3D) and must transfer to real benchmarks (Sintel, KITTI). A method that overfits the synthetic distribution is useless.
- **Efficiency.** Training optical-flow networks had become extraordinarily expensive — millions of iterations across multiple GPUs — and inference must be fast enough for video.

A solution must produce one coherent, high-resolution flow field that simultaneously handles a 1-pixel motion and a 200-pixel motion, recovers small objects, is sharp at boundaries, trains in a reasonable budget, and transfers across domains.

## Background

**Optical flow as energy minimization.** The classical formulation (Horn & Schunck 1981) writes flow as the minimizer of an energy with two terms: a *data* term that rewards aligning visually similar regions, and a *regularization* term that imposes a smoothness prior on the motion. Horn & Schunck solved a continuous variational problem by gradient steps, maintaining and refining a single dense flow field. Black & Anandan (1993) made the estimation robust to outliers (brightness inconstancy, motion discontinuities) with robust penalty functions. TV-L1 replaced the quadratic penalties with an $L_1$ data term and total-variation regularization, tolerating discontinuities and outliers.

These continuous methods share a structural commitment and a structural weakness. The commitment: a *single* estimate of the flow field, refined iteratively. The weakness: to keep the objective smooth, the data term is linearized by a first-order Taylor expansion of image intensities, which is only valid for *small* displacements. To reach large displacements anyway, the field is wrapped in a **coarse-to-fine** image pyramid — estimate large motions cheaply at low resolution, then upsample and refine the residual at successively higher resolutions. The coarse-to-fine cascade is the workhorse of classical flow, and it carries two persistent pathologies: small fast objects disappear at the coarse levels (their motion is never seen), and an error committed at a coarse level is hard to undo at finer levels.

**Cost volumes and discrete optimization.** A *cost volume* stores, for each source pixel, the matching cost against candidate target pixels. In stereo (1D disparity) it is the standard, highly discriminative representation. For 2D flow the full cost volume is four-dimensional — for each source pixel $(i,j)$ and target pixel $(k,l)$, a scalar similarity $C_{ijkl}$. Discrete-optimization flow methods exploit this. FullFlow (Chen & Koltun 2016) showed that, using the distance transform, global optimization over the full 4D space of flow fields is tractable. DCFlow (Xu, Ranftl & Koltun, CVPR 2017) constructed the full 4D cost volume from *learned* feature embeddings $F_1, F_2 \in \mathbb{R}^{MN\times d}$ (the cost is the distance between feature vectors), then regularized it with an adaptation of Semi-Global Matching (SGM) to four dimensions. DCFlow established that a full 4D volume over learned features is *fast* (built in a fraction of a second by vectorized products) and *accurate*.

**Deep direct prediction.** A second paradigm sidesteps the optimization problem: train a convolutional network to output flow directly from the image pair. FlowNet (Dosovitskiy et al. 2015) introduced two designs — FlowNetSimple (stack the two images, let a generic CNN figure it out) and FlowNetCorr, which adds an explicit **correlation layer** that compares feature patches between the two images. FlowNet2 stacked several FlowNet modules in series and reached parity with classical methods. SpyNet, PWC-Net, LiteFlowNet, and VCN then converged on a common recipe: a *learnable feature pyramid* plus a *warping layer* plus *partial cost volumes at each pyramid level* — i.e. the classical coarse-to-fine cascade, but with every component made learnable and end-to-end trainable. This recipe dominated the benchmarks at the time.

**Iterative refinement and weight tying.** Many flow networks include some form of iterative refinement, but they stack *distinct* modules with their own weights, so the number of refinement steps is baked into the architecture. Iterative Residual Refinement (IRR; Hur & Roth 2019) is the closest to a genuinely recurrent design: it *shares* weights across refinement stages, re-using FlowNetS or PWC-Net as the recurrent unit. Because the recurrent unit is itself a full flow network (FlowNetS is 38M parameters; PWC-Net's iterations are bounded by its number of pyramid levels), IRR is heavy and runs only a handful of iterations.

Separately, the sequence-modeling literature contributed a relevant idea about *depth*. TrellisNet (Bai, Kolter & Koltun 2018) ties weights across a large number of layers; Deep Equilibrium Models (DEQ; Bai, Kolter & Koltun 2019) observe that the hidden states of such weight-tied deep networks converge toward a fixed point, and exploit this by solving for the equilibrium directly. The transferable lesson: a *large number of weight-tied units with bounded, gated updates* can be driven to a stable fixed point.

**Learning to optimize.** A line of work embeds optimization into network architectures. OptNet and differentiable convex layers backpropagate through a solver. Adler & Öktem (2017, 2018) instead *learn the iterative updates directly from data*, motivated by the fact that first-order optimizers (e.g. Primal–Dual Hybrid Gradient) are sequences of update steps; they build a network that mimics those steps for inverse problems (denoising, tomography). TVNet implemented the TV-L1 algorithm as a computation graph so its parameters could be trained — but it operates on raw intensity gradients, not learned features, capping its accuracy on hard datasets.

**Motivating empirical facts about existing systems** (knowable before any new method):
- A coarse-to-fine cascade systematically loses small fast-moving objects, because their motion is absent at the coarse levels where large displacement is estimated.
- A coarse-to-fine cascade struggles to recover from an error made at a coarse level.
- Coarse-to-fine flow networks of the day required very large training budgets (often >1M iterations, sometimes 7M for FlowNet2).
- Warping the second frame by the current flow estimate (the standard way to build a *local* cost volume at each pyramid level) distorts the local image geometry near motion boundaries, creating false correspondences.
- A full 4D cost volume over learned features is cheap to build (DCFlow), but DCFlow's pipeline is *not* end-to-end differentiable: its SGM cost-volume processing has no gradient, so its feature network must be trained with a surrogate embedding/triplet loss instead of directly on flow error.

## Baselines

**Horn–Schunck / variational flow.** Minimize $E(\mathbf{f}) = E_{\text{data}}(\mathbf{f}) + \lambda E_{\text{reg}}(\mathbf{f})$, where the data term enforces brightness constancy (linearized by Taylor) and the regularizer penalizes spatial gradients of the flow. Solved by gradient descent on a single dense field. Limitation: the Taylor linearization is valid only for small motion, so large displacement requires a coarse-to-fine pyramid, with the attendant loss of small fast objects and inability to undo coarse-level mistakes.

**FlowNetC (correlation layer).** Builds two feature maps $f_1, f_2$ and compares patches with a correlation
$$c(\mathbf{x}_1, \mathbf{x}_2) = \sum_{o\in[-k,k]^2} \langle f_1(\mathbf{x}_1+o),\, f_2(\mathbf{x}_2+o)\rangle,$$
restricting $\mathbf{x}_2$ to a neighborhood of maximum displacement $d$ around $\mathbf{x}_1$ (output size $w\times h\times D^2$ with $D=2d+1$). Limitation: the correlation is computed only within a bounded displacement window, so motions larger than $d$ are invisible to it; the design must then be embedded in a multi-resolution stack to see large motion, and the network decodes flow from the cost volume in a single feed-forward pass with no notion of refining a maintained estimate.

**PWC-Net (learnable coarse-to-fine cost volume + warping).** At each pyramid level $l$: extract learnable features, *warp* the second-image features toward the first by the upsampled flow from level $l{+}1$, build a *partial* cost volume over a small search range $d$ (dimension $d^2\times H_l\times W_l$), and decode a flow update with a CNN; a context network post-refines. Limitation: by construction the cost volume at each level is *local* (range $d$) and *warped*, so the method only ever sees small residual displacements at each scale; large motion lives entirely in the coarse levels, re-introducing the small-fast-object loss and the coarse-error problem, and warping distorts geometry near motion boundaries.

**DCFlow (full 4D cost volume + SGM).** Learn feature embeddings $F_1, F_2$, populate the full four-dimensional cost volume by feature-vector distances, regularize it with 4D Semi-Global Matching, then upsample and inpaint. Limitation: the cost-volume *processing* (SGM, winner-take-all) is not differentiable, so the feature network cannot be trained directly on flow error — it is trained with a triplet/embedding loss as a surrogate — and the pipeline is a fixed feed-forward sequence rather than a learned refinement of a maintained flow field.

**IRR (weight-tied iterative refinement).** Re-uses a full flow network (FlowNetS or PWC-Net) as a recurrent unit with shared weights across refinement passes. Limitation: the recurrent unit is itself a large flow network, so the parameter cost is high (FlowNetS: 38M) and the number of iterations is small and bounded (PWC-Net variant limited by pyramid levels); it cannot be run for many lightweight steps.

**Devon.** Builds a cost volume without warping and updates at fixed resolution, handling large displacement with a *dilated* cost volume. Limitation: it has no recurrent unit, and a fixed dilation is its only mechanism for reaching large displacement, which couples the reachable range to a chosen dilation hyperparameter.

**FlowNet2.** Stacks multiple FlowNetS/FlowNetC modules in series, each with its own weights. Limitation: very large (≈162M parameters), very expensive to train (millions of iterations), and the number of refinement stages is fixed by the stack.

## Evaluation settings

- **Training data (synthetic, with dense ground-truth flow):** FlyingChairs (C) and FlyingThings3D (T). The standard protocol pretrains on C then T.
- **Benchmarks:** MPI Sintel (clean and final passes; final adds atmospheric effects, motion blur, defocus) and KITTI-2015 (driving scenes, semi-dense LiDAR ground truth, large illumination changes and occlusions). Additional finetuning data: HD1K. High-resolution qualitative video: DAVIS (1080p).
- **Metrics:** average end-point error (EPE, the mean Euclidean distance between predicted and ground-truth flow vectors over valid pixels); on KITTI, the F1-all error (percentage of pixels whose flow error exceeds 3 px or 5% of magnitude). Cross-dataset generalization is measured by training on C+T and evaluating on the Sintel/KITTI train splits without finetuning.
- **Protocol:** pretrain on synthetic data, optionally finetune on benchmark-specific data; report both the generalization (C+T) numbers and the finetuned leaderboard numbers. Photometric and spatial augmentation are standard; occlusion augmentation (random erasing in $I_2$) is used.

## Code framework

The primitives that already exist: PyTorch, a convolutional feature extractor, bilinear sampling (`F.grid_sample`), average pooling, a GRU/LSTM-style gated recurrent cell, the `F.unfold` operator, AdamW with gradient clipping and a one-cycle LR schedule, and the standard flow data pipeline with augmentation. The contribution will be filled into the empty slots below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureCNN(nn.Module):
    """Per-pixel feature extractor (residual blocks) mapping an image to a
    feature map at reduced (1/8) resolution."""
    def __init__(self, output_dim):
        super().__init__()
        # TODO: convolutional residual backbone producing H/8 x W/8 x D features
        pass
    def forward(self, image):
        pass

class SimilarityModule:
    """Turn the two feature maps into whatever representation of visual
    similarity the method will read during refinement, and provide a way to
    query it given a current flow estimate."""
    def __init__(self, fmap1, fmap2):
        # TODO: build the similarity representation we will design
        pass
    def query(self, flow):
        # TODO: produce similarity features for the current flow estimate
        pass

class RefinementModule(nn.Module):
    """The operator that, given similarity features, the current flow, and any
    context/state, proposes how to change the flow. The core slot to design."""
    def __init__(self, *dims):
        super().__init__()
        # TODO: the iterative update operator we will design
        pass
    def forward(self, state, context, similarity_feats, flow):
        # TODO: return updated state and a flow increment
        pass

def upsample_to_full_res(flow_low):
    """Map a low-resolution flow field up to input resolution."""
    # TODO: the upsampling we will design
    pass

class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = FeatureCNN(output_dim=256)
        # TODO: context features; refinement operator; upsampler
        pass
    def forward(self, image1, image2, iters):
        # extract features for both images (and context from image1)
        # build the similarity representation once
        # initialize a flow field and refine it for `iters` steps,
        #   collecting a prediction at each step
        # TODO: assemble the pieces above
        pass

def sequence_loss(flow_preds, flow_gt, valid):
    """A loss over a SEQUENCE of flow predictions (one per refinement step)."""
    # TODO: weight and aggregate the per-step prediction errors
    pass
```
