Two frames sit in front of me and I want, for every pixel of the first, the 2D vector that points to where it went in the second. Forty years of optical flow and the same cases still break everything: a bird flapping across the frame, a car edge against a guardrail, motion blur smearing the texture away. Let me start from the failure I keep watching, and ask what a method would have to do to not fail that way.

The way the classical people think about this is energy minimization. Write down $E(\mathbf{f}) = E_{\text{data}}(\mathbf{f}) + \lambda E_{\text{reg}}(\mathbf{f})$. The data term says: the flow should map visually similar regions onto each other. The regularizer says: nearby pixels should move similarly. Then you keep a single dense flow field and you push it downhill — Horn and Schunck did exactly this in 1981, gradient steps on a variational energy; Black and Anandan made the penalties robust to outliers; TV-L1 swapped in an $L_1$ data term and total-variation smoothness so motion discontinuities survive. There is something I genuinely like in this picture: you maintain *one* estimate and you *refine* it. It is honest about the fact that flow is a thing you converge to, not a thing you read off in one shot.

But there is a rot at the center of it. To make the energy smooth enough to differentiate, the data term gets a first-order Taylor expansion of the image intensities — $I_2(\mathbf{x}+\mathbf{f}) \approx I_2(\mathbf{x}) + \nabla I_2 \cdot \mathbf{f}$. That linearization is only any good when $\mathbf{f}$ is *small*. The moment a pixel moves more than a pixel or two, the approximation is lying. So the whole field has to estimate large motion at low resolution — shrink the image until the displacement is small in pixels, solve there, upsample, refine. Coarse-to-fine. And I have stared at enough coarse-to-fine output to know its two diseases by heart: a small fast object — the bird — is *gone* at the coarse level, its motion is never even seen, so it can never be recovered at the fine level; and an error you commit at the coarse level is almost impossible to undo as you go finer, because each fine level only ever estimates a small residual around whatever the coarse level handed it.

Now, the deep-learning crowd promised to escape all this by just training a network to spit out flow. And in a sense they did escape the hand-crafted energy. But look at what the best of them actually *are* — PWC-Net, LiteFlowNet, VCN, FlowNet2. Learnable feature pyramid, a warping layer, a partial cost volume at each pyramid level, decode a flow update, go up a level, repeat. That is the classical coarse-to-fine cascade wearing a CNN costume. They inherited the *exact same* two diseases. They still lose the bird. They still can't recover from a coarse mistake. And on top of that they cost a fortune to train — FlowNet2 took on the order of millions of iterations. So the deep methods didn't cure the disease, they just made it learnable.

Let me hold onto the part I liked — one field, refined iteratively — and ask: what is forcing the coarse-to-fine? It is *only* the Taylor linearization. The pyramid exists to keep the data-term approximation valid. If I never linearize — if I let the data term *directly* see large displacement — there is no reason left to go coarse-to-fine at all. I could maintain a single high-resolution flow field from start to finish.

So the question becomes: how do I let a single high-resolution update see *both* a 1-pixel motion and a 200-pixel motion at once?

The data term is "visual similarity between a source pixel and a candidate target pixel." If I want to see *all* displacements, I should just compute the similarity between a source pixel and *every* target pixel. Not a window. Everything. For source pixel $(i,j)$ with feature vector $g(I_1)_{ij}$ and target pixel $(k,l)$ with feature $g(I_2)_{kl}$, define
$$C_{ijkl} = \sum_h g(I_1)_{ijh}\, g(I_2)_{klh} = \langle g(I_1)_{ij},\, g(I_2)_{kl}\rangle.$$
That is a 4D object, $H\times W\times H\times W$. It is the data term for *every* possible displacement of *every* pixel, all at once. No window, no maximum displacement $d$, no warping.

Is this even computable? FlowNetC introduced a correlation layer but bounded the displacement to a window of size $D=2d+1$, precisely because comparing all $w^2 h^2$ pairs "yields a large result and makes efficient forward and backward passes intractable" — that was their stated reason for the window. But stare at the definition again. $C_{ijkl}$ is an inner product over the channel dimension. Flatten the spatial dims: $g(I_1)$ is $(HW)\times D$, $g(I_2)$ is $(HW)\times D$, and the full correlation is just
$$C = g(I_1)\, g(I_2)^\top,$$
a single $(HW)\times(HW)$ matrix multiply. On a GPU that is one of the most optimized operations in existence. The thing FlowNetC declared intractable in 2015 is a matmul. Compute it once — it does not depend on the flow estimate, so it is computed a single time no matter how many refinement steps I take. Let me sanity-check the cost: it is $O(N^2)$ in the number of pixels $N=HW$. For a $1088\times 1920$ video at $1/8$ resolution that is large, but it is a *single* matmul amortized over all the refinement steps, so I'd expect it to be a small fraction of total inference rather than the bottleneck. (And if it ever did become one, there is an escape hatch I'll come back to.) The all-pairs 4D volume is real and cheap. The window was never necessary; it was an artifact of nobody noticing the inner product is a matmul.

One small thing while I'm here: the raw inner products scale with the feature dimension $D=256$, which makes their magnitudes large and the downstream operator twitchy. Divide by $\sqrt{D}$ to keep the scale sane. Cheap insurance.

Now I have, for each source pixel, a full 2D response map over the whole target image — its similarity to everything. The update operator can't ingest the entire $HW$-dimensional response per pixel at every step; that's enormous and most of it is irrelevant. What it actually needs, at any given step, is the similarity *in the vicinity of where it currently thinks the pixel went*. So given the current flow $\mathbf{f}$, map source pixel $\mathbf{x}=(u,v)$ to its estimated correspondence $\mathbf{x}' = (u+f^1, v+f^2)$ and read the correlation in a small grid around $\mathbf{x}'$. Define the local set
$$\mathcal{N}(\mathbf{x}')_r = \{\mathbf{x}' + \mathbf{dx} \mid \mathbf{dx}\in\mathbb{Z}^2,\ \|\mathbf{dx}\|_1 \le r\},$$
the integer offsets within $L_1$ radius $r$. Since $\mathbf{x}'$ is a real number (the flow is continuous), the grid points land between integer correlation entries, so I bilinearly sample the 4D volume. This is the lookup operator $L_C$: it turns "where do I think this pixel went" into "what does the similarity look like right there."

But wait — a fixed radius $r$ around the current correspondence only sees displacements within $r$ pixels of the *current guess*. If the flow is initialized at zero and the true motion is 200 pixels, the first lookup is reading similarity around the origin, nowhere near the answer. The operator needs evidence that the answer is far away, not just "nothing matches nearby." A local window has re-smuggled in the small-displacement problem.

Here is the tension. I have the *full* large-displacement information sitting in the 4D volume. I just can't afford to feed all of it to the operator each step, so I window it — and the window kills the large-displacement reach. I want a local lookup that nonetheless carries information about far-away matches.

Pool the volume. The correlation $C_{ijkl}$ has two pairs of dimensions: the first pair $(i,j)$ indexes the *source* pixel, the second pair $(k,l)$ indexes the *target* pixel. If I average-pool over the *target* dimensions $(k,l)$, with kernels of size $\{1,2,4,8\}$ and matching strides, I get a 4-level pyramid $\{C^1,\dots,C^4\}$ where $C^k$ has shape $H\times W\times H/2^k\times W/2^k$. The crucial choice: I pool *only the last two dimensions*. I leave the source dimensions $(i,j)$ at full resolution. Why does that matter? Because the source resolution is what protects the bird. Keep $(i,j)$ at full res and even a 3-pixel-wide fast object still has its own response maps; it is never blurred away. The coarse-to-fine cascade pooled *everything*, source and target together, and that is precisely when small objects vanish. Here the source stays sharp and only the *target* (the displacement axis) gets coarsened.

Now do the local lookup on *every* level. At level $k$, index $C^k$ with the grid $\mathcal{N}(\mathbf{x}'/2^k)_r$ — the same radius $r$, but in the pooled coordinate system. A constant $r$ across levels means a wildly different *real-world* range per level: at the finest level $r$ steps cover $r$ cells of the $1/8$-resolution grid, but at the coarsest level each pooled cell has been shrunk by three successive stride-2 poolings, so one cell there spans $2^3=8$ cells of the $1/8$-resolution grid, which is $8\times 8 = 64$ pixels at the original resolution; with $r=4$ that level reaches $4\times 64 = \pm 256$ pixels at the original resolution. Concatenate the lookups from all levels into one feature map. Now a single local lookup, cheap, simultaneously carries fine-grained nearby similarity (from level 1) and coarse evidence about matches hundreds of pixels away (from level 4). The large-displacement information survives the windowing because the coarse levels squeeze a huge spatial range into the same small grid. That is the move I was missing: don't shrink the *image* to reach large motion (coarse-to-fine), shrink the *displacement axis of the cost volume* while keeping the image sharp.

Let me double-check the pooling is principled and not just convenient. The pooled value at level $m$ is
$$C^m_{ijkl} = \frac{1}{2^{2m}}\sum_{p}^{2^m}\sum_{q}^{2^m} \langle g^{(1)}_{ij},\, g^{(2)}_{2^m k+p,\, 2^m l+q}\rangle = \Big\langle g^{(1)}_{ij},\ \frac{1}{2^{2m}}\sum_{p}^{2^m}\sum_{q}^{2^m} g^{(2)}_{2^m k+p,\,2^m l+q}\Big\rangle,$$
where I pulled $g^{(1)}_{ij}$ out of the sum by linearity of the inner product. So pooling the correlation over the target window is *identical* to correlating the source feature with the *pooled* target features. That is a nice consistency check — the pyramid is averaging similarity over a target neighborhood, exactly the multi-scale displacement context I wanted. And it hands me the escape hatch I promised: I never have to materialize the $O(N^2)$ volume. I can precompute the pooled target feature maps and compute each correlation value *on demand, only when it is looked up* — $O(NM)$ for $M$ iterations instead of $O(N^2)$. I won't need it by default (the matmul isn't the bottleneck), but it's good to know the all-pairs choice doesn't paint me into a memory corner.

This is, by the way, why all-pairs beats a bounded correlation range even though a 128-pixel window would cover most Sintel motions. With all-pairs I never have to *choose* a range — the method adapts to whatever is in the scene — and it's trivial to implement as a matmul, so the whole thing stays in pure PyTorch. A local-range cost volume buys me nothing except a hyperparameter to get wrong.

So far: feature encoder $g_\theta$ at $1/8$ resolution (residual blocks; $1/8$ is the standard balance between keeping enough spatial detail and keeping the $N^2$ volume affordable), all-pairs 4D correlation as the data term, pooled on the target axis into a pyramid, queried by a local multi-level lookup at the current flow. That is the data term, fully learned, seeing all displacements, sharp on the source.

Now the engine. I kept "one field, refined iteratively" from the variational picture, but I deleted the analytic descent direction — there is no explicit energy to take the gradient of anymore, because I replaced the hand-crafted data+smoothness energy with a learned correlation volume. So what proposes the update? The learning-to-optimize people (Adler and Öktem, learned primal-dual for inverse problems; TVNet unrolling TV-L1) had the right instinct: a first-order optimizer *is* a sequence of update steps, so *learn* the update step from data instead of deriving it. Apply that here. Maintain flow estimates $\mathbf{f}_0 = \mathbf{0}, \mathbf{f}_1, \mathbf{f}_2,\dots$; at each step a learned operator looks at the correlation around the current guess and proposes an increment $\Delta\mathbf{f}$, and I apply $\mathbf{f}_{k+1} = \mathbf{f}_k + \Delta\mathbf{f}$. The operator never computes a gradient with respect to an objective; it *learns to propose the descent direction* from the similarity features. The smoothness prior, which used to be hand-written into $E_{\text{reg}}$, is now whatever the operator learns to do — it can look at neighboring correlations and decide how to regularize. Both the features and the motion prior are learned; nothing is hand-crafted.

What should the operator be, concretely? My first instinct is the obvious one: stack of convolutions, ReLU, output $\Delta\mathbf{f}$. Let me think about whether that survives being run many times. I want to run this operator a *lot* — that's the whole point of refining a maintained estimate — and I want it to *converge*, the sequence $\mathbf{f}_k$ should settle to a fixed point $\mathbf{f}^*$, not oscillate or blow up. A plain conv stack applied 50 times with no gating is a recipe for drift: nothing bounds the state, nothing decides when to stop changing a pixel that's already correct. I recall what the sequence-modeling people learned about depth: TrellisNet ties weights across a huge number of layers; DEQ observes that such weight-tied stacks *converge to a fixed point* and even solves for the equilibrium directly. The structural ingredients that make that work are (1) tied weights — the same transformation applied repeatedly — and (2) bounded, gated updates so the state can't run away. A GRU is exactly a gated update: it has an update gate $z$ that decides, per element, how much of the state to overwrite, and it keeps activations bounded through sigmoids and tanh. That gating is what should let a pixel that's already converged simply *stop* getting changed ($z\to 0$), while a pixel still far from its match keeps updating. So make the operator a convolutional GRU — replace the GRU's fully-connected layers with $3\times3$ convolutions so it respects spatial structure:
$$z_t = \sigma(\text{Conv}_{3\times3}([h_{t-1}, x_t], W_z)),$$
$$r_t = \sigma(\text{Conv}_{3\times3}([h_{t-1}, x_t], W_r)),$$
$$\tilde h_t = \tanh(\text{Conv}_{3\times3}([r_t \odot h_{t-1}, x_t], W_h)),$$
$$h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t.$$
The gating *is* the convergence mechanism: bounded activations, and an update gate that can shrink the change to zero. A plain three-conv-ReLU block has neither — nothing keeps its repeated application from drifting — so I expect the gated GRU to make the sequence of flow estimates converge much more cleanly, which is exactly the property I'm designing it for.

Now, *tie the weights across iterations* — every step uses the *same* GRU. This is not a parameter-saving trick I'm tacking on; it is the thing that makes the operator mean "one step of an optimization algorithm." If each step had its own weights, I'd be back to a fixed-depth cascade (and a giant one). By forcing every step to be the *same* update, I constrain the network to *learn a single update rule* that must work no matter how many times it's applied — which both shrinks the search space (better generalization, less overfitting to the synthetic training distribution) and, crucially, decouples the number of iterations from training: I can train with a modest number of unrolled steps and run as many as I like at test time. Untying the weights would blow the parameter count up by an order of magnitude *and* — since each step would no longer have to learn a single reusable update rule — I'd expect it to generalize worse, not better. The tied operator is doing real work, not just saving memory.

What goes into the operator's input $x_t$? Three things. The correlation lookup features around the current flow — that's the data evidence. The current flow itself — the operator should know where it is. And one more: features from the *first image only*, injected directly. Call it a context network $h_\theta$, same architecture as the feature encoder but reading only $I_1$. Why? Because the smoothness prior — "aggregate motion within object boundaries, don't bleed across them" — needs to know *where the boundaries are*, and that information is in the image appearance, not in the correlation. Feeding the operator the image context lets it propagate flow within a coherent region and stop at its edge. So the motion encoder runs a couple of convs over the correlation features, a couple over the flow, concatenates them, and the operator additionally consumes the context features. The hidden state $h$ is the GRU's running memory; I initialize it from the context too (split the context features into a $\tanh$-initialized hidden part and a $\text{ReLU}$ input part). The flow head is two convolutions on the hidden state producing the 2-channel $\Delta\mathbf{f}$ — a residual, not the absolute flow, because the whole design is about *refining* a maintained estimate.

A subtlety about the receptive field. A single $3\times3$ conv per gate has a small spatial reach, and propagating motion across an object wants a larger receptive field — but I don't want to pay for a $5\times5$ conv. Factor it: run the GRU twice per step, once with a $1\times5$ convolution and once with a $5\times1$ convolution. Two separable passes give me a $5\times5$-ish receptive field at a fraction of the parameters of a dense $5\times5$. (The small variant of the model just uses a single $3\times3$ GRU and bottleneck residual blocks in the encoder, when parameters matter more than the last fraction of accuracy.)

Two practical issues with training the recurrent chain. First, the update $\mathbf{f}_{k+1} = \mathbf{f}_k + \Delta\mathbf{f}$: if I backpropagate through *both* the $\Delta\mathbf{f}$ branch and the $\mathbf{f}_k$ branch, gradients flow back through the entire history of additions and the chain gets unstable to train. The fix is to only let the gradient through the $\Delta\mathbf{f}$ branch and *detach* the $\mathbf{f}_k$ branch — each step learns to produce a good increment given the current flow *as a constant*, which is exactly the optimizer-step semantics I want anyway. Second, how do I supervise a *sequence* of estimates rather than a single output? Supervise *all* of them. If only the last iterate is supervised, the intermediate steps have no pressure to actually be refinements. So define the loss over the full sequence $\{\mathbf{f}_1,\dots,\mathbf{f}_N\}$ against ground truth $\mathbf{f}_{gt}$, with exponentially increasing weight on later iterates:
$$\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i}\,\|\mathbf{f}_{gt} - \mathbf{f}_i\|_1,\qquad \gamma=0.8.$$
$L_1$ rather than $L_2$ because flow ground truth has outliers and occlusion artifacts and I don't want them dominating. The $\gamma^{N-i}$ weighting says every step should be heading toward the answer, but the later ones matter more — each prediction is a valid refinement, and the final one is the one I'll use. $\gamma=0.8$ is the discount that makes early steps count enough to learn from without swamping the final-step signal.

I can already feel a nice property falling out of this design — the same property the variational people wanted and the coarse-to-fine people gave up. Because I maintain a single high-resolution field and the operator is *the same every step*, I'm not locked to a fixed iteration count. I'll unroll a modest number during training, but at inference I can keep iterating until the updates $\|\Delta\mathbf{f}_k\|$ shrink to nothing — the field converges to a fixed point and stays there even if I push it to a hundred-plus steps. And because the field is at a single high resolution with no warping, I can do something coarse-to-fine simply *can't*: when processing video, initialize $\mathbf{f}_0$ not at zero but at the previous frame's flow, forward-projected to the new frame (occlusion gaps filled by nearest-neighbor). The coarse-to-fine pyramid has no single coherent field to warm-start; I do. By default, though, $\mathbf{f}_0 = \mathbf{0}$.

One thing I keep glossing: the operator outputs flow at $1/8$ resolution, and I need it at full resolution. The lazy answer is bilinear upsampling — multiply the flow by 8, interpolate. But bilinear interpolation across a motion boundary averages two different motions into a smear, exactly killing the boundary sharpness I worked to preserve by keeping the source dimensions at full resolution. The flow field is piecewise-smooth with hard edges; I want an upsampler that respects edges. So *learn* the upsampling. Each full-resolution pixel is the convex combination of a $3\times3$ neighborhood of its coarse-resolution neighbors — and let the network *predict the combination weights*. From the hidden state, predict a mask of shape $H/8 \times W/8 \times (8\times8\times9)$: for each of the $8\times8$ fine pixels inside a coarse cell, 9 weights over the $3\times3$ coarse neighbors. Softmax over the 9 so they form a convex combination, then take the weighted sum of the (×8-scaled) coarse flow over the $3\times3$ neighborhood, and reshape to $H\times W\times 2$. This is implementable directly with `unfold`. Because the weights are *predicted per pixel*, the network can choose, right at a motion boundary, to put all the weight on the neighbors belonging to the correct side — no averaging across the edge. (I scale the mask by $0.25$ before the softmax to keep the gradient magnitudes balanced with the rest of the network.) Where this should pay off is exactly where bilinear hurts most — motion boundaries and small fast objects — so that's what I'd want to check.

Let me also settle the visual-similarity question I glossed: should the operator read correlation, or should I warp $I_2$ features toward $I_1$ by the current flow and read the *warped residual* the way the coarse-to-fine methods do? Warping is the standard. But warping resamples the second image's features by the current (imperfect) flow, and near a motion boundary that resampling distorts the local geometry and can fabricate correspondences that aren't there — Devon avoided warping for exactly this reason. My lookup reads the correlation volume *directly* in the coordinate system of $I_2$, no warping of features, so the geometry near boundaries stays intact. So I'll read correlation rather than warp — and I'd expect the gap to be widest precisely on the harsh-displacement, hard-boundary scenes like KITTI, where warping's fabricated correspondences do the most damage.

Stepping back to make sure the whole machine is coherent and end-to-end. Feature encoder produces $g(I_1), g(I_2)$; matmul gives the all-pairs 4D volume; average-pooling the target axis gives the pyramid; the lookup is bilinear sampling (differentiable); the GRU update is differentiable; the convex upsampling is differentiable; the loss is on the upsampled flow. Every operation has a gradient, so the feature encoder is trained *directly on the final flow error* — not on a surrogate embedding loss the way DCFlow had to, because its SGM cost-volume processing was a dead end for gradients. That is the payoff for insisting the entire pipeline be differentiable: the features learn to be exactly the features that make the *flow* accurate, not the features that minimize some triplet margin and hope.

Let me write the forward pass the way it actually runs. I find it cleaner to represent the flow not as an explicit field but as the difference of two coordinate grids: $\text{coords0}$ is the identity grid (each pixel's own location), $\text{coords1}$ starts equal to it, and the flow is always $\text{coords1} - \text{coords0}$. Then "look up correlation at the current correspondence" is just "look up at $\text{coords1}$," and "apply the update" is $\text{coords1} \mathrel{+}= \Delta\mathbf{f}$ — which is identical to $\mathbf{f}_{k+1} = \mathbf{f}_k + \Delta\mathbf{f}$ since the identity grid is constant. It also makes the warm-start trivial: add the initial flow to $\text{coords1}$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- visual similarity: all-pairs 4D correlation + pyramid + lookup ----------
class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels, self.radius = num_levels, radius
        self.corr_pyramid = []
        corr = CorrBlock.corr(fmap1, fmap2)                 # all-pairs, one matmul
        b, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(b * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr)
        for _ in range(num_levels - 1):                     # pool ONLY the target dims
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):                             # lookup at current correspondence
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        b, h1, w1, _ = coords.shape
        out = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)         # local L-inf grid
            centroid = coords.reshape(b*h1*w1, 1, 1, 2) / 2**i           # same radius, coarser level => larger range
            coords_lvl = centroid + delta.view(1, 2*r+1, 2*r+1, 2)
            corr = bilinear_sampler(corr, coords_lvl)                    # real-valued grid => bilinear
            out.append(corr.view(b, h1, w1, -1))
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        b, dim, ht, wd = fmap1.shape
        f1 = fmap1.view(b, dim, ht*wd)
        f2 = fmap2.view(b, dim, ht*wd)
        corr = torch.matmul(f1.transpose(1, 2), f2)                      # C = g1 g2^T : ALL pairs
        corr = corr.view(b, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())             # scale by sqrt(D)

# ---------- the learned update operator: motion encoder + (separable) ConvGRU + heads ----------
class FlowHead(nn.Module):
    def __init__(self, in_dim=128, hid=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hid, 3, padding=1)
        self.conv2 = nn.Conv2d(hid, 2, 3, padding=1)                    # predicts residual delta-flow
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SepConvGRU(nn.Module):                                            # 1x5 then 5x1 => big RF, few params
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)                                   # horizontal gate (bounded update)
        z = torch.sigmoid(self.convz1(hx)); r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)));  h = (1-z)*h + z*q
        hx = torch.cat([h, x], dim=1)                                   # vertical gate
        z = torch.sigmoid(self.convz2(hx)); r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)));  h = (1-z)*h + z*q
        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super().__init__()
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)          # correlation features
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)                   # flow features
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv   = nn.Conv2d(64+192, 128-2, 3, padding=1)
    def forward(self, flow, corr):
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)

class BasicUpdateBlock(nn.Module):                                      # SAME weights every iteration (tied)
    def __init__(self, cor_planes, hidden_dim=128):
        super().__init__()
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hid=256)
        self.mask = nn.Sequential(                                      # convex-upsampling weights
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
    def forward(self, net, inp, corr, flow):
        motion = self.encoder(flow, corr)
        inp = torch.cat([inp, motion], dim=1)
        net = self.gru(net, inp)                                        # gated, bounded => converges
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)                                     # scale to balance gradients
        return net, mask, delta_flow

# ---------- the whole model ----------
class RAFT(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=128, corr_levels=4, corr_radius=4):
        super().__init__()
        self.hidden_dim, self.context_dim = hidden_dim, context_dim
        self.corr_radius = corr_radius
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')               # per-pixel features (both images)
        self.cnet = BasicEncoder(output_dim=hidden_dim+context_dim, norm_fn='batch')  # context (image1 only)
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        self.update_block = BasicUpdateBlock(cor_planes, hidden_dim=hidden_dim)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        return coords0, coords1                                          # flow = coords1 - coords0

    def upsample_flow(self, flow, mask):                                 # learned convex upsampling
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)                               # convex weights over 3x3 coarse nbrs
        up_flow = F.unfold(8 * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None):
        image1 = 2*(image1/255.0) - 1.0; image2 = 2*(image2/255.0) - 1.0
        fmap1, fmap2 = self.fnet([image1, image2])                      # features for similarity
        corr_fn = CorrBlock(fmap1.float(), fmap2.float(), radius=self.corr_radius)  # all-pairs once

        cnet = self.cnet(image1)                                        # context from image1
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net); inp = torch.relu(inp)                    # init hidden + input features

        coords0, coords1 = self.initialize_flow(image1)                 # flow initialized at 0
        if flow_init is not None:
            coords1 = coords1 + flow_init                               # warm-start (video)

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()                                  # detach f_k branch
            corr = corr_fn(coords1)                                     # lookup at current correspondence
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow                              # f_{k+1} = f_k + delta f
            flow_predictions.append(self.upsample_flow(coords1 - coords0, up_mask))
        return flow_predictions                                        # one prediction per step

# ---------- sequence loss over all iterates ----------
def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    n = len(flow_preds); flow_loss = 0.0
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n):
        w = gamma ** (n - i - 1)                                        # later iterates weighted more
        flow_loss += w * (valid[:, None] * (flow_preds[i] - flow_gt).abs()).mean()  # robust L1
    return flow_loss
```

The causal chain, start to finish: the only thing forcing classical and deep flow into a coarse-to-fine cascade was the small-displacement validity of a linearized data term, and that cascade is what loses small fast objects and can't undo coarse errors — so I refuse the linearization, compute visual similarity for *all* pixel pairs at once as a single matmul (a 4D correlation volume), and pool only its target axis so a cheap local lookup at any single high resolution still carries evidence about displacements hundreds of pixels away while the source resolution keeps small objects sharp; with the data term now a learned, fully-differentiable correlation, there is no explicit energy to descend, so I *learn* the update step — a weight-tied convolutional GRU whose gating bounds the activations and drives the maintained flow field to a fixed point, fed by the correlation lookup, the current flow, and image context so it can regularize within object boundaries — supervise the whole sequence of iterates with a discounted $L_1$ loss, upsample the $1/8$-resolution result with a learned per-pixel convex combination that stays sharp at boundaries, and because every operation is differentiable the feature encoder trains directly on flow error rather than a surrogate, and because every step shares weights I can iterate as many times as I want at test time and warm-start from the previous frame.
