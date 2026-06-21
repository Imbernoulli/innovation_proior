The thing that actually hurts is this: I train a convolutional network on digits, it reads an upright centred digit perfectly, and then I drop the same digit into a random corner of a larger canvas, shrink it, tilt it forty degrees, scatter a few clutter patches around it, and the network stumbles — even though nothing about the digit's *identity* changed, only its position, scale, and orientation, which are exactly uncorrelated with the label. The network is burning capacity learning to tolerate nuisances instead of learning to read. The root cause is architectural. Convolution is translation-equivariant — slide the input and the feature map slides with it — and the only mechanism that turns that equivariance into any *invariance* is max-pooling over a tiny window, typically $2\times2$. One pooling layer is invariant to a one-pixel jitter and nothing more; invariance to a large shift, let alone a change of scale or a rotation, has to be laundered out slowly across a deep stack of alternating convolution and pooling. And when one actually measures how a CNN's intermediate representations respond to a transformed input — fitting the linear map between the representation of an image and the representation of its rotated or scaled copy — the maps turn out simply *not* to be invariant to large transformations. The invariance I was relying on is local, small, and hand-wired into a fixed stencil; the network has no way to reorganise the spatial layout of a feature map in a way that depends on what it is looking at.

The standard remedy is data augmentation: rotate, scale, translate, and warp every training image many ways so the network is forced to average over the nuisance. It works after a fashion, but it is a crutch — it spends data and capacity memorising invariance with no guarantee the transformation space is covered, and at test time it does nothing to actually *fix* a tilted digit; it just hopes it has seen enough tilted ones. That is the wrong shape of solution. I do not want pose *tolerance*, I want pose *normalisation*: un-tilt the digit and hand a clean, upright version to the rest of the network. The alternatives that try to do better each fall short in a specific way. Building invariance into the *filters* — tying weights across a rotation group, or banking scaled filters — bakes a fixed, pre-chosen set of transformations into the feature extractor, pays for each transformation in compute, and cannot decide per image that *this* picture needs a thirty-degree correction while *that* one needs a two-times zoom; it manipulates the detectors, not the data. The capsule and transforming-parts line is closer in spirit, representing objects as parts carrying explicit 2D affine transforms, but it learns those transforms from *supervision* — they are handed to the network as input or target — and I only have the class label. And the attention glimpse models do select a region, but hard-cropping pixels 12-to-40 is not differentiable, so those models reach for high-variance reinforcement learning; the one differentiable attention I know reads through a grid of axis-aligned Gaussian kernels, which is a translation-and-scale window, not rotation, shear, or a free-form warp.

I propose the Spatial Transformer: a single, self-contained, differentiable module that looks at the incoming feature map, decides a spatial transformation conditioned on that map, and applies the transformation to the whole map to produce a warped output — with the entire chain differentiable, so the gradient of the ordinary task loss flows back through it and teaches it what transformation to pick, with no transformation labels at all. Because it is differentiable, it is just another layer: drop it anywhere, train it with the same SGD, and the "how to warp this input" knowledge gets cached in its weights. It has three pieces, in computation order. A *localisation network* maps the input feature map $U \in \mathbb{R}^{H\times W\times C}$ to transformation parameters $\theta$ through some hidden layers ending in a regression head — fully connected or convolutional, it does not matter, as long as the last layer emits the right number of parameters (six for affine, three for attention, eight for projective). A *grid generator* turns those parameters into a source coordinate for every output pixel. And a *differentiable sampler* reads $U$ at those source coordinates to produce the warped output $V$.

The hard word in all of this is *differentiable*, and everything reduces to making "warp a feature map by a predicted geometric transformation" a smooth function of the predicted parameters. The naive forward warp breaks immediately: pushing each input pixel to its mapped location $q = \mathrm{transform}(p)$ produces collisions when zooming out and holes when zooming in, and the rounding of $q$ to an integer output cell is a step function with no gradient in the parameters. So I go the other way, exactly as image resampling in graphics does — output→input. Lay down a fixed regular grid of output pixel locations and, for each, compute the *source* location in the input to read from. Every output pixel gets exactly one source coordinate: no holes, no collisions, the output is completely and uniquely defined. For the affine case, writing an output grid point as $(x_t^i, y_t^i)$ and the source point as $(x_s^i, y_s^i)$, the transform maps output to source as
$$\begin{pmatrix} x_s^i \\ y_s^i \end{pmatrix} = A_\theta \begin{pmatrix} x_t^i \\ y_t^i \\ 1 \end{pmatrix}, \qquad A_\theta = \begin{bmatrix} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{bmatrix},$$
i.e. $x_s = \theta_{11} x_t + \theta_{12} y_t + \theta_{13}$ and $y_s = \theta_{21} x_t + \theta_{22} y_t + \theta_{23}$. The appended $1$ and third column give the translation $(\theta_{13}, \theta_{23})$; the left $2\times2$ block does rotation, scale, and shear. A crucial design choice is the coordinate system: I use height- and width-normalised coordinates with $-1 \le x_t, y_t \le 1$ spanning the output and $-1 \le x_s, y_s \le 1$ spanning the input, rather than pixel units. In pixels, $A_\theta$ would mean a different geometric thing at $28\times28$ than at $42\times42$, and a module dropped at a different resolution would behave differently — brittle. In normalised coordinates the identity is just $A_\theta = [[1,0,0],[0,1,0]]$ at any resolution, and the same six numbers mean the same transformation everywhere. As a bonus, cropping falls out for free: if the left $2\times2$ block is a contraction with $|\det| < 1$, the mapped grid lands inside a sub-region of the input, so the output reads from a small patch — a crop-and-zoom, the attention behaviour I wanted, with no separate mechanism. The affine is not a limitation either: constrain it to $A_\theta = [[s,0,t_x],[0,s,t_y]]$ for three-parameter attention, or generalise as $T_\theta = M_\theta B$ with $M_\theta$ built from $\theta$ and a target-grid representation $B$ (the regular grid in homogeneous coordinates for the affine, but learnable in general), covering projective, piecewise-affine, and thin-plate-spline warps. The only requirement the whole scheme imposes is that $(x_s, y_s)$ be a differentiable function of $\theta$, because that is the link the gradient must cross.

The real obstacle is the read itself: the grid generator hands me a *fractional* source coordinate such as $(12.3, 7.8)$, but the input is defined only at integer pixel locations, and I need the value there as a differentiable function of $x_s$ and $y_s$. Write the read as a sampling sum over the input with a separable kernel,
$$V_i^c = \sum_n \sum_m U_{nm}^c \, k(x_s^i - m;\, \Phi_x)\, k(y_s^i - n;\, \Phi_y),$$
where $U_{nm}^c$ is the input at integer location $(m,n)$ in channel $c$ and the same warp is applied to every channel, which keeps the channels spatially aligned. The obvious kernel, nearest neighbour, is fatal: rounding $x_s$ to the nearest integer makes $V_i$ flat as $x_s$ moves within a pixel and discontinuous when it crosses a half-integer, so $\partial V_i / \partial x_s$ is zero almost everywhere and undefined at the jumps — there is no signal to tell the localisation network which way to slide the sampling point, which kills the very gradient the inverse-mapping machinery was built to obtain. The fix is bilinear interpolation, which makes the read vary continuously as $x_s$ moves between pixels: each neighbouring pixel is weighted by $\max(0,\,1 - |x_s - m|)$, which is $1$ when $x_s = m$, falls linearly to $0$ one pixel away, and is $0$ beyond. The sampler becomes
$$V_i^c = \sum_n \sum_m U_{nm}^c \, \max\!\big(0,\, 1 - |x_s^i - m|\big)\, \max\!\big(0,\, 1 - |y_s^i - n|\big),$$
which in practice is a four-tap read (only the four pixels surrounding $(x_s, y_s)$ get nonzero weight) and is piecewise-linear in $x_s$ and $y_s$, so it has a gradient. Working the gradients out is the crux. The sampler is linear in $U$, so
$$\frac{\partial V_i^c}{\partial U_{nm}^c} = \max\!\big(0,\, 1 - |x_s^i - m|\big)\, \max\!\big(0,\, 1 - |y_s^i - n|\big)$$
— the bilinear weight itself, which flows gradient back to whatever produced the input feature map, i.e. to layers *before* the module. For the source coordinate, holding the $y$ factor fixed and differentiating the $x$ factor, the term is clamped at zero outside its support, while inside it equals $1 - |x_s - m|$ with derivative $-\mathrm{sign}(x_s - m)$, which is $+1$ when $m \ge x_s$ and $-1$ when $m < x_s$:
$$\frac{\partial V_i^c}{\partial x_s^i} = \sum_n \sum_m U_{nm}^c \, \max\!\big(0,\, 1 - |y_s^i - n|\big) \cdot \begin{cases} 0 & |m - x_s^i| \ge 1 \\ +1 & |m - x_s^i| < 1 \text{ and } m \ge x_s^i \\ -1 & |m - x_s^i| < 1 \text{ and } m < x_s^i \end{cases}$$
with $\partial V_i / \partial y_s$ symmetric; the absolute value has a kink at $x_s = m$, where I simply take a sub-gradient, which SGD is perfectly happy with. Closing the loop to $\theta$ is then immediate from the affine, $\partial x_s / \partial \theta_{11} = x_t$, $\partial x_s / \partial \theta_{12} = y_t$, $\partial x_s / \partial \theta_{13} = 1$, and symmetrically for $y_s$, so the loss gradient chains all the way through,
$$\frac{\partial L}{\partial \theta} = \sum_i \frac{\partial L}{\partial V_i}\, \frac{\partial V_i}{\partial x_s^i}\, \frac{\partial x_s^i}{\partial \theta} + (\text{same with } y),$$
and since $\theta$ is just the output of an ordinary network the rest is standard backprop. The chain is unbroken from the task loss to the parameter-predicting network: no transformation labels anywhere, the geometry is learned purely because warping toward a canonical pose lowers the downstream loss. (The 3-D extension simply adds a $\max(0, 1 - |z_s - l|)$ factor and a $3\times4$ affine.)

Two practical choices make it actually train. First, there is an initialisation trap: a randomly initialised regression head predicts an arbitrary $\theta$ on the very first pass and warps the input into garbage — a wild rotation or a near-zero crop — before the rest of the network has learned anything, returning useless gradients. So I initialise that final regression layer to emit the identity transform: weights zero, bias $[1,0,0,0,1,0]$ so $A_\theta = [[1,0,0],[0,1,0]]$. The module then starts as a no-op, the host network trains like a perfectly ordinary CNN, and the module only *gradually* learns to deviate from identity as deviating helps — and normalised coordinates are what make this clean, since the identity bias is the same six numbers at any resolution. Second, the transform parameters are enormously high-leverage — a small change in $\theta$ can swing the sampled region across the whole image — so I run the localisation network at a lower learning rate than the rest of the network (around a tenth, far less on top of a large pretrained backbone) to keep it from overshooting. The payoff is architectural freedom: the module is cheap — a small localisation net plus a four-tap read per output pixel, a few percent overhead — so I can place one at the input to pose-normalise the raw image, several at increasing depth to warp progressively abstract feature maps, or several in parallel on the same map so each locks onto a different object or part (a bird's head, its body) with the outputs concatenated. Letting the output grid be smaller than the input crops *and* downsamples in one step, so I can attend to a small region of a high-resolution image and pass only a small canonical crop downstream — attention that saves computation rather than costing it (with mild aliasing if the downsampling factor is large, since four taps cannot average a big region). In the implementation the affine grid and bilinear sampler are the standard PyTorch primitives `F.affine_grid` (which builds the normalised output grid and applies $A_\theta$ output→input) and `F.grid_sample` (the bilinear read); the `align_corners` convention only fixes the boundary coordinate convention, so I set it explicitly rather than leave it to a version default.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # recognition network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # --- Spatial Transformer ---
        # localisation network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        # regress the 6 affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )
        # initialise to the identity transform (start as a no-op)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs).view(-1, 2, 3)     # A_theta per sample
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        x = self.stn(x)                            # warp into canonical pose
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

Trained with ordinary SGD on the task loss only:

```python
def train(model, loader, optimizer):
    model.train()
    for data, target in loader:
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()        # gradient reaches f_loc via the sampler sub-gradients
        optimizer.step()
```
