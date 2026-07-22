Let me start from the thing that actually hurts. I keep having to solve the same PDE again and again. Designing an airfoil, I solve the flow equations for one shape, nudge the shape, solve again — thousands of times. Calibrating a subsurface model, I propose a permeability field, solve for the pressure, compare to data, propose another field, solve again. Every one of these is a finite-element or finite-difference or pseudospectral solve on a fine mesh, and each is completely independent of the last. The solver knows nothing carried over from the previous run. And the meshes have to be fine: the phenomena I care about — boundary layers, turbulence — only show up when the discretization is fine enough to resolve them, and then a single solve is slow. For the 2-D incompressible flow I have in mind, even a tuned pseudospectral solver is around two seconds per instance on a GPU. Multiply by thirty thousand for a Bayesian inversion and I'm looking at the better part of a day for one experiment.

So what is the object I'm really after? Not a single solution. I want the *map*. For a parametric PDE there's a coefficient or initial condition — call it `a`, a function on a domain `D ⊂ R^d` — and a solution `u`, another function on `D`. The solver implements

    G† : a ↦ u,

a map between two spaces of functions, and that map is the same for the whole family. If I could learn `G†` once, then each new `a` is a single forward evaluation instead of a fresh solve. That's the prize: amortize the cost of the family across one training run.

The catch is that `a` and `u` are *functions*, living in infinite-dimensional spaces — say separable Banach spaces `A = A(D; R^{d_a})` and `U = U(D; R^{d_u})`. Most of what I know how to train maps a fixed-length vector to a fixed-length vector. I could discretize `D` on a grid of `n` points, stack the samples `a(x_1),…,a(x_n)` into a vector in `R^n`, and learn a map `R^n → R^n` with a convolutional network. People do exactly this, treating the field as an image and doing image-to-image regression. And it's fast at inference and it amortizes across instances, so it solves part of my problem. But it has a defect that I can't live with: it's welded to the grid. The learned filters encode a particular grid spacing. If I train at one resolution and evaluate at another, the error doesn't stay put — it drifts, and in fact for these problems it *grows* as I refine the test grid. I've seen the pattern: a fully-convolutional surrogate on the 1-D Burgers map sits around ten percent relative error at a 256-point grid and climbs past thirty percent by 8192 points. That's backwards. Refining the grid should give the *truth* more room, not make my model worse. The model isn't approximating an operator; it's approximating one finite-dimensional slice of it and falling apart off that slice. I can't query at a point that isn't on the training grid, I can't transfer between meshes. This is the wall with finite-dimensional surrogates: they are, by definition, discretization-dependent.

There's a different neural approach that's mesh-free: represent the *solution itself* as a neural network `u_θ(x)` and fit `θ` by driving down the PDE residual at sampled points — essentially replacing the finite-element basis with the space of networks. That's genuinely mesh-independent and accurate. But it models *one* instance. Hand it a new coefficient field and it has to solve a fresh optimization problem from scratch — same per-instance cost as the classical solver, just with a different inner loop. And it needs the equation written down. That's not an operator either; it's a per-instance solver wearing a neural coat.

So I want the best of both: mesh-free like the residual networks, amortized like the convolutional surrogate. The non-negotiable consequence is that I must define my model as a map between *function spaces*, and only discretize at the very end to touch data. If the parameters live in function space and are consistent in the continuum, then any grid is just a way of *resolving* the same operator, rather than a new input dimension that changes the model itself.

Now, how do I even parameterize a map between function spaces? Let me reason by analogy with an ordinary network, which alternates a linear map `Wv` with a pointwise nonlinearity `σ`. The nonlinearity carries over to functions trivially — apply `σ` pointwise, `σ(v)(x) = σ(v(x))`. The hard part is the linear map. In finite dimensions a linear map is a matrix-vector product, `(Wv)_i = Σ_j W_{ij} v_j`. What is the continuum analogue of summing `W_{ij} v_j` over `j`? It's integrating a kernel against the function:

    (K v)(x) = ∫_D κ(x, y) v(y) dy.

That's the natural infinite-dimensional linear operator — a kernel integral operator, with `κ(x,y)` playing the role of the matrix entry coupling location `x` to location `y`. Before I commit to it I want to check whether this is just a formal analogy or whether it actually shows up in the problem I'm solving. So let me look at what the solution operator of a linear PDE actually is. If `L u = f` with `L` linear, the solution is built from the Green's function `G`:

    u(x) = ∫_D G(x, y) f(y) dy.

The Green's function is the response at `x` to a unit point source at `y`, and linearity (superposition) says the response to a general source is the integral of the responses to all the point sources. So the solution operator of a linear PDE *is* a kernel integral operator, with the Green's function as its kernel. That settles the question for me: the integral operator isn't an arbitrary choice imported from finite-dimensional networks — for the linear case it's the literal form the answer takes. My operators are nonlinear, so this isn't a proof that the same primitive suffices for them, but it tells me the integral operator is at least the correct linear backbone to wrap nonlinearities around.

The wrapping I just gestured at is the same trick that lets ordinary networks approximate nonlinear functions out of linear pieces, and I'll need it: my operators are nonlinear (even for Darcy flow, where the equation is linear in `u`, the coefficient-to-solution map `a ↦ u` is not). So I'll build an iterative architecture. Lift the input function `a` to a higher-dimensional representation `v_0(x) = P(a(x))` with a local (pointwise-in-`x`) map `P` — a shallow network acting on the channel dimension, raising it to some width `d_v`. Then iterate updates `v_t ↦ v_{t+1}`, and at the end project back, `u(x) = Q(v_T(x))`, with another local map `Q`. Lifting first gives the per-layer linear operator a fat channel space to mix within, which is where its expressive power will come from; one scalar channel wouldn't have room.

What's the update? Compose the global linear integral operator with a pointwise nonlinearity. But a pure integral operator plus nonlinearity can blur away information that should stay purely local: boundary values, local coefficients, channel-wise corrections at the same point. I want a direct path that can transform the channel vector at `x` without forcing it through an all-to-all integral. So I also add a pointwise linear map `W` acting on the channel vector at each `x`. The update is

    v_{t+1}(x) = σ( W v_t(x) + (K(a) v_t)(x) ),    x ∈ D.

Here `W : R^{d_v} → R^{d_v}` is an ordinary matrix applied at each point, `σ` is a pointwise nonlinearity, and `K(a)` is the kernel integral operator, which I'll let depend on the input function `a`:

    (K(a) v_t)(x) = ∫_D κ(x, y, a(x), a(y)) v_t(y) dy,

with `κ` a learned neural network mapping `(x, y, a(x), a(y))` to a `d_v × d_v` matrix (it has to be a matrix, because it mixes the `d_v` channels of `v_t(y)` into the `d_v` channels of the output, just like `W_{ij}` was a scalar mixing scalar entries — here each "entry" is itself a channel vector). Stack a few of these layers and I have a learnable, mesh-free operator between function spaces. The kernel `κ` is shared across all `x` — it's a single finite set of parameters, a network whose weights say nothing about `n` — so the same `κ` can be evaluated against whatever set of sample points I'm handed. That is the property I was missing from the convolutional surrogate: the parameters here don't carry a grid spacing in them. Whether that translates into errors that actually stay flat under refinement is something I can only test once it runs, but at least the model is no longer defined on one fixed grid.

Now I try to actually run it and I hit the wall. Look at the cost of one layer. To get the output at a single point `x` I integrate over all of `D`. Discretize `D` with `n` points and the integral becomes a quadrature sum over all `n` of them. I need the output at all `n` points. That's `n` outputs, each a sum over `n` inputs: `O(n²)` kernel evaluations per layer. And each kernel evaluation is a forward pass of the network `κ`, producing a `d_v × d_v` matrix. For the coarse grids it's tolerable, but the whole point was fine grids — turbulence — and there `n²` is hopeless. This is exactly why this style of operator, evaluated by summing the kernel against every pair of points (message passing over a graph on the sample points), has been accurate but slow, and on the turbulent regime it just doesn't get there. The integral operator is the right object and the wrong bill. I need the same operator computed in near-linear time.

Stare at the integral. The expensive thing is that `κ(x, y, …)` depends on `x` and `y` separately, so there's no structure to exploit — every pair is its own computation. What if I throw away the dependence I don't strictly need and impose structure? Two moves. First, drop the explicit dependence on `a(x), a(y)` inside the kernel for now (I'll keep `a`'s influence through the lifting `P` and can add it back later if it helps). Second, assume the kernel is *translation invariant*:

    κ(x, y) = κ(x − y).

Why is this not just a convenient hack? Because that's precisely the structure the Green's function has when the differential operator has constant coefficients: the response depends only on the displacement `x − y`, not on the absolute positions. It's the same reason a physical impulse response is the same wherever you poke it. So I'm not pulling a restriction out of nowhere; I'm specializing to the translation-invariant case, the one the Green's function lives in for constant-coefficient `L`. It does give something up — genuinely spatially varying problems aren't constant-coefficient — but I'll see whether the local `W` branch and the channel mixing can absorb that, and the payoff is structure I can compute against. With `κ(x,y) = κ(x−y)` the integral becomes

    (K v)(x) = ∫_D κ(x − y) v(y) dy = (κ * v)(x),

a convolution. That hasn't made it cheaper yet — a convolution done directly is still `O(n²)`. But a convolution is exactly the thing the Fourier transform simplifies.

Let me write the Fourier coefficients for a function `f : D → R^{d_v}` on a periodic box, componentwise in the channel index `j`,

    (F f)_j(k) = ∫_D f_j(x) e^{−2πi⟨x,k⟩} dx,    f_j(x) = Σ_{k∈Z^d} (F f)_j(k) e^{2πi⟨x,k⟩},

with `i = √(−1)`. The convolution theorem says the Fourier transform turns convolution into pointwise multiplication: for each frequency `k`,

    F(κ * v)(k) = F(κ)(k) · F(v)(k).

Let me convince myself, because everything rides on it. Write `(κ * v)(x) = ∫ κ(x−y) v(y) dy` and Fourier transform in `x`:

    F(κ*v)(k) = ∫_x [∫_y κ(x−y) v(y) dy] e^{−2πi⟨x,k⟩} dx.

Substitute `z = x − y`, so `x = z + y` and `dx = dz`:

    = ∫_y ∫_z κ(z) v(y) e^{−2πi⟨z+y,k⟩} dz dy
    = (∫_z κ(z) e^{−2πi⟨z,k⟩} dz)(∫_y v(y) e^{−2πi⟨y,k⟩} dy)
    = F(κ)(k) · F(v)(k).

The exponential factors because `⟨z+y,k⟩ = ⟨z,k⟩ + ⟨y,k⟩`, and the double integral separates. So the convolution that cost `O(n²)` in physical space is, in Fourier space, a *pointwise* product across frequencies. Then

    (K v)(x) = (κ * v)(x) = F⁻¹( F(κ) · F(v) )(x).

I don't need to form `κ` in physical space and transform it. The only thing that enters is its Fourier transform `F(κ)`, frequency by frequency. So let me just *make that the learnable object*. Call it `R := F(κ)` and parameterize it directly in Fourier space:

    (K v)(x) = F⁻¹( R · F(v) )(x).

I never touch `κ(x−y)` at all. I learn a multiplier `R` that acts on the Fourier coefficients of `v`. The integral operator has turned into: Fourier transform `v`, multiply by `R` per frequency, transform back. Per frequency, `(F v)(k) ∈ C^{d_v}` is a channel vector and `R(k) ∈ C^{d_v × d_v}` is a complex matrix — exactly the channel-mixing role the matrix `κ` had, but now indexed by frequency instead of by position. So the structure I gave up (position-by-position kernel) reappears as freedom across frequencies, which is the natural place for it to live for a convolution.

Now, how many frequencies? `D` is bounded, and I'm assuming `κ` is periodic on it, so it has a Fourier *series* — the frequencies are discrete, `k ∈ Z^d`. If I kept all of them I'd have as many modes as grid points, and the parameter count would track the resolution again, which defeats the whole purpose. So I truncate: keep a small rectangular block of low modes. In the implementation I will treat `m_j` as a retained-mode count on axis `j`, not as an inclusive upper bound; on a full FFT axis that means indices `0,…,m_j−1` and the wrapped negative frequencies `s_j−m_j,…,s_j−1`. Abstractly I can call the retained block `Z_m`.

    Z_m ≈ ∏_j {−m_j,…,−1,0,…,m_j−1},    with |Z_m| independent of the grid resolution.

Is truncation defensible or am I just throwing away signal? I shouldn't take "the spectrum decays" on faith — I should measure what a low-mode cutoff actually costs on a field that looks like my data. Let me do that. Take a periodic 1-D field on `N = 256` points with a decaying spectrum — amplitudes `~(1+k²)^{−1}`, random phases, the kind of smoothness a Gaussian-random-field solution has. Transform it, zero everything outside a low-mode block keeping `m` modes per side (`0:m` and the wrapped negatives), transform back, and measure the relative `L²` error of the reconstruction against the original. The numbers I get:

    keep  4 modes/side -> rel L2 error 0.074
    keep  8 modes/side -> rel L2 error 0.026
    keep 12 modes/side -> rel L2 error 0.013
    keep 20 modes/side -> rel L2 error 0.005

So at twelve modes per side the reconstruction is already within about 1.3% of the original field, and at twenty it's half a percent — the tail genuinely carries little energy. That answers the worry concretely: the low modes do carry most of what matters here, the truncation is not throwing away signal so much as discarding noise, and a fixed low-mode budget is a defensible parameterization, not a desperate one. As a bonus it acts as a low-pass regularizer and gives a finite, resolution-independent parameter count in one stroke. Second — and this is the part that initially worried me — even though each `K` layer only outputs the retained modes, the *network as a whole* is not band-limited, because `σ` sits between the layers in physical space, and a pointwise nonlinearity in physical space spreads energy across frequencies — it manufactures new high modes out of products of low ones. (This is also why `σ` must live in physical space and not in Fourier space: a pointwise nonlinearity applied to Fourier coefficients would correspond to a *convolution* in physical space, which is meaningless as an activation and can't regenerate high frequencies.) And the final projection `Q`, being nonlinear, adds more. So I can use a small fixed budget per layer — twelve retained modes per spatial axis is a reasonable 2-D starting point — and still leave the full network a path to produce sharper physical-space structure.

So `R` is a complex tensor with one `d_v × d_v` channel-mixing matrix per retained frequency, and I drop the kernel notation entirely; `R` *is* the parameter. The physical representation is real-valued, so a full complex spectrum would have Hermitian symmetry. In code I do not store the full spectrum and tie every conjugate pair by hand; I store the one-sided real-FFT spectrum and let the inverse real FFT reconstruct the missing half along the last axis. On axes that the real FFT does not halve, I still need both the low positive and wrapped negative corners, and the implementation learns those corner tensors directly.

Now make it concrete on a grid, because that's where I touch data. Discretize `D` with `n` points, so `v_t ∈ R^{n × d_v}` and its discrete Fourier transform `F(v_t) ∈ C^{n × d_v}`. Since I only multiply by a multiplier supported on the retained block, I just *slice off* those low modes and discard the rest. The per-frequency multiply, written in indices with `k` the frequency, `l` the output channel, `j` the input channel, is

    ( R · (F v_t) )_{k,l} = Σ_{j=1}^{d_v} R_{k,l,j} (F v_t)_{k,j},    k ∈ Z_m,  l = 1,…,d_v.

That's a small dense `d_v × d_v` matrix-vector product done independently at each retained frequency — a batched matrix multiply, trivial to vectorize. On a uniform grid I compute `F` with the FFT. For a uniform discretization of resolution `s_1 × … × s_d = n`, the discrete transform and inverse are

    (F f)_l(k) = Σ_{x_1=0}^{s_1−1} … Σ_{x_d=0}^{s_d−1} f_l(x) e^{−2πi Σ_j x_j k_j / s_j},
    (F⁻¹ f)_l(x) = (1/n) Σ_{k_1=0}^{s_1−1} … Σ_{k_d=0}^{s_d−1} f_l(k) e^{+2πi Σ_j x_j k_j / s_j}.

where `n = s_1⋯s_d`; this is the normalization convention used by the default PyTorch FFT pair, and with another convention the constant can be absorbed into `R`.

One subtlety in indexing the retained modes on the FFT grid. The FFT returns frequencies in the order `0, 1, …, s/2, …, −2, −1` wrapped around, so a negative frequency `−m` lives at index `s − m`. A retained count `m_j` on a full FFT axis therefore means the two slices `0:m_j` and `s_j−m_j:s_j`. With a real FFT, the last axis is one-sided, so I keep `0:m_d` there and let `irfft` reconstruct the missing negative last-axis modes. The retained block therefore appears as corners of the stored array: two corners in 2-D, four in 3-D.

That wrapping is exactly the kind of thing I'll get wrong by an index if I don't check it, and the whole layer is silently broken if the corners pick the wrong frequencies. So let me trace it on a tiny grid before I trust it. Take `s = 8` on the first axis and feed in two pure cosines, one at frequency `k1 = 1` (a low mode I want to keep) and one at `k1 = 3` (a mode I want to drop with `m = 2`). Run `rfft2` and read off where the energy lands on axis 1:

    k1=1 (low)  -> nonzero axis-1 indices: [1, 7]
    k1=3 (high) -> nonzero axis-1 indices: [3, 5]

So the `k1=1` field shows up at index `1` and at index `7 = 8 − 1` (its conjugate `−1`), and the `k1=3` field at indices `3` and `5 = 8 − 3`. Now apply the two corner slices `[:2]` (indices 0,1) and `[-2:]` (indices 6,7) with `m = 2`, identity weights, and inverse-transform, then measure how much of each input survives, `||out|| / ||in||`:

    k1=1 (low):  survived fraction = 1.000
    k1=3 (high): survived fraction = 0.000

The low mode at indices 1 and 7 is caught exactly by the two corners and passes through untouched; the high mode at 3 and 5 falls in the gap between the corners and is annihilated. That is precisely the band-limiting I wanted, and it confirms the wrap-around indexing — `0:m` for the positives, `s−m:s` for the negatives — is right rather than off by one. (Had I written `s−m+1:s` or forgotten the negative corner, the `k1=1` survived fraction would have come out below one, since index 7 carries half its energy.)

I could instead define "low modes" by an `ℓ¹` bound on the wave number — that's the more canonical notion of low frequency — but the box/corners version is what slices cleanly out of the array for a parallel matmul, so I take the box.

What does this cost? The per-frequency multiply touches `|Z_m|` modes and does a `d_v × d_v` channel mix at each one, so the multiply is `O(|Z_m| d_v²)` with `|Z_m|` fixed by design. The transforms dominate: a naive DFT is `O(n²)`, but because I truncate I only ever need `|Z_m|` output modes, so even a direct transform is `O(n |Z_m|)`, and on a uniform grid the FFT does the whole transform in `O(n log n)`. That's the win I was after — the `O(n²)` integral has become quasi-linear, `O(n log n)`, for fixed width and fixed mode budget. The price is that the FFT wants a uniform grid; that's the one place I've narrowed the generality, and it's a fair trade for turbulence-scale `n`.

And the discretization-invariance I demanded at the start falls out for free. The parameters are `R`, living in Fourier space, with a fixed retained block that has nothing to do with `n`. To evaluate the operator at some resolution I just resolve the Fourier basis `e^{2πi⟨x,k⟩}` on whatever grid I'm handed — those basis functions are defined everywhere on `R^d`, at any `x`. So I can train at one resolution and evaluate at a finer one without retraining and without ever having seen fine data: zero-shot super-resolution. The same `R` is a *consistent* discretization of one continuum operator, so it should not inherit the finite-dimensional convolutional surrogate's resolution-change failure mode.

One more thing about `W`. The Fourier branch is periodic by construction — the inverse transform builds a periodic function. Real problems have non-periodic boundaries (the Darcy square, the time axis of the flow). The pointwise `W v_t(x)` branch, running alongside in physical space, is not constrained to be periodic, so it carries the boundary and the strictly-local content that the periodic convolution branch can't represent. So the two branches are complementary: `W` is the local/bias term keeping track of non-periodic structure, the Fourier term is the global translation-invariant operator. Adding `σ` on top makes the whole update nonlinear. This is the analogue, layer for layer, of "linear map plus activation," but the linear map is a global convolution realized as a multiply in Fourier space.

Let me also reconsider whether `R` should depend on `a`, to mirror the original `κ(x,y,a(x),a(y))`. I can let `R` be a function `R(k, (F a)(k))` — a parametric map from the frequency and the input's Fourier coefficient to the multiplier. A linear dependence is possible, and a small network is possible too. But `Z^d` is a discrete, unstructured index set, so a network reading `k` directly has little smooth geometry to exploit; a linear dependence adds cost while the lifted representation `v_0 = P(a)` already carries the input into every layer. So the simplest thing — learn `R` directly per retained mode, with no explicit `a`-dependence — is the right default, and `a` enters through the lifting `P`.

Now I can write the whole forward pass and it should match the architecture I reasoned to. Lift with `P`, run a few layers each of which adds the Fourier branch and the `W` branch and applies `σ` between layers, project with `Q`. Let me write the two-dimensional layer because it shows the indexing issue that matters in practice. The input is real, so I use the real FFT; it halves only the last axis, while the first axis still stores both its low-positive and wrapped negative frequencies. I allocate a zero one-sided spectrum, fill the two retained corners with the multiplied values, and inverse-transform.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        # one R per retained corner (the rfft halves the last axis; the first axis keeps
        # both its low-positive and its high-negative frequencies)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, weights):
        # (batch,in,x,y),(in,out,x,y) -> (batch,out,x,y)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        b = x.shape[0]
        x_ft = torch.fft.rfft2(x)  # (b, c, s1, s2//2+1)
        out_ft = torch.zeros(b, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                             device=x.device, dtype=torch.cfloat)
        # low-positive corner on axis 1
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # high (negative) corner on axis 1
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
```

Then I write the full operator network: lift, four layers of (Fourier branch `+` pointwise `W`, then `σ` between them), project. The pointwise `W` is just a 1×1 convolution — a per-point linear map on channels. I concatenate the grid coordinates onto the input so the pointwise maps have positional information, and I can pad the domain when the boundary is non-periodic.

```python
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1, padding=9):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding                    # pad if the domain is non-periodic
        self.fc0 = nn.Linear(in_channels + 2, width)  # lift P: values plus (x, y) coordinates
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)      # pointwise-in-space linear W (the local/bias branch)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)          # projection Q
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)   # append coordinates as channels
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)       # channels-first for the conv/FFT ops
        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for conv, w in [(self.conv0, self.w0), (self.conv1, self.w1),
                        (self.conv2, self.w2), (self.conv3, self.w3)]:
            x1 = conv(x)                          # K v: global Fourier branch
            x2 = w(x)                             # W v: local pointwise branch
            x = x1 + x2
            if conv is not self.conv3:            # nonlinearity in physical space, not after the last
                x = F.gelu(x)
        if self.padding:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))                   # Q
        return self.fc2(x)

    def get_grid(self, shape, device):
        b, sx, sy = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, sx, device=device).reshape(1, sx, 1, 1).repeat(b, 1, sy, 1)
        gy = torch.linspace(0, 1, sy, device=device).reshape(1, 1, sy, 1).repeat(b, sx, 1, 1)
        return torch.cat((gx, gy), dim=-1)
```

If I switch to one dimension, I keep only `out_ft[:, :, :modes1]`; if I move to space-time, I use `rfftn` and four retained corners because two axes remain full. Training is unremarkable and that's the point — the operator is just a module. I fit it with a relative `L²` loss, which is the natural function-space error (norm of the difference over norm of the target), Adam, and a step-halving schedule. When the fields are normalized, I decode both prediction and target before computing that relative norm, because the loss should measure error in the physical units of the function.

```python
def relative_l2(pred, target):
    diff = (pred - target).reshape(pred.size(0), -1).norm(dim=1)
    return (diff / target.reshape(target.size(0), -1).norm(dim=1)).mean()

def train(model, train_loader, y_normalizer=None, epochs=500, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    for _ in range(epochs):
        for a, u in train_loader:
            opt.zero_grad()
            pred = model(a).reshape_as(u)
            target = u
            if y_normalizer is not None:
                pred = y_normalizer.decode(pred)
                target = y_normalizer.decode(target)
            loss = relative_l2(pred, target)
            loss.backward()
            opt.step()
        sched.step()
```
