The central challenge is to learn the solution operator of a parametric family of PDEs, the map G† : a ↦ u that takes an input function such as a coefficient, forcing, or initial condition and returns the corresponding solution. Classical solvers produce one solution per expensive mesh-based run, so they cannot amortize cost across thousands of related instances. Finite-dimensional neural surrogates such as fully convolutional networks do amortize, but they commit to a fixed discretization: their filters are tied to the training grid spacing, and their error grows when the mesh is refined or changed. Physics-informed networks are mesh-free, yet they fit a single instance at a time and require a fresh optimization for every new input. What is missing is a learned operator that lives in function space, evaluates cheaply at inference, and transfers across resolutions without retraining.

The right primitive is a kernel integral operator, because the solution operator of a linear PDE already has exactly that form through its Green's function. A general learned integral operator, however, couples every pair of evaluation points and costs O(N²) per layer, which is prohibitive on the fine grids needed for turbulence. The key observation is that for constant-coefficient problems the Green's function is translation invariant, so the kernel depends only on the displacement x − y and the integral becomes a convolution. The convolution theorem then turns that expensive spatial coupling into a pointwise multiplication in Fourier space, which the Fast Fourier Transform computes in O(N log N).

The method is the Fourier Neural Operator, or FNO. It builds a learnable nonlinear operator between function spaces by alternating a global Fourier integral layer with a local pointwise linear map and a pointwise nonlinearity. The architecture first lifts the input function to a higher-dimensional channel representation with a shallow pointwise network, then iterates layers of the form v ↦ σ(Wv + Kv), and finally projects back to the output function space. The Fourier layer is the central piece. It transforms the representation to Fourier space with the FFT, keeps only a fixed block of low modes, multiplies each retained frequency by a learned complex matrix that mixes channels, and maps back with the inverse FFT. Because the parameters live in Fourier space and the retained mode block is fixed independently of the grid resolution, the same operator resolves consistently on any mesh, giving zero-shot super-resolution. The pointwise W branch, implemented as a 1×1 convolution, runs in physical space and carries local structure and non-periodic boundaries that the periodic Fourier branch cannot represent. The nonlinearity is applied in physical space, which is essential: a pointwise nonlinearity there regenerates high-frequency content that truncation removes, whereas a nonlinearity applied directly to Fourier coefficients would correspond to a spatial convolution and could not create new high modes.

Truncation to a small low-mode block is justified by the spectral decay of the data. PDE solution fields tend to concentrate energy in low frequencies, and the physical-space nonlinearities between layers give the full network enough expressive power to recover finer structure even though each individual Fourier layer is band-limited. The result is a model whose layer complexity is quasi-linear in the number of grid points and whose parameter count is resolution-independent. Training is standard: minimize the relative L² error, which measures the function-space norm of the residual, using Adam with weight decay and a step learning-rate schedule.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Fourier layer: rfft -> keep low modes -> complex channel-mix -> irfft."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        b = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1, padding=9):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for conv, w in [(self.conv0, self.w0), (self.conv1, self.w1),
                        (self.conv2, self.w2), (self.conv3, self.w3)]:
            x = conv(x) + w(x)
            if conv is not self.conv3:
                x = F.gelu(x)
        if self.padding:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        b, sx, sy = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, sx, device=device).reshape(1, sx, 1, 1).repeat(b, 1, sy, 1)
        gy = torch.linspace(0, 1, sy, device=device).reshape(1, 1, sy, 1).repeat(b, sx, 1, 1)
        return torch.cat((gx, gy), dim=-1)


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
