# DCGAN

## Method

DCGAN is the original GAN objective with a constrained convolutional architecture and tuned training recipe.

```text
min_G max_D V(D, G)
  = E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))].
```

Train `D` by minimizing binary cross-entropy on real images labelled `1` and generated images labelled `0`. Train `G` with the non-saturating update, minimizing binary cross-entropy on generated images labelled `1`, which is the same as maximizing `log D(G(z))`.

Architecture rules:

- Replace pooling with strided convolutions in `D` and transposed convolutions in `G`.
- Remove fully connected hidden layers. Project `z` into a spatial tensor at the generator input; collapse the discriminator's last spatial tensor to one sigmoid output.
- Use BatchNorm in both networks except on the generator output layer and discriminator input layer.
- Use ReLU in `G` except for a final `tanh`; use LeakyReLU with slope `0.2` in all hidden layers of `D`.
- Scale images to `[-1, 1]`; initialize convolution weights with mean `0` and std `0.02`.
- Original reported training uses minibatch size `128`, Adam with `lr=0.0002`, `beta1=0.5`, and a 100-dimensional uniform latent. Public Torch and PyTorch code defaults to batch size `64` and normal latent noise, with the Torch code also exposing a uniform option.

## Reference-Faithful PyTorch Core

This core follows the public PyTorch DCGAN example architecture and training loop, with an explicit `prior` switch to recover the originally stated uniform latent if desired.

```python
import torch
import torch.nn as nn
import torch.optim as optim

nz = 100
ngf = 64
ndf = 64
nc = 3

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def sample_z(batch_size, device, prior="normal"):
    if prior == "uniform":
        return torch.empty(batch_size, nz, 1, 1, device=device).uniform_(-1.0, 1.0)
    return torch.randn(batch_size, nz, 1, 1, device=device)

netG = Generator()
netD = Discriminator()
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

def train_step(real, prior="normal"):
    device = real.device
    dtype = real.dtype
    batch_size = real.size(0)
    real_label = torch.full((batch_size,), 1.0, dtype=dtype, device=device)
    fake_label = torch.full((batch_size,), 0.0, dtype=dtype, device=device)

    optimizerD.zero_grad(set_to_none=True)
    errD_real = criterion(netD(real), real_label)
    errD_real.backward()

    noise = sample_z(batch_size, device, prior=prior)
    fake = netG(noise)
    errD_fake = criterion(netD(fake.detach()), fake_label)
    errD_fake.backward()
    optimizerD.step()

    optimizerG.zero_grad(set_to_none=True)
    errG = criterion(netD(fake), real_label)
    errG.backward()
    optimizerG.step()

    return errD_real + errD_fake, errG
```

## Shape Checks

Generator transposed convolution output size is `(i - 1) * s - 2p + k`:

- `1 -> 4` with `k=4, s=1, p=0`.
- `4 -> 8 -> 16 -> 32 -> 64` with `k=4, s=2, p=1`.

Discriminator convolution output size is `floor((i + 2p - k) / s) + 1`:

- `64 -> 32 -> 16 -> 8 -> 4` with `k=4, s=2, p=1`.
- `4 -> 1` with `k=4, s=1, p=0`.

So the 64x64 RGB architecture is:

```text
G: z(100x1x1) -> 512x4x4 -> 256x8x8 -> 128x16x16 -> 64x32x32 -> 3x64x64
D: 3x64x64 -> 64x32x32 -> 128x16x16 -> 256x8x8 -> 512x4x4 -> 1
```
