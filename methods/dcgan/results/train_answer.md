The goal is to learn useful visual representations from unlabeled images. Supervised convolutional networks already build strong hierarchies of features, but they depend on labels that are expensive to collect. A purely unsupervised approach should exploit the abundance of raw images and produce a representation that can be reused by a simple linear classifier. Reconstruction-based methods such as autoencoders are stable, yet their pixel-level loss tends to reward blurry averages when many plausible outputs exist, and the features they learn are often tuned to low-frequency recovery rather than the kind of structure that matters for recognition. Plain generative adversarial networks are appealing because they avoid explicit densities and pixel reconstruction, but early attempts to make them deep and convolutional were notoriously unstable: training oscillates, samples become incoherent, or the generator collapses many latent inputs into the same image. The challenge is therefore to keep the adversarial pressure while making the convolutional architecture and training procedure stable enough to learn a hierarchy that actually transfers.

The solution is DCGAN, the Deep Convolutional Generative Adversarial Network. It keeps the original GAN objective but constrains the generator and discriminator so that the game can be played inside deep convolutional networks. The generator maps a simple 100-dimensional noise vector to a 64x64 RGB image, while the discriminator judges whether an image is real or generated. Training follows the standard two-player update: the discriminator is trained to maximize the probability of assigning the correct real or fake label, and the generator is trained to maximize the probability that the discriminator mistakes generated samples for real ones. In practice this means binary cross-entropy on real images labeled one, generated images labeled zero for the discriminator, and binary cross-entropy on generated images labeled one for the generator. This non-saturating generator loss avoids the weak gradients that come from minimizing log(1 - D(G(z))) early in training.

What makes DCGAN different from earlier unstable architectures is a set of design rules that remove hand-chosen operations and keep gradients flowing. First, all fixed pooling layers are replaced by learned resampling: strided convolutions in the discriminator learn how to downsample, and transposed convolutions in the generator learn how to upsample. This puts spatial geometry under gradient-based learning instead of baking it in. Second, fully connected hidden layers are removed. The generator begins by reshaping the latent vector into a small spatial tensor and then grows it through convolution, while the discriminator ends with a final convolution whose output is collapsed to a single score. Third, batch normalization is used in both networks, but it is deliberately withheld from the generator output and the discriminator input. Normalizing the generator output would distort the image statistics it is trying to match, and normalizing the discriminator input would hide the raw difference between real and generated pixels before learning begins. Fourth, the generator uses ReLU activations internally and a final tanh so that outputs lie in the same [-1, 1] range as the training images. The discriminator uses LeakyReLU with slope 0.2 so that negative activations still pass a small gradient back to the generator, preventing dead units from killing the generator update. Fifth, weights are initialized from a zero-mean normal with standard deviation 0.02, and batch-norm layers start near scale one and shift zero so that early updates are well behaved. Finally, the optimizer is Adam with learning rate 0.0002 and beta1 0.5; the reduced momentum keeps the optimizer from chasing gradients computed against a discriminator that has already changed.

The architecture closes cleanly. The generator starts from a 100x1x1 tensor, applies a transposed convolution with kernel four, stride one, and padding zero to reach 512x4x4, then repeatedly applies transposed convolutions with kernel four, stride two, and padding one to reach 256x8x8, 128x16x16, 64x32x32, and finally 3x64x64. The discriminator mirrors this path with ordinary strided convolutions and ends with a convolution that maps 512x4x4 down to a single scalar followed by a sigmoid. Because the whole pipeline is convolutional, the discriminator's hidden layers can be reused as a feature extractor for downstream tasks: one simply freezes the learned weights and trains a small classifier on top. The resulting model trains reliably, generates plausible images, and produces representations that capture meaningful visual structure without any labels.

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
