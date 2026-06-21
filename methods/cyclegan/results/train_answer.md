The goal is to translate images from one domain into another when no aligned input-output pairs exist. We may have thousands of landscape photos and Monet paintings, or horse and zebra images, but no pairing tells us which photo corresponds to which painting or which horse should become which zebra. An adversarial loss is the natural starting point: train a generator G to map inputs into the target domain and a discriminator to distinguish G(x) from real target images. In theory this pushes the marginal distribution of outputs to match the target distribution, but it does not constrain the joint relationship between inputs and outputs. Many different mappings can produce the same output distribution, so the optimizer can collapse to a degenerate solution that ignores the input entirely, sending every photo to the same plausible painting.

The fix is to require that the translation be invertible, or cycle-consistent. We learn not one mapping but two: G from domain X to domain Y and F from Y back to X. We then require that translating an image there and back returns something close to the original image, F(G(x)) ≈ x and G(F(y)) ≈ y. This replaces the missing paired supervision with a structural recoverability constraint. If G collapsed many different inputs onto the same output, F could not reconstruct all those distinct inputs, so the cycle loss directly penalizes information loss. Together with the inductive biases of convolutional residual networks, this makes the learned mapping preserve the content and structure of the input while re-rendering it in the style of the target domain.

The method is called CycleGAN. Its full objective combines two adversarial losses, one for each direction, with a cycle-consistency loss weighted by λ. In practice the adversarial losses use the least-squares GAN form, which gives stronger gradients than the saturating cross-entropy form and stabilizes training. The cycle losses use L1 reconstruction rather than L2, because L2 reconstruction tends to average plausible outputs into a blur, whereas L1 keeps reconstructions sharper. For tasks where global color can drift, such as painting-to-photo translation, an optional identity loss encourages each generator to leave an image unchanged when it is already in the target domain, anchoring the overall color palette. The generators are Johnson-style residual transformers with reflection padding and instance normalization: a 7×7 convolution, two stride-2 downsampling convolutions, six or nine residual blocks depending on image size, two fractionally-strided upsampling convolutions, and a final 7×7 convolution with Tanh. Residual blocks encode the prior that the output should equal the input plus a learned change, while instance normalization removes instance-specific contrast statistics so the network can impose the target domain's appearance. The discriminators are fully-convolutional 70×70 PatchGANs that judge local texture realism, which is both lightweight and focused on the high-frequency properties that matter. Training uses Adam with learning rate 2×10⁻⁴ and β₁ = 0.5, batch size 1, and a 50-image history buffer for discriminator updates to reduce oscillation. The cycle weight is set to λ = 10, with identity loss at 0.5λ when used.

```python
import torch
import torch.nn as nn
import itertools
import functools
import random


def use_conv_bias(norm_layer):
    if isinstance(norm_layer, functools.partial):
        return norm_layer.func == nn.InstanceNorm2d
    return norm_layer == nn.InstanceNorm2d


def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=use_bias), norm_layer(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_nc, out_nc, ngf=64, n_blocks=9,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)):
        super().__init__()
        use_bias = use_conv_bias(norm_layer)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_nc, ngf, 7, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_down = 2
        for i in range(n_down):
            m = 2 ** i
            model += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * m * 2), nn.ReLU(True)]
        m = 2 ** n_down
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * m, norm_layer, use_bias)]
        for i in range(n_down):
            m = 2 ** (n_down - i)
            model += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(ngf * m // 2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_nc, ndf=64, n_layers=3,
                 norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)):
        super().__init__()
        use_bias = use_conv_bias(norm_layer)
        kw, padw = 4, 1
        seq = [nn.Conv2d(in_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        mult, mult_prev = 1, 1
        for n in range(1, n_layers):
            mult_prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        mult_prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, target_is_real):
        target = pred.new_tensor(1.0 if target_is_real else 0.0).expand_as(pred)
        return self.loss(pred, target)


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size, self.num, self.images = pool_size, 0, []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for image in images:
            image = image.detach().unsqueeze(0)
            if self.num < self.pool_size:
                self.num += 1
                self.images.append(image)
                out.append(image)
            elif random.uniform(0, 1) > 0.5:
                i = random.randint(0, self.pool_size - 1)
                out.append(self.images[i].clone())
                self.images[i] = image
            else:
                out.append(image)
        return torch.cat(out, 0)


class CycleGAN:
    def __init__(self, in_nc=3, out_nc=3, lambda_cyc=10.0, lambda_idt=0.5, lr=2e-4):
        if lambda_idt > 0:
            assert in_nc == out_nc, "identity loss requires matching input/output channels"
        self.G_A = ResnetGenerator(in_nc, out_nc)   # X -> Y
        self.G_B = ResnetGenerator(out_nc, in_nc)   # Y -> X
        self.D_A = NLayerDiscriminator(out_nc)      # judges Y
        self.D_B = NLayerDiscriminator(in_nc)       # judges X
        for net in (self.G_A, self.G_B, self.D_A, self.D_B):
            init_weights(net)
        self.fake_A_pool, self.fake_B_pool = ImagePool(), ImagePool()
        self.criterionGAN = GANLoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        self.lambda_cyc, self.lambda_idt = lambda_cyc, lambda_idt
        self.opt_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                      lr=lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                      lr=lr, betas=(0.5, 0.999))

    def forward(self, real_A, real_B):
        self.fake_B = self.G_A(real_A)
        self.rec_A = self.G_B(self.fake_B)
        self.fake_A = self.G_B(real_B)
        self.rec_B = self.G_A(self.fake_A)

    def backward_G(self, real_A, real_B):
        lam, lam_i = self.lambda_cyc, self.lambda_idt
        loss_idt = 0.0
        if lam_i > 0:
            loss_idt = (self.criterionIdt(self.G_A(real_B), real_B) * lam * lam_i +
                        self.criterionIdt(self.G_B(real_A), real_A) * lam * lam_i)
        loss_G = (self.criterionGAN(self.D_A(self.fake_B), True) +
                  self.criterionGAN(self.D_B(self.fake_A), True))
        loss_cyc = (self.criterionCycle(self.rec_A, real_A) * lam +
                    self.criterionCycle(self.rec_B, real_B) * lam)
        (loss_G + loss_cyc + loss_idt).backward()

    def backward_D(self, netD, real, fake):
        loss = (self.criterionGAN(netD(real), True) +
                self.criterionGAN(netD(fake.detach()), False)) * 0.5
        loss.backward()
        return loss

    def optimize(self, real_A, real_B):
        self.forward(real_A, real_B)
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad = False
        self.opt_G.zero_grad(); self.backward_G(real_A, real_B); self.opt_G.step()
        for p in itertools.chain(self.D_A.parameters(), self.D_B.parameters()):
            p.requires_grad = True
        self.opt_D.zero_grad()
        self.backward_D(self.D_A, real_B, self.fake_B_pool.query(self.fake_B))
        self.backward_D(self.D_B, real_A, self.fake_A_pool.query(self.fake_A))
        self.opt_D.step()
```
