Deep discriminative models train beautifully today because they combine three well-behaved ingredients: backpropagation, dropout, and piecewise-linear units such as ReLU and maxout. Generative modeling has not had the same breakthrough, and the obstacle is structural rather than a matter of scale. The standard way to learn a generative model is maximum likelihood, and maximum likelihood forces you to write down, normalize, and differentiate an explicit probability density. That requirement is exactly where the trouble begins: energy-based models carry an intractable partition function and need MCMC to approximate it; variational methods replace the likelihood with a bound and require a separate inference network; and fully-visible belief networks, while tractable, sample one coordinate at a time and offer no latent code. Implicit models escape the density entirely, but until now they lacked a clean learning signal that could drive them toward the data distribution using only forward and backward propagation.

The failure of the closest alternative, noise-contrastive estimation, points the way out. NCE trains a classifier to distinguish data from a fixed noise distribution, using the model's own unnormalized density inside the logistic loss. The problem is that the noise never gets harder: as soon as the model captures a few features, the classifier saturates and the gradient goes slack. A fixed contrast is a pushover, and beating a pushover is not the same as matching the data distribution. What is needed is a contrast that improves in lockstep with the model, so the classification task stays difficult and informative all the way to the optimum.

The method I propose is the Generative Adversarial Network, or GAN. It keeps the implicit-model branch of the design tree and supplies the missing signal with a learned adversary. There are two networks. The generator G takes a noise vector z drawn from a simple prior p_z and maps it through a differentiable feedforward network to a sample x = G(z). Because the only stochasticity is the injected noise at the input, gradients flow cleanly from any downstream scalar through G and into its parameters by the reparameterization principle. The discriminator D is a separate network that takes a data-space point x and outputs a scalar D(x) in (0, 1), interpreted as the probability that x came from the real data rather than from G. D is trained to classify correctly, while G is trained to make D misclassify generated samples as real. The opponent is thus the generator itself, and because it is being optimized, the contrast never collapses into a trivial problem.

Formally, the two networks play the minimax game V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))], where D tries to maximize V and G tries to minimize it. For a fixed G, the optimal discriminator is D*_G(x) = p_data(x) / (p_data(x) + p_g(x)), the Bayes-optimal classifier between the two classes and, equivalently, the density ratio p_data / p_g squashed into the unit interval. Substituting this optimal discriminator into V reveals what G is actually minimizing: C(G) = -log 4 + 2 * JSD(p_data || p_g), where JSD is the Jensen-Shannon divergence. Since JSD is non-negative and zero only when the two distributions are equal, the unique global optimum is p_g = p_data, with D indifferent at 1/2 everywhere. The divergence is symmetric and bounded, which comes for free from the real-versus-fake formulation rather than from any hand-designed objective.

Two practical adjustments make the idea runnable. First, optimizing D to convergence at every step is too expensive and would overfit, so we take k discriminator gradient steps per generator step; k = 1 is the cheapest version that still keeps D tracking the moving optimum and prevents G from training too long against a stale discriminator. A stale D invites mode collapse: G discovers a small set of samples that fool the current classifier and stops exploring the full data distribution. Second, the raw minimax generator term log(1 - D(G(z))) has weak early gradients, because when G is poor D rejects fakes confidently and the logit-space derivative nearly vanishes. We therefore train G to maximize log D(G(z)) instead. This non-saturating loss shares the same fixed point but gives a strong gradient of roughly -1 when D(G(z)) is near zero, exactly when G most needs guidance.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Simple fully-connected GAN for MNIST.

class Generator(nn.Module):
    def __init__(self, z_dim=100, h_dim=256, x_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim=784, h_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_gan(z_dim=100, epochs=20, batch_size=128, lr=2e-4, device="cuda"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    bce = nn.BCELoss()
    opt_g = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    ones = torch.ones(batch_size, 1, device=device)
    zeros = torch.zeros(batch_size, 1, device=device)

    for epoch in range(epochs):
        for real, _ in loader:
            real = real.to(device)

            # Train discriminator: maximize log D(x) + log(1 - D(G(z))).
            z = torch.randn(batch_size, z_dim, device=device)
            fake = G(z).detach()
            loss_d = bce(D(real), ones) + bce(D(fake), zeros)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train generator: maximize log D(G(z)) (non-saturating).
            z = torch.randn(batch_size, z_dim, device=device)
            loss_g = bce(D(G(z)), ones)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    return G, D
```
