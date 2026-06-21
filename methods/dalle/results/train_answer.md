The goal is to generate high-fidelity images from free-form natural language, and to do so with a single general model rather than a stack of hand-engineered, task-specific modules. Earlier text-to-image systems were built from conditional GANs with multi-scale generators, attention blocks, auxiliary text-image matching losses, and extra conditioning signals, all tuned on small datasets like MS-COCO or CUB-200. Despite that machinery, their samples still suffered from distorted objects, illogical placement, and unnatural blending of foreground and background. The binding question is whether the real limitation is not architecture but scale: if one large, general model is trained on hundreds of millions of image-text pairs, will it subsume those specialized tricks and produce a flexible, language-controllable generator?

A transformer is the most natural candidate for such a general model, because it is modality-agnostic: it simply models a sequence of discrete tokens. The obstacle is that an image is not already a short token sequence. Modeling raw pixels directly is infeasible, a 256x256 RGB image is roughly two hundred thousand values, far too long for quadratic attention, and likelihood objectives over pixels waste most of their capacity on short-range texture rather than the low-frequency structure that makes objects recognizable. The way forward is to compress the image first into a short grid of discrete tokens, then model the joint text-plus-image token stream autoregressively.

The method is DALL-E. It is a two-stage generative model that treats text and image as a single sequence of tokens. In the first stage a discrete variational autoencoder, the dVAE, compresses a 256x256x3 image into a 32x32 grid of tokens, each drawn from a vocabulary of K=8192 entries. That reduces the image sequence length by a factor of 192, which finally makes transformer modeling practical, while the large vocabulary limits the information lost by the spatial downsampling. In the second stage a decoder-only sparse transformer is trained over the concatenation of BPE-encoded caption tokens and these 1024 image tokens. At generation time a caption is fed in and the image tokens are sampled autoregressively, then decoded back to pixels through the frozen dVAE. Multiple candidates can be drawn and reranked with a pretrained contrastive image-text model.

The two stages are not arbitrary; they are coordinate ascent on a single evidence lower bound over images, captions, and image tokens. Write the joint as p_theta,psi(x, y, z) = p_theta(x | y, z) p_psi(y, z), with an encoder q_phi(z | x). Maximizing over phi and theta while holding psi fixed trains the dVAE on images, and maximizing over psi while holding phi and theta fixed trains the transformer prior over tokens. The encoder outputs categorical distributions over the codebook, which are not differentiable, so DALL-E relaxes them with Gumbel-Softmax: add Gumbel noise to the logits and take a softmax at temperature tau. At high tau the sample is soft and reparameterizable; as tau is annealed toward zero the relaxation hardens into the true categorical. To keep the relaxed model from exploiting correlations in the soft codes that disappear after hardening, the encoder ends and the decoder begins with 1x1 convolutions around the bottleneck, shrinking the receptive field at the discrete layer.

For pixel reconstruction, DALL-E uses a logit-Laplace likelihood. A Laplace variable is pushed through a sigmoid so that the resulting distribution has bounded support on (0, 1), matching the valid pixel range instead of wasting mass outside it. The decoder emits six feature maps per pixel: three location parameters mu and three log-scale parameters ln b for the RGB channels. Pixels are first mapped from [0, 1] into a strictly interior interval to avoid the density blowing up at the boundaries. Surprisingly, raising the KL weight beta to about 6.6 improves both reconstruction and codebook usage, because a stronger prior pressure prevents the relaxation noise from collapsing the code distribution onto a small subset of codes. The transformer prior uses factorized row and column attention over the 32x32 image grid, alternating them across layers so that each image token can attend to its spatial neighbors far more cheaply than with dense attention, while every image token can always attend to all caption tokens so that the text actually steers the image.

```python
import torch
from torch import nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_out, n_layers):
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)
        self.id_path = nn.Conv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(
            nn.ReLU(), nn.Conv2d(n_in,  n_hid, 3, padding=1),
            nn.ReLU(), nn.Conv2d(n_hid, n_hid, 3, padding=1),
            nn.ReLU(), nn.Conv2d(n_hid, n_hid, 3, padding=1),
            nn.ReLU(), nn.Conv2d(n_hid, n_out, 1))
    def forward(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)

class Encoder(nn.Module):
    def __init__(self, n_hid=256, n_blk=2, in_ch=3, vocab_size=8192):
        super().__init__()
        n_layers = 4 * n_blk
        blk = lambda i, o: EncoderBlock(i, o, n_layers)
        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, n_hid, 7, padding=3),
            *[blk(n_hid, n_hid) for _ in range(n_blk)], nn.MaxPool2d(2),
            blk(n_hid, 2*n_hid), *[blk(2*n_hid, 2*n_hid) for _ in range(n_blk-1)], nn.MaxPool2d(2),
            blk(2*n_hid, 4*n_hid), *[blk(4*n_hid, 4*n_hid) for _ in range(n_blk-1)], nn.MaxPool2d(2),
            blk(4*n_hid, 8*n_hid), *[blk(8*n_hid, 8*n_hid) for _ in range(n_blk-1)],
            nn.ReLU(), nn.Conv2d(8*n_hid, vocab_size, 1))
    def forward(self, x):
        return self.blocks(x)  # [B, K, 32, 32]

class Decoder(nn.Module):
    def __init__(self, n_hid=256, n_blk=2, out_ch=6, vocab_size=8192):
        super().__init__()
        n_layers = 4 * n_blk
        blk = lambda i, o: EncoderBlock(i, o, n_layers)
        self.blocks = nn.Sequential(
            nn.Conv2d(vocab_size, 8*n_hid, 1),
            *[blk(8*n_hid, 8*n_hid) for _ in range(n_blk)],
            nn.Upsample(scale_factor=2, mode="nearest"),
            blk(8*n_hid, 4*n_hid), *[blk(4*n_hid, 4*n_hid) for _ in range(n_blk-1)],
            nn.Upsample(scale_factor=2, mode="nearest"),
            blk(4*n_hid, 2*n_hid), *[blk(2*n_hid, 2*n_hid) for _ in range(n_blk-1)],
            nn.Upsample(scale_factor=2, mode="nearest"),
            blk(2*n_hid, n_hid), *[blk(n_hid, n_hid) for _ in range(n_blk-1)],
            nn.ReLU(), nn.Conv2d(n_hid, out_ch, 7, padding=3))
    def forward(self, z):
        return self.blocks(z)  # [B, 6, 256, 256]

def gumbel_softmax_codes(logits, codebook, tau):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    soft = F.softmax((logits + g) / tau, dim=1)
    z = torch.einsum("bkhw,kd->bdhw", soft, codebook)
    return soft, z

def logit_laplace_nll(x, mu, ln_b):
    b = ln_b.exp()
    return (torch.log(2 * b * x * (1 - x)) + (torch.logit(x) - mu).abs() / b).mean()

def kl_uniform(soft):
    q = soft.mean(dim=(0, 2, 3))
    K = soft.shape[1]
    return (q * (q.add(1e-9).log() + torch.log(torch.tensor(float(K))))).sum()

def phi(img, eps=0.1):
    return (1 - 2 * eps) * img + eps

def anneal(step, start, end, steps):
    t = min(step / steps, 1.0)
    return start + (end - start) * (1 - torch.cos(torch.tensor(t * 3.14159265))) / 2

# Stage 1: train the dVAE on images alone (relaxed ELBO)
encoder = Encoder()
decoder = Decoder()
codebook = nn.Parameter(torch.randn(8192, 256) * 0.02)
opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + [codebook],
                        lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
for step, img in enumerate(loader):
    x = phi(img)
    logits = encoder(x)
    tau = anneal(step, 1.0, 1.0 / 16, 150_000)
    beta = anneal(step, 0.0, 6.6, 5_000)
    soft, z = gumbel_softmax_codes(logits, codebook, tau)
    mu, ln_b = decoder(z).chunk(2, dim=1)
    loss = logit_laplace_nll(x, mu, ln_b) + beta * kl_uniform(soft)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Stage 2: fit the transformer prior over concatenated text+image tokens
@torch.no_grad()
def image_tokens(encoder, img):
    return encoder(phi(img)).argmax(dim=1).flatten(1)  # [B, 1024]

transformer = TokenTransformer()  # decoder-only sparse transformer over shared text+image vocab
opt = torch.optim.AdamW(transformer.parameters(), lr=4.5e-4, betas=(0.9, 0.96),
                        eps=1e-8, weight_decay=4.5e-2)
for text_tok, img in loader:  # text_tok: [B, 256]
    img_tok = image_tokens(encoder, img)  # [B, 1024]
    stream = torch.cat([text_tok, img_tok + 16384], dim=1)  # image ids offset past text vocab
    logits = transformer(stream[:, :-1])
    tgt = stream[:, 1:]
    n_text = 256 - 1
    ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
    loss = (1 / 8) * ce[:, :n_text].mean() + (7 / 8) * ce[:, n_text:].mean()
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 4.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
```
