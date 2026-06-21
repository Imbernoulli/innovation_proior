# Textual Inversion

## Problem

A pretrained text-to-image model can render concepts that its language interface can name,
but a few casual images of a particular object, pet, toy, or style usually cannot be
compressed into a caption that fixes its identity. Finetuning the generator on 3-5 images is
the obvious adaptation path, but it risks overfitting and forgetting the broad prior that
makes prompts such as "on the beach" or "as an oil painting" work. The target is therefore a
personalized concept handle that can be used inside ordinary prompts while the generator and
text encoder stay fixed.

## Core Method

The prompt enters the model as tokens, then as per-token embeddings, then through a frozen
text encoder $c_\theta$, and finally conditions a frozen denoiser $\epsilon_\theta$. Textual
inversion chooses the embedding stage as the inversion space. Pick a placeholder string
$S_*$ that tokenizes to one existing token, and learn a replacement embedding vector $v_*$
for that token. During the text encoder's embedding lookup, every occurrence of the
placeholder token is replaced by $v_*$; all other token embeddings and all downstream model
weights remain unchanged.

The learned vector is initialized from the embedding of a single-token coarse descriptor,
such as "sculpture" or "cat". Training prompts are neutral templates derived from the CLIP
ImageNet template set, for example "a photo of a $S_*$", "a rendering of a $S_*$", and "a
close-up photo of the $S_*$". The template provides a syntactic context; the learned vector
must carry the visual concept.

For an LDM with image encoder $\mathcal{E}$, decoder $D$, noised latent $z_t$, sampled noise
$\epsilon \sim \mathcal{N}(0,1)$, text conditioning $c_\theta(y)$, and epsilon-prediction
denoiser $\epsilon_\theta$, the original training loss is

$$
L_{LDM}
= \mathbb{E}_{z\sim\mathcal{E}(x),\,y,\,\epsilon,\,t}
\left[\left\|\epsilon-\epsilon_\theta(z_t,t,c_\theta(y))\right\|_2^2\right].
$$

Textual inversion optimizes the same objective over the placeholder embedding only:

$$
v_*
= \arg\min_v
\mathbb{E}_{z\sim\mathcal{E}(x),\,y,\,\epsilon,\,t}
\left[\left\|\epsilon-\epsilon_\theta(z_t,t,c_\theta(y))\right\|_2^2\right],
$$

where $x$ is sampled from the few concept images and $y$ is a sampled template containing
$S_*$. The dependence on $v$ is only through the embedding substitution inside
$c_\theta(y)$. There is no sign change, contrastive term, classifier guidance term, or
caption loss in the default objective. The optional embedding regularizer toward the coarse
descriptor is available but disabled (weight `0.0`) by default.

## Implementation

The LDM uses a BERT-style text encoder, so the placeholder can be an existing single-token
string (default `"*"`) and the optimized vector kept as a separate `nn.Parameter` that
overwrites the embedded placeholder positions at runtime — no need to grow the tokenizer or
permanently edit the embedding table.

```python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def single_bert_token(tokenizer, text):
    ids = tokenizer(text)
    assert torch.count_nonzero(ids) == 3  # [CLS], token, [SEP]
    return ids[0, 1]


class EmbeddingManager(nn.Module):
    def __init__(self, text_encoder, placeholder="*", initializer="sculpture"):
        super().__init__()
        self.placeholder = placeholder
        self.placeholder_id = single_bert_token(text_encoder.tknz_fn, placeholder)
        init_id = single_bert_token(text_encoder.tknz_fn, initializer)

        with torch.no_grad():
            init = text_encoder.transformer.token_emb(init_id.cpu())
        self.v_star = nn.Parameter(init.unsqueeze(0))  # one learned 1280-d LDM word vector

    def embedding_parameters(self):
        return [self.v_star]

    def forward(self, tokenized_text, embedded_text):
        positions = torch.where(tokenized_text == self.placeholder_id.to(tokenized_text.device))
        embedded_text[positions] = self.v_star.to(embedded_text.device)
        return embedded_text


model = load_pretrained_ldm_1p4b()
model.first_stage_model.requires_grad_(False)  # autoencoder E, D
model.model.requires_grad_(False)              # diffusion denoiser epsilon_theta
model.cond_stage_model.requires_grad_(False)   # text encoder c_theta

embedding_manager = EmbeddingManager(model.cond_stage_model,
                                     placeholder="*",
                                     initializer="sculpture")

# Paper setup: base LR 0.005, scaled by number of GPUs and batch size.
optimizer = torch.optim.AdamW(embedding_manager.embedding_parameters(), lr=0.04)

templates = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "a photo of my {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
]

for step in range(5000):
    images = sample_concept_batch()  # 3-5 image set, repeatedly sampled
    prompts = [random.choice(templates).format("*") for _ in range(images.shape[0])]

    with torch.no_grad():
        posterior = model.encode_first_stage(images)
        z = model.get_first_stage_encoding(posterior)

    noise = torch.randn_like(z)
    t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=z.device)
    z_t = model.q_sample(x_start=z, t=t, noise=noise)

    # The frozen encoder performs its normal token lookup, then the manager swaps
    # the placeholder's embedding with v_star before the transformer layers run.
    c = model.cond_stage_model.encode(prompts, embedding_manager=embedding_manager)
    noise_pred = model.apply_model(z_t, t, c)

    loss = F.mse_loss(noise_pred.float(), noise.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The same `EmbeddingManager` supports the analysis variants by changing its construction and
the dataset prompts:

- `num_vectors_per_token > 1`: replace one placeholder with two or three learned vectors.
- `progressive_words=True`: expose the second vector after 2000 steps and the third after
  4000 steps.
- `embedding_reg_weight > 0`: add an L2 pull toward the coarse descriptor embedding.
- `per_image_tokens=True`: add image-specific placeholders and train prompts of the form
  "a photo of $S_*$ with $S_i$".
- `unfreeze_model=True`: also optimize text/generator weights as a pivotal-tuning-style
  variant.

Those branches are diagnostic variants, not the default method. The default is one learned
embedding vector, no embedding regularization, frozen autoencoder, frozen text encoder,
frozen denoiser, and the unchanged epsilon-prediction reconstruction loss.

## What The Analysis Shows

The natural GAN-inversion expectation is that extra latent capacity should improve
reconstruction. In this embedding space that expectation fails. A single learned vector gives
reconstruction comparable to multi-vector alternatives while preserving substantially better
editability. Multi-word, progressive, regularized, and per-image-token variants mostly move
the solution along a distortion-editability curve rather than improving the frontier.

The useful operating knob is the learning rate. Lower learning rates or explicit pulls
toward existing word embeddings keep the learned vector closer to the natural word cloud:
the concept is easier to edit but less faithful. Higher learning rates let it drift farther:
the concept is captured more strongly but becomes harder to edit. Human captions perform
poorly for the same reason long descriptions often weaken prompt control: the language
conditioner can focus on the object description and neglect the requested scene or style.

The reconstruction claim is semantic rather than pixel-exact. Concept-match is measured with
CLIP-space cosine similarity, which is relatively insensitive to exact shape, so precise
shape preservation remains an open limitation. Pivotal tuning and bipartite DDIM inversion
can improve or preserve some structure in special settings, but in this LDM guidance regime
they lose editability or prompt alignment enough that they do not replace the one-vector
frozen-model method.
