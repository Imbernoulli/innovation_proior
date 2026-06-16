# Textual Inversion

## The problem

A frozen, pretrained text-to-image diffusion model can render almost anything you can *name* — but only what you can name. Given 3–5 casual images of a single user-provided concept (a specific object, a particular pet, an artistic style), there is no caption that pins down its exact appearance: "a clay sculpture" yields *a* sculpture, never *your* sculpture, and piling on adjectives still leaves the appearance underdetermined. The goal of **personalized text-to-image generation** is to teach the frozen model this one new concept from those few images, so it can afterward be invoked compositionally with ordinary language — "a photo of $S_*$ on the beach", "an oil painting of $S_*$", "a drawing of $S^1_*$ in the style of $S^2_*$" — with the model's existing knowledge and editing ability left fully intact. The hard constraints: only a few images, no text description of the concept, and no degradation of the prior (finetuning the weights on so few images overfits and causes catastrophic forgetting, so editability collapses).

## The core idea

Do not touch the model. The conditioning path is: prompt string → tokenizer → integer indices → **per-token embedding lookup table** → frozen text transformer $c_\theta$ → frozen denoiser $\epsilon_\theta$. The embedding lookup is the first point where the discrete input becomes a continuous, directly-optimizable vector. So a "word", to the frozen model, *is* a vector in that table, and a new concept is a new vector.

Designate a placeholder string $S_*$ for the concept, reserve it a single new row $v_*$ in the embedding table, and learn that one vector — **everything else frozen** (the autoencoder $\mathcal{E},D$, the text transformer $c_\theta$, the denoiser $\epsilon_\theta$, and every other embedding row). At inference, any prompt containing $S_*$ is tokenized, $S_*$ maps to $v_*$, and the rest of the unchanged pipeline treats it as an ordinary word, so composition with the model's existing vocabulary comes for free.

Because generation is a *visual* task, $v_*$ is optimized with the model's **own denoising objective** rather than a contrastive or language-completion loss (those capture coarse semantics but never force the vector to encode enough to redraw the object, so they produce visual corruption when used for synthesis). For a Latent Diffusion Model the per-example denoising loss is

$$L_{LDM} = \mathbb{E}_{z\sim\mathcal{E}(x),\,y,\,\epsilon\sim\mathcal{N}(0,1),\,t}\big[\lVert \epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\rVert_2^2\big],$$

with $z=\mathcal{E}(x)$ the image latent, $z_t$ that latent noised to timestep $t$, and $\epsilon$ the sampled noise. The optimization re-uses this exact loss, changing nothing but what is trainable:

$$v_* = \arg\min_v \; \mathbb{E}_{z\sim\mathcal{E}(x),\,y,\,\epsilon\sim\mathcal{N}(0,1),\,t}\big[\lVert \epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\rVert_2^2\big],$$

where $x$ ranges over the few concept images and $y$ is a neutral template containing $S_*$. The dependence on $v$ is only through the embedding lookup for that placeholder inside $c_\theta(y)$; $c_\theta$ and $\epsilon_\theta$ are held fixed. This reconstruction task is what pushes concept-specific visual detail into $v_*$.

Practical choices that fall out of this:
- **Neutral prompt templates.** $y$ is sampled from generic, content-free CLIP-ImageNet-style templates ("a photo of a $S_*$", "a rendition of a $S_*$", "a close-up photo of the $S_*$", …) so $v_*$ must carry the concept's appearance regardless of the surrounding boilerplate instead of entangling with one phrasing.
- **Initialization.** $v_*$ is initialized from the embedding of a single-word coarse descriptor of the concept (e.g. "sculpture", "cat"), giving the optimization a running start in a sensible region of the space.
- **Same training scheme.** The original LDM hyperparameters are kept; the only knob with real leverage is the learning rate.

## Reconstruction–editability tradeoff and single-word capacity

Borrowing the GAN-inversion playbook — multiple embedding vectors per concept, progressively introduced vectors, an L2 regularizer pulling $v_*$ toward real word embeddings, per-image auxiliary tokens — buys nothing here or actively hurts. A **single** vector already reconstructs about as well as the multi-vector setups (its semantic reconstruction is on par with sampling real images from the training set), while the multi-word variants are markedly *worse* on editability. The textual embedding space is far more flexible than the single-vector bottleneck would suggest.

The deeper structure is a **distortion–editability tradeoff**: embeddings kept near the real-word distribution (via regularization, fewer vectors, or a lower learning rate) stay editable but miss the concept's details; embeddings that drift far from it (more vectors, higher learning rate, no regularization) capture details but become hard to edit. The GAN-inspired "improvements" merely move along this same curve, mostly the wrong way. A single embedding already sits at an appealing point, and the operating point is controlled simply by the **learning rate** — lower keeps $v_*$ near the word cloud (more editable, less faithful), higher lets it roam (more faithful, less editable) — so the user can choose. (Reconstruction is measured by CLIP *semantic* similarity, which is forgiving about exact shape, so "on par with real images" is a semantic claim, not pixel-exact; precise shape fidelity remains open.) Verbose human captions are worse still: vision-language encoders attend selectively to a subset of tokens, so a long object description crowds out the requested edit, *reducing* editability. A single pseudo-word spends one token on identity and leaves the rest of the prompt free to steer the edit.

## Runnable PyTorch

Insert one trainable row for $S_*$ into the otherwise frozen embedding table, run the unchanged LDM denoising loss, and after each step restore every embedding row except $v_*$ so only that row can persistently move.

```python
import random
import torch
import torch.nn.functional as F

# --- frozen pretrained text-to-image LDM ---
vae, unet, text_encoder, tokenizer, scheduler = load_pretrained_ldm()
vae.requires_grad_(False)            # encoder E / decoder D : frozen
unet.requires_grad_(False)           # denoiser epsilon_theta : frozen
text_encoder.requires_grad_(False)   # freeze c_theta, including all old text weights

# --- inject the pseudo-word S_* and seed v_* from a coarse descriptor ---
placeholder = "<my-sculpture>"                      # the placeholder string S_*
tokenizer.add_tokens([placeholder])
text_encoder.resize_token_embeddings(len(tokenizer))  # adds a fresh embedding row
ph_id   = tokenizer.convert_tokens_to_ids(placeholder)
init_id = tokenizer.encode("sculpture", add_special_tokens=False)[0]  # coarse word
token_emb = text_encoder.get_input_embeddings()
with torch.no_grad():
    token_emb.weight[ph_id] = token_emb.weight[init_id].clone()  # initialize v_*

orig_embeds = token_emb.weight.detach().clone()     # snapshot to restore every old row
token_emb.weight.requires_grad_(True)               # gradients may reach the table...
optimizer = torch.optim.AdamW(
    [token_emb.weight], lr=5e-3, weight_decay=0.0)  # ...but no extra regularizer/loss

# neutral, content-free templates -> v_* must hold the concept regardless of phrasing
templates = ["a photo of a {}", "a rendition of a {}",
             "a cropped photo of the {}", "a close-up photo of the {}",
             "a bright photo of the {}", "a good photo of a {}"]  # ... full CLIP set

for step in range(5000):                            # ~5000 optimization steps
    img = sample_concept_image().to(token_emb.weight.device)  # one of the 3-5 images x
    with torch.no_grad():
        z = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor  # z = E(x)
    eps = torch.randn_like(z)                        # epsilon ~ N(0, 1)
    t   = torch.randint(0, scheduler.config.num_train_timesteps,
                        (z.shape[0],), device=z.device)
    z_t = scheduler.add_noise(z, eps, t)             # latent noised to time t

    prompt = random.choice(templates).format(placeholder)  # y contains S_*
    ids    = tokenizer(prompt, padding="max_length", truncation=True,
                       return_tensors="pt").input_ids.to(z.device)
    c      = text_encoder(ids)[0]                    # frozen c_theta turns v_* into conditioning

    eps_pred = unet(z_t, t, c).sample
    # the SAME, unchanged LDM denoising objective, restricted to v_*:
    #   v_* = argmin_v  E || eps - eps_theta(z_t, t, c_theta(y)) ||^2
    loss = F.mse_loss(eps_pred.float(), eps.float())
    loss.backward()
    with torch.no_grad():
        keep = torch.ones(token_emb.weight.shape[0], dtype=torch.bool,
                          device=token_emb.weight.device)
        keep[ph_id] = False
        token_emb.weight.grad[keep] = 0              # only S_*'s row receives an update
    optimizer.step()
    optimizer.zero_grad()

    # restore every embedding row except v_* -> gradient effectively trains only S_*'s row,
    # so the prior is untouched and nothing else can be forgotten.
    with torch.no_grad():
        keep = torch.ones(token_emb.weight.shape[0], dtype=torch.bool,
                          device=token_emb.weight.device)
        keep[ph_id] = False
        token_emb.weight[keep] = orig_embeds[keep]
```

Although the optimizer is attached to the embedding matrix for ordinary lookup semantics, the gradient mask and per-step restore make the only persistent trainable degree of freedom $v_*$. The net effect is exactly $v_* = \arg\min_v L_{LDM}$ with everything else frozen — one new word in the model's vocabulary, learned from a few images by the model's own denoising loss.

## Verification

Each load-bearing claim, checked against the derivation it rests on:

- **Frozen everything else; only $v_*$ trained.** The pseudo-word is found by re-using the original LDM training scheme while keeping both the text encoder $c_\theta$ and the denoiser $\epsilon_\theta$ fixed, leaving the generative model untouched. In code, `vae`, `unet`, and the full text encoder are set `requires_grad_(False)`; only the input-embedding matrix is re-enabled, its gradient is zeroed on every row but `ph_id` before the optimizer step, and all non-placeholder rows are restored afterward. The single persistent degree of freedom is row `ph_id` $= v_*$. Consistent.
- **Single-embedding LDM objective.** The objective is the LDM denoising loss $\mathbb{E}[\lVert\epsilon-\epsilon_\theta(z_t,t,c_\theta(y))\rVert_2^2]$ with $z=\mathcal{E}(x)$, $z_t$ the noised latent, $\epsilon\sim\mathcal{N}(0,1)$, minimized over $v$ alone: $v_*=\arg\min_v\mathbb{E}[\lVert\epsilon-\epsilon_\theta(z_t,t,c_\theta(y))\rVert_2^2]$. The dependence on $v$ enters only through the embedding lookup of $S_*$ inside $c_\theta(y)$. The code's `F.mse_loss(eps_pred, eps)` is exactly this loss with `weight_decay=0.0`, so no extra term is introduced. Consistent.
- **Embedding-space inversion / pseudo-word $S_*\!\to\!v_*$.** The text encoder first tokenizes, then retrieves a per-token embedding by index lookup; the method designates placeholder $S_*$ and replaces its vector with a learned $v_*$, injecting the concept into the vocabulary. `add_tokens` + `resize_token_embeddings` create exactly that row. Consistent.
- **3–5 images, neutral CLIP templates, reconstruction objective.** A small set of 3–5 images is used; prompts are neutral CLIP-ImageNet templates ("a photo of a $S_*$", etc.); because this is a reconstruction task it drives fine visual detail into $v_*$. The code samples one of the few images per step and a random template. Consistent.
- **Initialization, learning rate, steps.** $v_*$ is initialized from the embedding of a single-word coarse descriptor (e.g. "sculpture", "cat"); base learning rate $0.005$; $5{,}000$ optimization steps. Code: `init_id` from "sculpture", `lr=5e-3`, `range(5000)`. Consistent.
- **Single-word capacity and distortion–editability tradeoff.** A single word embedding suffices; its reconstruction is comparable to multi-word setups and to sampling real training images, with improved editability; the operating point on the tradeoff curve is moved by the learning rate alone; verbose human captions reduce editability through selective attention; CLIP reconstruction is semantic, not shape-exact. All transcribed faithfully. Consistent.
- **In-frame.** No source-paper citation line, authors, venue, or arXiv id; "Textual Inversion" is named as the artifact being built; only prior-art ancestors (GAN inversion, LDM) appear as background.
