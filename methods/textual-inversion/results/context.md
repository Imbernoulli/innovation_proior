# Context

## Research question

We now have text-to-image generators that genuinely *reason* over natural language: type a sentence and out comes a vivid, coherent scene in any style you ask for. But the interface is the bottleneck. Everything the model can render, it can only render through words it already knows. If I want a picture of *my* specific clay sculpture — the headless one on my desk, with its particular proportions and surface — there is no phrase that pins it down. "A photo of a clay sculpture" produces *a* sculpture, never *mine*. Even the most painstaking caption underdetermines the appearance: dozens of distinct objects all satisfy the same description.

So the question is: given a frozen, pretrained text-to-image model and a handful (3–5) of casual photos of one specific concept — an object, a particular pet, an artistic style — how do we teach the model this *one new concept* so that we can then invoke it compositionally, with language? I want to write "a photo of [my sculpture] on the beach", "an oil painting of [my sculpture]", "[my sculpture] in the style of [some artist]", and have it work — the new concept slotting into arbitrary prompts exactly as an ordinary word would. The hard constraints: only a few images, no text description of the concept, and — crucially — the model's existing knowledge and compositional ability must survive intact. A solution that learns my sculpture but forgets how to render "beach" or "oil painting" is useless.

## Background

**Large text-to-image generators and their semantic prior.** A class of generators trained on hundreds of millions of image–caption pairs has learned an extraordinarily rich joint prior over images and language: they compose unseen combinations, transfer styles, and place objects in novel contexts. The knowledge that makes them useful is *exactly* the thing few-shot adaptation tends to destroy.

**Latent Diffusion Models (LDMs).** One such generator operates in two stages. First, an autoencoder is pretrained on a large image corpus: an encoder $\mathcal{E}$ maps an image $x$ to a spatial latent $z=\mathcal{E}(x)$ (regularized by a KL penalty or vector quantization), and a decoder $D$ inverts it, $D(\mathcal{E}(x))\approx x$. Second, a diffusion model is trained to generate codes in that latent space, conditioned on the output $c_\theta(y)$ of a text-embedding model applied to a prompt $y$. With $z_t$ the latent noised to timestep $t$ and $\epsilon$ the sampled noise, the denoiser $\epsilon_\theta$ is trained on
$$L_{LDM}=\mathbb{E}_{z\sim\mathcal{E}(x),\,y,\,\epsilon\sim\mathcal{N}(0,1),\,t}\big[\lVert \epsilon-\epsilon_\theta(z_t,t,c_\theta(y))\rVert_2^2\big].$$
At inference, a random latent is iteratively denoised to $z_0$ and decoded, $x'=D(z_0)$. A publicly available 1.4B-parameter instance, trained on a 400M-image web dataset, realizes $c_\theta$ with a BERT-style text encoder.

**The first stage of the text encoder.** Before any transformer layer, an input string is tokenized into indices, and each index retrieves a learned embedding vector by lookup; these per-token embedding vectors are continuous representations of language that sit ahead of the transformer stack. Prior work showed this embedding space is expressive enough to capture basic image semantics — but those demonstrations used contrastive or language-completion objectives, neither of which demands precise *visual* reconstruction, and they fail to capture the appearance of a specific concept well enough for synthesis.

**GAN-inversion lineage.** The established discipline for "find a latent that reproduces *this* image" is GAN inversion. Its central, hard-won lesson is a **distortion–editability tradeoff**: codes pushed far from the generator's natural latent distribution reconstruct the target faithfully but become brittle and uneditable; codes kept near the training distribution stay editable but lose identity. Specific techniques grew around this — extended/multi-code latent spaces ($\mathcal{W}+$-style), progressive code introduction, regularizing inverted codes toward the natural distribution, and two-stage "pivotal tuning" (invert into an editable pivot, then lightly finetune the generator around it). These are the menu of moves one naturally reaches for when inverting into any new latent space.

## Baselines

- **Direct finetuning of the generator on the few images.** The obvious approach. With only a handful of examples it overfits and triggers *catastrophic forgetting*: the rich prior that made the model worth using degrades, and editability collapses.

- **Frozen-model adapter / transformation modules.** Freeze the generator and train small modules to adapt its behavior to new concepts. Less destructive, but still prone to forgetting prior knowledge, or to being unable to access old knowledge and the new concept concurrently.

- **Human-written captions.** Just describe the concept in words ("a retro yellow alarm clock with a white face…"). This is the honest baseline for "use the interface as intended." It fails on identity — many objects match any description — and, paradoxically, long detailed captions appear to *hurt* editability rather than help, the requested edit getting lost.

- **Contrastive / language-completion embedding methods.** Prior work that learns a representation in this same embedding space, using contrastive or language-completion objectives. They capture coarse semantics but produce visual corruption when pushed to synthesize.

- **GAN-inversion moves transplanted into the embedding space** — multi-vector codes, progressive vectors, regularization toward real word embeddings, per-image auxiliary codes, pivotal-style generator tuning. Each is a candidate; the open question is whether any of them actually helps in this *textual* latent space or just imports the same tradeoff (or makes it worse).

## Evaluation settings

- **Concepts.** A small collection of user-supplied concepts, each given by 3–5 (and in a size-sensitivity study, 1–25) casual images spanning varied backgrounds/poses.
- **Reconstruction metric.** For each concept, generate 64 images from a neutral "A photo of [concept]" prompt; report the average pairwise CLIP-space cosine similarity between generated images and the concept's training images. (CLIP measures *semantic* similarity and is relatively insensitive to exact shape.)
- **Editability metric.** Generate 64 samples (50 DDIM steps) for a battery of prompts of escalating difficulty — background change ("[concept] on the moon"), style change ("an oil painting of [concept]"), composition ("Elmo holding [concept]"); embed each sample in CLIP space, average, and take cosine similarity to the CLIP embedding of the *prompt with the concept marker removed*. Higher = more faithful to the requested edit.
- **Reference anchors.** "Image only" (always emit a training image, ignore the prompt) upper-bounds reconstruction; "Prompt only" (render the prompt without the concept) upper-bounds editability.
- **Human study.** Two questionnaires — rank methods by similarity to reference images, and rank by similarity to a target text — totaling 1,200 responses.

## Code framework

Known, pre-existing primitives: a frozen pretrained LDM with encoder `vae`, denoiser `unet`, tokenizer, and a text encoder whose first operation is an embedding lookup table; a noise scheduler; an optimizer; the standard diffusion training step. The contribution will fill the single empty slot below.

```python
import torch, torch.nn.functional as F

# --- frozen pretrained text-to-image LDM (already exists) ---
vae          = load_pretrained_autoencoder()      # E, D ; frozen
unet         = load_pretrained_denoiser()         # epsilon_theta ; frozen
tokenizer    = load_pretrained_tokenizer()        # string -> token ids
text_encoder = load_pretrained_text_encoder()     # embedding table + transformer
scheduler    = load_noise_scheduler()             # alphas, add_noise

# === THE SLOT THE METHOD WILL FILL ========================================
def adapt_to_new_concept(images, coarse_word):
    """Given a few images of one concept, make the frozen model able to
    render that concept on demand inside arbitrary prompts.
    TODO: what is the target of adaptation, what objective, what stays frozen?"""
    pass
# ==========================================================================

def build_conditioning(prompt):
    # tokenize -> embedding lookup -> transformer -> conditioning vectors
    # TODO: how does the new concept enter this pipeline?
    pass

def training_step(images, prompt):
    z = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    eps = torch.randn_like(z)
    t   = torch.randint(0, scheduler.config.num_train_timesteps, (z.shape[0],))
    z_t = scheduler.add_noise(z, eps, t)
    c   = build_conditioning(prompt)
    eps_pred = unet(z_t, t, c).sample
    loss = F.mse_loss(eps_pred, eps)          # the unchanged LDM objective
    # TODO: which parameters does the gradient actually update?
    return loss
```
