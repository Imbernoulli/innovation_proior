The method I am describing is textual inversion, a way to teach a frozen text-to-image model a new visual concept from only a handful of images without changing any of the model's weights. The problem it solves is easy to state but hard to fix: I have a specific object, pet, or style in front of me, I can take three to five casual photos of it, and I want to be able to write prompts like "a photo of my thing on the beach" or "an oil painting of my thing" and have the model actually render that exact thing in the requested context. Ordinary captions fail because dozens of different objects match any description I can write, and finetuning the whole model on so few images destroys the rich prior that makes compositions like "on the beach" work in the first place.

My starting point is the observation that a word, as far as the frozen text encoder is concerned, is not really a string; it is the continuous embedding vector that the tokenizer pulls from the embedding table before the transformer layers run. That embedding table is the first place in the entire pipeline where the prompt becomes differentiable, and it is the only place I can introduce new information without touching the autoencoder, the denoiser, or the text encoder itself. So I reserve a placeholder token, conventionally written as "*", and I replace its ordinary embedding vector with a new learnable vector v*. Any prompt containing the placeholder now injects v* into the same compositional space as real words, which means the frozen transformer can combine it with ordinary tokens exactly as it would combine "cat" or "sculpture" with "beach" or "oil painting."

The next question is what objective should train v*. Since the downstream task is generation, not classification or contrastive retrieval, I use the model's own reconstruction objective. For a latent diffusion model, the standard training loss predicts the noise added to a latent code, conditioned on the text encoder output. I keep that loss unchanged, sample timesteps and noise in the usual way, and make only v* receive gradients. The training images are encoded by the frozen autoencoder into latents, those latents are noised, and the denoiser is asked to recover the original noise given a prompt like "a photo of a *". Because the loss is pixel-level reconstruction in latent space, v* is forced to encode not just coarse category information but the actual visual appearance that lets the denoiser redraw the concept.

I initialize v* from the embedding of a single coarse word such as "sculpture" or "cat." This is not because the coarse word is a good description; it clearly is not, since any such description underdetermines the specific instance. But the coarse word is a sensible point in embedding space near the right semantic region, and it gives optimization a much better starting point than a random vector. The prompts during training are neutral templates randomized from a small set: "a photo of a {}", "a rendering of a {}", "a close-up photo of the {}", and so on. Randomizing the template prevents v* from overfitting to a single sentence and instead pushes it to represent the concept independent of the surrounding syntactic scaffolding.

A natural impulse is to add capacity: why stop at one vector? GAN inversion has taught us that extended latent spaces often reconstruct better, so I consider variants with two or three learned vectors, progressive schedules that add vectors partway through training, per-image tokens to absorb background variation, and regularizers pulling the learned vectors back toward real word embeddings. But extra capacity only pays for itself if a single vector is genuinely capacity-limited, and every one of these moves buys whatever it buys the same way: by letting the embedding drift farther from the cloud of ordinary word embeddings, which is precisely the direction editability degrades along. That reframes the choice as a distortion-editability tradeoff rather than a missing-capacity problem: a vector kept near the word cloud edits easily but may leave fine detail uncaptured; a vector allowed to drift farther away can capture more detail but becomes harder to edit. Under that framing, the simplest knob that moves along the curve is the learning rate itself, not a pile of auxiliary vectors, so I default to a single embedding and let the learning rate alone decide where on that curve it sits.

Compared with human-written captions, the single pseudo-word has a separate advantage that has nothing to do with capacity: a long object description competes for the text encoder's attention with the rest of the prompt, since these encoders are known to attend selectively rather than spreading attention evenly across every token, so requested edits like "on the moon" risk getting crowded out. One token for identity leaves the remaining tokens free to steer composition, style, and background, whatever that one token's embedding encodes. The main limitation to keep in mind is that reconstruction, wherever it is measured, is measured semantically rather than pixel-exactly; CLIP similarity is forgiving about exact shape, so precise geometry preservation remains an open direction. Heavier techniques such as pivotal tuning or DDIM inversion at high guidance scales are set aside for a structural reason rather than a measured one: pivotal tuning finetunes the generator itself, which breaks the frozen-weights constraint the whole method depends on, and DDIM-based inversion only holds an object's structure through a text change at guidance scales below the range this model normally uses for editing, so it fights the same regime the rest of the method lives in. Both are best treated as orthogonal directions rather than replacements.

Here is the implementation exactly as it runs inside the latent diffusion model. An `EmbeddingManager` wraps the frozen text encoder: it looks up the single BERT token id for the placeholder string and for the initializer word, clones the initializer's embedding into the learnable parameter `v_star`, and on every forward pass swaps `v_star` into the embedded sequence wherever the placeholder token appears, leaving every other embedding row untouched. The training script freezes the autoencoder, the diffusion denoiser, and the text encoder outright, builds the manager with placeholder `"*"` and initializer `"sculpture"`, and optimizes only `v_star` with AdamW at learning rate 0.04. Each step samples a batch from the handful of concept images, encodes them into the frozen latent space, adds noise at a random timestep, sends a template prompt containing the placeholder through the frozen text encoder — with the manager's swap applied — into the frozen denoiser, and backpropagates the ordinary noise-prediction MSE loss into `v_star` alone.

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

The same `EmbeddingManager` is what supports the capacity variants I dismissed above — passing more vectors per token, staggering their introduction after 2000 and 4000 steps, adding an embedding regularizer toward the coarse descriptor, or adding per-image tokens are all changes to how the manager is constructed and to the training prompts, not changes to the frozen backbone. None of them is the default. The default, and the one that matters, is exactly what the loop above runs: one learned vector, the unchanged epsilon-prediction reconstruction loss, and the autoencoder, denoiser, and text encoder all frozen. That is textual inversion in its canonical form.
