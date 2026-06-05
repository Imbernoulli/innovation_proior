Let me start from the thing that actually frustrates me, not from any architecture. I have this text-to-image model that is genuinely astonishing — I type "an oil painting of a cottage in a thunderstorm" and it just *does* it, in the right style, with the right mood. And yet I cannot get it to draw the one clay sculpture sitting on my desk. There is no sentence that picks it out. "A clay sculpture" gives me *a* sculpture, never mine. I can pile on adjectives — "a headless seated clay figure with rough texture and a greenish patina" — and I get something vaguely in the neighborhood, but a hundred different objects satisfy that description equally well. The appearance is underdetermined by any caption I can write. So the goal: take 3–5 casual photos of this specific thing, and afterward be able to write "a photo of [it] on the beach" and have it work.

The obvious move is to finetune the model on my five photos. I can already feel why that's wrong. Five images, a model with a billion-plus parameters — it will memorize the five images and, worse, it will forget. The whole reason this model is valuable is the prior it learned from hundreds of millions of image–caption pairs: how to render "beach", "oil painting", "thunderstorm", how to compose things it never saw together. Few-shot finetuning on my sculpture will drag those weights toward my five photos and the prior frays. Catastrophic forgetting. And the symptom I'd see is editability collapsing — I'd get my sculpture but I couldn't put it on the moon anymore. So: do not touch the model weights. That has to be a hard rule, because the prior *is* the product.

If the weights are frozen, then whatever I learn has to live in the *input*. Where does the prompt enter? Let me trace the pipeline backwards from the image. The denoiser is conditioned on $c_\theta(y)$, the output of the text encoder on prompt $y$. The text encoder is a transformer, frozen. Before the transformer, there's a tokenizer that turns the string into integer indices, and then — this is the part I keep coming back to — each index is used to look up a vector in an embedding table. The discrete word "cat" becomes a continuous vector $v_{\text{cat}}$, and *that* vector is what flows into the network. The string is discrete and unoptimizable, but the embedding vector is continuous. It's the first place in the whole stack where the input becomes a thing I can take gradients with respect to.

So here's the reframing. A "word", as far as the frozen model is concerned, *is* a vector in this embedding table. The vocabulary the model can talk about is exactly the set of vectors that table holds. To teach the model a new concept, I don't need a new sentence and I don't need new weights — I need a new *vector*. I'll designate a placeholder string, call it $S_*$, reserve a slot for it in the table, and put a learnable vector $v_*$ there. Then any prompt containing $S_*$ — "a photo of $S_*$", "an oil painting of $S_*$" — gets tokenized, $S_*$ maps to $v_*$, and the rest of the frozen pipeline treats it as an ordinary word. I'm injecting a pseudo-word into the model's vocabulary. Composition comes for free, because $v_*$ sits in the same space as every real word and the frozen transformer already knows how to combine words.

Now, what objective do I optimize $v_*$ with? This is where I have to be careful, and where I think prior attempts went wrong. There's earlier work that put representations in this same embedding space using contrastive or language-completion losses. But think about what those losses *require*. A contrastive loss asks: is this embedding close to the image in some joint space? A completion loss asks: does this embedding predict plausible surrounding text? Neither one ever forces the embedding to encode enough to *redraw* the object pixel by pixel. They capture that it's "a sculpture-ish thing", coarse semantics, but not the specific geometry and surface. And generation is a *visual* task — I'm going to ask the model to synthesize the appearance. So I should optimize the embedding with the exact objective the model uses for synthesis: the denoising reconstruction loss. If the embedding is good, then conditioning on it should let the frozen denoiser reconstruct my images from noise. That's a reconstruction objective, and reconstruction is precisely what forces fine visual detail into the vector. The objective should match the downstream use.

Concretely, I reuse the model's own training loss, unchanged:
$$v_* = \arg\min_v \; \mathbb{E}_{z\sim\mathcal{E}(x),\,y,\,\epsilon\sim\mathcal{N}(0,1),\,t}\big[\lVert \epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\rVert_2^2\big],$$
where $x$ ranges over my few images, $z=\mathcal{E}(x)$ is its latent, $z_t$ is that latent noised to time $t$, and the prompt $y$ is a short neutral template containing $S_*$. Both $c_\theta$ (text encoder) and $\epsilon_\theta$ (denoiser) stay frozen; the *only* thing the gradient updates is $v_*$. Same training scheme as the original model, same loss, one trainable vector. There's something satisfying about that — I'm not inventing a new loss, I'm asking "what single embedding, plugged into your existing machinery, makes you able to denoise images of my concept?"

The prompts $y$: I don't want to overfit to one phrasing, and I don't have a caption for the concept anyway. So I sample neutral context templates — "a photo of a $S_*$", "a rendition of a $S_*$", "a cropped photo of the $S_*$", "a close-up photo of the $S_*$", and so on — the kind of generic, content-free scaffolding used for class templates. Randomizing over them means $v_*$ has to carry the concept's appearance regardless of the surrounding boilerplate, rather than entangling with one specific sentence.

Initialization. Starting $v_*$ from random noise seems wasteful — I actually know a coarse category for the concept ("sculpture", "cat"). The embedding of that single coarse word already sits in a sensible region of the space and already pulls the right kind of prior. So initialize $v_*$ with the embedding of a one-word descriptor. It gives the optimization a running start in the right neighborhood instead of wandering in from nowhere.

Let me sanity-check the optimization itself. I have one vector, a few hundred dimensions, and a handful of images. The loss is the standard diffusion loss, so I keep the original hyperparameters; the only knob with real leverage is the learning rate. A few thousand steps should be plenty for one vector. Fine.

Now — is one vector enough? My instinct says I'm leaving capacity on the table. A single embedding has to encode an entire object's appearance, and intuitively that's a tight bottleneck. This is exactly where the GAN-inversion playbook is screaming at me, because GAN inversion is the established craft of "find a latent that reproduces this image," and it learned hard lessons I should not have to relearn. Let me bring its moves over and actually try them, because the embedding space here is uncharted and I have no idea which of its intuitions transfer.

First move: extended, multi-vector latent space. In GAN inversion, going from a single $w$ to an extended set of per-layer codes ($\mathcal{W}+$) buys a lot of reconstruction fidelity. The analogue: represent $S_*$ with two or three learned vectors instead of one — equivalently, describe the concept with several pseudo-words. More capacity, should reconstruct better. Let me hold that thought and try the others before judging.

Second move: progressive introduction. Rather than optimizing all the extra vectors at once, start with one, add a second after a couple thousand steps, a third later. The hope is the first vector grabs the coarse identity and the later ones refine details, instead of all of them redundantly fighting over the same information.

Third move: regularization toward the natural distribution. This is the big one from GAN inversion, tied to the distortion–editability tradeoff. The lesson there: codes that drift far from the distribution the generator saw in training reconstruct well but become uneditable and brittle; codes kept near that distribution stay editable. The textual analogue: keep $v_*$ close to the cloud of *real* word embeddings. I can add an L2 penalty pulling $v_*$ toward the embedding of its coarse descriptor. If the tradeoff exists here too, this should improve editability.

Fourth move: per-image tokens. My few images differ in background, pose, lighting. If I force one shared vector to explain all of them, it might waste capacity memorizing incidental background. So introduce, alongside the shared $S_*$, a per-image placeholder $S_i$ with its own embedding $v_i$, and train on sentences like "a photo of $S_*$ with $S_i$", matching each image to its own $S_i$. The intent: the model offloads the shared concept into $S_*$ and dumps per-image junk into $S_i$, cleaning up the shared vector.

And then there's the heavier two-stage idea from GAN inversion — pivotal tuning: invert into an editable pivot code, *then* lightly finetune the generator so the pivot reconstructs better, supposedly getting fidelity without losing editability. But that breaks my hard rule — it touches the weights. Let me note it as a tempting escape hatch and come back only if the lightweight approaches stall.

So now I have a menu, and the honest thing is to evaluate it, which means I need to *measure* reconstruction and editability separately, because the whole GAN-inversion framing predicts they trade off. For reconstruction: generate a batch of images from "a photo of $S_*$" and measure their average CLIP-space cosine similarity to my training images — a semantic match score. For editability: take harder prompts ("on the moon", "an oil painting of $S_*$", "Elmo holding a $S_*$"), generate samples, and measure CLIP similarity to the prompt *with $S_*$ deleted* — i.e., did the model actually do the background/style/composition I asked for. Two axes. And I'll anchor the scale with two references: "image only" (just return a training image — perfect reconstruction, zero editability) and "prompt only" (render the prompt ignoring the concept — perfect editability, zero identity).

Here's where I expected to be vindicated and instead got humbled. The single-vector method reconstructs about as well as the multi-vector setups — its semantic reconstruction is already on par with simply sampling real images from the training set. So the extra capacity bought me essentially nothing on the axis I thought it would help. And on editability, the multi-word setups are *worse* — markedly worse. The progressive and per-image variants, the same story: complexity that either does nothing or actively hurts. The textual embedding space is far more flexible than I gave it credit for; a single vector already captures the concept with high accuracy.

And the regularization experiment is the one that reorganizes my whole picture. The distortion–editability tradeoff *is here*, exactly as GAN inversion warned. Embeddings that stay near the real-word distribution — because I regularized them, or used fewer vectors, or a lower learning rate — are easy to edit but miss the concept's details. Embeddings that drift far away — more vectors, higher learning rate, no regularization — nail the details but become hard or impossible to edit. So all my GAN-inspired "improvements" weren't improvements; they were just *moving along the same tradeoff curve*, mostly in the wrong direction or to no benefit. The right framing isn't "add capacity to reconstruct better." It's "find a good operating point on a curve, with a single vector, and the knob that moves you along it is simply the learning rate." A lower learning rate keeps $v_*$ near the word cloud (editable, less faithful); a higher one lets it roam (faithful, less editable). The user gets to choose. That's a much cleaner story than a pile of auxiliary vectors.

The human-caption comparison closes the loop and surprises me a second time. Replacing $S_*$ with a careful human description not only fails to capture the likeness — it also *reduces* editability. At first that's counterintuitive: more words should mean more control. But it lines up with how these vision-language encoders behave: they attend selectively, focusing on a subset of the semantically loaded tokens. A long object description hogs that attention, so the model fixates on describing the object and ignores my requested "on the moon". My single pseudo-word minimizes that risk — one token for identity, leaving the rest of the prompt's tokens free to steer the edit. So the single vector isn't just simpler; it's *better* for editing than verbose human text, for a concrete attentional reason.

One caveat I want to keep honest about: my reconstruction scores match real images, but the metric is CLIP semantic similarity, which is forgiving about exact shape. So "reconstruction on par with real images" should be read as semantic, not pixel-exact — shape fidelity is still an open frontier, not something I get to claim.

Let me also pin down the pieces I deliberately left out. Pivotal tuning, if applied naïvely — invert a pseudo-word, then finetune the generator to reconstruct better — does improve shape preservation, but it collapses editability at the high guidance scales this model uses. And the bipartite DDIM-inversion route (find the initial noise that maps to my image, then change the text while freezing that noise) drifts in structure at this model's typical guidance scales (5–10): at those scales the denoiser won't hold the object's structure across prompt changes, and only at much lower guidance (~2, the regime more powerful models use) does structure survive — but then prompt-matching is poor. Both break the frozen-weights rule or fight the guidance regime, so neither belongs in the core method. The lightweight single-vector inversion stands on its own.

Now the code. The skeleton is just the model's own training step with two surgical changes: add a slot to the embedding table for $S_*$, and make sure the gradient updates *only* that slot.

```python
import torch, torch.nn.functional as F

vae, unet, text_encoder, tokenizer, scheduler = load_pretrained_ldm()
vae.requires_grad_(False)
unet.requires_grad_(False)
# freeze the text transformer; only the embedding table will have one live row
text_encoder.text_model.encoder.requires_grad_(False)
text_encoder.text_model.final_layer_norm.requires_grad_(False)
text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

# --- inject the pseudo-word S_* and seed it from a coarse descriptor ---
placeholder = "<my-sculpture>"
tokenizer.add_tokens([placeholder])
text_encoder.resize_token_embeddings(len(tokenizer))
ph_id   = tokenizer.convert_tokens_to_ids(placeholder)
init_id = tokenizer.encode("sculpture", add_special_tokens=False)[0]  # coarse word
embeds  = text_encoder.get_input_embeddings().weight.data
embeds[ph_id] = embeds[init_id].clone()           # initialize v_* from "sculpture"

orig_embeds = embeds.clone()                       # to restore every other row
# the ONLY trainable parameter is the embedding table; we zero out all but one row
optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=5e-3)

templates = ["a photo of a {}", "a rendition of a {}", "a close-up photo of the {}", ...]

for step in range(5000):
    img = sample_concept_image()                   # one of the 3-5 photos
    z   = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
    eps = torch.randn_like(z)
    t   = torch.randint(0, scheduler.config.num_train_timesteps, (z.shape[0],))
    z_t = scheduler.add_noise(z, eps, t)

    prompt = random.choice(templates).format(placeholder)   # S_* in neutral context
    ids    = tokenizer(prompt, return_tensors="pt").input_ids
    c      = text_encoder(ids)[0]                   # frozen transformer turns v_* into conditioning

    eps_pred = unet(z_t, t, c).sample
    loss = F.mse_loss(eps_pred.float(), eps.float())   # the unchanged LDM objective
    loss.backward()
    optimizer.step(); optimizer.zero_grad()

    # keep every embedding fixed except the one we are learning -> no forgetting
    with torch.no_grad():
        keep = torch.ones(len(tokenizer), dtype=torch.bool); keep[ph_id] = False
        text_encoder.get_input_embeddings().weight[keep] = orig_embeds[keep]
```

The causal chain, start to end: I can't describe my concept in words, and finetuning would forget the prior, so I freeze the model and look for the most directly optimizable place the input enters — the per-token embedding table. A "word" to this model is just a vector there, so a new concept is a new vector $v_*$. Because generation is visual, I optimize that vector with the model's own denoising-reconstruction loss over a few images in neutral templates, seeding it from a coarse-word embedding and updating only that one row so nothing else moves. Reaching for GAN-inversion tricks — more vectors, progressive growth, regularization, per-image tokens — buys nothing or hurts, because the real structure is a distortion–editability tradeoff, and a single vector already sits at a good point on it whose location I tune with the learning rate alone; verbose human captions are worse still, since selective attention lets a long description crowd out the requested edit. One learnable vector, frozen everything else: a new word in the model's vocabulary.
