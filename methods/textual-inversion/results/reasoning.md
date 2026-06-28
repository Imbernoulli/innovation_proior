Let me start from the thing that actually frustrates me, not from any architecture. I have this text-to-image model that is genuinely astonishing — I type "an oil painting of a cottage in a thunderstorm" and it just *does* it, in the right style, with the right mood. And yet I cannot get it to draw the one clay sculpture sitting on my desk. There is no sentence that picks it out. "A clay sculpture" gives me *a* sculpture, never mine. I can pile on adjectives — "a headless seated clay figure with rough texture and a greenish patina" — and I get something vaguely in the neighborhood, but a hundred different objects satisfy that description equally well. The appearance is underdetermined by any caption I can write. So the goal: take 3–5 casual photos of this specific thing, and afterward be able to write "a photo of [it] on the beach" and have it work.

The obvious move is to finetune the model on my five photos. I can already feel why that's wrong. Five images, a model with a billion-plus parameters — it will memorize the five images and, worse, it will forget. The whole reason this model is valuable is the prior it learned from hundreds of millions of image–caption pairs: how to render "beach", "oil painting", "thunderstorm", how to compose things it never saw together. Few-shot finetuning on my sculpture will drag those weights toward my five photos and the prior frays. Catastrophic forgetting. And the symptom I'd see is editability collapsing — I'd get my sculpture but I couldn't put it on the moon anymore. So: do not touch the model weights. That has to be a hard rule, because the prior *is* the product.

If the weights are frozen, then whatever I learn has to live in the *input*. Where does the prompt enter? Let me trace the pipeline backwards from the image. The denoiser is conditioned on $c_\theta(y)$, the output of the text encoder on prompt $y$. The text encoder is a transformer, frozen. Before the transformer, there's a tokenizer that turns the string into integer indices, and then — this is the part I keep coming back to — each index is used to look up a vector in an embedding table. The discrete word "cat" becomes a continuous vector $v_{\text{cat}}$, and *that* vector is what flows into the network. The string is discrete and unoptimizable, but the embedding vector is continuous. It's the first place in the whole stack where the input becomes a thing I can take gradients with respect to.

That suggests a reframing I want to test rather than assume: that a "word", as far as the frozen model is concerned, just *is* a vector at this lookup boundary, and the vocabulary the model can talk about is exactly the set of vectors that enter the transformer. If that's true, then to teach the model a new concept I don't need a new sentence and I don't need new weights — I need a new *vector*. The plan would be: designate a placeholder string, call it $S_*$, make sure it tokenizes to one token, and intercept that token's embedding lookup with a learnable vector $v_*$. Then any prompt containing $S_*$ — "a photo of $S_*$", "an oil painting of $S_*$" — gets tokenized, the placeholder position receives $v_*$, and the rest of the frozen pipeline treats it as an ordinary word.

Before I get attached to this, I should check the two mechanical claims it rests on, because if either is false the whole approach is dead. Claim one: the placeholder really does collapse to a single token I can target. With a BERT-style tokenizer, a tokenized string comes back padded to a fixed length with zeros, so a one-token word should look like `[CLS] token [SEP] 0 0 …`. Counting nonzeros is then a clean test — three for a genuine single token, four if the "word" secretly split into two pieces. Let me run that out:

```
single-token  ids = [101, 1037,  102,    0,   0,   0]  -> count_nonzero = 3
two-token     ids = [101, 1037, 2061,  102,   0,   0]  -> count_nonzero = 4
ids[0,1] = 1037   # the token I'd actually grab
```

So `assert count_nonzero == 3` is exactly the guard I want, and `ids[0,1]` picks out the placeholder's token id. Good — the single-token assumption is enforceable, not just hoped for.

Claim two is the substitution itself: when a prompt contains $S_*$ at some arbitrary position, can I swap *that slot's* embedding for $v_*$ and leave everything else untouched, across a whole batch where the placeholder lands in different positions? Let me trace it concretely. Take two prompts in a batch, placeholder token id 7, and a tiny embedding dimension of 4 so I can read the tensors:

```
tokenized = [[1, 2, 3, 7, 99],     # placeholder in slot 3
             [1, 2, 7, 4, 99]]      # placeholder in slot 2
positions = where(tokenized == 7) = (rows [0,1], cols [3,2])
v_star    = [-1, -2, -3, -4]
```

After `embedded[positions] = v_star`:

```
row 0:  [ 0  1  2  3]      row 1:  [20 21 22 23]
        [ 4  5  6  7]              [24 25 26 27]
        [ 8  9 10 11]              [-1 -2 -3 -4]   <- slot 2 replaced
        [-1 -2 -3 -4]   <- slot 3 replaced   [32 33 34 35]
        [16 17 18 19]              [36 37 38 39]
```

That's what I wanted: `torch.where` finds the placeholder in each row independently — slot 3 in the first, slot 2 in the second — and `v_*` is broadcast into precisely those positions while every other embedding is byte-for-byte unchanged. So I really am injecting a pseudo-word into the model's vocabulary, at whatever position it appears, without changing the tokenizer or the old embedding table. And whether composition "comes for free" is now a concrete empirical question rather than a slogan: $v_*$ sits in the same space as every real word, so *if* the frozen transformer already knows how to combine words, it should combine this one too — but that's something the editability experiments will have to actually show, not something I get to declare here.

Now, what objective do I optimize $v_*$ with? This is where I have to be careful. There's earlier work that put representations in this same embedding space using contrastive or language-completion losses, and at first that seems like the natural precedent to follow. But think about what those losses *require*. A contrastive loss asks: is this embedding close to the image in some joint space? A completion loss asks: does this embedding predict plausible surrounding text? Neither one ever forces the embedding to encode enough to *redraw* the object pixel by pixel. They can be fully satisfied by an embedding that means "a sculpture-ish thing" — coarse semantics — while staying blind to the specific geometry and surface. And the thing I'm going to ask the model to do is *synthesize the appearance*. The objective should match the downstream use. So I want to optimize the embedding with the exact objective the model uses for synthesis: the denoising reconstruction loss. If the embedding is good, then conditioning on it should let the frozen denoiser reconstruct my images from noise. Reconstruction is the one objective that can't be satisfied without putting fine visual detail into the vector — you cannot denoise back to *my* sculpture's silhouette and texture from a vector that only knows "sculpture-ish".

Concretely, I reuse the model's own training loss, unchanged:
$$v_* = \arg\min_v \; \mathbb{E}_{z\sim\mathcal{E}(x),\,y,\,\epsilon\sim\mathcal{N}(0,1),\,t}\big[\lVert \epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\rVert_2^2\big],$$
where $x$ ranges over my few images, $z=\mathcal{E}(x)$ is its latent, $z_t$ is that latent noised to time $t$, and the prompt $y$ is a short neutral template containing $S_*$. Both $c_\theta$ (text encoder) and $\epsilon_\theta$ (denoiser) stay frozen; the *only* thing the gradient should update is $v_*$.

But "should update only $v_*$" is itself a claim I should verify, not assume — $v_*$ flows through the frozen encoder and the frozen denoiser before it ever reaches the loss, and it would be easy to be wrong about whether gradient quietly leaks into those frozen modules. Let me build a stripped-down analogue: a "frozen encoder" matrix $W_{\text{enc}}$ and "frozen denoiser" $W_{\text{den}}$, both with `requires_grad=False`, one trainable leaf $v_*$ substituted into a slot of the embedding, push it through both matrices to a prediction, take the MSE against a noise target, and backprop:

```
v_star.grad is not None : True
v_star.grad nonzero     : True
W_enc.requires_grad / W_den.requires_grad : False  False
W_enc.grad : None     W_den.grad : None
```

So gradient does reach $v_*$ (non-None and actually nonzero — the update is real), and the two frozen matrices receive `grad = None` even though they sit squarely on the forward path. Freezing them with `requires_grad_(False)` is enough; nothing else moves. That's the property the whole method depends on, and it holds. Same training scheme as the original model, same loss, one trainable vector. There's something satisfying about that — I'm not inventing a new loss, I'm asking "what single embedding, plugged into your existing machinery, makes you able to denoise images of my concept?"

The prompts $y$: I don't want to overfit to one phrasing, and I don't have a caption for the concept anyway. So I sample neutral context templates — "a photo of a $S_*$", "a rendition of a $S_*$", "a cropped photo of the $S_*$", "a close-up photo of the $S_*$", and so on — the kind of generic, content-free scaffolding used for class templates. Randomizing over them means $v_*$ has to carry the concept's appearance regardless of the surrounding boilerplate, rather than entangling with one specific sentence.

Initialization. Starting $v_*$ from random noise seems wasteful — I actually know a coarse category for the concept ("sculpture", "cat"). The embedding of that single coarse word already sits in a sensible region of the space and already pulls the right kind of prior. So initialize $v_*$ with the embedding of a one-word descriptor. It gives the optimization a running start in the right neighborhood instead of wandering in from nowhere.

Let me sanity-check the optimization itself. I have one embedding vector — 1280 dimensions in this LDM text encoder — and a handful of images. The loss is the standard diffusion loss, so I keep the original hyperparameters; the only knob with real leverage is the learning rate. A few thousand steps should be plenty for one vector. Fine.

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

Now the code. The skeleton is just the model's own training step with two surgical changes I've already traced in isolation: intercept the embedding lookup for the placeholder token (the `torch.where` substitution above), and make sure the optimizer sees *only* the replacement vector (the gradient-isolation check above).

```python
import random
import torch, torch.nn as nn, torch.nn.functional as F

ldm = load_pretrained_ldm()
ldm.first_stage_model.requires_grad_(False)  # autoencoder
ldm.model.requires_grad_(False)              # denoiser
ldm.cond_stage_model.requires_grad_(False)   # text encoder

def one_bert_token(tokenizer, text):
    ids = tokenizer(text)
    assert torch.count_nonzero(ids) == 3      # [CLS], token, [SEP]
    return ids[0, 1]

class EmbeddingManager(nn.Module):
    def __init__(self, text_encoder, placeholder="*", initializer="sculpture"):
        super().__init__()
        self.ph_id = one_bert_token(text_encoder.tknz_fn, placeholder)
        init_id = one_bert_token(text_encoder.tknz_fn, initializer)
        with torch.no_grad():
            init = text_encoder.transformer.token_emb(init_id.cpu())
        self.v_star = nn.Parameter(init.unsqueeze(0))

    def forward(self, tokenized_text, embedded_text):
        loc = torch.where(tokenized_text == self.ph_id.to(tokenized_text.device))
        embedded_text[loc] = self.v_star.to(embedded_text.device)
        return embedded_text

emb = EmbeddingManager(ldm.cond_stage_model, "*", "sculpture")
optimizer = torch.optim.AdamW([emb.v_star], lr=0.04)  # 0.005 base, scaled by 2 GPUs x batch 4

templates = ["a photo of a {}", "a rendering of a {}", "a close-up photo of the {}", ...]

for step in range(5000):
    img = sample_concept_batch()                   # the 3-5 photos, repeatedly sampled
    with torch.no_grad():
        posterior = ldm.encode_first_stage(img)
        z = ldm.get_first_stage_encoding(posterior)
    eps = torch.randn_like(z)
    t   = torch.randint(0, ldm.num_timesteps, (z.shape[0],), device=z.device)
    z_t = ldm.q_sample(x_start=z, t=t, noise=eps)

    prompts = [random.choice(templates).format("*") for _ in range(z.shape[0])]
    c = ldm.cond_stage_model.encode(prompts, embedding_manager=emb)

    eps_pred = ldm.apply_model(z_t, t, c)
    loss = F.mse_loss(eps_pred.float(), eps.float())   # the unchanged epsilon objective
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The causal chain, start to end: I can't describe my concept in words, and finetuning would forget the prior, so I freeze the model and look for the most directly optimizable place the input enters — the per-token embedding boundary, where I checked that a single placeholder token can be isolated and its embedding swapped position-independently across a batch. A "word" to this model is just a vector there, so a new concept is a new vector $v_*$. Because generation is visual, I optimize that vector with the model's own denoising-reconstruction loss over a few images in neutral templates, seeding it from a coarse-word embedding and optimizing only that standalone replacement parameter — and I confirmed the gradient really does reach only $v_*$ while the frozen modules on its path stay at `grad = None`. Reaching for GAN-inversion tricks — more vectors, progressive growth, regularization, per-image tokens — buys nothing or hurts once I measure reconstruction and editability separately, because the real structure is a distortion–editability tradeoff, and a single vector already sits at a good point on it whose location I tune with the learning rate; verbose human captions are worse still, since selective attention lets a long description crowd out the requested edit. One learnable vector, frozen everything else: a new word in the model's vocabulary.
