Let me think about why our language systems are so narrow. The standard move is: pick a task, gather a big supervised dataset, train a model with a task-specific architecture and head, and you get something that's good on that task's distribution and brittle everywhere else. Even the transfer recipe — pretrain a representation, then fine-tune with a supervised head per task — still demands labeled data and a fresh adaptation for every single task. What I actually want is one model that, trained once with no supervision, can do many tasks with no parameter updates and no architectural surgery. I want to find out whether that's even possible, and if so what the data, the tokenizer, and the model have to be.

Start from what a language model is. It estimates the joint distribution over a symbol sequence, factorized autoregressively: p(x) = Πᵢ p(sᵢ | s₁,…,sᵢ₋₁). That's it — predict the next symbol given the prefix. Now a single supervised task is estimating a conditional p(output | input). A general system that should perform *many* tasks ought to condition on the task too: p(output | input, task). How do you condition on a task? You could build it into the architecture — separate encoders and decoders per task — or into the algorithm, like a meta-learner with an inner and outer loop. But there's a far simpler medium sitting in front of me: language itself can specify the task, the input, and the output all as one sequence of symbols. A translation example is just the sequence ("translate to french", english text, french text). A reading-comprehension example is ("answer the question", document, question, answer). The task description, the input, and the target are all tokens.

Stare at that for a second, because something falls out. If tasks, inputs, and outputs are all just token sequences, then p(output | input, task) is *also* just a conditional of the same autoregressive language model — it's p(next tokens | preceding tokens) where the preceding tokens happen to spell out a task and an input. The supervised objective for a task is the same as the unsupervised language-modeling objective, only evaluated on a subset of the sequence — the output tokens. So the global minimum of the unsupervised objective is also the global minimum of every one of these supervised objectives. The task behavior isn't something I need to bolt on; it's *contained* in the language-modeling solution. A model that truly learns to predict the next token over text that contains task demonstrations has, at its optimum, learned to perform those tasks. The only question is whether I can reach that optimum in practice — whether a big enough model on enough varied text actually picks up the multitask behavior, or whether it's just a theoretical fixed point. Preliminary toy experiments say it does begin to happen, just more slowly than explicit supervision. So the bet is: scale the model and the data until zero-shot task behavior emerges, with no fine-tuning at all.

If that's the plan, the data is now the most important design choice, because the model can only do a task zero-shot if natural demonstrations of that task appear somewhere in the training text. Translation pairs, question-answer pairs, summaries followed by articles — these have to occur "in the wild." Most language models train on a single domain: news, or Wikipedia, or fiction. That won't surface a broad range of task demonstrations. I need text that is large *and* diverse. The obvious source is a web scrape like Common Crawl — orders of magnitude bigger than any curated corpus — but its quality is terrible; huge fractions are unintelligible, and curating it by hand would be impossibly expensive. I need a cheap proxy for "a human found this page worth reading." Reddit gives me one for free: scrape only the pages that are *outbound links posted to Reddit and that received at least three karma*. The upvotes are a crowd-sourced signal that the link was interesting, educational, or at least funny — a human filter I didn't have to build. Take all such links, pull the article text out of the HTML with a couple of content extractors, deduplicate, and lightly clean. That gives a diverse, reasonably high-quality corpus of millions of documents and tens of gigabytes. One more thing: I must *remove* Wikipedia from it, because Wikipedia is the training or test source for so many of the benchmarks I'll evaluate on zero-shot, and leaving it in would let the model memorize evaluation text and contaminate the comparison.

Now the input representation, and this matters more than it looks. My whole premise is that one model assigns probability to arbitrary text and can be evaluated on *any* dataset without retraining. But standard pipelines lowercase, apply language-specific tokenization, and emit an out-of-vocabulary token for anything unseen — every one of those choices restricts which strings the model can even represent and ties the evaluation to the preprocessing. I want a tokenizer that can encode literally any string with no preprocessing and no unknown token.

The fully general option is bytes: every string is a sequence of UTF-8 bytes, so a byte-level model can represent anything with a base alphabet of just 256. But byte-level language models, in my own attempts and in the literature, are not competitive with word-level ones on large datasets — operating at the byte granularity wastes the model's capacity on low-level structure. Word-level models are accurate but require the lossy preprocessing I'm trying to escape. So I want the middle ground: byte-pair encoding, which merges frequent symbol pairs and so behaves word-like for common sequences and character-like for rare ones. There's a trap, though. Reference BPE operates on Unicode *code points*, not bytes — and to cover all of Unicode the base vocabulary alone would exceed 130,000 symbols before a single merge, which is far larger than the 32K–64K vocabularies BPE is meant to give. So apply BPE to the *byte* sequence instead: the base vocabulary is just 256, and I keep full generality.

But naively running BPE on raw bytes by greedy frequency produces a different pathology. I watched it happen: the merges produce many near-duplicate tokens for the same word — "dog." and "dog!" and "dog?" — because the punctuation is frequent enough to merge onto the word. That squanders limited vocabulary slots and model capacity on what are really the same word plus different trailing punctuation. The fix is to forbid BPE from merging across *character categories* — letters don't merge with punctuation, punctuation doesn't merge with digits — so "dog" stays one token regardless of what follows it. I'll make a single exception for spaces, which improves compression a lot while barely fragmenting words. The result is the efficiency of word-level BPE with the universality of bytes: a vocabulary of about 50,000 that can encode any Unicode string, no UNK, no preprocessing, evaluable on any dataset.

Now the model. It's a decoder-only Transformer language model, largely the same generative-pretraining architecture as before, but I'm about to make it much deeper — up to dozens of layers — and naive deep Transformers are hard to optimize, so I need to fix the normalization and initialization. In the original design the layer normalization sits *after* each sub-block, on the residual pathway. That means the identity shortcut, the thing that's supposed to let gradients and signal flow cleanly through depth, gets a normalization stuck on it at every layer. The lesson from very deep residual networks is to keep the shortcut a clean identity and move the normalization to the *input* of each sub-block — pre-normalization. So each block becomes: normalize, then attend, then add back to the input; normalize again, then feed-forward, then add back. x ← x + attn(norm(x)); x ← x + mlp(norm(x)). The residual path is now an unobstructed sum, which is what keeps a forty-eight-layer stack trainable.

That refactor has a consequence I have to patch. Because the normalization now lives at each block's input and never on the residual path, the output of the whole stack — the running sum of all the residual contributions — comes out *un*normalized. So I add one extra layer normalization after the final block, before the output projection, to normalize the accumulated representation.

And one more issue with depth at initialization. The residual path is a sum of N contributions, one per residual layer, and at initialization those contributions add up — their variance accumulates along the path, so by the top of a deep stack the activations are inflated. To keep the accumulated signal controlled at initialization, scale the weights of the residual layers by 1/√N, where N is the number of residual layers. The √N is exactly right because N independent contributions of unit scale sum to something of scale √N, so dividing each by √N brings the total back to unit scale.

A couple of capacity choices follow from the goal. I want to model long-range structure — web documents are long, and the long-dependency benchmarks need it — so I extend the context window from the previous 512 tokens to 1024. The vocabulary is the ~50,257 from the byte-level BPE. And since the whole thesis is that zero-shot ability emerges with scale, I train a series of log-uniformly spaced sizes to watch the trend: 12 layers at width 768 (the same size as the original generative-pretraining model), 24 layers at 1024 (matching the largest contemporary bidirectional encoder), 36 layers at 1280, and the largest at 48 layers and width 1600. I expect each task's zero-shot performance to climb with capacity. I tune only the learning rate per size, on a small held-out slice of the corpus, and I note these models still *underfit* the corpus — held-out perplexity is still improving — which says the bottleneck is capacity and compute, not data, exactly as the thesis predicts.

Let me write the architecture, filling the slots: the pre-norm block, the byte-level BPE, the curated corpus, and the residual-scaled initialization.

```python
import tensorflow as tf, numpy as np

def gelu(x):
    return 0.5*x*(1 + tf.tanh(np.sqrt(2/np.pi)*(x + 0.044715*tf.pow(x, 3))))

def norm(x, scope, axis=-1, epsilon=1e-5):
    # layer normalization: zero mean, unit variance, then learned gain g and bias b
    with tf.variable_scope(scope):
        n = x.shape[-1].value
        g = tf.get_variable('g', [n], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        return (x - u) * tf.rsqrt(s + epsilon) * g + b

def attn(x, scope, n_state, *, past, hparams):
    # causal self-attention: q,k,v from a conv1d, scale by 1/sqrt(d_k),
    # mask out future positions (lower-triangular), softmax, then project
    ...

def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        h = gelu(conv1d(x, 'c_fc', n_state))     # 4x inner width
        return conv1d(h, 'c_proj', x.shape[-1].value)

def block(x, scope, *, past, hparams):
    # PRE-norm: normalization at the INPUT of each sub-block, clean identity residual
    with tf.variable_scope(scope):
        a, present = attn(norm(x, 'ln_1'), 'attn', hparams.n_embd, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', 4*hparams.n_embd, hparams=hparams)
        x = x + m
        return x, present

def model(hparams, X, past=None, scope='model'):
    with tf.variable_scope(scope):
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],   # token embedding (tied)
                              initializer=tf.random_normal_initializer(stddev=0.02))
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],     # position embedding
                              initializer=tf.random_normal_initializer(stddev=0.01))
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past))
        for layer in range(hparams.n_layer):
            # residual-layer weights initialized scaled by 1/sqrt(N) inside the blocks
            h, _ = block(h, 'h%d' % layer, past=None, hparams=hparams)
        h = norm(h, 'ln_f')                      # EXTRA final layer norm after the last block
        logits = tf.matmul(tf.reshape(h, [-1, hparams.n_embd]), wte, transpose_b=True)  # tied
        return logits

def lm_loss(logits, X):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=X[:, 1:], logits=logits[:, :-1]))    # plain next-token prediction
```

```python
# --- byte-level BPE: base 256, no merges across character categories (space excepted) ---
def bytes_to_unicode():
    # reversible map from bytes to printable unicode so BPE can operate on a byte stream
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
# regex splits text by character category (letters/digits/punct/space) BEFORE merging,
# so "dog", "dog.", "dog!" share the token "dog"; vocabulary ~50,257; encodes ANY string, no UNK
```

```python
# --- WebText: diverse, human-filtered, no Wikipedia ---
def build_webtext():
    links = reddit_outbound_links(min_karma=3)        # human upvotes = quality proxy
    docs = [extract_article(url) for url in links]    # Dragnet + Newspaper extractors
    docs = dedupe(docs)
    docs = [d for d in docs if not is_wikipedia(d)]   # avoid contaminating zero-shot eval
    return docs                                       # ~8M docs, ~40GB
```

So the causal chain: I wanted one unsupervised model to do many tasks zero-shot, and the unlock was noticing that p(output | input, task) is just an ordinary conditional of an autoregressive language model when the task, input, and output are all written as tokens — and that the supervised objective is the unsupervised one restricted to a subset of the sequence, so task behavior is contained in the language-modeling optimum and should emerge with scale. That made the data the lever: I need diverse natural task demonstrations, so I built a corpus from Reddit-upvoted outbound links (cheap human quality filter) with Wikipedia removed to protect evaluation. To evaluate on any string with no preprocessing or unknown tokens I used byte-level BPE, forbidding merges across character categories so the vocabulary isn't wasted on punctuation variants of words. And to make the stack deep enough for the thesis to bite, I moved layer normalization to each sub-block's input for a clean residual path, added a final normalization to handle the now-unnormalized stack output, and scaled residual-layer initialization by 1/√N so signal doesn't blow up along the residual sum — then scaled the model from 117M up to 1.5B parameters and watched zero-shot task performance climb with capacity.
