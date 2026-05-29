OK, let me think this through from scratch. I want word vectors — one dense vector per word — that are *good*, and I have a fairly sharp notion of "good": not only should similar words sit near each other, but the offsets between words should be meaningful and consistent. I keep coming back to the thing people have been noticing: take vectors from a recurrent language model and compute vec("King") − vec("Man") + vec("Woman"), and the nearest word is "Queen". That is not a fluke of one pair; the same "feminine" offset, the same "plural" offset, the same "capital-of" offset seem to recur across many pairs. So there is real *linear* structure latent in these spaces. If I could get a space where that linear structure is crisp, then an enormous amount of downstream NLP becomes trivial vector arithmetic.

Two facts gnaw at me. First, the regularity seems to sharpen only when I push *both* the amount of data and the number of dimensions up — bump just the dimensionality and accuracy plateaus, bump just the data and it plateaus, push both together and it keeps climbing. Second, and this is the operative constraint, every method that currently produces these vectors is a neural language model, and every one of them is too slow to reach the scale where the regularity would really sharpen. Nobody has trained on more than a few hundred million words; dimensionality is stuck at 50–100. So this is not really a modeling-quality problem. It is a *cost* problem. If I want the regularity, I need orders of magnitude more data and dimensions, which means I need to cut the per-word training cost by orders of magnitude. Let me start there — where exactly is the cost going.

Let me write the per-example cost as Q (the number of parameters touched per training word; total training cost is just epochs × words × Q). Take Bengio's feedforward language model: embed the previous N words through a shared matrix (cost N·D), concatenate them into a projection layer, run a tanh hidden layer of width H over that dense projection (cost N·D·H), then a softmax over the whole vocabulary V (cost H·V). So Q = N·D + N·D·H + H·V. With V in the millions and H in the hundreds, H·V alone is hundreds of millions of multiply-adds *per training word*. And N·D·H is also large. The recurrent model trades the projection→hidden term for a recurrent H·H term, Q = H·H + H·V — same V bottleneck on the output, plus a dense recurrent matrix. So there are two cost centers staring at me: the **output softmax over V**, and the **dense non-linear hidden layer**.

Take the hidden layer first. Why is it there? In a *language model* the tanh hidden layer is doing real work — modeling the complicated, non-linear interactions among context words to predict the next word well. But I keep reminding myself: I don't actually want a great language model. I want great *vectors*. The vectors live in the projection layer, below the hidden layer. The hidden non-linearity is downstream of the thing I care about. And here's the suggestive part — the property I want is *linear*. The regularities are vector offsets, additions, subtractions. If what I ultimately want out is a linearly-structured space, maybe a model that is itself *linear* in the word vectors is not just adequate but actually *better suited* to producing that structure. The non-linear hidden layer may be paying a huge cost (N·D·H) for representational capacity that the word-vector task simply doesn't need, and that might even be working against the linearity I'm after. Let me commit to that bet: rip out the non-linear hidden layer entirely. Make the model log-linear — a model where the score of a word given its context is a simple inner product, no hidden layer in between. The capacity I lose, I'll buy back many times over by training on far more data with far bigger vectors.

There's precedent that reassures me. There was a two-step recipe — first learn word vectors with a single-hidden-layer network, then train an n-gram model on top of the *frozen* vectors — which already showed word vectors can be learned by a simpler standalone model, decoupled from building a full LM. I'm going to push that further: make the "simple model" as simple as possible, no hidden layer at all, and pour everything into scale.

So now, what is the simplest log-linear setup? I have a center word and the words around it. Two natural directions. One: use the surrounding context to predict the center word. Two: use the center word to predict each surrounding word. Let me sketch both as log-linear models and see which is cheaper and which is better.

Direction one — context predicts center. Take the 2c context words, look up each one's input vector, and... how do I combine them with no hidden layer? The cheapest combination that respects nothing about order is to just *sum or average* them. Average them into a single vector h_t, the same dimensionality D as a word vector, and use h_t to predict the center word through a softmax. On a corpus position t with selected context C_t, the objective is simply log p(w_t | h_t), where h_t = |C_t|^{-1}Σ_{j∈C_t} v_{w_j}. Because I'm averaging the context vectors into one shared position, the order of the context words is thrown away — it's a "bag of words", but a *continuous* one (distributed vectors, not one-hots). For a language model throwing away order would be a sin, but for learning *which words keep similar company* it's fine, maybe even helpful: it lets all the context positions reinforce the same target with shared parameters. If the output layer costs only a tree path, the per-word cost becomes Q = N·D + D·log₂V: N·D to gather the context, then about log₂V inner products of size D to predict the center. Call this the bag-of-words direction. If I add the future words too, not just the past — say four words on each side, predict the middle — I get more signal per center word at no real extra cost.

Direction two — center predicts context. Use the center word's input vector, and for each surrounding word run a softmax to predict it. Each surrounding word is a separate prediction sharing the same input vector, so for a fixed radius c the objective around position t is Σ_{−c≤j≤c, j≠0} log p(w_{t+j} | w_t). With a window of c on each side that's up to 2c predictions per center word, so it's more expensive per center word, but it manufactures far more (input, target) training pairs, and I suspect that's exactly what helps rare words and semantic structure: each word gets used as a predictor of many neighbors. With a tree output layer the usual cost expression is Q = C·(D + D·log₂V): for each selected context position I touch the input vector and then a path of output vectors. One refinement falls out naturally here. Distant words are usually less related to the center than near words, so I'd like to weight near context more. The clumsy way is to attach distance weights. The cheap way: for each center word, draw a window radius R uniformly from 1 to c, and use only the R words on each side. A word at distance 1 is included for every R; a word at distance c only when R = c. So nearer words get sampled into training more often — I've implemented a soft distance-weighting with zero extra parameters, just by randomizing the window size. I'll take that.

I'll keep both architectures — call them CBOW (the continuous-bag-of-words, context→center) and Skip-gram (center→context) — and let the data decide. CBOW is cheaper per center word; Skip-gram makes more pairs and I expect it to be stronger on semantics. Now both of them still have the *other* cost center sitting in them: that softmax over V. With the hidden layer gone, the softmax normalization is now the *entire* cost. So everything hinges on making it cheap. Let me write it down honestly.

For either direction the expensive piece is the same. Given the input representation h and target word w_O, the full softmax is

  p(w_O | h) = exp(v'_{w_O} · h) / Σ_{w=1}^{V} exp(v'_w · h).

For Skip-gram h is v_{w_I}; for the bag-of-words direction h is the averaged context vector. Note I've already written two different vectors per word: v_w, the *input* vector (the embedding I'll keep), and v'_w, the *output* vector used only in this prediction layer. I'll come back to why they're separate. The cost of the gradient ∇ log p(w_O | h) is dominated by that denominator: it sums over all V words, and the gradient touches every output vector v'_w. With V between 10^5 and 10^7, that's 10^5–10^7 inner products *per prediction*. That is the whole ballgame. I cannot afford to normalize over the vocabulary.

So how do I get the benefit of the softmax — a proper probability that pushes the right word up and all the wrong words down — without summing over V? Two ideas exist; let me work through both, because they have different strengths and I think I'll want both available.

First idea: keep it a genuine normalized probability, but factor the normalization into a tree so I never touch all V words at once. Build a binary tree with the V words at the leaves. To "predict a word" is to walk from the root to its leaf, and at each inner node make a binary decision: go left or go right. If at each inner node I have a logistic unit that, given h, outputs the probability of one branch, then the probability of a leaf is the product of the binary decisions along its path. Concretely, let the path to word w have inner nodes point_j in root-to-leaf order, and let code_j ∈ {0,1} be the stored branch bit. I will make code_j = 0 the positive branch, so its probability is σ(v'_{point_j} · h); code_j = 1 gets the opposite branch, σ(−v'_{point_j} · h). Because σ(x) + σ(−x) = 1, the two branch probabilities sum to 1 at every node. So:

  p(w | h) = Π_j σ( (1 − 2·code_j) · v'_{point_j} · h ).

Let me check this is actually a distribution. At the root the two children's probabilities sum to 1; descending, each node splits its incoming probability mass into two pieces summing to it; so the leaf masses sum to 1 over the whole vocabulary. Good — Σ_w p(w | h) = 1 holds exactly, it's a real softmax, just *hierarchically* factored. And the cost? To compute p(w_O | h) and its gradient I only walk the path to w_O: I touch L(w_O) inner-node vectors, not V leaf vectors. On a roughly balanced tree L ≈ log₂ V. So I've gone from V to log₂ V — for a million-word vocabulary, from ~10^6 to ~20. That's the win.

Now I get to choose the tree, and the choice isn't free — the shape determines the path lengths. A balanced tree gives every word a path of log₂ V. But word frequencies are wildly skewed: a handful of words ("the", "of", "a") account for a huge fraction of tokens. If I give *those* short paths and rare words longer paths, the *expected* path length over the token stream drops well below log₂ V. That's exactly what a Huffman code does — it's the optimal prefix code, assigning short codes to frequent symbols. So build the tree as a Huffman tree over word frequencies. The expected number of nodes per training word becomes about log₂ of the unigram perplexity rather than log₂ V; for a million-word vocab that's a further ~2× speedup. Bookkeeping note on parameters: in this hierarchical scheme each *word* keeps its input vector v_w, but the output side is now one vector per *inner node* (there are V−1 of them), not one per word. That's fine; the inner-node vectors are exactly the logistic-regression weights of the path decisions.

Let me nail the gradient, because I'll be coding it. Each node on the path is a binary logistic regression: target is "which child", prediction is f = σ(h · v'_n). If I encode the path with a bit code_j ∈ {0,1} at node j (the Huffman code bit telling me which child the true word's path takes), then maximizing log p means the per-node loss is the logistic loss with label (1 − code_j). The gradient of the logistic log-likelihood is (label − prediction) times the other factor, so with step size α: g = (1 − code_j − f)·α; then accumulate the input-representation gradient as g·v'_n using the old node vector, update the node vector v'_n += g·h, and apply the accumulated gradient to the input vector or context vectors at the end. Clean — it's just a chain of logistic regressions sharing the same input representation.

That's a complete, cheap, *normalized* solution. But it carries a tree, the inner-node bookkeeping, and the Huffman construction, and its cost still scales with path length. Let me see if I can get away with something even simpler; that pushes me to the second idea, which abandons normalization entirely.

What do I actually need from the model? I don't need a calibrated probability distribution over the vocabulary. I need the geometry to come out right: words that genuinely co-occur should have input/output vectors that point together; random pairs should not. That's a *discrimination* task, not a density-estimation task. Can I train the vectors by just teaching them to tell "real neighbor" from "random word"?

This is exactly the spirit of Noise-Contrastive Estimation, so let me recall how NCE works and then see how far I can strip it down. NCE's problem is fitting an unnormalized model without computing its partition function Z. Its trick: convert density estimation into binary classification. Take a real example, and mix in k "noise" samples per real one, drawn from a known noise distribution p_n. Train a classifier to say "data" vs "noise". By Bayes' rule, given a sample u that came from this mixture (1 part data to k parts noise), the posterior probability that it was data is

  P(D=1 | u) = p_d(u) / ( p_d(u) + k·p_n(u) ).

NCE plugs the *model* in for p_d — and crucially it treats the normalizer as just another free parameter (it doesn't have to be the true Z), because the classifier only cares about ratios. So with a model score s(w,c) = exp(v'_c · v_{w}) for "is c a context of w", and folding the normalizer into the score (you can even fix it to 1 and let the vectors absorb it),

  P(D=1 | w, c) = exp(v'_c · v_w) / ( exp(v'_c · v_w) + k·p_n(c) ).

NCE then maximizes the classifier's log-likelihood — log P(D=1) on real pairs plus k·E_{noise}[log P(D=0)] on noise pairs — and the beautiful theorem is that as you do this, the model converges to the properly normalized softmax. But that theorem is also what makes NCE heavier than I want: to get it I need the actual numeric values of p_n(c) sitting in the denominator, and I'm carrying all this machinery in service of recovering a softmax likelihood I've already decided I don't care about.

So let me ask the irreverent question: I only want good vectors, not a likelihood guarantee. What if I just throw away the part of the denominator that ties this back to the softmax? Look at where p_n appears: the term k·p_n(c). What if I simply set k·p_n(c) = 1 — i.e., pretend the noise term is a constant? Then

  P(D=1 | w, c) = exp(v'_c · v_w) / ( exp(v'_c · v_w) + 1 ).

Divide numerator and denominator by exp(v'_c · v_w)... actually just recognize the form directly: x/(x+1) with x = exp(z) is exactly e^z/(e^z+1) = 1/(1+e^{−z}) = σ(z). So

  P(D=1 | w, c) = σ(v'_c · v_w).

The whole apparatus collapses into a plain logistic unit on the inner product. No partition function, no p_n values in the denominator, no tree. The training objective for a single real (input w_I, output w_O) pair, with k noise words w_i drawn from p_n, becomes

  log σ(v'_{w_O} · v_{w_I}) + Σ_{i=1}^{k} E_{w_i ∼ p_n}[ log σ(−v'_{w_i} · v_{w_I}) ].

Read it: push the real output word's vector to align with the input vector (drive σ(v'_{w_O}·v_{w_I}) toward 1), and push k random words' vectors to *anti*-align (drive σ(−v'_{w_i}·v_{w_I}) toward 1, i.e. their dot product negative). This replaces every "log p(w_O | w_I)" term in the objective. I'll call it negative sampling. Let me be honest about what I gave up: by setting k·p_n = 1 I broke NCE's guarantee — this no longer maximizes the softmax log-likelihood, and it's no longer a consistent density estimator. But I don't need it to be. I traded a likelihood guarantee for radical simplicity, and the only thing I'm betting on is that the *vectors* come out good. And the practical payoff is real: I now need only *samples* from p_n, never its numeric probabilities — strictly less than NCE requires.

How many negatives, k? Each negative is one extra inner product and one extra vector update per real pair, so k directly sets the cost multiplier. Intuitively, with little data I want more negatives to get a clean contrastive signal, and with a lot of data the signal is plentiful so I can afford fewer. So k in the range 5–20 for small corpora, 2–5 for large ones. The cost per prediction is now k+1 inner products — even cheaper than the tree's log₂ V when k is small, and far simpler to implement.

Now the noise distribution p_n itself. The obvious candidates are uniform over the vocabulary, or the unigram distribution U(w) (sample negatives in proportion to how often words actually occur). Uniform under-samples frequent words as negatives, so frequent words rarely get pushed away from anything and their geometry stays mushy. Pure unigram over-samples the handful of ultra-frequent words so hard that rare words almost never appear as negatives and never learn to be distinct. I want something between them — sample frequent words more than uniform, but not as brutally as unigram. Raising the unigram to a fractional power does exactly that: U(w)^α with 0 < α < 1 flattens the distribution toward uniform; α=1 is unigram, α=0 is uniform. The three-quarters power is the compromise I want: p_n(w) ∝ U(w)^{3/4}. It boosts the relative sampling weight of rare words (a rare word with unigram mass m gets m^{3/4}, which is proportionally larger for small m) while still letting common words dominate the negatives more than uniform would. So p_n(w) = U(w)^{3/4} / Z. The gradient mirrors the hierarchical case, just over a flat list instead of a path: for the 1 positive and k negatives, with label ∈ {1, 0}, prediction f = σ(v_{w_I} · v'_{target}), set g = (label − f)·α, update v'_{target} += g·v_{w_I}, accumulate the input gradient g·v'_{target}, and apply it to v_{w_I} at the end.

Let me circle back to something I deferred: why two separate vector tables, an input v_w and an output v'_w, for each word? Why not share a single vector per word for both roles? Picture sharing: the dot product that scores "is w its own neighbor" would be v_w · v_w = ‖v_w‖² — always large and positive. But a word rarely sits right next to *itself* in text, so the model would constantly be fighting to keep v_w · v_w small to avoid predicting a word as its own context, which directly shrinks the norms and corrupts the geometry I'm trying to build. Keeping input and output vectors separate dissolves the pathology: the score of w predicting itself is v'_w · v_w, two different vectors, with no built-in tendency to be large. So I keep two tables, and the embeddings I ship are the input vectors v_w (the ones on the projection side, shared across context positions).

One more nuisance in the data itself, and it's the same frequency skew that motivated Huffman trees, now biting from a different angle. The token stream is dominated by "the", "a", "in". When "France" appears, seeing it next to "Paris" teaches the model a lot; seeing it next to "the" teaches almost nothing, because *everything* appears next to "the". Yet because "the" is so frequent, the overwhelming majority of training pairs involve these near-empty-information words, wasting compute and, worse, drowning the content-word co-occurrences in noise. Also, the vectors of ultra-frequent words stop changing after a few million examples — continuing to train on them is pure waste. So I want to *thin out* the frequent words in the stream before forming windows. I'll discard each token with a probability that grows with its frequency. I want a rule that leaves rare words essentially untouched, kicks in aggressively once a word's frequency exceeds some threshold t, and preserves the relative ordering of frequencies. The form

  P(discard w) = 1 − √( t / f(w) ),

where f(w) is the word's relative frequency, does this: when f(w) ≤ t the square root is ≥ 1 so the discard probability is ≤ 0 (keep it always); as f(w) grows past t the discard probability climbs toward 1, and because it's monotone in f it never reorders frequencies. It's a heuristic, but it has the three properties I wanted. The payoff is double: training should spend far fewer updates on near-empty frequent-word pairs, *and* rare-word vectors should see cleaner neighborhoods — with the frequent words thinned out, the effective window reaches across more content words, so a center word now sees more of its genuinely informative neighbors within the same window radius. In the low-level trainer I will use the keep-probability min(1, (√(f/t)+1)·t/f) = min(1, √(t/f) + t/f), a slightly less aggressive form with the same leading √(t/f) behavior for very frequent words.

Let me also pin the small but load-bearing optimization choices, because at this scale they matter. SGD with backprop; learning rate starts around 0.025 and decays *linearly* as raw corpus tokens are consumed — simple, and it spends the early high-rate phase moving fast and the late phase settling even when subsampling removes many updates. Initialize the input vectors to small uniform random values in roughly [−0.5/D, 0.5/D]: small so that initial dot products land in the near-linear middle region of σ where gradients are healthy, and scaled by 1/D so the dot product of two D-dim vectors doesn't blow up with dimension. Initialize the *output* vectors to exactly zero: then at the very first step every σ(v'·v) = σ(0) = 0.5, so there's no arbitrary initial push in any direction — the model starts perfectly agnostic and lets the data write the geometry. And since σ saturates, precompute it on a grid over, say, [−6, 6] and clamp outside that range to 0 or 1; outside ±6, σ is within 0.0025 of its asymptote, so clamping costs nothing and saves an exp() on every single one of the billions of updates.

Now let me assemble the whole thing into the training loop, because the pieces interlock in a specific order. For each sentence: first thin it with the frequent-word discard. Then slide over positions; at each center word, draw the dynamic window radius R uniformly in [1, c]. For CBOW: average the input vectors of the R-on-each-side context words into h, run the cheap output layer (hierarchical-softmax path *or* negative sampling) with h as the input vector and the center word as the positive target, accumulate the hidden-vector error, and apply that same accumulated error to every context word's input vector, which matches the low-level scatter update scale. For Skip-gram: for each context word in the window, treat the *center* word's input vector as the input and that context word as the positive target, run the cheap output layer, and update. The cheap output layer is itself swappable: hierarchical softmax walks the Huffman path of the target word doing the per-node logistic updates; negative sampling does one positive plus k draws from U^{3/4} doing the per-sample logistic updates. Decay α linearly against the raw token count. Ship the input vectors.

Here is the implementation, mirroring the structure of the standard C trainer (syn0 = input vectors = the embeddings we keep; syn1 = hierarchical-softmax inner-node vectors; syn1neg = negative-sampling output vectors). I'll write it in Python for clarity but keep it faithful to that structure.

```python
import numpy as np

MAX_EXP = 6.0
def build_sigmoid_table(size=1000):
    # precompute sigma on [-MAX_EXP, MAX_EXP]; clamp outside (sigma within 0.0025 of 0/1)
    xs = (np.arange(size) / size * 2 - 1) * MAX_EXP
    return 1.0 / (1.0 + np.exp(-xs)), size
EXP_TABLE, EXP_SIZE = build_sigmoid_table()
def sigm(x):
    if x >= MAX_EXP:  return 1.0
    if x <= -MAX_EXP: return 0.0
    return EXP_TABLE[int((x + MAX_EXP) * (EXP_SIZE / MAX_EXP / 2))]

# ---- frequent-word subsampling: discard with prob 1 - sqrt(t/f) -------------
def subsample_keep(count, train_words, t, rng):
    if t <= 0: return True
    fr = count / train_words
    keep_p = (np.sqrt(fr / t) + 1.0) * (t / fr) if fr > 0 else 1.0  # C trainer's keep form
    return keep_p >= 1.0 or rng.random() < keep_p

# ---- Huffman tree for hierarchical softmax: short codes for frequent words --
def build_huffman(counts):
    # Counts are in descending vocabulary order, as in the C trainer.
    V = len(counts)
    count = list(counts) + [10**15] * (V - 1)     # internal nodes start "infinite"
    binary = [0] * (2 * V - 1); parent = [-1] * (2 * V - 1)
    pos1, pos2 = V - 1, V
    def take_min():
        nonlocal pos1, pos2
        if pos1 >= 0 and count[pos1] < count[pos2]:
            idx = pos1; pos1 -= 1
        else:
            idx = pos2; pos2 += 1
        return idx
    for a in range(V - 1):                          # merge two smallest each step
        min1, min2 = take_min(), take_min()
        count[V + a] = count[min1] + count[min2]
        parent[min1] = parent[min2] = V + a; binary[min2] = 1
    root = 2 * V - 2
    codes, points = [], []                          # per word: path bits + inner-node ids
    for a in range(V):
        code, path, b = [], [], a
        while True:                                  # climb from leaf to root
            code.append(binary[b]); path.append(b); b = parent[b]
            if b == root: break
        # store root->leaf order; one inner-node point per code bit
        codes.append(list(reversed(code)))
        inner_nodes = [root] + list(reversed(path[1:]))
        points.append([n - V for n in inner_nodes])
    return codes, points

# ---- the two cheap output layers (each fills the "predict cheaply" slot) ----
def hs_update(h, word, syn1, codes, points, alpha):
    grad = np.zeros_like(h)
    for code_bit, node in zip(codes[word], points[word]):
        f = sigm(np.dot(h, syn1[node]))
        g = (1 - code_bit - f) * alpha            # (label - prediction)*alpha, label = 1-code
        grad += g * syn1[node]                     # accumulate input-side gradient
        syn1[node] += g * h                        # learn inner-node vector
    return grad

def neg_update(h, word, syn1neg, unigram_table, k, alpha, rng):
    grad = np.zeros_like(h)
    for d in range(k + 1):
        if d == 0:
            target, label = word, 1                # the true (positive) pair
        else:
            target = unigram_table[rng.integers(len(unigram_table))]  # ~ U(w)^{3/4}
            if target == word: continue
            label = 0
        f = sigm(np.dot(h, syn1neg[target]))
        g = (label - f) * alpha
        grad += g * syn1neg[target]
        syn1neg[target] += g * h
    return grad

def build_unigram_table(counts, size=10**8, power=0.75):
    w = np.power(np.asarray(counts, dtype=np.float64), power); w /= w.sum()  # U(w)^{3/4} / Z
    table, i, cum = np.empty(size, dtype=np.int64), 0, w[0]
    for a in range(size):
        table[a] = i
        if a / size > cum and i < len(w) - 1: i += 1; cum += w[i]
    return table

# ---- fill the generic scaffold slots ---------------------------------------
def predict(h, target, syn1, syn1neg, codes, points, table, k, hs, neg, alpha, rng):
    grad = np.zeros_like(h)                           # input-side gradient for this prediction
    if hs:  grad += hs_update(h, target, syn1, codes, points, alpha)
    if neg: grad += neg_update(h, target, syn1neg, table, k, alpha, rng)
    return grad

class TrainingState:
    def __init__(self, counts, V, D, window, k, arch, hs, neg, sample, table_size, rng):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.train_words = float(self.counts.sum())
        self.window, self.k, self.arch = window, k, arch
        self.hs, self.neg, self.sample = hs, neg, sample
        self.syn0 = (rng.random((V, D)) - 0.5) / D    # syn0: input vectors kept as embeddings
        self.syn1 = np.zeros((V - 1, D)) if hs else None      # syn1: HS inner-node vectors
        self.syn1neg = np.zeros((V, D)) if neg else None      # syn1neg: NEG output vectors
        self.codes, self.points = build_huffman(self.counts) if hs else (None, None)
        self.table = build_unigram_table(self.counts, size=table_size, power=0.75) if neg else None

def configure_training(counts, V, D, options, rng):
    return TrainingState(counts, V, D,
                         window=options.get("window", 5),
                         k=options.get("k", 5),
                         arch=options.get("arch", "skipgram"),
                         hs=options.get("hs", False),
                         neg=options.get("neg", True),
                         sample=options.get("sample", 1e-3),
                         table_size=options.get("table_size", 10**8),
                         rng=rng)

def prepare_sentence(sentence, state, rng):
    return [w for w in sentence
            if subsample_keep(state.counts[w], state.train_words, state.sample, rng)]

def window_context(sentence, pos, window, rng):
    b = int(rng.integers(window)) if window > 0 else 0  # dynamic radius = window - b
    return [sentence[i] for i in range(pos - window + b, pos + window + 1 - b)
            if 0 <= i < len(sentence) and i != pos]

def training_step(sentence, pos, state, alpha, rng):
    center = sentence[pos]
    ctx = window_context(sentence, pos, state.window, rng)
    if not ctx: return
    if state.arch == "cbow":
        h = state.syn0[ctx].mean(axis=0)               # average context -> predict center
        grad = predict(h, center, state.syn1, state.syn1neg, state.codes,
                       state.points, state.table, state.k, state.hs, state.neg, alpha, rng)
        for w_ctx in ctx:                              # C-style scatter, no /len(ctx)
            state.syn0[w_ctx] += grad
    else:                                              # skip-gram: center -> each context
        for w_ctx in ctx:
            grad = predict(state.syn0[center], w_ctx, state.syn1, state.syn1neg,
                           state.codes, state.points, state.table, state.k,
                           state.hs, state.neg, alpha, rng)
            state.syn0[center] += grad

def linear_alpha(alpha0, seen, total):
    return max(alpha0 * (1 - seen / total), alpha0 * 1e-4)

def train(corpus, counts, V, D=100, window=5, k=5, arch="skipgram",
          hs=False, neg=True, alpha0=0.025, sample=1e-3, table_size=10**8,
          epochs=5, seed=1):
    rng = np.random.default_rng(seed)
    state = configure_training(counts, V, D,
                               {"window": window, "k": k, "arch": arch, "hs": hs,
                                "neg": neg, "sample": sample, "table_size": table_size},
                               rng)
    total, seen = epochs * state.train_words, 0
    for _ in range(epochs):
        for sentence in corpus:
            raw_len = len(sentence)
            alpha = linear_alpha(alpha0, seen, total)
            sent = prepare_sentence(sentence, state, rng)
            for pos, _ in enumerate(sent):
                training_step(sent, pos, state, alpha, rng)
            seen += raw_len
    return state.syn0   # the learned word embeddings
```

So the causal chain, start to finish: I wanted crisp *linear* regularities in a word space, the empirical clue said they appear only at a scale the expensive neural language models can't reach, so the real adversary was per-word cost; I traced the cost to a non-linear hidden layer and a vocabulary-sized softmax, killed the hidden layer to get a log-linear model (which also fits the linearity I'm after), built two minimal log-linear architectures (CBOW averaging context to predict the center, Skip-gram using the center to predict each neighbor with a randomized window for cheap distance-weighting), then attacked the softmax two ways — a Huffman-tree hierarchical softmax that keeps an exact normalized distribution at O(log V) cost, and, by starting from NCE's data-vs-noise classifier and brutally setting k·p_n = 1 so the posterior collapses to a plain σ(v'·v), a negative-sampling objective that needs only samples from a flattened U^{3/4} noise distribution; finally I thinned the frequency-dominated stream with the 1 − √(t/f) discard rule, kept input and output vectors separate to avoid the self-prediction pathology, and pinned the SGD/init/σ-table details — leaving a model cheap enough to train on billions of words, which is exactly the scale the linear regularities were waiting for.
