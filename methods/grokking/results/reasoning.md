OK, let me think about what I actually want to study and work backward to an experiment, because the thing I care about — why an overparameterized network generalizes instead of just memorizing — is almost impossible to see cleanly on natural data. On ImageNet or a language corpus, the data has rich and partly-unknown structure, train and test differ in tangled ways, a single run takes days, and whatever generalization effect I'm chasing is mild and buried under confounds. I can't iterate, and I can't tell *what* the network learned. So before any theory, I need a testbed: a task small enough to train to death on one GPU, where I know the exact underlying structure, and where "memorized the training set" and "found the real pattern" are cleanly distinguishable.

What kind of task has a *known* underlying structure and a *finite, fully specified* answer key? A mathematical binary operation. Pick an operation a∘b=c — modular addition, modular multiplication, composition of permutations — and it defines a complete table: every pair (a,b) has exactly one correct c. Now the task writes itself: show the network some of the (a,b)↦c entries and ask it to fill in the rest. That's exactly filling in the blanks of the operation table, like a Sudoku. Train on a random fraction of all the equations, hold out the rest as validation, and generalization is *literally* predicting the unseen slots of the same table. Train and validation are disjoint cells of one object, so generalization is perfectly decoupled from training performance — there's no distribution shift to argue about, just "did it infer the cells it wasn't shown."

Now the subtle design choice, and it's the one that makes the testbed actually measure what I want. How do I present the elements a, b, c to the network? The tempting thing is to present them in their natural form — numbers as decimal digits, permutations in one-line notation. But watch what that lets the network do: if it sees "37" as the digits 3 and 7, it can exploit the *surface form* — carry rules, digit patterns — and I'd never know whether it learned the operation or just learned arithmetic on the notation. That contaminates the memorize-vs-generalize question. So strip all internal structure: assign every distinct element its *own abstract symbol*, a bare token with no relationship to any other token a priori. The network sees "⟨symbol_37⟩" not "37"; it sees a permutation as a single opaque token, not as a line of mappings. Then the *only* way it can predict an unseen slot is to have inferred the properties of each element purely from how that element interacts with others across the equations it did see. There's no shortcut through notation. That's the whole point — abstract symbols force the network to recover the operation's structure or fail.

So an equation becomes a short token sequence: ⟨a⟩ ⟨∘⟩ ⟨b⟩ ⟨=⟩ ⟨c⟩, five tokens, each element and each of "∘" and "=" a distinct symbol. The model: a standard decoder-only transformer with causal masking, predicting the sequence left to right. But I only *care* about predicting c — the answer. The a, ∘, b, = tokens are the question, the given context; the supervised target is the c after the equals sign. So compute loss and accuracy *only on the answer token*. Validation accuracy = fraction of held-out equations whose c the model gets right. The model can be tiny — a couple of transformer layers, modest width, a few attention heads — because the task is small; tiny means I can train it for an absurd number of steps cheaply, which (I'll see) turns out to matter enormously.

Let me pick operations spanning easy-to-hard and abelian-to-non-abelian, all modulo a prime p (say 97) or in the permutation group S₅: x+y, x−y, x/y mod p; some polynomials like x²+y², x²+xy+y², x³+xy mod 97; and group products in S₅ like x·y, x·y·x⁻¹. With a prime modulus there's a nice consistency check baked in — every nonzero residue mod a prime is a power of a primitive root, so modular addition mod (p−1) and modular multiplication mod p are *the same group up to renaming elements*. Since I present elements as structureless symbols, the network literally cannot tell those two tasks apart — they're isomorphic as symbol-interaction problems. So I'd predict x−y and x/y require essentially the *same* amount of data to crack. That's a free sanity check on whether the testbed is behaving as the abstract-symbol framing says it should.

Now let me actually run the basic experiment in my head and watch what happens, because this is where it gets strange. Take modular division with, say, half the table as training data. Train the transformer. Early on — within maybe a few hundred to a thousand optimization steps — training accuracy shoots up to nearly 100%. The network has *memorized* the training equations. Validation accuracy at this point? Chance. Flat. The network has fit everything it was shown and learned nothing transferable; classically this is the textbook moment to stop, because the network is overfitting and validation isn't moving. If I followed the conventional script I'd early-stop here and conclude the network memorized and cannot generalize on this little data.

But suppose I *don't* stop. Suppose I keep training — not for a few more epochs, but for orders of magnitude longer, into the hundreds of thousands or a million steps, long, long past the point where training accuracy saturated and there's essentially nothing left to learn on the training set. What I see is that validation accuracy stays pinned at chance for a very long time — and then, abruptly, far later, it lifts off and climbs all the way to perfect generalization. Memorization happened around 10³ steps; generalization doesn't even *begin* until around 10⁵ steps, roughly a *thousandfold* later. The network sits there having perfectly fit the training data, apparently doing nothing, and then — well past the point everyone would call overfitting — it suddenly *gets it*. The generalization is real and complete; it figures out the operation.

Let me make sure I'm reading the loss curves consistently, because the accuracy story should have a loss signature. Training loss collapses to near zero early (matching the fast memorization). Validation loss *rises* for a stretch after that — the network is becoming more confidently wrong on held-out data, classic overfitting — and then, far later, validation loss turns around and *descends a second time*, bottoming out as validation accuracy completes its climb. A second descent of validation loss. That rhymes with the double-descent stories, but I want to be careful not to conflate them: the double descent I know of is along the *model capacity* axis and comes with a non-monotonic *accuracy* peak. Here the second descent is purely along the *training-time* axis, it happens tens of thousands of epochs past the first time training loss went to zero, and the *accuracy* is monotone — there's no validation-accuracy peak, it just climbs once and stays. So this is a distinct thing: delayed generalization, decoupled from training performance, far past memorization. The network "groks" the pattern long after overfitting.

Why would this happen, and what controls *when*? Let me probe the dependence on dataset size, because that's the cheapest knob. Reduce the training fraction. Two things could happen: the converged generalization could get worse (the usual story — less data, worse final accuracy), or something else. What I find is the something else: within a range of training fractions the *converged* accuracy stays at 100% — the network still fully groks — but the *time* it takes to get there grows fast as I shrink the dataset. Near the small end (say 25–30% of the table for S₅), shaving off another 1% of the training data can blow up the median steps-to-generalize by 40–50%; it looks roughly exponential. Meanwhile the time to memorize the *training* set barely moves — it stays in the 10³–10⁴ range no matter how little data I use. So small datasets don't prevent generalization; they make me *pay for it in optimization time*. That's a genuinely useful axis: you can trade compute for performance on smaller data. And it tells me grokking is most dramatic right near the *minimal* dataset size that still generalizes within budget — push the training fraction up and train/validation curves track each other closely and the drama disappears; sit near the threshold and the delay is enormous.

This immediately tells me something about the optimization budget as an experimental parameter: if I only train to convergence-on-the-training-set, I will *systematically miss generalization*, because it happens orders of magnitude later. To see the dramatic modular-division case I have to train to ~10⁶ steps. Any short budget would have me reporting "this network memorizes and never generalizes," which is exactly the wrong conclusion. The phenomenon is invisible unless you train absurdly long.

Now, generalization *does* eventually happen for many operations — but not all. Some operations (a messy one like x³+xy²+y mod 97) never generalize within any reasonable budget at any training fraction up to 95%; the network just memorizes and, to it, the held-out cells look random. And there's structure to *which* operations are easy: operations symmetric in their operands (x+y, x·y, x²+y², x²+xy+y²) need *less* data than their non-symmetric cousins (x−y, x/y, x²+xy+y²+x). That's at least partly an artifact of the architecture — a transformer can represent a symmetric function of the operands by simply ignoring the positional embedding that distinguishes the first operand from the second, so symmetry is "cheap" for it to express. And the primitive-root prediction checks out: x−y and x/y take about the same data, as the relabeling-isomorphism said they must. So the testbed is internally consistent.

The deepest question is *what makes the network grok at all*, and rather than theorize I can run interventions and let the data efficiency curves rank them. Fix a task (S₅), fix a budget, and sweep the optimization recipe: full-batch gradient descent, SGD, large vs. small learning rate, residual dropout, weight decay, gradient noise — and see which most reduces the amount of data needed to generalize. The standout, by a wide margin, is *weight decay*: adding it more than *halves* the samples needed compared to most other interventions. That's a strong clue about the mechanism. Why would an ℓ2 penalty pulling weights toward the origin specifically help a network *find the pattern* rather than just memorize?

Let me push on weight decay, because "it regularizes" is too glib. There are two stories. One: weight decay encodes a *prior* that near-zero weights are the right kind of solution for these small algorithmic tasks — a low-norm function is the "simple" one, and the simple function is the one that generalizes (memorizing the exceptions costs norm). Two: it's about the *geometry* of the minimum the optimizer settles into. I can partially separate these. Try weight decay toward the *initialization* instead of toward the *origin*. If the benefit were purely "stay small," both should help equally; if decay-toward-origin is *better* than decay-toward-init — which is what I find — then the "approximately zero weights are suitable" prior explains *part* of the effect but not all of it, because the specific pull to zero matters beyond just penalizing movement. So weight decay is doing something more than implementing a small-norm prior.

What's the rest? Look at the *other* interventions that help: adding noise — minibatch gradient noise, or Gaussian noise injected into the weights before or after the gradient step — also improves generalization. Noise during optimization is the canonical way to bias SGD toward *flat* minima: regions where the loss barely changes under parameter perturbations, which are exactly the minima a long line of work associates with better generalization. So a coherent reading emerges: the recipes that help grokking (weight decay toward origin, gradient/weight noise) are the ones that push optimization out of the sharp, high-norm minimum that *memorizes* the training table and into a flatter, lower-norm minimum that encodes the *operation*. The network memorizes fast because memorization is an easy sharp minimum to reach; it groks late because escaping that minimum and drifting to the flat, generalizing one takes a very long time of small noisy steps, accelerated by weight decay's pull toward simple low-norm solutions. Learning rate has to sit in a narrow band (within about an order of magnitude) for this to happen — too small and the drift never completes in budget, too large and it doesn't settle.

I can get more evidence for the flat-minimum reading without proving it. Train many networks with different seeds on S₅ for a fixed number of steps, until roughly half of them have grokked and half haven't, and measure the *sharpness* of each one's minimum (sensitivity of the trained network to parameter perturbations). If the flat-minimum story is right, the ones that generalized should sit in flatter minima. The sharpness score and validation accuracy come out strongly *anti*-correlated — Spearman around −0.8 — so grokking is happening preferentially in the flat regions, consistent with the idea that the network has to reach a flat part of the landscape before it generalizes.

And I should check that grokking isn't secretly "the network couldn't memorize so it was forced to generalize." Inject k outlier equations — equations whose answers I've deliberately corrupted to wrong values — into the training set, and watch. If the network's capacity were the binding constraint, a noisy training set might force it into a denoising solution. Instead: it *always* reaches 100% training accuracy, fully fitting the corrupted set including the k lies, and the point at which it does is barely affected by k. Small numbers of outliers (up to ~1000) hardly dent generalization; large k narrows the range of training fractions for which it generalizes. So the network has capacity to spare — far more than needed to memorize all the labels, true outliers included — which means generalization happening *at all* is not a capacity story; it genuinely requires the slow drift to a structured, low-norm solution, and demands a non-trivial explanation.

Finally, can I *see* that the grokked network found the real structure, not some accidental fit? Visualize the learned symbol embeddings — t-SNE the rows of the output (unembedding) layer. For modular addition, the embeddings lay themselves out on a *circle* — the natural topology of arithmetic mod p — and stepping "+8" walks you around it like a number line wrapped into a ring. For S₅, the embeddings cluster into *cosets* of a subgroup (and its conjugates) — the actual group structure of the permutations. And the structure is *cleaner* in networks trained with weight decay. So the network that groks has genuinely recovered the algebraic object; the abstract-symbol framing worked exactly as intended, and weight decay both speeds the discovery and sharpens the structure.

Let me write the experimental harness, grounded in how this is actually built — a binary-operation dataset of tokenized equations split by training fraction, a small decoder-only transformer scoring only the answer token, and an AdamW recipe with the long budget and weight decay that the ablations selected.

```python
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- the task: a binary operation table as tokenized equations ----
EQ, OP = "=", "∘"

def all_equations(operation, modulus):
    # every (a, b) -> c ; elements are ABSTRACT SYMBOLS (rendered, but used only as distinct tokens)
    eqs = []
    for a in range(modulus):
        for b in range(modulus):
            if operation == "+":  c = (a + b) % modulus
            elif operation == "-": c = (a - b) % modulus
            elif operation == "/": 
                if b == 0: continue
                c = (a * pow(b, -1, modulus)) % modulus            # x / y mod p
            else: raise ValueError(operation)
            eqs.append((a, OP, b, EQ, c))                          # <a> <op> <b> <=> <c>
    return eqs

def make_dataset(operation, modulus, train_fraction, rng):
    eqs = all_equations(operation, modulus)
    # vocabulary: every distinct symbol gets its own token (no internal structure exposed)
    vocab = {tok: i for i, tok in enumerate([EQ, OP] + list(range(modulus)))}
    encode = lambda eq: torch.tensor([vocab[t] for t in eq])
    data = torch.stack([encode(eq) for eq in eqs])
    perm = torch.randperm(len(data), generator=rng)
    n_train = int(round(train_fraction * len(data)))
    return data[perm[:n_train]], data[perm[n_train:]], len(vocab)   # train, val disjoint slots of the table

# ---- a small decoder-only transformer; loss only on the answer token ----
class GrokTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, max_len=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           activation="gelu", batch_first=True)   # d_ff = 4*d_model
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        T = tokens.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)  # causal mask
        h = self.embed(tokens) + self.pos[:T]
        h = self.blocks(h, mask=mask)
        return self.unembed(h)                                  # logits at each position

def answer_loss_acc(logits, tokens):
    # the answer c is the LAST token; predicted from the position before it (after '=')
    pred = logits[:, -2, :]                                     # logits that should predict the answer
    target = tokens[:, -1]
    loss = F.cross_entropy(pred, target)
    acc  = (pred.argmax(-1) == target).float().mean()
    return loss, acc

# ---- training: AdamW with weight decay, warmup, and a LONG budget ----
def train(model, train_data, val_data, num_steps=10**5, lr=1e-3, weight_decay=1.0,
          batch_size=512, warmup=10):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,  # weight decay -> origin
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))  # linear warmup
    bs = min(batch_size, len(train_data))
    history = []
    for step in range(num_steps):                               # train ORDERS OF MAGNITUDE past memorization
        idx = torch.randint(0, len(train_data), (bs,))
        batch = train_data[idx]
        loss, train_acc = answer_loss_acc(model(batch), batch)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0:
            with torch.no_grad():
                _, val_acc = answer_loss_acc(model(val_data), val_data)
            history.append((step, train_acc.item(), val_acc.item()))  # train_acc ~1 early; val_acc lifts off late
    return history
```

The causal chain, start to finish: to study why overparameterized nets generalize rather than memorize, I need a fast, fully-known testbed where memorization and generalization are cleanly separable — so I use binary operation tables (fill-in-the-blanks of a known algebraic object), presenting every element as an abstract structureless symbol so the only route to a held-out cell is to infer the operation. Training a small decoder-only transformer (scored only on the answer token) on a fraction of the table, training accuracy hits ~100% within ~10³ steps while validation stays at chance — and only if I keep training a thousandfold longer does validation suddenly climb to perfect generalization, with validation loss showing a late second descent and monotone accuracy, distinct from capacity-axis double descent. Shrinking the dataset keeps converged accuracy at 100% but blows up the optimization time roughly exponentially, so small data trades compute for generalization and the effect is sharpest near the minimal generalizing fraction; this is why a long optimization budget is essential — short budgets miss it. Sweeping interventions, weight decay toward the origin most reduces the data needed (more than halving it), and since decay-toward-origin beats decay-toward-init, a small-norm prior explains part but not all; gradient/weight noise also help, and sharpness anti-correlates with generalization — pointing to a picture where the network reaches a sharp memorizing minimum fast and slowly drifts, under noise and low-norm pressure, to a flat generalizing minimum that — as the circular modular-addition and coset-structured S₅ embeddings confirm — has genuinely recovered the algebraic structure.
