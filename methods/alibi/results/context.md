## Research question

When a transformer language model is built, one design number dominates cost: the training input length, call it $L$. The self-attention sublayer compares every token with every other token in its input, so both compute and memory scale quadratically in $L$. Longer inputs give each prediction more left context and lower its loss, but they make every training step dramatically more expensive. The natural wish is to have it both ways: train on short subsequences (cheap), then at inference feed sequences *longer* than $L$ and still predict well.

Call this ability **extrapolation**: a model continues to perform well as the number of input tokens at validation grows beyond the number it was trained on. The recurrent language models that preceded transformers were routinely trained on short sequences and simply run on longer ones at test time, and this was assumed to generalize. Transformers were hoped to do the same. The precise question here: does a transformer LM, trained at length $L$, keep its perplexity (ideally improve it) when evaluated at length $L_\text{valid} > L$? If not, *why* not, and can the position-representation mechanism be changed so that it does — **without** paying extra runtime, memory, or parameters relative to the cheapest existing method?

A solution must clear three bars at once: (1) it must actually extrapolate to much longer inputs; (2) it must be as fast and memory-light as the cheapest current position method; (3) ideally it adds no learned parameters, so it transfers across model sizes and datasets without retuning.

## Background

A transformer LM maps a list of token vectors to the same number of output vectors. The sublayers that make up a layer — embedding lookup, the position-wise feedforward network, the softmax classifier, and the attention sublayer itself — are all agnostic to how many input vectors they receive; none of their parameters depend on input length. Causal (left-to-right) language modeling is enforced with a causal mask added to the attention scores before the softmax, so position $i$'s prediction sees only tokens $1..i$.

Because the layers are length-agnostic, sequential order has to be injected explicitly. The standard route is to add a **position signal** to the token embeddings at the bottom of the network, so that two identical tokens at different positions enter the stack as different vectors. Everything the model learns about "where" a token is, it learns from how this injected signal behaves over the positions $[0, L)$ that appear in training.

**Inference on long text.** Training and perplexity evaluation both process many predictions at once under the causal mask. A document longer than $L$ is handled by segmenting it into $L$-length pieces and scoring each independently — **nonoverlapping inference**. A token's prediction therefore only sees context back to the start of its own segment. An alternative, **sliding-window inference**, advances a window of width $L$ by a small stride $S$ so that every prediction has up to $L$ left-context tokens; it is far more accurate at segment boundaries but much slower, since the same tokens are re-encoded many times.

**The early token curse.** Under nonoverlapping inference, the predictions near the *start* of each segment have very little left context (the relevant tokens sit at the end of the previous segment, which was thrown away). These context-starved predictions have high loss and inflate the reported perplexity. Crucially, feeding longer subsequences at inference is one way to dilute this curse: with longer $L_\text{valid}$, a smaller *fraction* of predictions are context-starved. So a model that can simply *accept* longer inputs at inference gains perplexity even if it learns nothing new about long-range structure — provided it does not break when given more tokens than it trained on.

**The diagnostic finding that frames everything.** A transformer LM with the standard added position signal, trained at $L$ and then evaluated at $L+k$, improves perplexity only for very small $k$ (a few dozen tokens) and then *degrades sharply* as $k$ grows. The model does not gracefully use the extra length; it breaks. The cause is traceable to the position mechanism: at positions beyond $L$, the position signal the model is asked to interpret is one it never saw during training — it is out of distribution. The model has learned responses tuned to the specific signals occurring in $[0, L)$, and the new signal lies outside that support. Holding the architecture, data, seed, and training budget fixed and changing *only* the position method changes the extrapolation behavior — which pins the failure on the position representation, not on the transformer as such.

A second, more architectural observation matters for the design space. In the original added-signal scheme, position information is mixed into the *values* and thus flows into every layer's output. In some later relative schemes, position influences only the query–key comparison (the attention weights) and never the values, so the attention output — a weighted average of value vectors — carries no explicit absolute-position component. Segregating position out of the values in this way appears to help a model tolerate lengths it did not train on.

## Baselines

**Learned absolute position embeddings.** A separate trainable vector is stored for each position $0..L-1$ and added to the token embedding there. Simple and effective inside the training range. The fatal limitation for the present goal: there is no vector for any position $\geq L$, so the method has *no defined behavior* beyond the training length — extrapolation is impossible by construction.

**Sinusoidal position embeddings.** Instead of learning per-position vectors, define fixed, non-learned vectors whose coordinates are sines and cosines of the position at a geometric range of frequencies, and add them to the token embeddings at the input. No parameters, cheap, and — because the function is defined for every position — it *can* in principle produce a signal at positions $> L$. It was hoped this would extrapolate. Empirically it barely does: trained at $L$, perplexity improves for only roughly the first $\sim$20 extra tokens, holds briefly, then degrades. The continued-but-novel combinations of sinusoid phases past $L$ are still out of distribution for the trained model. Leaves the gap: cheap, parameter-free, but does not extrapolate.

**Rotary position embeddings (Su et al., 2021; popularized by GPT-J).** Rather than adding a vector at the input, rotate the query and key vectors in each attention layer by an angle that depends on their absolute position, using a geometric range of rotation frequencies. Because a query at $i$ and a key at $j$ end up interacting through a rotation by $i-j$, the scheme is effectively *relative*, and it injects position at *every* layer rather than only the first, and never into the values. It extrapolates somewhat better than sinusoidal (perplexity keeps improving for roughly the first hundred-plus extra tokens), but the per-layer rotations make training and inference slower and more memory-hungry than the sinusoidal baseline. Leaves the gap: better extrapolation, but not efficient.

**Relative position bias on attention scores (Shaw et al., 2018; the T5 variant of Raffel et al., 2020).** Add no signal to the token embeddings at all. Instead, after computing each query–key dot-product score, add a *learned scalar bias* that depends only on the relative distance between query and key, shared across the network and tuned per head. Distances are bucketed: nearby distances each get their own learned bias, while distances beyond a cutoff are pooled into shared buckets — which intuitively should help at lengths past training, since a never-before-seen distance falls into an already-learned far bucket. Like rotary, it injects position at every layer through the query–key comparison only, never the values. Of the existing methods it extrapolates best (perplexity improving for several hundred extra tokens). Its limitation is efficiency: computing and gathering the relative bias makes it at least about twice as slow as the sinusoidal baseline in a PyTorch/GPU setting, plus it adds learned parameters. So even though it proves that *changing the position method can buy extrapolation*, it cannot deliver the intended payoff — train short and cheap, then extrapolate — because the method itself is expensive.

The ladder, then: extrapolation is demonstrably reachable by swapping the position method (the relative-bias result shows it), but every method that extrapolates well is slower and heavier than the cheapest method that does not. The open slot is a position method that extrapolates *and* costs no more than sinusoidal *and* adds no parameters.

## Evaluation settings

- **WikiText-103** (Merity et al., 2016): about 103M tokens of English Wikipedia. The standard LM here is the adaptive-input model of Baevski & Auli (2018): 16 transformer layers, model dimension 1024, 8 attention heads, feedforward inner dimension 4096, with tied input-embedding and softmax matrices (Press & Wolf, 2017; Inan et al., 2017), trained for 205 epochs. The natural way to study extrapolation is to train at one length $L_\text{train}$ and evaluate at a range of lengths $L_\text{valid}$.
- **Toronto Book Corpus** (Zhu et al., 2015): a books-domain corpus used to check that conclusions reached on Wikipedia transfer to a different domain with the same hyperparameters.
- **CC100 + RoBERTa corpus**: a 461 GB combination of the RoBERTa training data (Book Corpus, English Wikipedia, CC-News, OpenWebText, Stories; 161 GB) and the English portion of CC-100 (300 GB). Used for a billion-parameter model: 25 layers, dimension 2048, 16 heads, feedforward 8192 (~1.3B parameters), one epoch (50k updates) on 128 V100 GPUs.
- **Metric and protocol.** Perplexity on held-out text. Nonoverlapping inference is the default; sliding-window inference (with a small stride) serves as a slow but accurate analysis tool, used to separate "the model truly uses longer context" from "longer inputs merely reduce the early token curse." Extrapolation is measured by holding $L_\text{train}$ fixed and increasing $L_\text{valid}$.
- **Stack.** Fairseq (PyTorch) on GPUs.

## Code framework

The pre-existing pieces are a decoder-only (causal) transformer LM and its training/eval harness. The open slot is the additive attention mask: it already enforces causality, and the design work is to decide whether that same score-level object can also carry the missing order information.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x

def build_attn_mask(seq_len, n_heads, device=None):
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    # TODO: decide whether the additive score mask should contain any order signal.
    return causal

class LMModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight        # tied embeddings

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.tok(tokens)
        mask = build_attn_mask(T, self.layers[0].attn.n_heads, device=x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.ln_f(x))

def lm_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Eval: segment a long document into L-length pieces (nonoverlapping inference),
# score each independently, accumulate perplexity. To study extrapolation, train
# at L_train and run this loop at several L_valid > L_train.
```
