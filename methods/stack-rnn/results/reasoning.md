OK, let me think this through from scratch. I have a stream of symbols spat out by some short deterministic algorithm — `a^n b^n`, or well-nested brackets, or "remember this and echo it" — and I want a next-symbol predictor that, trained on short examples, keeps predicting the forced symbols correctly on much longer ones. The thing I keep tripping on is that "much longer" is not a detail; it is the entire test. A model that nails everything up to the training length and then falls to chance hasn't learned the rule, it has learned the strings. So before I write a single equation I should be honest about what failure I'm trying to escape.

Start with the only model I really trust here, the Elman recurrent net: `h_t = sigmoid(U x_t + R h_{t-1})`, `y_t = softmax(V h_t)`. I read in the current one-hot token, fold it into a hidden state, and predict the next token. Train it by backprop through time and SGD. On `a^n b^n` it'll do fine for a while — fit the training `n`, even look like it's "counting." But stare at what `h_t` actually is. It's a vector in `R^m`, and `m` is fixed the moment I instantiate the net. Everything the model knows about the entire prefix it has read — including, on `a^n b^n`, exactly how many `a`s have gone by — has to be crammed into those `m` real numbers. For small `n` it can encode the count in some smooth region of the hidden space and ride the dynamics down. But there is no count it can represent that grows without bound; once `n` exceeds what the trained dynamics tiled, the trajectory wanders into a region the net never shaped, and the prediction goes to noise. That's the Wiles–Elman result honestly read: it counts without a counter, over a *limited* range, and the careful follow-ups found it's mostly memorization near the trained regime. So the SRN's wall isn't a tuning problem; it's that a fixed-width vector can't be an unbounded counter.

Does the LSTM escape this? Its gates can make a unit linear, and a linear unit can add and subtract a constant each step, so it really can hold a count and length-generalize on `a^n b^n`, `a^n b^n c^n` — one-turn counter languages. Good, but a counter is not what these patterns ultimately need. Take nested brackets: `( [ ( ) ] )`. To know which closer is legal I have to remember not *how many* things are open but *what* is open and *in what order* — the last opener has to be matched first. That's last-in-first-out, and a counter, even several counters, can't represent the *order* of an arbitrarily deep nesting. And on verbatim memorization the LSTM is back to memorizing trained lengths. So even the LSTM's fixed-width cell, clever as it is, is the wrong *shape* of memory for the order-sensitive, depth-unbounded case.

So what shape do I actually need? The patterns I care about are the toy core of the context-free languages, and the machine that recognizes context-free languages is the pushdown automaton: a finite controller plus a stack. The stack is the missing piece — it's the data structure whose whole job is "remember things in order, match them last-in-first-out, and grow as deep as you need." If I give the recurrent net a stack and let the net learn when to PUSH and when to POP, then the *depth* of memory is no longer pinned to `m`; it grows with the input. That's the structural fix: don't enlarge the fixed vector, attach a memory that grows.

And this isn't a new idea in the abstract — Das, Giles & Sun in '92 bolted an external stack onto an RNN, the neural-network pushdown automaton, and Zeng et al. did similar. The idea is exactly right. So why didn't it just win? Their stack is *discrete*: at each step the controller picks PUSH or POP or nothing, one of them, and acts. And the reported pain is that the discrete external stack is hard to learn on long sequences. Let me make sure I understand *why* it's hard, because the whole design is going to hinge on it. If the action is a hard discrete choice, then "the loss as a function of the controller's parameters" is piecewise constant — nudge a weight a hair and, until the argmax flips, the chosen action is identical, so the output is identical, so the loss is identical: the gradient is zero almost everywhere and undefined where the argmax flips. Backprop has nothing to push on. I could reach for reinforcement-style estimators to get a gradient through the discrete choice, but on long sequences the credit-assignment variance is brutal — a single wrong PUSH near the start poisons the whole rollout and the reward signal is far too noisy to learn a precise stack discipline. So the discreteness is the wall, dead center.

What about Graves' Neural Turing Machine? That's differentiable — a tape with soft attention for read and write, end-to-end trainable, no discrete-action problem. But it has two costs for *my* problem. The tape is fixed size, so it doesn't grow with the input the way an unbounded stack does. And full content/location addressing is enormously more machinery than a stack-shaped task needs — a lot of attention weights to learn and, on these simple problems, hard and unstable to train. I don't want a general read/write memory; I want specifically a stack, and I want it to *grow*.

So the real question crystallizes: how do I get a stack — push, pop, grow without bound — that is differentiable, so plain SGD with backprop can learn the controller end to end, with no supervision on the actions? The discreteness is the only thing standing between Das-and-Giles's right idea and a trainable model. Kill the discreteness without killing the stack.

Here's the move I keep circling back to. The reason the discrete choice has no gradient is that I *commit* to one action. What if I don't commit? What if, instead of choosing PUSH or POP, the controller emits a *probability* for each, and the stack does *all of them* and blends the results by those probabilities? Let me write it and see if it's actually a stack or just mush.

Let the controller read the hidden state and produce an action distribution. Two actions to start, PUSH and POP, so a softmax over two:
`a_t = softmax(A h_t)`, with `A` a `2 x m` matrix; call the entries `a_t[PUSH]` and `a_t[POP]`, summing to 1. Why softmax and not just two independent sigmoids? Because these are mutually exclusive operations and I want their weights to form a convex combination — a proper interpolation on the simplex — so that when I blend the candidate stacks I get a weighted *average*, not an arbitrary rescaling that could inflate the values. Softmax gives me exactly the simplex.

Now the stack itself. Store it as a vector `s_t` of length `p`, with the top at index 0, the next element at index 1, and so on. Crucially `p` doesn't have to be fixed — I can let it grow on demand as the model pushes deeper, which is the unbounded-memory property I'm after. Think about what each of the two pure actions does to this vector:

- A pure POP shifts everything *up*: the new top is what was at index 1, the new index 1 is the old index 2, etc. So in a pure POP, `s_t[i] = s_{t-1}[i+1]`.
- A pure PUSH shifts everything *down* and writes a new value on top: the new index 1 is the old top, new index 2 is old index 1, etc. So for `i > 0`, `s_t[i] = s_{t-1}[i-1]`, and `s_t[0]` is a freshly computed value.

Since I refuse to commit, the actual update is the probability-weighted blend of these two shifted versions. For an element below the top, `i > 0`:
`s_t[i] = a_t[PUSH] * s_{t-1}[i-1] + a_t[POP] * s_{t-1}[i+1]`.
And for the top:
`s_t[0] = a_t[PUSH] * (new value) + a_t[POP] * s_{t-1}[1]`.
That's it — each cell of the new stack is a convex combination of the cell above it and the cell below it from the previous step, weighted by PUSH and POP. Let me check this is genuinely differentiable: every operation is a multiply and an add by a continuous probability, and the probabilities are a smooth softmax of a linear function of `h_t`. So `s_t` is a smooth function of the parameters; gradients flow through the stack back into `A`, into the controller, into everything. The discreteness wall is gone — and I didn't have to give up the stack structure, because when `a_t[PUSH]` saturates to 1 the update *is* a pure push and when `a_t[POP]` saturates to 1 it *is* a pure pop. The soft version contains the hard stack as its limit, and training will push the softmax toward those corners on its own because committing is what minimizes the prediction loss. The continuous relaxation is a path *to* discrete behavior, not a replacement for it.

Now, what's the "new value" I push on top? It should be a learned summary of what the controller currently knows, i.e. a function of `h_t`. Let me make it a bounded nonlinearity of a learned projection: `sigmoid(D h_t)`, where `D` is a `1 x m` row (one number on top, for the simplest single-dimensional stack). Why bounded, and why sigmoid specifically? Because I'm going to be summing and re-summing these values across many time steps as they get shifted around the stack; if the pushed value were unbounded, repeated pushes and blends could let stack entries grow without limit and destabilize training. A sigmoid pins every pushed value into `(0, 1)`, so the stack stays in a fixed, well-behaved range no matter how deep or how long. (Later I'll want the stack cells to be vectors rather than single numbers, to carry more than one bit of state per slot — same idea, `D` becomes a matrix and the top is a vector — but one dimension is enough to see the mechanism.)

One subtlety I almost skated past: what happens at the *bottom*, when I pop past the last element? `s_{t-1}[i+1]` runs off the end of the vector. I need a defined value to read there — and more than that, I'd like the controller to be *able to tell* that the stack is empty, because several of these patterns hinge on noticing "the stack just ran out." So fill the bottom / empty slot with a sentinel constant, and pick it *outside* the range of pushed values. Pushed values live in `(0, 1)`; let the empty value be `-1`. Now an empty-stack read is unambiguously distinguishable from any real pushed value, and the controller can learn to key off it. (I'll see this pay off concretely: on `a^n b^{2n}` a stack reads `-1` to know it has been exhausted and it's time to switch behavior.)

Now wire the stack back into the controller, or it's a write-only memory and useless. The hidden update should see the top of the stack:
`h_t = sigmoid(U x_t + R h_{t-1} + P s_{t-1}^k)`,
where `s_{t-1}^k` is the top `k` elements of the stack and `P` is an `m x k` matrix mapping them into the hidden space. Why feed `k` of them and not just the single top? Because handing the controller a small window — the top two, say `k = 2` — lets it base its next action on a little more context than the one topmost number, and that extra peek empirically makes the discipline easier to learn; `k = 2` is a fine default. And note `R`: on the toy counting tasks I'll actually set `R = 0`, killing the ordinary hidden-to-hidden recurrence, precisely so that *all* the long-range memory has to flow through the stack — that isolates whether the stack mechanism is doing the work, rather than letting the fixed-width recurrence quietly memorize. (On a real language-modeling task I'd turn `R` back on; there I want both.)

Let me now run the single-stack version forward in my head on `a^n b^n` to see whether the mechanism can even express the right program. The controller wants to PUSH on every `a` and POP on every `b`. Reading `a`s, the stack grows: top after the first `a`, two deep after the second, ... `n` deep after the last `a`. Then on each `b` it pops. The instant it has popped `n` times the stack reads the empty sentinel, and *that* is the signal that the `b`s should stop — the model can now correctly predict the switch back to `a` (the next sequence in the stream). The depth it climbed to is `n`, and `n` was never baked into any weight matrix — the *same* push-on-`a`, pop-on-`b`, watch-for-empty program runs identically whether `n` is 5 or 50. That is exactly the length-independence the SRN couldn't have: the SRN had to encode the whole count in `h_t`; here the count *is* the stack depth, and depth is free to grow. So this architecture can, in principle, length-generalize where the fixed-vector models structurally cannot.

But one stack is thin. Look at `a^n b^{2n}`: I need to emit twice as many `b`s as `a`s, which means tracking two related quantities at once. Or `a^n b^m c^{n+m}`: I have to remember `n`, then `m`, and produce `n + m`. A single stack does exactly *one* action per time step — one push or one pop — so it can maintain one count-like quantity, and these patterns need more bookkeeping than that in a single step. The fix is to run several stacks *in parallel*, each with its own action controller reading the shared hidden state, and let them interact *through* that hidden layer. Two stacks can divide the labor: I can actually watch this happen on `a^n b^{2n}` — the first stack pushes on every `a` and pops on every `b`, so it empties after the first `n` `b`s; the second stack pushes on every `a` too but pops on `b` *only once the first stack is empty*, which is why I needed the empty sentinel to be detectable. The two stacks, coordinating through `h_t`, jointly count out `2n`. With enough parallel stacks the model has, in theory, the expressive power of a Turing-complete system; in practice a handful suffices for these patterns.

There's one more pure action I've been ignoring, and the two-stack example just told me I need it. When the second stack is supposed to *wait* — sit still while the first stack drains — neither PUSH nor POP is the right move; it should hold its top unchanged. So add a third action, NO-OP, and extend the softmax to three: `a_t = softmax(A h_t)` with `A` now `3 x m`. The update for the top gains a term, and so does every deeper cell:
`s_t[0] = a_t[PUSH] * sigmoid(D h_t) + a_t[POP] * s_{t-1}[1] + a_t[NO-OP] * s_{t-1}[0]`,
`s_t[i] = a_t[PUSH] * s_{t-1}[i-1] + a_t[POP] * s_{t-1}[i+1] + a_t[NO-OP] * s_{t-1}[i]`.
NO-OP just blends in the *unchanged* stack — index `i` maps to index `i`. It's optional: pure counting like `a^n b^n` never needs a stack to idle, so I leave it off there; but for the coordinated multi-stack patterns it's essential, so it's a free switch. Notice the three actions now form a clean picture: each new cell is the convex combination of the cell above it (PUSH), the cell at it (NO-OP), and the cell below it (POP) from the previous step — a three-way superposition, weighted by a proper probability distribution. That's the whole memory.

Training. Everything is continuous, so I just do SGD with backprop through time. A couple of practical things fall out. Truncate the BPTT to a window — about 50 steps — so the recurrence is tractable; hard-clip the gradients to fight the explosion that long recurrences are prone to; start the learning rate at something like `0.1` and halve it whenever the validation entropy stops dropping. And grow `n` during training — feed shorter sequences first and incrementally lengthen — a curriculum, so the controller can lock in the easy short-range version of the program before it has to hold it together at depth.

I do hit a real wall on the harder patterns, though, and I should be honest about it rather than pretend SGD just works. On something like `a^n b^n`, the bigram statistics alone give an entropy quite close to the optimum, so there's a shallow local minimum where the model predicts the *easy* parts well and never bothers to learn the stack discipline that the *forced* parts need — SGD slides into that basin and sits. Plain gradient descent on a non-convex model has no way out of it on its own. The patch is to wrap a search around the SGD: random restarts — train several models from different initializations and keep the one that escaped — and more elaborate search schemes help further. It's not elegant, but it acknowledges that continuous optimization of a discrete-flavored program is genuinely hard, and a little discrete search on top of the continuous training is what gets the harder tasks to converge.

And one last numerical wrinkle that only shows up at the thing I care most about — long test sequences. The stack operations are continuous, so each step introduces a tiny imprecision; the softmax never goes *exactly* one-hot, so a "push" leaves a sliver of mass on pop, a "pop" leaves a sliver on push, and over hundreds of steps these slivers accumulate and smear the stack. The fix is to discretize at test time: take the argmax action — round the controller to a hard PUSH/POP/NO-OP — so the stack executes the *exact* program the soft training discovered, with no drift, on sequences far longer than anything seen. The soft relaxation was the vehicle for *learning*; at inference I can collapse it back to the crisp stack it was always converging toward.

Let me also sanity-check the gradient actually carries the stack's error back into the controller, since that's the only reason this is better than the discrete version. Take the top cell's PUSH term, `a_t[PUSH] * sigmoid(D h_t)`. Its derivative with respect to `D` is `a_t[PUSH] * sigmoid'(...) * h_t` — nonzero whenever there's any push probability — so the pushed value gets a real learning signal. Its derivative with respect to the action logits flows through the softmax — an error on a stack cell pushes the PUSH/POP/NO-OP probabilities directly. And the deeper cells `a_t[PUSH] * s_{t-1}[i-1] + ...` send error both into the actions and back into `s_{t-1}`, i.e. one step further back in time, so a mistake at the bottom of a deep stack can be traced all the way to the push that put it there. That is precisely the credit assignment the discrete stack couldn't provide. The relaxation didn't just make the forward pass smooth; it made the *backward* pass informative.

Now let me write the actual module, the way it goes into a standard next-token harness. I'll carry vector-valued stack cells (each slot is a `stack_dim` vector, not a single number — same superposition, just richer per slot) and use a `tanh` controller, which is the form that works cleanly in this setting. I'll implement the up/down shifts as fixed band matrices so a whole stack-shift is one matmul, and inject the new top into row 0. Here's the core:

```python
import torch
import torch.nn as nn


class StackRNN(nn.Module):
    """Simple-RNN controller coupled to a differentiable stack.
    At each step the controller emits a softmax over PUSH / POP / NO-OP and a
    new top-of-stack vector; the next stack is the probability-weighted
    superposition of the push-shifted, pop-shifted, and unchanged stacks. The
    effective top is fed back into the controller, giving an explicit,
    length-independent memory of the bracket/symbol stack."""

    def __init__(self, vocab, hidden, stack_dim, stack_depth):
        super().__init__()
        self.hidden = hidden
        self.stack_dim = stack_dim
        self.stack_depth = stack_depth
        self.embed = nn.Embedding(vocab, hidden)
        # controller cell over [embed, top_of_stack, prev_hidden] -> hidden
        self.cell = nn.Linear(hidden + stack_dim + hidden, hidden)
        # 3-way action logits (PUSH / POP / NO-OP) + new-top candidate vector
        self.stack_update = nn.Linear(hidden, 3 + stack_dim)
        # read out from [hidden, top_of_stack]
        self.head = nn.Linear(hidden + stack_dim, vocab)

        # push: row i <- row i-1 (everything shifts DOWN, top freed for new value)
        # pop:  row i <- row i+1 (everything shifts UP, bottom freed)
        push_shift = torch.zeros(stack_depth, stack_depth)
        pop_shift = torch.zeros(stack_depth, stack_depth)
        if stack_depth > 1:
            eye = torch.eye(stack_depth - 1)
            push_shift[1:, :-1] = eye
            pop_shift[:-1, 1:] = eye
        top_slot = torch.zeros(1, stack_depth, 1)   # selects row 0 to receive new top
        top_slot[:, 0, :] = 1.0
        self.register_buffer("push_shift", push_shift, persistent=False)
        self.register_buffer("pop_shift", pop_shift, persistent=False)
        self.register_buffer("top_slot", top_slot, persistent=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        h = torch.zeros(B, self.hidden, device=device)
        stack = torch.zeros(B, self.stack_depth, self.stack_dim, device=device)
        embeddings = self.embed(input_ids)
        states = []
        for t in range(T):
            emb = embeddings[:, t]
            top = stack[:, 0]                                   # top of stack, fed back
            # controller: h_t = tanh(W [emb, top, h_{t-1}])
            h = torch.tanh(self.cell(torch.cat([emb, top, h], dim=-1)))
            update = self.stack_update(h)
            action_logits, push_raw = update[:, :3], update[:, 3:]
            a = torch.softmax(action_logits, dim=-1).view(B, 3, 1, 1)
            push_p, pop_p, noop_p = a[:, 0], a[:, 1], a[:, 2]   # convex weights
            new_top = torch.tanh(push_raw)                      # bounded pushed value

            # candidate stacks for the three pure actions:
            pushed = torch.matmul(self.push_shift, stack) + self.top_slot * new_top.unsqueeze(1)
            popped = torch.matmul(self.pop_shift, stack)
            # superposition: blend by the action probabilities
            stack = push_p * pushed + pop_p * popped + noop_p * stack
            states.append(torch.cat([h, stack[:, 0]], dim=-1))  # read out [h, new top]
        return self.head(torch.stack(states, dim=1))
```

So the causal chain, start to finish: a fixed-width recurrent net can only memorize counting/nesting up to its trained range because the whole prefix has to fit in `h_t`; the data structure that fixes this is a stack, whose depth grows with the input, but a discrete stack has no usable gradient and so can't be learned by SGD; relaxing the discrete PUSH/POP/NO-OP choice into a softmax and making the next stack the probability-weighted superposition of the push-shifted, pop-shifted, and unchanged stacks makes the whole thing differentiable while still containing the hard stack as its saturated limit; bounding the pushed value with a squashing nonlinearity and marking the empty slot with an out-of-range sentinel keeps it stable and lets the controller detect an empty stack; feeding the top back into the controller and running several stacks in parallel (with NO-OP so a stack can wait) gives enough coordinated memory for the multi-count patterns; and because the learned program is a finite controller acting on an unbounded stack, the *same* program runs at any depth — so, trained short with curriculum, random restarts to dodge the bigram local minimum, and discretized at test time to kill numerical drift, it predicts the forced symbols on sequences far longer than it ever saw.
