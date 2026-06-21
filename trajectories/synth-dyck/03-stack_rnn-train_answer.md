The LSTM confirmed the diagnosis I closed on by being almost right and failing in exactly one place. Removing the absolute-position table did everything I bet it would for length generalization: on `dyck-k2-m3` it is essentially perfect (`ood_token_acc` $0.99994$, `ood_string_acc` $0.997$), and on `dyck-length-ood` it jumps from the Transformer's $0.73$ to $0.974$ token accuracy with `ood_string_acc` climbing off the floor to $0.026$, so the headline geometric mean rose from $0.785$ to $0.962$. But the third environment is the tell. On `dyck-k8-m5` — $8$ types, depth $5$ — the LSTM gets `ood_token_acc` $0.924$ yet `ood_string_acc` $0.001$ and `id_string_acc` only $0.126$. Token accuracy near $0.92$ with string accuracy near zero means the model is right at most positions and wrong at a *scattering* of them — and on Dyck the positions that matter are the closers, where the model must name *which* of $8$ closing brackets matches the symbol on top of the stack. A fixed-width vector of $64$ units, sliding a depth-$5$ stack of $8$-way symbols, has to pack the top-of-stack identity cleanly enough that the read-out picks the right one of $8$; it mostly does, but it *smears under load*. This is no longer a length problem — even `id_string_acc` is $0.126$ — it is a *precision* problem: a dense vector is the wrong *shape* of memory for an order-sensitive, multi-type stack. The fix is not more recurrence; it is to stop asking a dense vector to emulate a stack and give the model an *explicit* one.

I propose a **stack-augmented RNN**: a simple-RNN controller coupled to a differentiable stack with soft PUSH/POP/NO-OP actions. The machine that recognizes context-free languages — and Dyck is the toy core of that class — is the pushdown automaton: a finite controller plus a stack whose entire job is "remember things in order, match last-in-first-out, and read the top." If I give the controller a real stack and let it learn when to PUSH (on an opener) and POP (on a closer), the top-of-stack symbol is held *explicitly* in its own slot rather than superposed into a dense state, the read-out reads it directly, and the order is maintained by the structure itself.

The obstacle to a literal stack is differentiability: a hard PUSH/POP/NO-OP choice makes the loss piecewise-constant in the controller's parameters — nudge a weight and, until the argmax flips, the action, the output, and the loss are unchanged, so the gradient is zero almost everywhere, and reinforcement-style estimators have brutal credit-assignment variance over long sequences. So I relax the discreteness without killing the stack. The controller emits a *probability* for each action via a **3-way softmax**, and the next stack is the probability-weighted *superposition* of the three pure operations applied to the current stack. Softmax rather than three independent sigmoids, because the actions are mutually exclusive and I want a convex combination on the simplex, so blending the candidate stacks gives a weighted *average* rather than an arbitrary rescaling.

The stack is a tensor of shape $[B, \text{depth}, \text{stack\_dim}]$, top at row $0$, and the three pure actions are shifts of this tensor. A pure PUSH shifts every row *down* (row $i \leftarrow$ row $i-1$) and writes a fresh candidate into row $0$; a pure POP shifts every row *up* (row $i \leftarrow$ row $i+1$); a pure NO-OP leaves it unchanged. I implement the shifts as fixed band matrices — `push_shift` carrying the sub-diagonal identity, `pop_shift` the super-diagonal identity — so a whole shift is one `matmul`, and inject the new top with a one-hot `top_slot` selecting row $0$. With $\mathrm{push}_p, \mathrm{pop}_p, \mathrm{noop}_p$ the softmax probabilities, the update is
$$\text{pushed} = \text{push\_shift}\cdot\text{stack} + \text{top\_slot}\cdot\text{new\_top}, \quad \text{popped} = \text{pop\_shift}\cdot\text{stack},$$
$$\text{stack} = \mathrm{push}_p\cdot\text{pushed} + \mathrm{pop}_p\cdot\text{popped} + \mathrm{noop}_p\cdot\text{stack}.$$
Every operation is a multiply-add by a continuous probability, so the stack is a smooth function of the parameters and gradients flow through it into the action logits and the controller. When a probability saturates to $1$ the update *is* the corresponding pure action, so the soft stack contains the hard stack as its limit and minimizing the prediction loss drives the softmax toward the corners on its own — the relaxation is a path *to* discrete behavior, not a replacement for it.

The controller is a single fused linear over the concatenation of the token embedding, the *top of the stack*, and the *previous hidden state*, squashed by $\tanh$: $h_t = \tanh\!\big(W[\,\text{emb}_t,\ \text{top}_{t-1},\ h_{t-1}\,]\big)$. I keep the ordinary hidden-to-hidden recurrence rather than zeroing it — on abstract counting probes one might kill it to *isolate* the stack, but here I am language-modeling for accuracy, so the controller may use both its dense state and the explicit stack. I feed back only the single top slot ($k=1$, not a window), because the Dyck next-token rule depends on exactly *one* thing — the symbol on top — so one slot is the right amount of context and keeps the fused projection small. The new top candidate is $\tanh(\text{push\_raw})$, bounded so repeated pushes and blends keep entries in a fixed range across a long string. The read-out is a linear over the concatenation of the hidden state and the *new* top, $\text{head}([h_t, \text{top}_t])\to\text{logits}$, so the prediction sees both the controller's summary and the explicit top.

Two deliberate departures from the unbounded-stack-with-curriculum constructions, both because this harness is bounded-depth Dyck trained for $100$ steps with no curriculum and no test-time surgery. First, the depth is *bounded*: `stack_depth = config.m + 2`. The language guarantees nesting never exceeds $m$, so a stack of depth $m$ plus a small margin holds every legal configuration; an unbounded growing stack is unnecessary, and a fixed-depth tensor is far cheaper and trains in one shot — the $+2$ gives slack for the empty-stack and one-over-full transients. Second, I do *not* use an out-of-range sentinel ($-1$) for the empty slot or a test-time argmax discretization. The unbounded constructions needed the sentinel so the controller could detect an exhausted stack on tasks like $a^n b^{2n}$, and needed discretization to kill numerical drift over hundreds of soft steps; here the stack is zero-initialized, the bounded depth gives drift fewer slots to accumulate in, and the model is evaluated with its trained soft actions exactly as trained. The harness scores $\arg\max(\text{logits})$, so any residual softness in the *actions* only matters if it changes the *token* prediction, which the read-out is trained to get right. I am betting the bounded depth and the explicit top slot make the soft stack crisp enough without the extra machinery.

The parameter budget is the smallest on the board — embedding, one fused controller linear over $\text{hidden}+\text{stack\_dim}+\text{hidden}$ inputs, a small $3+\text{stack\_dim}$ action/candidate projection, and a read-out over $\text{hidden}+\text{stack\_dim}$, with $\text{hidden}=\text{stack\_dim}=64$, on the order of $18{,}000$–$20{,}000$ parameters — because the stack *tensor* is activation, not weights, so explicit storage buys precision without buying parameters. The sharp, falsifiable claim is on `dyck-k8-m5`: if the explicit stack genuinely holds the top symbol in its own slot rather than superposing it, the *string* accuracies there should jump from near zero toward near one as the scattered closer errors disappear; if they stay low, the explicit stack is not crisper than the dense state and my read of the precision failure is wrong. On the two environments the LSTM already nearly solved I expect the stack to match and close the remaining gap to $1.0$ — bounded depth and a length-independent transition give it the LSTM's generalization plus explicit memory.

```python
def build_model(config: TaskConfig) -> DyckModel:
    """Stack-augmented RNN (Joulin & Mikolov 2015 + Softmax variant)."""

    class StackRNN(DyckModel):
        def __init__(self, vocab: int, hidden: int, stack_dim: int, stack_depth: int):
            super().__init__()
            self.hidden = hidden
            self.stack_dim = stack_dim
            self.stack_depth = stack_depth
            self.embed = nn.Embedding(vocab, hidden)
            # Equivalent to an RNNCell over [embed, top_of_stack], but one
            # fused projection avoids two tiny matmuls per token.
            self.cell = nn.Linear(hidden + stack_dim + hidden, hidden)
            # 3-way action (PUSH / POP / NO-OP) plus new top candidate.
            self.stack_update = nn.Linear(hidden, 3 + stack_dim)
            self.head = nn.Linear(hidden + stack_dim, vocab)
            push_shift = torch.zeros(stack_depth, stack_depth)
            pop_shift = torch.zeros(stack_depth, stack_depth)
            if stack_depth > 1:
                eye = torch.eye(stack_depth - 1)
                push_shift[1:, :-1] = eye
                pop_shift[:-1, 1:] = eye
            top_slot = torch.zeros(1, stack_depth, 1)
            top_slot[:, 0, :] = 1.0
            self.register_buffer("push_shift", push_shift, persistent=False)
            self.register_buffer("pop_shift", pop_shift, persistent=False)
            self.register_buffer("top_slot", top_slot, persistent=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            B, T = input_ids.shape
            device = input_ids.device
            h = torch.zeros(B, self.hidden, device=device)
            stack = torch.zeros(B, self.stack_depth, self.stack_dim, device=device)
            embeddings = self.embed(input_ids)
            states = []
            for t in range(T):
                emb = embeddings[:, t]
                top = stack[:, 0]
                h = torch.tanh(self.cell(torch.cat([emb, top, h], dim=-1)))
                update = self.stack_update(h)
                action_logits, push_raw = update[:, :3], update[:, 3:]
                a = torch.softmax(action_logits, dim=-1).view(B, 3, 1, 1)
                push_p, pop_p, noop_p = a[:, 0], a[:, 1], a[:, 2]
                new_top = torch.tanh(push_raw)

                pushed = torch.matmul(self.push_shift, stack) + self.top_slot * new_top.unsqueeze(1)
                popped = torch.matmul(self.pop_shift, stack)
                stack = push_p * pushed + pop_p * popped + noop_p * stack
                states.append(torch.cat([h, stack[:, 0]], dim=-1))
            return self.head(torch.stack(states, dim=1))

    return StackRNN(
        vocab=vocab_size(config.k),
        hidden=config.hidden_dim,
        stack_dim=config.hidden_dim,
        stack_depth=config.m + 2,
    )
```
