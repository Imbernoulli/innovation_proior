**Problem.** The LSTM length-generalizes but fails on *precision*: on `dyck-k8-m5` (8 types, depth 5) it
gets `ood_token_acc` 0.924 yet `ood_string_acc` 0.001 — a dense vector emulating a multi-type stack by
superposition blurs at the closers, so almost no full string survives. The next rung must give the model an
*explicit* stack instead of asking a vector to emulate one.

**Key idea.** A simple-RNN controller coupled to a **differentiable stack** with soft PUSH/POP/NO-OP
actions. The controller emits a 3-way softmax over actions plus a new top candidate; the next stack is the
probability-weighted *superposition* of the push-shifted, pop-shifted, and unchanged stacks (shifts as
fixed band matmuls; new top injected into row 0). The top slot is fed back into the controller and read by
the head, so the top-of-stack symbol is held *explicitly* in its own slot. The relaxation is differentiable
(plain SGD learns the controller end-to-end) and contains the hard stack as its saturated limit.

**Harness-specific choices (vs. the unbounded constructions).** Depth is *bounded* at `m + 2` (Dyck-(k,m)
nesting never exceeds `m`), so a fixed-depth tensor suffices and trains in 100 steps — no curriculum. Stack
is zero-initialized (no `−1` empty sentinel) and evaluated with its trained soft actions (no test-time
argmax discretization); bounded depth keeps drift small. The controller keeps its hidden-to-hidden
recurrence and feeds back only the single top slot (`k=1`, the Dyck rule depends on one symbol). New top is
`tanh`-bounded so entries stay in range across long strings.

**Why it should top the ladder.** Explicit per-slot storage removes the closer-precision smearing, so
`dyck-k8-m5` string accuracy should jump from ~0; bounded depth + length-independent transition keep the
LSTM's length generalization. Smallest model on the board (~18–20k params; the stack is activation, not
weights).

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
