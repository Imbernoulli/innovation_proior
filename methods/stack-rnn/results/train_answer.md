The data is a stream of discrete symbols produced by short deterministic generators — $a^n b^n$, $a^n b^n c^n$, $a^n b^{2n}$, $a^n b^m c^{n+m}$, bracket-like recursive strings, reverse-order memorization, and binary addition with the answer emitted in reverse — concatenated into one stream with no sequence boundaries and, for most of the toy problems, no labels marking which symbols are forced by the generator. What is scored is whether the predictor reproduces those forced symbols at lengths and nesting depths beyond the training range. That is the crux: the easy reading of next-token prediction, that many $a$'s precede more $a$'s and many $b$'s precede more $b$'s, is exactly the part that does not transfer. To pass, the model must infer the compact rule behind the stream and execute it for larger $n$ — know when the $b$ run has exactly matched the preceding $a$ run, know which opener is most recent in nested brackets, remember stored symbols in last-in-first-out order for reversal. These are demands on the *organization* of memory, not merely on longer-range correlation.

The Elman simple recurrent network is the natural starting point, $h_t = \mathrm{sigmoid}(U x_t + R h_{t-1})$ and $y_t = \mathrm{softmax}(V h_t)$, and it captures many temporal regularities, but the entire prefix is compressed into one fixed-width vector in $\mathbb{R}^m$ chosen before the sequence length is known. On short $a^n b^n$ the trajectory of $h_t$ can imitate counting, because the training lengths carve out a small region of state space where "after this many $a$'s, expect this many $b$'s" happens to hold — and the test deliberately leaves that region. If the rule needs a resource that grows with the count or the nesting depth, a fixed vector simply has no such resource. The LSTM line weakens this objection without removing it: with gates and a linear cell a unit can hold near-constant error flow and learn to increment or decrement a quantity, so it really can act like a counter on one-turn languages. But a counter answers "how many closers remain?", which is not enough for "( [ ( ) ] )", where the next legal closer is fixed by the last unmatched opener, not by the number of open brackets; and reversal makes the point cleanest of all, because the last symbol stored must be the first one read back. The missing memory is last-in-first-out. The formal-language reading confirms the diagnosis — a finite-state controller remembers which phase it is in but cannot keep an unbounded count or stack, while a pushdown automaton adds exactly one ingredient, a stack that grows and whose top is read first — yet it also warns that the needed memory is *structured*: a fully general random-access memory does far more than these languages require, a slow bag-like context unit does too little, and a hand-built symbolic recognizer solves only one chosen grammar. Earlier neural pushdown automata establish the right bias but lean on specialized error functions, hints, isolated grammars, or hard stack actions, so they are not a clean drop-in predictor trained from the raw stream; the Neural Turing Machine is trainable end-to-end but exposes a fixed-size matrix with a read/write/addressing interface far broader than a stack problem needs. The open question is whether a narrower memory topology can be learned directly while keeping gradient training.

I propose the Stack RNN: a finite recurrent controller coupled to one or more differentiable stacks, where the controller emits a softmax distribution over stack actions and the next stack is the probability-weighted superposition of the pure action outcomes. The obstruction to training a hard stack is precisely the discreteness — if the controller selects PUSH or POP by argmax, a small weight change usually leaves the chosen action, the forward behavior, and the loss unchanged, so the gradient through the action is zero, and reinforcement-style credit assignment is high variance because one wrong early action corrupts a long rollout. The escape is to stop committing to a single action during training and let the memory execute the convex mixture. The controller produces $a_t = \mathrm{softmax}(A h_t)$ over PUSH and POP. The stack is a vector $s_t$ with its top at index $0$; a push shifts every old cell one slot deeper, a pop shifts every old cell one slot toward the top, so the geometry forces the indices. The new top is

$$s_t[0] = a_t[\text{PUSH}]\,\mathrm{sigmoid}(D h_t) + a_t[\text{POP}]\,s_{t-1}[1],$$

and a deeper cell $i>0$ reads the cell above it under a push and the cell below it under a pop,

$$s_t[i] = a_t[\text{PUSH}]\,s_{t-1}[i-1] + a_t[\text{POP}]\,s_{t-1}[i+1].$$

The destination $i$ reads old $i-1$ because push shifts down, and old $i+1$ because pop shifts up; the signs and indices are not free choices. The pushed value $\mathrm{sigmoid}(D h_t)$ is a deliberate one too — not an arbitrary unbounded write but a scalar summary of the hidden state clipped through a sigmoid into $(0,1)$, which keeps the stack in a stable numeric range. That range immediately raises the empty-slot question: if a pop past the bottom reads $0$, an empty cell is indistinguishable from a real pushed value near zero, so I fill emptiness with $-1$, outside the pushed range, letting the controller detect emptiness. An all-empty stack then has every slot at $-1$, and a pop that frees the bottom writes $-1$ into that freed slot — the interior formula covers $i<p-1$, but the new last cell has no old $i+1$ and so receives the sentinel; this bottom case is easy to drop and essential to keep.

The stack feeds back into the controller through the previous top window: the hidden update becomes

$$h_t = \mathrm{sigmoid}(U x_t + R h_{t-1} + P\,s_{t-1}^{k}),$$

with $s_{t-1}^k$ the top $k$ entries and $k=2$ in the experiments, giving the controller a small view into the stack rather than only the single top, while the output stays $y_t = \mathrm{softmax}(V h_t)$ — there is no need to splice the stack into the output head, since it has already shaped $h_t$. For the synthetic algorithmic tasks I set $R=0$ so all long-range state must pass through the structured memory; for language modeling I restore the ordinary hidden recurrence. One more action earns its place: a single stack solves $a^n b^n$ by pushing through the $a$ phase and popping through the $b$ phase without ever pausing, but once several stacks run in parallel one may need to hold steady while another drains, so I add NO-OP — the exact unchanged-cell case, not a new write — giving

$$s_t[0] = a_t[\text{PUSH}]\,\mathrm{sigmoid}(D h_t) + a_t[\text{POP}]\,s_{t-1}[1] + a_t[\text{NO-OP}]\,s_{t-1}[0],$$
$$s_t[i] = a_t[\text{PUSH}]\,s_{t-1}[i-1] + a_t[\text{POP}]\,s_{t-1}[i+1] + a_t[\text{NO-OP}]\,s_{t-1}[i],$$

with the three weights summing to one, a convex superposition of the pushed, popped, and unchanged stacks. Multiple stacks are not decoration — one stack performs one action per step, so harder patterns need parallel bookkeeping — and they interact only through the shared finite hidden layer. Because the action probabilities, the pushed value, the hidden update, and the output are all differentiable, backpropagation sends error into the pushed value, into the action softmax, into earlier stack entries, and backward through time to the decision that created a bad entry: exactly the gradient path the hard-action stack lacked.

Training is still not effortless, because the local statistics of these streams form shallow basins where a bigram-like predictor captures most of the entropy while ignoring the rare forced boundary that actually tests the algorithm; so I keep ordinary SGD with BPTT truncated to about 50 steps, clip gradients hard, start near learning rate $0.1$ and halve it when validation entropy stalls, grow the length parameter during training as a curriculum, and accept random restarts as a search wrapper over SGD for the harder patterns, since the learned behavior is program-like enough that initialization can decide whether the controller finds the right discipline. A final issue surfaces only at very long strings: the soft action distribution need never be exactly one-hot, and a push at probability $0.999$ still leaves a faint pop/no-op trace whose hundreds of copies smear the stack. At test time I take the argmax action and execute the corresponding hard stack move — not changing the learned program but collapsing the relaxed controller to the discrete stack behavior it has learned, so the same finite controller runs crisply at depths beyond training. That is the complete method: a recurrent controller whose action distribution defines a differentiable superposition of stack operations, with sentinel emptiness, top-of-stack feedback, optional no-op, and multiple parallel stacks when the algorithm needs more than one piece of last-in-first-out bookkeeping.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


EMPTY_STACK_VALUE = -1.0


def clipped_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.clamp(x, -50.0, 50.0))


class CanonicalStackRNN(nn.Module):
    def __init__(
        self,
        vocab: int,
        hidden: int,
        n_stacks: int,
        stack_size: int,
        depth: int = 2,
        use_noop: bool = False,
        mod: int = 1,
    ):
        super().__init__()
        assert depth >= 1
        assert mod in (0, 1, 2)
        self.hidden = hidden
        self.n_stacks = n_stacks
        self.stack_size = stack_size
        self.depth = depth
        self.use_noop = use_noop
        self.mod = mod
        self.n_actions = 3 if use_noop else 2

        self.in2hid = nn.Embedding(vocab, hidden)
        self.hid2hid = nn.Linear(hidden, hidden, bias=False)
        self.hid2act = nn.ModuleList(
            nn.Linear(hidden, self.n_actions, bias=False) for _ in range(n_stacks)
        )
        self.hid2stack_top = nn.ModuleList(
            nn.Linear(hidden, 1, bias=False) for _ in range(n_stacks)
        )
        self.stack2hid = nn.ModuleList(
            nn.Linear(depth, hidden, bias=False) for _ in range(n_stacks)
        )
        self.hid2out = nn.Linear(hidden, vocab, bias=False)

        if mod != 2:
            nn.init.zeros_(self.hid2hid.weight)

    def initial_state(self, batch: int, device: torch.device):
        h = torch.zeros(batch, self.hidden, device=device)
        stack = torch.full(
            (batch, self.n_stacks, self.stack_size),
            EMPTY_STACK_VALUE,
            device=device,
        )
        return h, stack

    def forward(self, input_ids: torch.Tensor, hard_actions: bool = False) -> torch.Tensor:
        batch, time = input_ids.shape
        h, stack = self.initial_state(batch, input_ids.device)
        outputs = []

        for t in range(time):
            prev_h = h
            prev_stack = stack

            h_pre = self.in2hid(input_ids[:, t])
            if self.mod != 0:
                for s in range(self.n_stacks):
                    h_pre = h_pre + self.stack2hid[s](prev_stack[:, s, : self.depth])
            if self.mod == 2:
                h_pre = h_pre + self.hid2hid(prev_h)
            h = clipped_sigmoid(h_pre)

            next_stack = torch.zeros_like(prev_stack)
            for s in range(self.n_stacks):
                action = F.softmax(self.hid2act[s](h), dim=-1)
                if hard_actions:
                    action = F.one_hot(action.argmax(dim=-1), self.n_actions).to(action.dtype)

                push_w = action[:, 0:1]
                pop_w = action[:, 1:2]

                # PUSH: old row i-1 moves to row i; row 0 receives sigmoid(D h_t).
                next_stack[:, s, 1:] += prev_stack[:, s, :-1] * push_w
                new_top = clipped_sigmoid(self.hid2stack_top[s](h))
                next_stack[:, s, 0] += (new_top * push_w).squeeze(-1)

                # POP: old row i+1 moves to row i; freed bottom receives the empty sentinel.
                next_stack[:, s, :-1] += prev_stack[:, s, 1:] * pop_w
                next_stack[:, s, -1] += EMPTY_STACK_VALUE * pop_w.squeeze(-1)

                if self.use_noop:
                    noop_w = action[:, 2:3]
                    next_stack[:, s, :] += prev_stack[:, s, :] * noop_w

            stack = next_stack
            outputs.append(self.hid2out(h))

        return torch.stack(outputs, dim=1)
```
