# Stack-Augmented Recurrent Network

A Stack RNN is a simple recurrent controller coupled to one or more differentiable stacks. The
controller emits a softmax distribution over stack actions; the next stack is the probability-weighted
superposition of the pure PUSH, POP, and optionally NO-OP outcomes. The relaxation gives a gradient
through the controller while preserving the hard stack as the one-hot limit.

## Equations

Base recurrent predictor:

```
h_t = sigmoid(U x_t + R h_{t-1})
y_t = softmax(V h_t)
```

For each stack, action probabilities:

```
a_t = softmax(A h_t)                  # PUSH/POP, or PUSH/POP/NO-OP
```

Two-action stack, top at index `0`:

```
s_t[0] = a_t[PUSH] * sigmoid(D h_t) + a_t[POP] * s_{t-1}[1]
s_t[i] = a_t[PUSH] * s_{t-1}[i-1]   + a_t[POP] * s_{t-1}[i+1],  i > 0
```

With NO-OP:

```
s_t[0] = a_t[PUSH] * sigmoid(D h_t) + a_t[POP] * s_{t-1}[1] + a_t[NO-OP] * s_{t-1}[0]
s_t[i] = a_t[PUSH] * s_{t-1}[i-1]   + a_t[POP] * s_{t-1}[i+1] + a_t[NO-OP] * s_{t-1}[i]
```

The empty stack value is `-1`. In the finite implementation, popping also writes `-1 * a_t[POP]`
into the freed bottom slot. The hidden state reads the previous stack top window:

```
h_t = sigmoid(U x_t + R h_{t-1} + P s_{t-1}^k),  k = 2 in the experiments
y_t = softmax(V h_t)
```

Multiple stacks run in parallel with separate action/write/read matrices and interact through the
shared hidden state. For synthetic tasks, `R` can be disabled so recurrence flows only through the
stacks; for language modeling, the full hidden recurrence is used.

## Canonical Implementation Shape

This PyTorch port mirrors the official C++ `StackRNN.h` update logic: scalar stack slots, no linear
biases, sigmoid hidden/top values, all stack slots initialized to `-1`, pop-bottom sentinel injection,
optional hard argmax actions, and output logits from `h_t` only.

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

Training follows the reference setup: SGD with BPTT truncated to 50 steps, hard gradient clipping
at 15, initial learning rate `0.1`, halving when validation entropy stops improving, and a curriculum
that increases the generator length. Random restarts are used for harder patterns. At validation or
test time, `hard_actions=True` implements the reference rounding step that replaces action
probabilities with the argmax one-hot action to reduce long-horizon numerical drift.
