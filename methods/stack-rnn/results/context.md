## Research Question

The data is a stream of discrete symbols made by short deterministic generators: `a^n b^n`,
`a^n b^n c^n`, `a^n b^{2n}`, `a^n b^m c^{n+m}`, bracket-like recursive strings, memorization in
reverse order, and binary addition with the answer emitted in reverse. Training sees only a
concatenated stream. The model is not told where one generated sequence ends, and for the main
toy problems it is not told which future symbols are forced by the generator. At evaluation time,
the only score that matters is whether it predicts those forced symbols on lengths and nesting
depths beyond the training range.

The hard part is not next-token prediction in the ordinary statistical sense. A local model can learn
that many `a` tokens are followed by more `a` tokens and many `b` tokens by more `b` tokens. The
test asks for something stronger: infer the compact rule behind the stream and execute that same
rule for larger `n`. In `a^n b^n`, the predictor must know when the `b` run has exactly matched
the previous `a` run. In nested brackets, it must know which opener is most recent. In reversal, it
must remember the symbols in last-in-first-out order. These are memory-organization demands, not
just longer-range correlations.

## Existing Recurrent Machinery

The standard sequence model is the Elman simple recurrent network. For a one-hot input token
`x_t`, hidden width `m`, and vocabulary size `d`, it keeps a hidden state

```
h_t = sigmoid(U x_t + R h_{t-1})
y_t = softmax(V h_t).
```

The recurrent edge makes the previous internal state available at the next step, so time is
represented implicitly in the trajectory of `h_t`. This is enough to learn many temporal
regularities and N-gram-like predictive structures, but the prefix is compressed into a fixed
`R^m` vector chosen before seeing the sequence length. If the task needs a resource that grows
with the count or nesting depth, the ordinary hidden state has no such resource.

The long-memory recurrent line weakens this objection but does not remove it. LSTM memory cells
and gates can preserve error flow over long time lags and can behave like counters on some
one-turn languages. Constrained recurrent matrices and slowly changing context units similarly
make longer summaries easier to keep. These mechanisms show that structure in the recurrence
matters, and that gradient descent can train useful long-memory dynamics. They still leave open
the shape of memory needed for order-sensitive recursion: a counter or slow summary is not the
same thing as an addressable last-in-first-out store.

## Formal-Language Pressure

The motivating generators sit near the boundary between regular-looking local statistics and
context-free structure. A finite-state controller can remember which phase it is in, such as "still
reading `a`" versus "now reading `b`", but it cannot keep an unbounded count or an unbounded
nesting stack. A pushdown automaton adds exactly one ingredient: a stack that can grow and whose
top is read first. That makes it the natural abstract machine for many context-free patterns.

This observation cuts both ways. It explains why fixed recurrent vectors are a poor fit for the
unbounded case, but it also warns that the useful memory operation is structured. A fully general
random-access memory can do far more than the toy languages require; a bag-like slow context unit
does too little; and a hand-designed symbolic recognizer solves only a chosen grammar. The missing
piece is a way to put a simple structured memory under a learned neural controller.

## Prior Memory-Augmented Baselines

Early neural pushdown automata already paired recurrent controllers with external stack memory and
showed that grammatical structure can be learned with stack actions. That work establishes the
right bias: a recurrent state machine plus a stack can represent the relevant rules. Its drawback
for this setting is trainability and generality. The earlier systems often used specialized error
functions, hints, isolated grammars, or hard action mechanisms, so they do not yet give a clean
drop-in next-symbol predictor trained from the raw stream alone.

The Neural Turing Machine gives another path: attach a differentiable memory matrix to a neural
controller and use soft attention for reading and writing. It is trainable by gradient descent and
can learn algorithmic behaviors such as copy and sort. But the memory is a fixed-size matrix, and
the read/write/addressing interface is much broader than a last-in-first-out problem demands. For
the simple generators here, the question is whether a narrower memory topology can be learned more
directly while retaining gradient-based training.

## Evaluation and Code Frame

The controlled benchmark is next-symbol prediction on generated streams. Training uses small
parameters, commonly `n < 20` for the counting tasks, and testing pushes to longer cases such as
`n` up to 60. The score is sequence-level correctness on the deterministic part of the generated
string. For binary addition, the deterministic output positions are supplied; for the other toy
streams, sequence boundaries and deterministic-position labels are absent. A standard language
modeling sanity check on Penn Treebank can be used to ensure that the model is not only a toy
recognizer.

The implementation frame is an ordinary recurrent next-symbol harness: embed the current token,
update a recurrent controller state, project the state to vocabulary probabilities, and train with
BPTT and SGD. The open slot is the controller update and whatever memory it is allowed to access;
everything around that slot is conventional.

```python
import torch
import torch.nn as nn


class StreamPredictor(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab, hidden)
        self.controller = None  # TODO: define the recurrent state and any allowed memory.
        self.out = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = []
        state = None
        for token in input_ids.transpose(0, 1):
            x_t = self.in_embed(token)
            state = self.controller(x_t, state)
            states.append(state.hidden)
        return self.out(torch.stack(states, dim=1))
```
