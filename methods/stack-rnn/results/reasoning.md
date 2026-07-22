I start with the simplest recurrent predictor and try to make the failure precise. I read a token `x_t`, update `h_t = sigmoid(U x_t + R h_{t-1})`, and predict `y_t = softmax(V h_t)`. This is a perfectly good way to let the past influence the present, but every fact about the prefix has to live inside one fixed-width vector. On a short `a^n b^n` stream the trajectory can look as if it is counting, because the training lengths carve out a small region of state space where "after this many `a`s, expect this many `b`s" works. But the test deliberately leaves that region. If the rule asks for arbitrary depth, the memory resource itself has to grow with depth; otherwise I am asking a fixed vector to behave like an unbounded data structure.

The LSTM makes me more careful. It is not fair to say recurrent nets cannot remember. With gates and a linear cell, a unit can keep nearly constant error flow and can learn to increment or decrement a quantity. That really can behave like a counter on one-turn languages. But now the shape of the task matters. A counter is enough for "how many closers remain?" when the alphabet and order are simple. It is not enough for "( [ ( ) ] )", where the next legal closer is determined by the last unmatched opener, not just by the number of open brackets. Reversal makes the same point in a cleaner way: I need the last symbol I stored to be the first symbol I read back. The missing memory is last-in-first-out.

That points me at a finite recurrent controller with a stack hung off the side of it: a finite state plus a stack is a pushdown automaton, which is the abstract machine for context-free-style structure, and earlier neural pushdown automata already put an external stack beside a recurrent controller and learned small grammars. So the bias is not arbitrary. The hard part is that I need the whole thing to train from next-token loss by backpropagation, and a hard stack action is the obstruction. If the controller chooses exactly PUSH or exactly POP with an argmax, a small change in a weight usually leaves the chosen action unchanged. The forward behavior is identical, the loss is identical, and the useful gradient through the action is zero. Reinforcement-style credit assignment is possible in principle, but a wrong early stack action corrupts a long rollout, which is exactly the high-variance setting I am trying to avoid.

So let me try to stop choosing one action during training. The controller produces a softmax distribution over actions and the memory executes the weighted mixture of the pure action outcomes. I write the two-action version first because the indices are easiest to audit. The controller emits `a_t = softmax(A h_t)`, with entries for PUSH and POP. The stack is a vector `s_t` with top at index 0. If I push, the new top is a learned value and every old cell moves one slot deeper. If I pop, every old cell moves one slot toward the top. I want to read off where each destination cell sources its value from. For the top, a push writes the new value and a pop pulls up what was below the top, so `s_t[0] = a_t[PUSH] sigmoid(D h_t) + a_t[POP] s_{t-1}[1]`. For a deeper cell `i > 0`, a push moves down whatever sat one slot above (`i-1`), and a pop moves up whatever sat one slot below (`i+1`): `s_t[i] = a_t[PUSH] s_{t-1}[i-1] + a_t[POP] s_{t-1}[i+1]`. Push shifts down so destination `i` reads old `i-1`; pop shifts up so destination `i` reads old `i+1`. I keep wanting to swap those two index offsets, so before I trust them I should just run them.

I take a width-6 stack, a single push value `0.7` standing in for the "saw an `a`" marker, and walk `a^3 b^3`. Three pushes, then three pops, applying the formulas literally:

```
init      [-1, -1, -1, -1, -1, -1]
push #1   [0.7, -1, -1, -1, -1, -1]
push #2   [0.7, 0.7, -1, -1, -1, -1]
push #3   [0.7, 0.7, 0.7, -1, -1, -1]
pop  #1   [0.7, 0.7, -1, -1, -1, -1]
pop  #2   [0.7, -1, -1, -1, -1, -1]
pop  #3   [-1, -1, -1, -1, -1, -1]
```

That is the behavior I needed and the offsets are right: the pushes stack the marker up from index 0, and the pops peel it back off the top, one per step. The thing I did not anticipate but now find reassuring is the final row. After exactly three pops the stack is all `-1` again — its starting value — so the top cell reads `-1` precisely when the `b` run has consumed every `a`. That is the boundary the predictor has to detect in `a^n b^n`, and it falls out of the dynamics for free rather than needing a separate counter. The same trace also tells me what a pop reads when it frees the bottom: the new last cell has no old `i+1` to pull from, so it has to receive something, and in the walk above the freed bottom is taking the value `-1`. The bottom case is easy to omit from the interior formula, so I make it explicit — a pop writes the empty value into the slot it frees.

But which empty value? I picked `-1` above almost by reflex; the obvious cheaper choice is to fill empty slots with `0`, since `zeros_like` is the natural initialization. So let me check whether `0` actually works. Push a small but legitimate value — `sigmoid(D h_t)` lives in `(0,1)`, so `0.05` is a perfectly possible pushed top — then pop it:

```
zero-fill:  push 0.05 -> [0.05, 0, 0, ...]   pop -> [0, 0, 0, ...]   top reads 0.0
-1 sentinel: push 0.05 -> [0.05, -1, -1, ...] pop -> [-1, -1, ...]   top reads -1.0
```

The zero-fill case is the problem: after the pop the top reads `0.0`, which is inside the range a real push can produce. The controller cannot distinguish "empty" from "I pushed a value very near zero". With the `-1` sentinel, empty sits outside the entire `(0,1)` range of pushable values, so emptiness is a signal the controller can read. So zero-fill is the cheaper option that quietly breaks the one cue I most need, and `-1` is the right choice for a reason I can point at rather than a preference.

The top value `sigmoid(D h_t)` is doing the rest of the work on the write side. It is not an arbitrary unbounded write; it is a scalar summary of the current hidden state clipped through a sigmoid into `(0,1)`, which is exactly what keeps the pushed values in a stable, bounded range and makes the `-1` sentinel separable from them.

Now I feed the stack back into the controller. The hidden update becomes `h_t = sigmoid(U x_t + R h_{t-1} + P s_{t-1}^k)`, where `s_{t-1}^k` is the top `k` stack entries. I use `k = 2`, which gives the controller a small window into the stack rather than only the single top. For the synthetic algorithmic tasks I can set `R = 0` and force long-range state to pass through the structured memory; for language modeling I can restore the ordinary hidden recurrence. The output stays `y_t = softmax(V h_t)`. I do not concatenate the stack top into the output head, because the stack has already influenced `h_t`.

I also want a no-op action. A single stack that solves `a^n b^n` can push during the `a` phase and pop during the `b` phase without ever waiting. But as soon as I run several stacks in parallel, one stack may need to hold steady while another drains. NO-OP is exactly the unchanged-cell case, not a new kind of write: the top becomes `s_t[0] = a_t[PUSH] sigmoid(D h_t) + a_t[POP] s_{t-1}[1] + a_t[NO-OP] s_{t-1}[0]`, and each deeper cell becomes `s_t[i] = a_t[PUSH] s_{t-1}[i-1] + a_t[POP] s_{t-1}[i+1] + a_t[NO-OP] s_{t-1}[i]`. The three action weights come from a softmax so they sum to one, which makes each cell a convex combination of the pushed, popped, and unchanged stacks.

There are two things I am taking on faith here that I should actually check, because the whole point of the relaxation is that it has to agree with a real stack at the limit and blend sensibly away from it. First, the one-hot limit: when the action distribution is exactly `[1,0]` or `[0,1]`, the soft update should reproduce a literal hard push or hard pop. I run a mixed sequence of pushes and pops with random pushed values through both the soft formula at one-hot weights and an explicit array push/pop, step by step, and they agree at every step. So the hard stack is genuinely the corner of this relaxed family, not just a nearby approximation. Second, the blend itself. Take a stack `[0.2, 0.5, -1, -1, -1, -1]`, push-weight `0.6`, pop-weight `0.4`, pushed value `0.9`:

```
push_only -> [0.9,  0.2,  0.5, -1, -1, -1]
pop_only  -> [0.5, -1,   -1,  -1, -1, -1]
blend     -> [0.74, -0.28, -0.1, -1, -1, -1]
```
