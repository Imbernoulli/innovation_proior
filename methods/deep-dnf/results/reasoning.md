I start with the ordinary threshold neuron because it is the tempting baseline. If I want `x1 and x2 and x3`, I can put weight one on those three inputs and set a bias so that only the all-one case crosses the threshold. But now the bias is carrying the count test, and that is exactly what I cannot read cleanly after training. A trained real-valued vector plus a trained real-valued bias does not tell me, without an arbitrary thresholding convention, which variables are members of the rule.

So I stop trying to make addition behave like conjunction. Addition counts; conjunction asks whether every required condition is true. Multiplication already has that behavior on Boolean inputs: one zero factor kills the product, and all-one factors keep it at one. If I can make the product include a variable only when a learned flag says to include it, I can remove the bias and make the flag itself be the symbolic decision.

For one input bit `x_i` and one membership flag `m_i`, the factor for a conjunction has to pass four cases. If `m_i = 0`, the variable is absent, so the factor must be `1` for both `x_i = 0` and `x_i = 1`. If `m_i = 1`, the variable is present, so the factor must be `0` when `x_i = 0` and `1` when `x_i = 1`. The formula `1 - m_i(1 - x_i)` gives exactly those four values: `(0,0)->1`, `(1,0)->1`, `(0,1)->0`, `(1,1)->1`, where the pair is `(x_i,m_i)`.

That gives the conjunction neuron

`O_conj(x) = product_i (1 - m_i(1 - x_i))`.

The membership cannot be an unconstrained real number, so I use `m_i = sigmoid(c w_i)` or a clipped hard thresholding function, with `c >= 1` to sharpen the transition. If `m_i` goes to zero, the factor goes to one and the variable disappears. If `m_i` goes to one, the factor becomes `x_i` and the variable is required. The trained membership vector is now the rule: large memberships are included literals, small memberships are absent inputs.

The disjunction is the dual calculation. A selected input should contribute `x_i` to an OR, while an unselected input should contribute nothing. The contribution is `m_i x_i`, so the OR over selected inputs is

`O_disj(x) = 1 - product_i (1 - m_i x_i)`.

Again the four cases are right. `m_i = 0` makes the product factor `1`, so the input has no effect. `m_i = 1` makes the factor `1 - x_i`, so the final `1 - product` is true as soon as at least one selected input is true. This is also the product/noisy-OR version of De Morgan's law.

Now I can compose layers. A bank of conjunction neurons produces candidate clauses. A disjunction neuron over those clause outputs selects which clauses enter the final rule. That is a differentiable DNF:

`DNF(x) = DISJ(CONJ(x))`.

The reverse order gives a CNF, and an outer OR can combine a CNF branch with a DNF branch when one normal form is much shorter than the other. But for a rule written as several clauses with one head, conjunctions followed by a disjunction are the direct shape.

I need to be careful about the gradient story. For a conjunction factorization,

`d O_conj / d m_i = -(1 - x_i) product_{k != i}(1 - m_k(1 - x_k))`.

So the derivative is zero when `x_i = 1` and negative when `x_i = 0`, up to the product of the other factors. A sample with `x_i = 0` is the only kind of sample that can tell the learner whether requiring `x_i` would suppress an overly broad clause. Under a loss that wants the current conjunction output to go down on such a sample, gradient descent moves `m_i` upward. If the loss instead wants the conjunction to remain high while `x_i = 0`, the same sign pushes `m_i` downward. The derivative itself does not say "include" or "exclude"; the target error supplies that direction.

There is a numerical wall in the product. If many memberships begin in the middle, then many factors are below one, the product becomes tiny, and the derivative carries the same tiny product. The initialization therefore has to be sparse for wide inputs: most memberships start near zero so most factors are near one, with either a small random subset near one or a negative-mean initialization of the underlying logits. If the product is long enough to underflow, I can compute the same quantity as `exp(sum log(epsilon + factor))`; for moderate layers the direct product is the same mathematical object.

For negated literals I do not need a different selector. I can make `1 - x_i` or `not atom` part of the input vocabulary and let the same scalar membership mechanism select it. Then a clause can include positive or negated atoms, but the layer still has one scalar membership for each input item it sees.

The final clean-up is interpretability. If exact satisfiability has been reached, extra memberships may still be one even though removing them would not change the examples. Those are pruned by testing whether turning a membership from one to zero changes the loss. When exact binary memberships are not reached, an optional penalty like `m(1-m)` pushes memberships toward zero or one. The core method remains the same throughout: scalar memberships, multiplicative AND, De Morgan OR, DNF by composition, and formula extraction by thresholding the learned memberships.
