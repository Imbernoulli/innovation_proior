The canonical method I am presenting is Turing's proof of the unsolvability of the halting problem, also known in its original form as Turing's negative solution to Hilbert's Entscheidungsproblem. The question is whether there exists a definite general method that decides, for every formula of first-order logic, whether it is provable. Because Gödel's completeness theorem makes provability coincide with validity, this is the same as asking whether validity is decidable by a uniform mechanical procedure. My goal is to show that no such procedure exists, but to do that I first have to say precisely what a mechanical procedure is.

I start by watching a human being compute. Stripped to its essentials, the activity is a person following fixed rules while reading and writing symbols on paper. The two-dimensional page is not essential; everything can be written on a one-dimensional tape divided into squares. Only finitely many distinct symbols can be used, because if the alphabet were infinite two symbols would eventually be too close to tell apart. At each moment the computer attends to only a bounded region, which I can idealize as a single scanned square. What happens next depends only on the symbol in that square and on the computer's current internal condition, or state of mind, which must also be finite; otherwise two states would be indistinguishable. In one elementary step the computer may change the symbol on the scanned square, shift attention by one square to the left or right, and enter a new state. This analysis yields the a-machine, now called a Turing machine: a finite set of m-configurations, a finite alphabet, a tape, a scanned square, and a transition table that fixes every step from the pair of scanned symbol and current m-configuration. The reason this captures all effective computation is that each clause was forced by the requirement that a human could actually carry the procedure out by rote.

Next I encode machines so that they can be treated as data. Each line of a machine's table is standardized to the form "in m-configuration q_i scanning symbol S_j, print S_k, move L or R or N, and go to m-configuration q_l." I encode q_i as D followed by i copies of A, encode S_j as D followed by j copies of C, join the lines with semicolons, and map the resulting standard description to a string of digits. The result is a single integer, the description number of the machine. Because every machine is a finite string over a finite alphabet, there are only countably many machines, and hence only countably many computable sequences. That already tells me that if I try a naive Cantor diagonal over the computable sequences something must go wrong, because the computable sequences are enumerable.

The false step in the naive diagonal is the assumption that I can tell which description numbers belong to circle-free machines, where circle-free means a machine that keeps printing output figures forever, as opposed to a circular machine that eventually stops producing figures. If I had a total decider D that labeled every integer as satisfactory when it is the description number of a circle-free machine and unsatisfactory otherwise, I could combine D with a universal machine U to build a new machine H. The universal machine is a single machine that, given the standard description of any machine M on its tape, simulates M step by step. It works because both the program and the running state of M are just strings of symbols that U can mark, compare, copy, and erase. Using D and U, the machine H scans the integers in order, uses D to pick out the satisfactory ones, and for the n-th satisfactory machine uses U to compute its n-th output figure and prints that figure. By construction H itself is circle-free, because every stage finishes in finite time and there are infinitely many satisfactory numbers.

Let K be the description number of H. In the K-th stage of its run, H examines K, which is its own description. If D says K is unsatisfactory, then D has misclassified a circle-free machine as circular, which is impossible. If D says K is satisfactory, then H is told that it is the R(K)-th circle-free machine, and its next instruction is to compute the first R(K) figures of that machine and write down the R(K)-th. But that machine is H itself, so to produce its own R(K)-th figure H must first run itself up to the production of that same figure, which can never be completed. H stalls and becomes circular, contradicting both the assumption that H is circle-free and D's verdict. Since both possible answers lead to a contradiction, the decider D cannot exist. No machine decides, from a description number, whether the described machine is circle-free.

This result propagates. First, no machine can decide whether an arbitrary machine ever prints a particular symbol, such as 0. If such a decider E existed, I could build an auxiliary machine that tests whether a given machine prints 0 infinitely often by striking successive zeros and consulting E, and a machine prints 0 or 1 infinitely often exactly when it is circle-free. That would make circle-free decidable, which is impossible. So ever-prints-0 is also undecidable.

Then I reduce the logical decision problem to ever-prints-0. For each machine M I construct a single first-order formula Un(M) whose provability in the functional calculus K is equivalent to M eventually printing 0. The formula uses predicates that encode complete configurations of M: R_S(x,y) says that in configuration x square y bears symbol S; I(x,y) says that in configuration x square y is scanned; K_q(x) says that in configuration x the machine is in m-configuration q; and F(x,y) is the successor relation, used both for successive instants and for neighboring tape squares. Each instruction of M becomes a sentence stating that if the machine is in the right state and scans the right symbol, then the next configuration has the appropriate new symbol, scanned square, and m-configuration, and every other square keeps its symbol. The conjunction of these instruction sentences, together with axioms for the initial blank tape and the successor structure, forms the antecedent of Un(M), while the consequent asserts that at some configuration some square bears the symbol 0.

The two directions of the equivalence are an induction and a soundness argument. If M really does print 0, I can prove Un(M) by walking the logic in lockstep with M's computation: for each n I derive the formula describing the n-th complete configuration from the previous one using the corresponding instruction sentence, and once 0 appears I instantiate the existential consequent. Conversely, if Un(M) is provable, then by soundness it is true in the intended interpretation, where the antecedent describes the actual run of M; hence the consequent is true and M really does print 0. Therefore a decision procedure for first-order provability, applied to the formulas Un(M), would decide whether M ever prints 0, and no such decision procedure exists. So the Entscheidungsproblem is unsolvable.

This is a different kind of result from Gödel's incompleteness theorems. Gödel showed that inside a fixed formal system there are sentences that are neither provable nor refutable. What Turing proved is that there is no general method at all for deciding, of an arbitrary formula, whether it is provable. The diagonal argument is the engine: not the unflipped digit equation 1 minus phi_n(n), but the self-application that turns a verdict of "runs forever" into an actual stall.

```python
# A tiny runnable illustration of the self-referential diagonal at the heart
# of Turing's halting undecidability.  A perfect halting decider D cannot exist,
# because we can always build a program H that does the opposite of what D
# predicts H will do.

def make_H(D):
    """Build the diagonal program from a hypothetical halting decider D."""
    def H(P):
        # Ask D whether P halts when given itself as input.
        if D(P, P):
            # D predicts P(P) halts, so H deliberately loops.
            return "LOOPS_FOREVER"
        else:
            # D predicts P(P) loops, so H halts immediately.
            return "HALTS"
    return H

# Suppose, for contradiction, that a perfect decider D exists.
def D_placeholder(program, input_):
    # A placeholder that merely returns a boolean verdict.
    return True

H = make_H(D_placeholder)
verdict = D_placeholder(H, H)  # D's prediction for H(H)
actual = H(H)                   # what H actually does

print(f"D predicts H(H) halts: {verdict}")
print(f"H(H) actually does:    {actual}")
print("If D says True, H loops; if D says False, H halts.")
print("Either way D is wrong, so no perfect decider D can exist.")
```
