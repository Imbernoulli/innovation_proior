Induction is the problem of moving from a finite observed history to a prediction about what comes next. In ordinary Bayesian inference I begin by choosing a hypothesis class and a prior over it, then update by Bayes' rule. That procedure is coherent, but its conclusions depend heavily on the class I chose at the outset. If the true explanation lies outside that class, no amount of data will ever resurrect it. Solomonoff induction, which I take as the canonical method here, removes this hand-selection step by making the hypothesis class universal in a precise sense: it ranges over all computable explanations.

The key move is to identify an explanation with a program for a fixed universal prefix Turing machine. Before I see any data I commit to a reference machine U. Once that machine is fixed, any computable way of generating a sequence is represented by some program p. A law of nature, a simulation, a compact rule, or even a long lookup table can all be written as programs. This gives the largest controlled hypothesis class I can use without stepping outside the theory of algorithms.

Having turned explanations into programs, I need a prior over them. Solomonoff induction assigns each program p a prior weight of 2^{-|p|}, where |p| is the length of the program in bits. This is not an arbitrary complexity penalty added to a loss; it is the natural measure of prefix-free code space. Each extra bit of description length costs a factor of two in prior mass. The result is a formal version of Occam's razor: simpler explanations receive exponentially more weight.

Crucially, the method does not keep only the shortest program that is consistent with the observations. If the observed prefix is x, I sum 2^{-|p|} over every program p whose output begins with x. This sum defines the universal semimeasure M_U(x). The summation matters because it turns Solomonoff induction into a Bayesian mixture rather than a hard model-selection rule. Every computable explanation that fits the data remains alive, and its posterior influence is determined by its length-weighted mass.

Prediction then follows by conditioning this universal prior. To predict the probability of a continuation y after observing x, I compare the mass of programs that output xy with the mass of programs that output x. The conditional probability is proportional to M_U(xy) relative to M_U(x). In this way the prior over programs is filtered by consistency with the data, and future predictions are obtained by averaging over the surviving hypotheses with their original length-based weights.

The conceptual shift is from asking "which model family should I choose?" to asking "what happens if every computable explanation is already in the prior?" Standard Bayesian modeling forces me to decide in advance whether to use polynomials, Markov chains, neural networks, grammars, or some other family. Anything outside that choice has probability zero forever. Solomonoff induction replaces that modeling decision with computability itself. If an explanation can be implemented as a program, it is already included.

This unifies three ideas that usually appear separately. It is Bayesian because prediction is posterior averaging over hypotheses. It is Occamian because shorter descriptions receive exponentially larger prior mass. And it is computational because the hypotheses are programs and the universality claim is tied to universal Turing machines and algorithmic information theory. The insight is not merely that simpler is better; it is that simplicity can be measured by program length inside a universal Bayesian prior.

There is an unavoidable cost. Exact Solomonoff induction is incomputable. To evaluate M_U(x) exactly I would need to know which programs eventually output a string beginning with x, and that requires solving the halting problem in general. This does not make the theory useless, but it changes its role. It is not a practical algorithm that I can run on real data in its exact form. It is an ideal standard against which compression methods, model-selection criteria, and universal prediction schemes can be compared.

Machine dependence is real but bounded in the usual algorithmic-information sense. A different universal machine changes program lengths by at most an additive compiler constant, so prior weights change by multiplicative constants rather than by a data-dependent refitting of the language. Still, the reference machine must be chosen before seeing the data. That requirement is essential; without it the universal prior could become a disguised way to encode the answer after the fact.

In practice I can only approximate the ideal. The code below illustrates the principle with a small finite collection of simple programs, each assigned a description length. Given an observed binary prefix, it weights every compatible program by 2^{-|p|} and predicts the next bit by conditioning that mixture. It is not Solomonoff induction in full, because the true universal class is infinite and incomputable, but it captures the same structure on a toy scale.

```python
import math

def generate(rule, n):
    """Generate the first n bits of a simple computable sequence."""
    if rule == "zeros":
        return "0" * n
    if rule == "ones":
        return "1" * n
    if rule == "alt01":
        return ("01" * ((n // 2) + 1))[:n]
    if rule == "alt10":
        return ("10" * ((n // 2) + 1))[:n]
    if rule == "repeat_0011":
        return ("0011" * ((n // 4) + 1))[:n]
    if rule == "fibonacci_parity":
        a, b = 0, 1
        out = ""
        for _ in range(n):
            out += str(a % 2)
            a, b = b, a + b
        return out
    return ""

rules = {
    "zeros": 2,
    "ones": 2,
    "alt01": 4,
    "alt10": 4,
    "repeat_0011": 6,
    "fibonacci_parity": 8,
}

def finite_solomonoff_predict(x, rules):
    """Weight each compatible rule by 2^{-description length} and condition."""
    mass_0 = 0.0
    mass_1 = 0.0
    total = 0.0
    for rule, length in rules.items():
        n = len(x) + 1
        gen = generate(rule, n)
        if gen.startswith(x):
            weight = 2.0 ** (-length)
            total += weight
            if gen[len(x)] == "0":
                mass_0 += weight
            else:
                mass_1 += weight
    if total == 0:
        return 0.5, 0.5
    return mass_0 / total, mass_1 / total

if __name__ == "__main__":
    observed = "01010"
    p0, p1 = finite_solomonoff_predict(observed, rules)
    print(f"Observed prefix: {observed}")
    print(f"P(next = 0) = {p0:.4f}")
    print(f"P(next = 1) = {p1:.4f}")
```

So the final summary of Solomonoff induction is this. I fix a universal machine before seeing data. I assign every program a prior weight that decreases exponentially with its length. I sum those weights over all programs compatible with the observed prefix to obtain the universal semimeasure. I condition that measure to predict future observations. The method is incomputable in its exact form, but it provides a clean ideal theory of induction in which the only restrictions on hypotheses are computability and a fixed reference machine. It is best understood not as another model-selection criterion, but as a gold-standard answer to the question of how to generalize from evidence without secretly choosing the answer in advance.
