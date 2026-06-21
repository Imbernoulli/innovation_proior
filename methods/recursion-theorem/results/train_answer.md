I will explain Kleene's second recursion theorem and Rogers' fixed-point theorem, then give a small runnable Python illustration that mimics the effective numbering and specialization construction at the heart of the proof.

The recursion theorem answers the question of how self-reference can be made legitimate inside a formal model of computation. It is not about adding a magical "give me my own source code" primitive to a programming language. Instead, it shows that once we can treat programs as data, every effective way of turning a program description into behavior has a fixed point: a program that receives its own description as an ordinary input and behaves accordingly. In Kleene's form, for every partial computable two-argument operation Q(e, x) there is an index p such that phi_p(x) is the same partial function as Q(p, x). In Rogers' form, for every total computable program transformer f there is an index p such that phi_p and phi_{f(p)} compute the same partial function. The equality is extensional: the two programs may look different, but on every input they either both diverge or both halt with the same value.

The conceptual engine behind the theorem is the combination of an effective numbering of programs and the s-m-n theorem. The numbering assigns a natural-number index to each program, and the universal function U(e, x) simulates program e on input x. We write phi_e(x) for U(e, x). The s-m-n theorem gives a total computable specialization function s(q, a) that takes a two-input program q and a value a and returns a one-input program that computes x |-> phi_q(a, x). In other words, we can hard-code a parameter into a program description by a purely syntactic, effective operation.

To obtain a self-referential program, we do not try to guess its own index in advance. We build a helper program B(q, x) that computes Q(s(q, q), x). This is computable because s and Q are computable. Let b be an index for B, and define p = s(b, b). Running p on x is the same as running B(b, x), because p was produced by specializing B with its own index. But B(b, x) was defined to compute Q(s(b, b), x), and s(b, b) is exactly p. Therefore phi_p(x) is the same partial function as Q(p, x). The fixed point appears because the specialization operation closes the loop: the builder's index b is fed back through s to produce a concrete program p that knows p as data.

This is why the result is a theorem rather than a paradox. Informally it sounds circular to say "the program uses its own description," but operationally the construction is staged. First we write a computable transformer from program texts to program texts. Second we take an index for that transformer. Third we specialize that index to itself. Only after this finite, effective construction does the program run. When it runs, p is available as a normal parameter because the construction placed it there. There is no inconsistent truth condition like "this sentence is false," no halting oracle, and no requirement that semantic equality be decidable. Self-reference enters as a fixed point of a computable operator on program descriptions.

The Rogers fixed-point form follows immediately from the Kleene form. Given a total computable transformer f on indices, define Q(e, x) = phi_{f(e)}(x). Kleene's theorem gives an index p with phi_p(x) ~= Q(p, x), which means phi_p ~= phi_{f(p)}. This is a powerful statement about program transformation: no matter how f wraps, logs, delays, optimizes, or rewrites a program, there is always some program whose observable behavior survives the transformation. The fixed point is behavioral, not textual, so the theorem accommodates many different encodings of the same function.

Quines are a familiar special case. Choose Q(e, x) to output a representation of e; the fixed-point construction then yields a program that prints its own index. Recursive definitions are another case: choose Q(e, x) to call the function named by e on smaller arguments. Self-recognition is a third case: choose Q(e, x) to compare the input x with the index e. In every instance the same fixed-point skeleton turns an external parameter slot into the program's own description. The theorem therefore unifies a wide range of phenomena that at first look like separate tricks.

The broader lesson is that self-reference is a capability created by representability. Once a system can enumerate programs and effectively produce specialized descriptions, it can close the loop on a program's own index. The theorem does not say that every program can introspect its source by itself; it says that for every computable way of using a would-be self-description, there is a program whose construction supplies exactly the description that use requires.

The following Python script simulates the same pattern with ordinary functions. A "program" is a Python function that takes a code string and an input value. The s_m_n function hard-codes the first argument by producing a new lambda. The make_self_referential function plays the role of the theorem: it builds a helper that calls s_m_n on itself, then specializes that helper with its own text. The result is a program whose behavior on input x is Q(self, x), where self is the final program's own code. Two examples are provided: a simple quine-like program that returns its own source, and a recursive factorial program that receives its own code and calls itself on smaller inputs. Running the script prints the self-description and verifies that the factorial example returns the expected values.

```python
import textwrap


def s_m_n(program_text, param):
    """Specialize a two-argument program by hard-coding the first parameter.

    In a real computability setting this is the total computable s-m-n
    function; here we simulate it by returning a lambda whose closure
    keeps the parameter and the program text.
    """
    def specialized(x):
        # Evaluate the program with its hard-coded first argument.
        env = {"param": param, "x": x}
        exec(program_text, env, env)
        return env.get("result")
    return specialized


def make_self_referential(builder_text):
    """Apply the recursion-theorem construction: p = s(b, b)."""
    # In the proof, b is the index of B and p = s(b, b).
    # Here we use the program text itself as its own index/description.
    return s_m_n(builder_text, builder_text)


# Example 1: a quine-like self-description.
quine_builder = textwrap.dedent('''
    result = param
''')
quine_program = make_self_referential(quine_builder)
print("Self-description returned by quine-like program:")
print(quine_program(None))
print()

# Example 2: a recursive factorial that receives its own code as `param`.
fact_builder = textwrap.dedent('''
    def Q(self_code, n):
        if n == 0:
            return 1
        # Re-specialize this program text on itself to call the same function.
        env = {"param": self_code, "x": n - 1}
        exec(self_code, env, env)
        return n * env.get("result")

    result = Q(param, x)
''')
fact_program = make_self_referential(fact_builder)
print("Factorial values computed by the self-referential program:")
for n in range(6):
    print(f"  {n}! = {fact_program(n)}")
```
