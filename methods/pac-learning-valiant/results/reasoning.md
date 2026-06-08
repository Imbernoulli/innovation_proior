Let me start from the thing that actually bothers me. We have a clean theory of what can be *computed*, and the reason we have it is that Turing handed us a precise model of mechanical calculation. Once the model was on the table, "what can be computed?" stopped being a vague philosophical question and became something you could prove theorems about — this is computable, that is not, this reduces to that. Learning has nothing like that. I say a child learns to recognize a table, a doctor learns to diagnose, a program learns a concept, and not one of those sentences is attached to anything I could prove. So the question I want is the mirror of computability: is there a precise model of *acquiring a recognizer without being handed the program for it*, a model sharp enough that I can prove whole classes of concepts learnable, and prove others not?

What would I even demand of such a model? Three things at once, and I think the difficulty is that they pull against each other. I want to be able to prove that a *whole characterizable class* of concepts can be acquired — not "here's a heuristic that often works," but a theorem about a class. I want the concepts in that class to be rich enough to matter for real knowledge. And I want the acquisition to be *feasible* — to take a number of steps that doesn't blow up. The third one is non-negotiable for me because it's exactly what gave computability its teeth: "feasible" has a precise meaning now, polynomial time, and if I don't bake that in I'll just recover the same vague "eventually, somehow" that I'm trying to escape.

So let me look hard at what's already been tried, because the gaps are going to tell me what the model has to do.

The most rigorous prior thing is Gold's identification in the limit. The setup: a target language, and the learner is fed a growing presentation of it — either a text, which just enumerates the strings that are *in* the language, or an informant, which labels strings in or out. After each new datum the learner emits a guessed grammar. It "identifies in the limit" if after finitely many wrong guesses it locks onto a correct grammar and never changes again. I admire the rigor, but stare at what it demands. First, the convergence is *exact* — it has to settle on a grammar that is exactly right. Second, it never knows it's done; a contradicting example can show up arbitrarily far in the future. Third, there's no clock at all — convergence is purely asymptotic, "in the limit," which is the precise opposite of feasible. And the result that falls out of demanding exactness is brutal: any class with an infinite ascending chain of languages whose union is also in the class — superfinite classes, the regular languages, the context-free languages — is *not* identifiable from text. So the most rigorous model I have says the interesting classes are unlearnable, and it says nothing about how long anything takes. That can't be the right notion of "learnable." The exactness is the trap. A human who's learned to recognize tables hasn't computed the exact table-predicate over all conceivable inputs; they get it right on the tables they actually meet.

What else. There's a big statistical pattern-recognition literature — Duda and Hart and that line. It does have two ingredients I'm going to need: there's a probability distribution over inputs, and there's an error rate. But it doesn't ask my question. It doesn't ask which *classes of concepts* admit a recognizer you can deduce in polynomially many steps; there's no characterization theorem, no notion of feasibly deducing a symbolic program. And there's the AI concept-learning tradition — learning by example, by analogy, by being told, Michalski and Carbonell and Mitchell, the Barr–Feigenbaum handbook — which is rich about the *diversity* of human learning but gives me no guarantee that the rule it induces is even approximately right, and no boundary on what such methods can ever acquire.

Now the pieces I want to keep and the piece I want to throw away start to separate. From the statistical line I want to keep: a distribution over inputs, and error measured against it. From Gold I want to throw away: exactness, and the absence of a clock. And the throwing-away of exactness is the crux, so let me push on *why* exactness has to go, not just that it's inconvenient.

Suppose I insist on exact deduction of the concept from a polynomial number of examples. Take the simplest nontrivial target imaginable: a single monomial, an AND of some literals. I draw polynomially many positive examples — vectors on which the monomial is true. Can I pin the monomial down for certain? No. The distribution is nature's, and it's arbitrary. Nature could put essentially all the mass on a tiny corner of the space and feed me examples that never reveal whether some particular variable is in the monomial or not. An arbitrary set of polynomially many positive examples simply cannot be relied on to determine even a single monomial in any reliable way. So exact, certain deduction is *impossible* from a feasible sample, even for the most trivial class. The naive baseline — just memorize the positive examples you've seen and answer by their disjunction — generalizes to nothing for the same reason: all the unseen mass is unconstrained.

That impossibility is actually the clue. If certainty is unattainable, stop demanding it. Two separate relaxations are forced, and they're forced for two different reasons.

The first: I cannot demand the hypothesis be *exactly* right, so let it be *approximately* right — let it disagree with the target on some small fraction of inputs. But fraction measured how? Here's where the distribution does real work, and it has to be the *same* distribution that generates the examples. The error of a hypothesis g against the target f is the probability, under nature's D, that g and f disagree on a freshly drawn input. That's the only honest yardstick, because D is exactly the relative frequency with which inputs actually occur. If I only ever see inputs where D has mass, then getting those right is all that "learned" can possibly mean — behavior on inputs that never occur is irrelevant. So the accuracy parameter, call it ε, says: the probability-weight of the region where g and f disagree is at most ε.

And notice this same move *dissolves* Gold's impossibility. The reason superfinite classes were unlearnable was the demand to be right *everywhere*, including on an adversarially chosen tail. Measuring error under D means I only have to be right where the mass is. The places I can't pin down are exactly the places that rarely occur, so they contribute little error. The relaxation from "exact" to "ε-approximate under D" is not a cop-out; it's what makes the achievable and the meaningful coincide.

The second relaxation: even ε-approximation I cannot *guarantee*, because my sample is random and could be unlucky — nature could hand me a freak run of examples that all look the same and tell me nothing. So I cannot promise success with certainty; I promise it with high probability. Let δ be the failure probability: with probability at least 1 − δ over the draw of the sample, the hypothesis I output is ε-good. Two parameters, ε for accuracy and δ for confidence, and they're logically independent — one bounds how wrong g is allowed to be, the other how often I'm allowed to fail to achieve that. (I could fold them into a single knob to make statements cleaner, but really they're two.)

So the notion crystallizing is: *probably* (1 − δ) *approximately correct* (error ≤ ε). And now the third leg, feasibility. I want the whole thing — the number of examples I draw *and* the time I spend deducing — to be polynomial. Polynomial in what? In 1/ε and 1/δ, because finer accuracy and higher confidence should cost more but only gracefully; in the size of the target concept; and in the number of variables. Actually I'll want to be careful and say the number of variables that ever *appear* in examples, not the whole universe of available variables — a learner might have an enormous vocabulary of predefined and previously-learned predicates available, and the cost of learning a new concept should scale with the variables that actually show up in its examples, not with the size of that whole vocabulary.

There's one more thing the model has to nail down before it's a model: where do the examples come from, and what can the learner ask? This is a design choice about the teacher's power, and I have to calibrate it from both sides. Give the teacher too much power and the problem is trivial and meaningless: if the teacher could hand me a *premeditated sequence* of example vectors, even with repetitions, it could just *encode the target program in binary* across that sequence — two distinct vectors suffice as a binary alphabet — and then "learning" is just decoding a transmitted program, which is cheating; that's programming, not learning. Give the teacher too little and it's hopeless: if I get no typical examples at all, and the concept is true on just one obscure total vector, only exponential search would ever find it.

So I want a source of *typical* positive examples — typical meaning drawn by nature's distribution, not chosen by a teacher to communicate — and that's exactly the distribution D I already need for measuring error. Call it EXAMPLES: no input, returns a vector v with F(v)=1, and the probability that any particular v comes out is D(v). That single routine carries the whole probabilistic structure. And separately I'll allow, when needed, a membership test: present a vector and ask whether it positively exemplifies the concept. Call it ORACLE. In a real system the oracle could be a human expert, a database of past cases, some deduction system. EXAMPLES gives me what's typical; ORACLE lets the deduction procedure probe a specific vector it has decided is critical.

I should pin down what a "vector" is, because I want concepts that don't mention irrelevant variables. With t propositional variables, a vector assigns each variable a value in {0, 1, *}, where * means undetermined; it's *total* if every variable is in {0,1}. A Boolean function is a map from the 2^t total vectors to {0,1}. I extend a function F to a *concept* on all vectors like this: for a partial vector v, F(v)=1 iff F(w)=1 for *every* total completion w of v. That extension is exactly what lets me not mention the variables a concept doesn't depend on — a partial vector that already forces the function true is a legitimate positive example without specifying the rest. (This is also why ORACLE's natural question is "do all completions make F true?" — a necessity reading; later I might want a possibility reading, "does some completion make it true?", which suggests different oracle semantics, but the necessity oracle is the basic one.)

Good. Now I have a model. A class X of programs is *learnable* with respect to this protocol if there's a deduction procedure A using EXAMPLES and ORACLE such that: A runs in time polynomial in the accuracy/confidence parameters, the target's size, and t; and for *every* program f in X and *every* distribution D over the vectors where f is 1, A outputs, with probability at least 1 − δ, a hypothesis g in X with error at most ε under D. For the classes I'm about to chase I can even ask for *one-sided* error — g never says yes when it should say no (g(v)=1 ⇒ f(v)=1), and only errs by saying no on a set of positive vectors of D-mass at most ε. One-sided isn't essential to the framework — in general I'd allow two-sided error and put a distribution on the negative side too — but the classes I have in mind don't need it, and it makes the bookkeeping clean.

The whole thing hangs on one quantitative question, and if I can answer it once, abstractly, I can reuse it everywhere: if I have a process that, every so often, makes "progress," and I know the process can only make progress S times before it's done, *how many random trials* do I need so that I'm almost certain to have used up all S chances for progress — i.e. so that the residual "still could make progress" probability has dropped below my threshold? Let me make that precise and prove a bound, because it's the engine.

Define L(h, S): for a real h > 1 and a positive integer S, let L(h,S) be the smallest number of independent Bernoulli trials, each with success probability at least h⁻¹, such that the probability of seeing *fewer than S* successes is below h⁻¹. (I'm using h⁻¹ for both the per-trial success floor and the failure threshold — same h playing two roles, which keeps the statement compact.) Intuition for why this is the right object: picture an urn with marbles of S types; I want a sample that contains a representative of all but a 1%-mass of the types. If at any point types making up at least 1% of the mass remain unseen, then each draw is a Bernoulli trial with success probability at least 1% of grabbing a not-yet-seen type, and the success probability stays at least that floor *regardless of the history* of previous draws — which is exactly the condition I need, because in my learning algorithms what counts as "progress" depends on the current hypothesis, but the *floor* on its probability doesn't.

Claim: L(h,S) ≤ 2h(S + ln h), so it's essentially linear in both h and S. Let me actually prove it, because the constant 2 matters and I want to know it's right.

The tool is the Chernoff lower tail. In m independent trials each with success probability at least p, the probability of at most k successes, when k < mp, is at most

    e^{−mp+k} · (mp/k)^k.

(This is the standard multiplicative lower-tail form; I'll take it as the lever.) Now substitute the values I'm guessing: m = 2h(S + ln h), p = h⁻¹, k = S. First check k < mp: mp = 2(S + ln h) > S = k since S ≥ 1 and ln h > 0 for h > 1, good. Compute the pieces. mp = 2S + 2ln h, so −mp + k = −2S − 2ln h + S = −S − 2ln h. And mp/k = (2S + 2ln h)/S = 2(1 + (ln h)/S). So the bound is

    e^{−S − 2ln h} · [2(1 + (ln h)/S)]^S.

Split the bracket: [2(1+(ln h)/S)]^S = 2^S · (1 + (ln h)/S)^S. The second factor is the one to tame. Write it as ((1 + (ln h)/S)^{S/ln h})^{ln h}, and use the elementary inequality (1 + 1/x)^x < e with x = S/ln h: the inner base is below e, so the whole thing is below e^{ln h} = h. So

    < e^{−S − 2ln h} · 2^S · h = (2/e)^S · e^{−2ln h} · h = (2/e)^S · h^{−2} · h = (2/e)^S · h^{−1}.

Since 2/e < 1 and S ≥ 1, (2/e)^S ≤ 2/e < 1, so the whole thing is < h⁻¹. That's exactly the defining property of L: with m = 2h(S + ln h) trials the chance of fewer than S successes is below h⁻¹. Therefore L(h,S) ≤ 2h(S + ln h). The constants checked: the −2S+S that leaves −S, the h⁻² from e^{−2ln h} eaten by the one h to give h⁻¹, the (2/e)^S < 1. Linear in h, linear in S. That's the engine.

Now the first real target, and I'll take the simplest thing that could possibly be a "concept": a conjunction. A single AND of literals — a literal being a variable or its negation. I want to learn an unknown conjunction f from positive examples. Here's the move that makes it work, and it comes straight out of wanting the one-sided invariant. I want my hypothesis g to never wrongly accept — g(v)=1 ⇒ f(v)=1 — so I want g to be at least as *restrictive* as f at all times, an over-constrained guess that I gradually loosen. The most restrictive conjunction there is, is the AND of *all* 2t literals (every variable and its negation). That hypothesis accepts essentially nothing — it's the safest possible over-constraint. So start there: g = AND of all literals.

Now feed it positive examples. When EXAMPLES hands me a v with f(v)=1, every literal in the *true* f must be made true by v. If v sets x_i=1, then ¬x_i cannot be in f; if v sets x_i=0, then x_i cannot be in f; and if v leaves x_i undetermined, neither literal can be in f, because some completion would falsify either choice. So any literal currently in g that v does not make true cannot possibly be in f — keep it and g would reject this genuine positive v, contradicting f(v)=1. Delete exactly those literals. That's the whole algorithm — start with all literals, and on each positive example delete the literals it does not make true.

Why does this give one-sided error and converge? Every literal of the true f survives forever — every positive example must make a true literal true — so g always contains all of f's literals, meaning g is always at least as restrictive as f: g(v)=1 ⇒ f(v)=1. One-sided, automatically, by the invariant. The only way g errs is by *carrying an extra literal* that isn't in f — a literal that makes g reject some v that f accepts. Call such a leftover a *bad literal*. A bad literal is one sitting in g but not in f, and it survives only because no positive example I've drawn happened to leave it untrue.

So the error of g is exactly the D-mass of positive vectors where some surviving bad literal is not made true, and I want that below ε. Let me bound how many examples force it. There are 2t = 2n literals in all (each variable positive or negated); I'll track that count honestly. Pin attention on one particular literal and call it "individually bad" if the probability — under D, over positive examples — that this literal is not made true (and would therefore be deleted) is at least ε/2n. If a literal is individually bad in that sense, then each example deletes it with probability ≥ ε/2n; the chance it survives all m examples is at most (1 − ε/2n)^m ≤ e^{−εm/2n}. If *no* surviving literal is individually bad, then each of the ≤ 2n surviving literals contributes error < ε/2n, so by the union bound the total error is < 2n·(ε/2n) = ε — exactly what I want. So the only failure mode is some individually-bad literal surviving all m examples. Union bound over the ≤ 2n candidate literals: probability of failure ≤ 2n · e^{−εm/2n}. Set that ≤ δ:

    2n · e^{−εm/2n} ≤ δ  ⟺  εm/2n ≥ ln(2n/δ)  ⟺  m ≥ (2n/ε) ln(2n/δ) = O((n/ε) ln(n/δ)).

That's polynomial in n, 1/ε, 1/δ. Processing each example is linear in n, so total time O(mn), polynomial. The conjunctions are learnable, from positive examples alone, with no oracle. Good — the simplest concept class is in.

Let me sanity-check that same conclusion from a completely different angle, because I distrust a bound I've only derived one way. The class of conjunctions over n variables is *finite*: each variable is either present-positive, present-negated, or absent — three choices — so there are at most 3^n distinct conjunctions; monotone (no negations) gives 2^n. Whenever the hypothesis class H is finite and my algorithm just returns *some* hypothesis consistent with the sample, I can bound the sample size generically. Fix a "bad" hypothesis h with error > ε. The chance one random example is consistent with it is the chance the example lands where h agrees with f, which is < 1 − ε. The chance it's consistent with all m independent examples is < (1 − ε)^m ≤ e^{−εm}. There are at most |H| bad hypotheses, so by the union bound the chance that *any* bad hypothesis survives as consistent is < |H| · e^{−εm}. Force that ≤ δ:

    |H| e^{−εm} ≤ δ  ⟺  εm ≥ ln(|H|/δ)  ⟺  m ≥ (1/ε) ln(|H|/δ) = (1/ε)(ln|H| + ln(1/δ)).

This is the master finite-class bound, and it's lovely because it's pure counting: any time my hypotheses are drawn from a finite pool and I output a consistent one, ln|H| examples per unit ε suffice. Drop in |H| = 3^n for conjunctions: m ≥ (1/ε)(n ln 3 + ln(1/δ)); monotone, |H| = 2^n: m ≥ (1/ε)(n ln 2 + ln(1/δ)). The delete-unsatisfied-literals algorithm always outputs a hypothesis consistent with the positive sample (it only ever removes a literal when forced, and a surviving conjunction accepts every positive example seen), so it qualifies, and the two derivations agree up to constants — the direct bad-literal count and the ln(3^n) count are the same n/ε scaling. That cross-check makes me trust it.

Now generalize the conjunction trick. A conjunction is a degenerate CNF — a product of clauses where each clause is a single literal. The natural class above it: k-CNF, a product of clauses where each clause is an OR of at most k literals, k a fixed integer. Can I lift the exact same "start maximally restrictive, delete what positive examples forbid" idea? The atoms now are *clauses* of up to k literals instead of single literals. Tautological clauses containing both x_i and ¬x_i impose no restriction, so I can leave them out. How many remaining clauses are there over t variables? A clause of exactly k literals: at most (2t)^k ways. Up to k literals: (2t) + (2t)^2 + ⋯ + (2t)^k < (2t)^{k+1}. So there are fewer than (2t)^{k+1} possible clauses — a polynomial number for fixed k. That's the key: the number of building blocks is polynomial, so I can afford to start with *all* of them.

Initialize g = product of all clauses of up to k literals. Then call EXAMPLES; for each positive v, delete from g every clause that v does *not* satisfy — every clause that has no literal made true by v. Same logic as before: if f is true at v, every clause of the true f is satisfied by v, so any clause v fails to satisfy can't be in f, drop it. Repeat L times.

Why is the result a correct ε-approximation? Let B be the product of *all* up-to-k clauses c with the property "for every v, if v⊨f then v⊨c." No clause of B is ever deleted, because deletion only happens at a positive v that fails the clause, and clauses of B are satisfied by every positive v. So the accepted set of g always sits inside the accepted set of B. And B computes the *same Boolean function as f*: trivially f⊨B (every clause of B is implied by f by construction); and B⊨f because f, being itself a k-CNF, has each of its clauses among the up-to-k clauses satisfied by all positive v, hence each clause of f is in B, so B is at least as restrictive as f. (In fact B is the maximal k-CNF equivalent to f.) So g is squeezed between f and B≡f from the restrictive side: g only over-restricts, never wrongly accepts — one-sided again.

Now the convergence. Let X be the current error: the D-mass of vectors v with v⊨f but v⊭g — positive vectors g wrongly rejects. X is monotone decreasing as clauses leave g (removing a clause can only make g accept more). Every time EXAMPLES produces a v with v⊭g, at least one clause gets deleted — that's a unit of progress, and it can happen at most (2t)^{k+1} times since that's all the clauses there are. The probability that a fresh example triggers a deletion is exactly the current X. So this is precisely my urn process: at every history where X is still large, the next independent example has a success probability bounded below, and at most (2t)^{k+1} successes are possible. To use the engine with separate ε and δ, set h_0 = max{1/ε, 1/δ}; then h_0⁻¹ ≤ ε and h_0⁻¹ ≤ δ. Run to completion and there are two outcomes. Either at some point X drops below h_0⁻¹ — then g is already an ε-approximation, done. Or X never drops below h_0⁻¹ across all L examples — but then I have L trials whose conditional success probability is always at least h_0⁻¹ and fewer than (2t)^{k+1} successes, which by the L bound has probability below h_0⁻¹ ≤ δ when L = L(h_0, (2t)^{k+1}). Plugging the engine: L = L(h_0,(2t)^{k+1}) ≤ 2h_0((2t)^{k+1} + ln h_0), polynomial in t for fixed k and polynomial in 1/ε and 1/δ. k-CNF is learnable from EXAMPLES alone, no oracle. The conjunction case is the k = 1 structural case, and the direct bad-literal proof simply uses the sharper exact count 2t instead of the rough (2t)^2 clause bound.

Now the harder, dual direction: DNF, a sum of monomials. This is the form humans find easiest to read — "it's an elephant if (grey and large and trunked) or (…)" — so any practical learner has to handle it. But the asymmetry bites immediately. With CNF I could start with *all* clauses and delete; the number of clauses was polynomial for bounded k. With DNF, starting with all monomials and deleting won't work the same way, and worse, there's a computational landmine I have to respect: deciding whether a partial vector forces a general DNF formula to be true is the *tautology* problem — taking the all-stars vector, "does the undetermined vector imply the function?" is exactly asking if the formula is a tautology, which is co-NP-hard by Cook. So for *unrestricted* DNF I can't even reliably test v⊨g for nontotal v; the membership test I rely on is itself intractable. That tells me unrestricted DNF, as a *concept* (with partial vectors), is out of reach by these means, and the most I can hope for unrestricted is to learn the *function* under distributions that only put mass on total vectors — a real but weaker statement.

So restrict to the *monotone* case — no variable negated — where this difficulty evaporates, because for a monotone DNF I can always evaluate the associated concept on a vector by substitution. Build g the other way: start with g ≡ 0 (the empty sum, accepting nothing — the maximally restrictive DNF, mirroring how I started CNF with the maximally restrictive product). Now I want to *add* genuine monomials of f, one at a time, and here EXAMPLES alone isn't enough — a positive vector v tells me f is true on it, but it's a whole vector; I need to distill it down to a *prime implicant* of f, a minimal monomial that already forces f true. That distillation is what the ORACLE buys me. Take a v with v⊭g (g currently misses it). First ignore every coordinate not set to 1, since a monotone monomial can only use positive variables. Then walk the 1-coordinates: for each p_i set in v, tentatively undetermine it (set it to *) and ask the necessity ORACLE whether the loosened vector still implies f; if it does, that variable was inessential, drop it; if not, keep it. What's left is a minimal product of positive literals that still forces f — a prime implicant m of f. Add m to g.

Each m I add is genuinely a prime implicant of f, so every monomial of g implies f — g⊆f, one-sided yet again, and g is monotone so I can always evaluate it. Each added m is new (if it weren't, v would already have satisfied g, contradicting v⊭g), and the number of distinct prime implicants of a monotone f is its degree d (the monotone prime DNF is unique — the sum of all prime implicants — so d is well-defined). So I add a monomial at most d times, and the inner distillation calls ORACLE at most t times each, ≤ dt oracle calls total.

Convergence is the same engine once more. Let X be the D-mass of w with w⊨f but w⊭g; it starts at 1 and decreases monotonically as monomials enter g. Each EXAMPLES call that yields a v⊭g causes a monomial to be added — a success — with probability exactly the current X, and there can be at most d successes. With h_0 = max{1/ε, 1/δ}, either X drops below h_0⁻¹, giving an ε-approximation, or it never does across L examples, which means L trials with conditional success probability at least h_0⁻¹ yield fewer than d successes — probability below h_0⁻¹ ≤ δ when L = L(h_0, d). Monotone DNF is learnable using L = L(h_0,d) ≤ 2h_0(d + ln h_0) examples and at most dt oracle calls — polynomial in d, t, 1/ε, 1/δ. The conjunction-from-positive-examples and the monotone-DNF-with-oracle are the two extremes of the same template: one starts maximally restrictive and *deletes* using EXAMPLES, the other starts maximally restrictive and *adds* prime implicants using EXAMPLES to flag the gap and ORACLE to distill, and both ride the L(h,S) progress-counting bound.

Let me step back and name what just happened, because the algorithms are almost beside the point — the real object is the *definition*. A concept class is learnable when there's a procedure that, from polynomially many examples drawn from an arbitrary unknown distribution, outputs in polynomial time a hypothesis that is, with probability at least 1 − δ, of error at most ε under that same distribution, *for every distribution*. Probably approximately correct. That single definition is what converts "can machines learn?" from a mood into a complexity question I can win or lose. And I can now exhibit the answers on both sides: conjunctions, k-CNF, monotone DNF — *learnable*, provably, with the bounds above. And there's pressure from the other side too: if good cryptographic functions exist — a cipher E_k secure against chosen-plaintext attack, where seeing E_k on polynomially many chosen inputs leaves you unable to predict it on a new one — then by definition that function is *not* learnable, so the conjectured existence of strong, easy-to-compute ciphers implies some easy-to-compute functions are unlearnable. The class of feasibly learnable concepts is genuinely circumscribed, with simple Boolean families inside the boundary and cryptographically hard ones outside it. That two-sidedness — provable membership *and* provable (conditional) exclusion — is exactly the parallel to computability I was after.

Here is the conjunction learner, the cleanest instance, in the form it actually takes:

```python
# Learn an unknown conjunction (single AND of literals) over n variables,
# from positive examples alone, distribution-free.
# Invariant kept throughout: g is at least as restrictive as the target f,
# so g never wrongly accepts (one-sided error). Only literals NOT in f ("bad
# literals") can survive and cause error, and each positive example that
# does not make such a literal true deletes it.

import math

def num_examples(n, eps, delta):
    # Union bound over the <= 2n candidate literals (bad-literal argument),
    # each "bad" literal not made true w.p. >= eps/2n per example:
    #   2n * (1 - eps/2n)^m <= delta,  using (1 - x) <= e^{-x}  =>
    #   m >= (2n/eps) * ln(2n/delta).
    # Cross-checks with the finite-class master bound (1/eps)*ln(|H|/delta),
    # |H| = 3^n for conjunctions:  (1/eps)*(n*ln 3 + ln(1/delta)).
    return math.ceil((2 * n / eps) * math.log(2 * n / delta))

def learn_conjunction(EXAMPLES, n, eps, delta):
    m = num_examples(n, eps, delta)
    # Start maximally restrictive: g = AND of all 2n literals.
    # Represent g by which literals are still "in": for each variable i,
    # require_pos[i] = "literal x_i is in g", require_neg[i] = "literal ~x_i is in g".
    require_pos = [True] * n
    require_neg = [True] * n
    for _ in range(m):
        v = EXAMPLES()                 # a vector with f(v) = 1, drawn from D
        for i in range(n):
            # Delete every literal this positive example does not make true:
            # it cannot be in f, else v would not force f.
            if v[i] == 1:
                require_neg[i] = False  # ~x_i is not true here -> drop ~x_i
            elif v[i] == 0:
                require_pos[i] = False  # x_i is not true here -> drop x_i
            else:
                require_pos[i] = False  # neither literal is forced by '*'
                require_neg[i] = False
    def g(x):
        # g accepts x iff every surviving literal is satisfied by x.
        for i in range(n):
            if require_pos[i] and x[i] != 1:
                return 0
            if require_neg[i] and x[i] != 0:
                return 0
        return 1
    return g
```

And the k-CNF generalization, the same delete-what-positive-examples-forbid idea over the polynomially-many up-to-k-literal clauses, with the sample count coming from the combinatorial engine L(h, (2t)^{k+1}):

```python
# Learn an unknown k-CNF (product of clauses, each an OR of <= k literals)
# from positive examples alone, k fixed. Same template as the conjunction
# (which is just k = 1): start with the product of ALL <= k-literal clauses,
# delete every clause a positive example fails to satisfy.

import itertools, math

def all_clauses_up_to_k(n, k):
    # Literals: (i, 1) means x_i, (i, 0) means ~x_i. A clause is a frozenset
    # of literals and never contains both signs of the same variable; there
    # are fewer than (2n)^{k+1} clauses of size 1..k.
    lits = [(i, b) for i in range(n) for b in (0, 1)]
    clauses = []
    for size in range(1, k + 1):
        for combo in itertools.combinations(lits, size):
            if len({i for (i, _) in combo}) != size:
                continue
            clauses.append(frozenset(combo))
    return clauses

def L_bound(h, S):
    # L(h, S) <= 2h(S + ln h): the smallest # of Bernoulli trials with per-trial
    # success prob >= 1/h so that Pr[fewer than S successes] < 1/h. Proven via
    # the multiplicative Chernoff lower tail; linear in h and S.
    return math.ceil(2 * h * (S + math.log(h)))

def clause_satisfied(clause, v):
    # OR of literals: satisfied if some literal is made TRUE by v.
    for (i, b) in clause:
        if v[i] == b:
            return True
    return False

def learn_kCNF(EXAMPLES, n, k, eps, delta):
    h = max(1.0 / eps, 1.0 / delta)          # single parameter playing both roles
    g = all_clauses_up_to_k(n, k)            # maximally restrictive product
    L = L_bound(h, (2 * n) ** (k + 1))       # <= (2n)^{k+1} possible deletions
    for _ in range(L):
        v = EXAMPLES()                       # f(v) = 1
        # Delete clauses NOT satisfied by v: a true positive can't violate a
        # clause of the true f, so any clause it violates is not in f.
        g = [c for c in g if clause_satisfied(c, v)]
    def hypothesis(x):
        return 1 if all(clause_satisfied(c, x) for c in g) else 0
    return hypothesis
```

The causal chain, start to finish: computability got its theory from a precise model, so I wanted a precise model of learning; exact identification (Gold) was the wrong target because exact certain deduction from a feasible sample is provably impossible even for a single monomial; that impossibility forced two independent relaxations — approximate error measured under the *generating* distribution (which also dissolves Gold's worst-case impossibility, since I only have to be right where mass lives), and high-probability rather than certain success; bolting on the complexity-theoretic demand of polynomial examples and polynomial time turned "learnable" into a sharp distribution-free complexity question, *probably approximately correct*; a single combinatorial bound L(h,S) ≤ 2h(S + ln h), proven by the Chernoff lower tail, counts how many random examples force a progress-making process to finish; and feeding the maximally-restrictive-hypothesis-with-monotone-progress template through that bound proves conjunctions and k-CNF learnable from positive examples and monotone DNF learnable with a membership oracle, with cryptography marking the far boundary where learnability provably fails.
