Let me start from what actually bugs me. We have this clean notion of a proof — NP — where the prover writes one string down and I, the polynomial-time verifier, check it. That captures proofs you can put in a book. But it feels too narrow, because the way people actually convince each other is by *talking*: I ask a question at the spot where I'm suspicious, you answer, I ask again. Goldwasser, Micali, Rackoff, and Babai already turned that into a model — interactive proofs. I'm a probabilistic polynomial-time verifier with my own private coins; you, the prover, are infinitely powerful but I don't trust you. We exchange messages, and at the end I accept or reject. The rules are: if the statement is true, some prover should make me accept with high probability; if it's false, *no* prover should make me accept with more than tiny probability, where "probability" is over my coins. The class of statements I can be convinced of this way is IP.

The question that won't leave me alone is: how big is IP, really? Because the obvious examples are barely past NP. Graph non-isomorphism — that one I can do, and it's not even known to be in NP. I pick one of the two graphs at random, scramble it with a random permutation, and send you the scrambled graph; if the two graphs are genuinely different, you (being all-powerful) can tell which one I started from and you name it; if they're secretly isomorphic, a random scramble of either looks identical, so you're just guessing and I catch you half the time. Nice. But it's a one-off trick, and it leans entirely on you *not* seeing my coins. It doesn't tell me how to prove that a formula is *unsatisfiable*, which is the coNP-complete thing I'd really like to handle.

And here's the discouraging part. Everyone believes IP is not much bigger than NP, and specifically that coNP-complete languages are *not* in IP. It's not just a vibe — Fortnow and Sipser built an oracle `D` under which coNP^D is not contained in IP^D. So in a relativized world, interaction does not buy you coNP. And basically every containment we know how to prove relativizes — it goes through under any oracle. So this oracle is a wall: it says any proof that coNP ⊆ IP must somehow *not* relativize, must crack open the actual structure of the formula instead of treating it as a black box. I don't have a non-relativizing technique. Nobody does. That's the state of things.

So let me hold that thought — "must use internal structure, can't be black-box" — and look for a place where a coNP-ish statement has structure I can exploit.

The thing that has tons of structure is *counting*. Unsatisfiability is "the number of satisfying assignments is zero." Counting satisfying assignments is `#SAT`, and the general counting class is `#P`. The crown jewel of `#P` is the permanent of a 0/1 matrix, `per(A) = Σ_σ ∏_i a_{i,σ(i)}` — the determinant without the signs, the number of perfect matchings of a bipartite graph. Valiant proved it's `#P`-complete. And Toda proved the whole polynomial hierarchy reduces to `#P`. So if I could interactively verify a claimed permanent value, I'd reach not just coNP but all of PH. That's a huge prize. The catch is the same wall: how on earth do I check `per(A) = s` when computing the permanent is intractable and I can't recompute it?

Let me stare at the permanent and look for the structure. Two things jump out. First, it's *multilinear* — degree exactly one in each matrix entry. Second, it satisfies the cofactor recursion exactly like the determinant: `per(A) = Σ_{i=1}^{N} a_{1i} · per(A_{1,i})`, where `A_{1,i}` is `A` with the first row and `i`-th column struck out. So the permanent of an `N×N` matrix is a fixed linear combination of permanents of `(N−1)×(N−1)` matrices. That's a way to shrink the problem — but it also *fans out*: one permanent becomes `N` of them. If I just recurse I get an exponential tree. I need to collapse the fan-out, not follow it.

The fan-out only hurts because I am carrying several claims at once. If two claimed permanents can be bound to one random question, the recursion can stay narrow. Suppose you've told me the permanents of two `r×r` matrices `C` and `D`, claiming `per(C)=c` and `per(D)=d`, and I suspect at least one claim is a lie but I don't know which. I take the line through them in matrix space: `D(x) = (1−x)C + xD`. The entries of `D(x)` are linear functions of `x`, and the permanent is multilinear in the entries, so `f(x) := per(D(x))` is a univariate polynomial in `x` of degree at most `r`. It interpolates the two endpoints: `D(0)=C` so `f(0)=per(C)`, and `D(1)=D` so `f(1)=per(D)`.

Now I ask you for the polynomial `f` — say its `r+1` coefficients. Call what you send `g`. I check the endpoints: does `g(0)=c` and `g(1)=d`? If not, you've already contradicted your own earlier claims, I reject. So suppose `g(0)=c`, `g(1)=d`. If at least one of your claims was wrong — say `per(C) ≠ c` — then your `g` cannot be the true `f`, because the true `f` has `f(0)=per(C) ≠ c = g(0)`. So `g ≠ f` as polynomials. Now I pick a *random* point `a ∈ F_p`, send it to you, and collapse both checks into the single new claim `per(D(a)) = g(a)`. When does this let you escape? Only if your lie `g` happens to agree with the true `f` at the random `a` I picked. But `g` and `f` are distinct polynomials of degree at most `r`, so `g − f` is a nonzero polynomial of degree at most `r`, and it has at most `r` roots. The probability that my random `a` is one of those roots is at most `r/p`. Make `p` large and that's tiny.

That's the whole engine, and it's the thing that beats NP: I converted "check two evaluations of a low-degree polynomial" into "check one evaluation at a *random* point," and the few-roots fact over a field turned the prover's freedom to lie into a near-certain way to get caught. Randomness is doing real work here — if I'd checked at a *fixed* point instead of a random one, the prover could have rigged `g` to match `f` there and I'd be back to a deterministic check, i.e. back to NP. It has to be a field, and it has to be big, for the root count to give me the bound.

Once this two-claim compression exists, I can use it every time the cofactor recursion fans out. I start with the single claim `per(A)=s`. I expand `A` into the `N` minors `A_{1,i}`, ask you for their permanents, check the linear identity `s = Σ_i a_{1i} q_i`, and now I have a list of `N` claims of size `(N−1)`. Then I repeatedly shrink the list two at a time with the line-through-them check, each shrink costing one random point and exposing any lie with probability at least `1 − r/p`. I alternate expansion and shrinking until I reach a single `1×1` matrix, whose permanent I can read off and check directly. If you are honest throughout, every check passes and I accept with probability 1. If `per(A) ≠ s`, then a false claim can become all-true only when one shrink point lands on a root, which happens with probability at most `r/p ≤ N/p`. There are fewer than `N²` shrink steps, so the total chance you ever escape is at most `N³/p`. I also need the field value to be the integer permanent value, and for a 0/1 matrix the permanent is at most `N!`; so I choose a prime `p > max(N!, 3N³)`, with a succinct primality certificate. All arithmetic is mod `p`, and the soundness error is below `1/3`.

I have an interactive proof for a `#P`-complete problem. Since coNP and all of PH reduce to `#P`, I've just put coNP — UNSAT, the complement of 3-coloring, everything — inside IP. And notice *why* it doesn't relativize: I used that the permanent has a low-degree polynomial structure, that it lives off the Boolean cube as an algebraic object. A black-box oracle has no such structure, which is exactly why the Fortnow–Sipser oracle doesn't block me. The wall came down because I refused to treat the computation as a black box.

Now, do I even need the permanent? The permanent was a convenient vehicle because it's `#P`-complete and obeys that nice recursion. But the real content was: take something defined on `{0,1}`, *extend it to a low-degree polynomial over a field*, and verify a sum/count of it by stripping it down at random points. Let me redo it directly for `#SAT`, which makes the engine cleaner and severs the dependence on Valiant's theorem.

I want to verify "the 3-CNF `φ(x_1,…,x_n)` has exactly `K` satisfying assignments." First, **arithmetize** `φ`: replace truth values by field elements `{0,1}` and the connectives by the lowest-degree polynomials that agree with them on the cube. `x ∧ y ↦ X·Y`. `¬x ↦ 1−X`. `x ∨ y ↦ 1 − (1−X)(1−Y)`, and a 3-literal clause `x∨y∨z ↦ 1−(1−X)(1−Y)(1−Z)` — by De Morgan, "the clause is satisfied" is "not all literals false," and a negated literal `¬z` just contributes its own arithmetization `1−Z`, so a clause `x∨y∨¬z ↦ 1−(1−X)(1−Y)Z`. A clause becomes a degree-3 polynomial; the conjunction of `m` clauses is their product, a polynomial `Φ(X_1,…,X_n)` of degree at most `3m`. On Boolean inputs `Φ` equals `φ` (1 for satisfying assignments, 0 otherwise) — and I keep `Φ` as the *product form*, size `O(m)`, never multiplied out. The number of satisfying assignments is then exactly the sum over the cube:
`#φ = Σ_{b_1∈{0,1}} Σ_{b_2∈{0,1}} … Σ_{b_n∈{0,1}} Φ(b_1,…,b_n)`.
That sum has `2^n` terms — I can't compute it. But the prover can, and I'll verify it by peeling one variable per round. The prover picks a prime `p ∈ (2^n, 2^{2n}]` and proves to me it's prime; since `#φ` is between `0` and `2^n`, the identity over the integers holds iff it holds mod `p`, so I work in `F_p`.

This stripping protocol — let me call it sum-check, because that's what it does — handles any claim of the shape `K = Σ_{b∈{0,1}^n} g(b)` for a polynomial `g` of degree at most `d` in each variable that I can evaluate at a point. If only one variable remains, I do not need help: I evaluate `g(0)+g(1)` myself and compare it with the running claim. If more than one variable remains, define `h_1(X_1) = Σ_{b_2,…,b_n ∈{0,1}} g(X_1, b_2, …, b_n)`. For each fixed setting of the other variables, `g` is a degree-`d` univariate in `X_1`, so `h_1` is a univariate of degree at most `d`. If the claim is true, then summing `h_1` over the two values of `X_1` gives back `K`: `h_1(0) + h_1(1) = K`. So I ask you for `h_1` (its `d+1` coefficients), you send some `s_1`, and I check `s_1(0) + s_1(1) = K`; reject if not. Then I pick a random `r_1 ∈ F_p`, send it, and recurse on the smaller claim `s_1(r_1) = Σ_{b_2,…,b_n} g(r_1, b_2, …, b_n)`. The recursion keeps peeling until one live variable is left, where I evaluate the two Boolean endpoints directly.

Completeness: an honest prover sends the true `h_j` every round, every check passes, I accept with probability 1.

Soundness — and I want the exact bound, not a wave of the hand. Claim: if the original sum is *not* `K`, then `Pr[I reject] ≥ (1 − d/p)^n`. Induct on `n`. Base `n=1`: with one variable left I evaluate `g(0)+g(1)` directly and compare it to `K`; if they differ I reject with probability 1, and `(1−d/p)^1 ≤ 1`. Inductive step: suppose the claim is false. If you send the true `h_1`, then `h_1(0)+h_1(1)` is the true sum `≠ K`, so I reject immediately. So to survive round one you must send `s_1 ≠ h_1`. But `s_1 − h_1` is a nonzero polynomial of degree at most `d`, so it has at most `d` roots, so over my random `r_1`,
`Pr[s_1(r_1) = h_1(r_1)] ≤ d/p`, i.e. `Pr[s_1(r_1) ≠ h_1(r_1)] ≥ 1 − d/p`.
When `s_1(r_1) ≠ h_1(r_1)`, you're now stuck proving the claim `s_1(r_1) = Σ_{b_2,…} g(r_1,b_2,…)` — but the right-hand side is `h_1(r_1)`, which differs from `s_1(r_1)`, so this is a false `(n−1)`-variable claim, and by the induction hypothesis you fail it with probability at least `(1−d/p)^{n−1}`. Multiply: `Pr[I reject] ≥ (1 − d/p)·(1 − d/p)^{n−1} = (1 − d/p)^n`. With `d ≤ 3m` and `p ≫ mn`, this is close to 1. So `#SAT ∈ IP`, no permanent required. The same algebraic engine, stated in its natural generality.

Now the real target. coNP and PH are inside `P^{#P}`, but the universe of polynomial-space computation is bigger — PSPACE. I keep asking myself: I beat the relativization wall once; how far up does this engine actually reach? Why couldn't IP hit the ceiling, PSPACE itself?

The PSPACE-complete problem is TQBF: decide whether `Ψ = Q_1 x_1 Q_2 x_2 … Q_n x_n φ(x_1,…,x_n)` is true, with `φ` a 3-CNF and the `Q_i` alternating `∀`/`∃`. Any PSPACE language reduces to this in polynomial time, so an interactive proof for TQBF gives me all of PSPACE. I already arithmetize `φ` into `Φ`. The new thing is the quantifiers. I want the whole formula to become an arithmetic expression whose top value is exactly `1` when `Ψ` is true and `0` when it is false.

`∀x_n ψ` means `ψ` holds for both settings of `x_n`, so its truth value is the *product*:
`∏_{x_n ∈ {0,1}} ψ(…,x_n) := ψ(…,0)·ψ(…,1)` — equals 1 iff both factors are 1.
`∃x_n ψ` means `ψ` holds for *some* setting, which is the Boolean OR of the two, so:
`⊔⊔_{x_n ∈ {0,1}} ψ(…,x_n) := 1 − (1 − ψ(…,0))(1 − ψ(…,1))` — equals 1 iff some factor is 1.
(I deliberately do *not* use a sum `Σ` for `∃`: a sum of 0/1 values over the cube can exceed 1 — it would *count* witnesses and blow past the 0/1 structure I need at the top, where I just want a truth value. `1 − ∏(1−·)` is exactly OR, and it stays 0/1 on the cube.) Stacking the operators from inside out, `Ψ` is true iff
`Q'_1 Q'_2 … Q'_n Φ(x_1,…,x_n) = 1`,
where each `Q'_i` is the `∏` or `⊔⊔` operator over `x_i`, and the whole thing evaluates to the Boolean truth value, which is `0` or `1`. So I want to verify the value of this fully-stripped expression. Strip the operators one at a time, exactly like sum-check — each round you send me the appropriate low-degree univariate, I check consistency, bind the variable to a random `r_i`, recurse.

And then I hit it. Multiplication, unlike summation, *raises the degree*. When `∏_{x_i}` passes over the expression, it multiplies the `x_i=0` branch by the `x_i=1` branch — and each of those branches already contains the variable `x_1` (and every earlier variable) to some degree, so the degree of `x_1` *doubles* every time a product operator goes by. With `n` nested operators, the degree of `x_1` in an intermediate polynomial can reach `2^n · 3m`. A polynomial of degree `2^n` cannot even be written down in polynomial time — it has exponentially many coefficients. The prover can't send it; I can't read it. This isn't a soundness problem, it's a *completeness* problem: even the honest prover is dead. The naive lift of sum-check to quantifiers collapses under its own degree.

So I have to keep the degrees small as I go. Stare at where the blow-up comes from: it's that the same variable `x_1` keeps getting *squared and re-squared* by successive products, climbing to degree `2, 4, 8, …`. But here's the thing — on the Boolean cube, `x^k = x` for every `k ≥ 1`. The high powers are pure waste: `x_1^4` and `x_1` are the *same function* on `{0,1}`, and the only place I ever care about the truth value is on the cube. So after each operator I can legally collapse every variable's degree back down to 1 without changing a single value on `{0,1}`.

Make that an operator. For a variable `x_i`, define the **linearization** (degree-reduction) operator
`L_i(P)(…,x_i,…) = x_i · P(…, x_i=1, …) + (1 − x_i) · P(…, x_i=0, …)`.
Check it: this is linear in `x_i` by construction; and at `x_i = 1` it gives `P(…,1,…)`, at `x_i = 0` it gives `P(…,0,…)` — so `L_i(P)` is the unique polynomial that is degree ≤ 1 in `x_i` and *agrees with `P` at both Boolean values of `x_i`*. (It's exactly the Lagrange interpolation of `P` in the single variable `x_i` through the two points `0,1`.) Applying `L_i` for every live variable after each quantifier operator pulls all per-variable degrees back to ≤ 1, and since it preserves all cube values, the top-level truth value of the whole expression is untouched. The exponential degree was an illusion created by carrying powers that don't matter on the cube; linearization throws them away. (Shamir's first version of this used auxiliary dummy variables to hold the degree down — a rewrite of `Ψ` into a logically equivalent formula whose natural arithmetization never exceeds degree 2; the linearization operator, which Shen found, is the same effect done in-line, and it's cleaner, so I'll carry it.)

The object I actually verify is a long operator string with reductions interleaved:
`1 = Q'_1 L_1 Q'_2 L_1 L_2 Q'_3 L_1 L_2 L_3 … Q'_n L_1 L_2 … L_n Φ(x_1,…,x_n)`,
where each `Q'_i` is `∏_{x_i}` or `⊔⊔_{x_i}`, matching the original quantifier, and after the `i`-th quantifier in the stripping order I re-linearize `x_1, …, x_i`. The total number of operators is `O(n²)` — `n` quantifier operators plus `1+2+…+n` reductions — so still polynomially many rounds. The polynomial sent in a quantifier round is degree at most 1, because the next operators include the relevant linearization. A reduction round sees degree at most 2, except for the innermost `n` reductions where `Φ` first enters and the per-variable degree can be as high as `3m`. The exponential degree is gone; the largest message has degree `3m`, not `2^n·3m`.

Now I strip the operators one at a time. Abstract the string as `O_1 O_2 … O_ℓ Φ`. I maintain a running claimed value `v_k` for "what the suffix `O_{k+1} … O_ℓ Φ` evaluates to, at the variables already bound to random points." I start with `v_0 = 1`. At round `k+1` the next operator `O_{k+1}` is one of three kinds.

If `O_{k+1} = ∏_{x_i}`, the `∀` case, you send me the degree-1 univariate `P̂(x_i)` you claim is the value of the suffix as a polynomial in `x_i`. I check that the product over the Boolean values reproduces my current claim: `P̂(0)·P̂(1) = v_k`. If not, I reject. Otherwise I pick a fresh random `r_i ∈ F_p`, set `v_{k+1} = P̂(r_i)`, and go on with `x_i` bound to `r_i`.

If `O_{k+1} = ⊔⊔_{x_i}`, the `∃` case, the message is again degree 1, but the consistency check is the OR polynomial: `1 − (1 − P̂(0))(1 − P̂(1)) = v_k`. If that holds, I bind `x_i` to a fresh random `r_i` and set `v_{k+1} = P̂(r_i)`.

If `O_{k+1} = L_i`, I am undoing a degree reduction at the current binding of `x_i`. You send the bounded-degree polynomial `P̂(x_i)` for the expression before linearization — degree at most `2`, except degree at most `3m` for the innermost reductions that touch `Φ`. I check the defining interpolation identity at the current binding: `r_i·P̂(1) + (1−r_i)·P̂(0) = v_k`. Then I pick a new random `r_i`, reset the binding, and set `v_{k+1} = P̂(r_i)`.

At the very end the expression is fully stripped and all variables are bound to random points; `Φ` is a product of `m` clause-polynomials I can evaluate myself in polynomial time, so I compute `Φ(r_1,…,r_n)` and check it against `v_ℓ`. Accept iff every check held.

Completeness: if `Ψ` is true, the honest prover sends, each round, the genuine univariate for the suffix as a function of the operator's variable, with the current random bindings substituted; every consistency check holds by definition of the operators, and the final `Φ` evaluation matches, so I accept with probability 1.

Soundness, exactly. Each round works just like sum-check's inductive step. If, going into round `k+1`, the claim `v_k` is false, then the polynomial that would make the suffix true cannot also pass the consistency check for the false `v_k`. To survive the check you must send a `P̂` different from the true univariate `P(x_i)`. Then `P̂ − P` is a nonzero polynomial, so my fresh random point lands on a root, letting your lie propagate as a true-looking sub-claim, with probability at most the degree divided by `p`. The degree is at most `1` in each of the `n` quantifier rounds, at most `3m` in each of the innermost `n` reduction rounds where `Φ` enters, and at most `2` in each of the other reduction rounds. There are `Σ_{i=1}^{n-1} i` of those other reductions, so the union bound gives
`Pr[falsely accept] ≤ n/p + 3mn/p + (2/p)·Σ_{i=1}^{n-1} i = (3mn + n²)/p`.
Pick `p` a prime of polynomially many bits with `p ≫ 3mn+n²`, and the soundness error is negligible. (`Ψ`'s arithmetic value can be huge inside the expression before the top truth value is obtained, but a polynomial-size prime still works because the protocol only needs the corresponding field identity and the root-count bound.) So TQBF ∈ IP, and therefore **PSPACE ⊆ IP**.

I should make sure I'm not about to claim something *false* — that IP is somehow bigger than PSPACE. So the other direction: **IP ⊆ PSPACE**. Given any verifier `V`, the best a prover can do is determined by a game tree. Picture the tree of conversations: nodes at depth `i` are the message-histories after `i` messages; the depth is polynomial (poly rounds), and each node has at most `2^{poly}` children (poly-length messages). A leaf is `1` if `V` accepts that transcript, `0` if it rejects. At a node where the *prover* sends next, the value is the *maximum* over children (the prover picks its best message); at a node where the *verifier* tosses coins and sends next, the value is the *average* over children (weighted by the coin probabilities). The root value is exactly the maximum probability with which any prover makes `V` accept on this input. I can compute that value in polynomial space — depth-first recursion over a tree of polynomial depth, reusing the same work space down each branch, never storing the whole (exponential) tree. If the root value is `> 2/3`, the input is in the language; if `< 1/3`, it isn't. So `IP ⊆ PSPACE`.

Both directions together: **IP = PSPACE**. The class of statements a polynomial-time randomized verifier can be talked into believing, through interaction with an untrusted all-powerful prover, is *exactly* the statements decidable in polynomial space — vastly beyond NP, well past coNP and the whole polynomial hierarchy. And the engine the whole thing runs on is arithmetization: lift the Boolean object to a low-degree polynomial over a large prime field, where a single random evaluation pins it down because two distinct low-degree polynomials can't agree at many points — and the verifier strips the formula one quantifier or one sum at a time, keeping the degree small by linearizing on the cube, so that at each step a cheating prover who has lied is forced to lie about a low-degree polynomial and gets caught with probability `1 − degree/p`. The pain was that a coNP/PSPACE statement seems to have no short certificate and a relativization wall said interaction couldn't help; the resolution was to refuse the black box, arithmetize the actual structure, and let randomness over a field do the policing.

I can write the verifier as the checks I have just forced.

```python
# IP = PSPACE.  PSPACE-complete target: TQBF  Psi = Q_1 x_1 ... Q_n x_n  phi(x),
# phi a 3-CNF (m clauses).  All arithmetic in F_p, p a poly-bit prime (verifier
# checks primality via a succinct certificate / randomized test).

# --- Arithmetization: Boolean object -> low-degree polynomial over F_p ----------
# x AND y -> X*Y ; NOT x -> 1 - X ; x OR y -> 1 - (1-X)(1-Y).  A clause is satisfied
# iff not all its literals are false: (l1 v l2 v l3) -> 1 - (1-A1)(1-A2)(1-A3), where
# Aj is the literal's arithmetization (X if positive, 1-X if negated).  Phi = AND of
# the m clauses, kept as a PRODUCT => degree <= 3m, Phi == phi on {0,1}^n, size O(m).
def lit_val(binding, lit, F):          # lit = (var_index, negated?)
    X = binding[lit.var]
    return F.sub(1, X) if lit.negated else X
def Phi(binding, clauses, F):          # evaluate the arithmetized 3-CNF at a point
    val = 1
    for clause in clauses:
        false_all = 1
        for lit in clause:             # (1-A1)(1-A2)(1-A3) = all literals false
            false_all = F.mul(false_all, F.sub(1, lit_val(binding, lit, F)))
        val = F.mul(val, F.sub(1, false_all))   # clause = 1 - (all false); AND = product
    return val

# --- Operator string with degree reduction on the Boolean cube -------------------
# forall x_i  ->  PROD : v = Phat(0)*Phat(1)
# exists x_i  ->  OR   : v = 1 - (1-Phat(0))*(1-Phat(1))
# L_i (degree reduction): replace P by x_i*P(1)+(1-x_i)*P(0), using x^k=x on the cube.
# Sequence: Q'_x1 L1  Q'_x2 L1 L2  ...  Q'_xn L1..Ln  Phi, with Q'_xi in {PROD, OR}.
# Quantifier messages have degree <= 1; ordinary L messages degree <= 2; final L messages
# that touch Phi have degree <= 3m.  Total operators/rounds: O(n^2).

def verify_TQBF(Psi, prover, F):
    ops = build_operator_string(Psi)   # list of ('PROD'|'OR'|'L', var_index)
    binding = {}                       # variables already fixed to random r_i
    v = 1                              # claim: the whole stripped expression == 1 (true)
    for op, i in ops:
        bound = degree_bound(op, i, Psi)                 # 1, 2, or 3m as above
        Phat = prover.send_univariate(op, i, binding)
        if degree(Phat) > bound: return False
        if op == 'PROD':                                # forall : product over {0,1}
            if F.mul(eval1(Phat,0,F), eval1(Phat,1,F)) != v:        return False
        elif op == 'OR':                                # exists : 1-(1-.)(1-.)
            lhs = F.sub(1, F.mul(F.sub(1,eval1(Phat,0,F)), F.sub(1,eval1(Phat,1,F))))
            if lhs != v:                                            return False
        else:                                           # L_i : linearization check
            r = binding[i]                              # current binding of x_i
            lhs = F.add(F.mul(r, eval1(Phat,1,F)), F.mul(F.sub(1,r), eval1(Phat,0,F)))
            if lhs != v:                                            return False
        r_i = F.random()               # FRESH random point => independent per-round error
        binding[i] = r_i               # (for L_i this re-binds x_i; otherwise binds it)
        v = eval1(Phat, r_i, F)        # carry the claim to the reduced sub-problem
    # Completeness: honest prover => every check holds, accept w.p. 1.
    # Soundness:    if Psi false, a cheating prover lies about some round's low-degree
    #   univariate; it disagrees with the true one except on <= deg roots, caught w.p.
    #   >= 1 - deg/p.  Union bound: error <= (3mn + n^2)/p -> 0.
    # all variables bound to random points; verifier evaluates Phi itself and checks.
    return Phi(binding, Psi.clauses, F) == v

# --- The matching easy direction:  IP  subseteq  PSPACE --------------------------
def value_of_game_tree(history, V):     # poly depth; reused space => PSPACE
    if is_leaf(history):
        return 1 if V.accepts(history) else 0
    if prover_moves_next(history):                      # prover maximizes
        return max(value_of_game_tree(history + msg, V) for msg in prover_messages())
    else:                                               # verifier averages over coins
        return weighted_average(value_of_game_tree(history + c, V) for c in coin_outcomes())
# root value > 2/3 => accept,  < 1/3 => reject.  Hence IP subseteq PSPACE,
# and with PSPACE subseteq IP above,  IP = PSPACE.
```
