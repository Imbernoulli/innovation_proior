# Synthesis — Karatsuba multiplication (KML)

Grounded entirely in retrieved sources in `refs/` (this run):
- `karatsuba-ofman-1962-doklady-145-original-russian.pdf` — THE PRIMARY SOURCE (Doklady AN SSSR 145(2):293–294, 1962). Russian full text extracted to `ko1962-textlayer.txt`.
- `karatsuba-1995-complexity-of-computations.pdf` — Karatsuba's own retrospective (Proc. Steklov Inst. Math. 211, 1995, pp.169–189). OCR in `karatsuba-1995-ocr.txt`. Discovery story IN HIS OWN WORDS: §5 conjecture, §6 disproof, §7 full KML derivation.
- `ofman-1962-algorithmic-complexity-discrete-functions-russian.pdf` — Ofman's companion paper (Doklady 145(1):48–51, 1962): automaton cost model + Theorem-1 ancestor. Russian text in `ofman-1962.txt`.
- `bernstein-multidigit-multiplication-for-mathematicians.pdf` — analysis (algebraic eval/interpolation view; Knuth x+1 variant; history note that K. couldn't generalize it).
- `uchicago-karatsuba-handout.pdf` — analysis (clean recursive pseudocode; (X1−X2)(Y1−Y2) variant; recurrence M(n)=3M(n/2)).
- `iacr-2007-393-overlap-free-ko.pdf` — analysis (canonical three-product recursion, eq (1)).

## Pain / research question
Estimate bit operations sufficient to multiply two n-digit binary numbers a, b. Schoolbook long multiplication (Karatsuba: OML, Ordinary Multiplication; ancient, >4000 yrs) costs O(n²). For millennia nothing faster was known. Kolmogorov (≈1956 at Moscow Math. Society; restated autumn-1960 seminar at MSU Mech-Math) CONJECTURED M(n)=Ω(n²) is a true lower bound — "the n² conjecture." Rationale (1995 §5): "throughout the history of mankind people have used OML whose complexity is O(n²), and if a more economical method existed, it would have already been found." The wall is historical/psychological, NOT proven. The broader Kolmogorov program (Ofman 1962): get lower bounds S_f(n); multiplication is the hard test case. Ofman (quoted 1995 §3): "Difficulties appear even in estimating the algorithmic complexity of the ordinary multiplication of binary n-digit numbers."

## Tools on the table (antecedents)
1. **OML / schoolbook**, O(n²). Karatsuba's own bound (1995 §4): a with ≥ n/2 ones ⇒ partial-product table ≥ n²/2 bits ⇒ 4n ≤ M(n) ≤ 8n². Upper O(n²), only trivial Ω(n) lower.
2. **EML (Egyptian)**, also O(n²): successive doubling + add; OML = EML with appended zeros (1995 §9–10). Context flavor, not load-bearing for the leap.
3. **Automaton / bit-operation cost model** (Ofman 1962): input on k=2^m links; one bit op = write a symbol 0,1,+,−,(,) or add/sub/multiply two bits. N = #ops, T = depth. The yardstick.
4. **Ofman's Theorem 1** (the 1962 paper's Teorema 1, credited to Ofman): split the multiplier into groups of s bits; serial within a group, parallel across groups ⇒ N ≍ m²/s, T ≍ s log m. s=1: N≍m², T≍log²m (eq 1); s=m: N≍m, T≍m log m (eq 2). Trades work vs depth but **work N never drops below m²** — never breaks n². It frames the open question: can N go below m²?
5. **CSS / residue number system** (Svoboda–Valach 1955): digitwise multiply, O(n log n) — but base conversion is costly and magnitudes can't be compared; Kolmogorov dismissed it (Vitushkin: "if people lived in CSS the n² conjecture would not exist"; Kolmogorov: number systems exist for measuring/comparing). A non-answer.
6. **Squaring↔multiplication identity** (1995 §3; "immediately pointed out by Kolmogorov"): ab = ¼[(a+b)² − (a−b)²]. So M(n) = squaring complexity up to a constant; ÷4 is trivial in binary. Lets you study one n-digit square instead of two numbers.

## The leap — Karatsuba's actual path (1995 §6)
Autumn 1960 seminar; Kolmogorov states the n² conjecture. Karatsuba: "I began to think actively about the n² conjecture, and exactly within a week I found that the algorithm with whose aid I HOPED TO DERIVE A LOWER ESTIMATE for M(n) provided an estimate of the form M(n)=O(n^1.585)." He was TRYING TO PROVE Kolmogorov RIGHT — to build the construction that pins the lower bound — and that very construction produced an *upper* bound BELOW n². THE SELF-CORRECTION: the tool meant to confirm the wall demolished it. Reaction: Kolmogorov "very agitated"; presented K.'s method himself at the next meeting; TERMINATED the seminar. Paper submitted by Kolmogorov (with Ofman) 13 Feb 1962; K. "learned about the article only when given its reprints." Credited to Karatsuba alone in the text (Bernstein).

## The method — two equivalent derivations

### A. Squaring form (THE form in the 1962 primary paper + 1995 eqs 1–3)
Square 2m-digit a = 2^m a1 + a2 (a1,a2 m-digit):
  a² = 2^{2m} a1² + 2^{m+1} a1 a2 + a2².   (naive: a1², a2², and the cross 2a1a2)
Recover the cross WITHOUT its own multiplication:
  2 a1 a2 = (a1+a2)² − a1² − a2².
⇒ a² = 2^{2m} a1² − 2^m a1² + 2^m (a1+a2)² + a2² − 2^m a2².   (1962 main formula / 1995 eq 1)
So squaring a 2m-digit number ⇐ THREE squarings of m-digit numbers (a1, a2, a1+a2) + shifts/adds.
Carry subtlety (primary footnote + 1995 eq 2): a1+a2 can be (m+1)-digit. Write a1+a2 = ε + 2 a3, ε∈{0,1}, a3 m-digit; then (a1+a2)² = ε² + 4 ε a3 + 4 a3² (footnote (2a3+ε)² = 4a3²+4a3ε+ε²). So only m-digit numbers are ever squared.
Lemma (1962): r-digit square cost N_r,T_r ⇒ N_{r+1}=3N_r + c·2^r, T_{r+1}=T_r + c·r ⇒ (induction) N ≍ m^{log2 3}, T ≍ log²m. (1995 §7: φ(n) ≤ 210 n^{log2 3}; constants deliberately loose.)

### B. Direct two-number form (1995 eq 7, "alternative version"; the canonical modern KML)
a = 2^m a1 + a2, b = 2^m b1 + b2.
  ab = 2^{2m} a1 b1 + 2^m (a1 b2 + a2 b1) + a2 b2.   (naive: 4 mults)
Middle: a1 b2 + a2 b1 = (a1+a2)(b1+b2) − a1 b1 − a2 b2.
⇒ ab = 2^{2m} a1 b1 + 2^m[(a1+a2)(b1+b2) − a1 b1 − a2 b2] + a2 b2.   (THREE products)
Recurrence φ(n) ≤ 3 φ(n/2) + c n ⇒ φ(n) ≤ c1 n^{log2 3}.
Knuth variant (Bernstein): (a1−a2)(b1−b2) ⇒ middle = a1b1+a2b2−(a1−a2)(b1−b2). UChicago handout: W=(X1−X2)(Y1−Y2), Z=U+V−W.

Forward directions (1995 §14, posterior — for synthesis only, NOT for context.md): split into r+1 parts ⇒ Toom–Cook; FFT (Cooley–Tukey 1965); Schönhage–Strassen 1971 O(n log n log log n); Strassen 1969 (2×2 matrix blocks 8→7). Bernstein: Karatsuba could NOT generalize KML; "apparently he did not realize that it amounted to evaluation and interpolation."

## Why the 3-multiplication trick is the whole game (design rationale)
- Recurrence T(n)=k·T(n/2)+O(n): master theorem ⇒ n^{log2 k} for k>2. k=4 (naive split) ⇒ n^{log2 4}=n²: SPLITTING ALONE BUYS NOTHING; you just rederive O(n²). The only lever is k. Dropping k from 4→3 moves the exponent from 2 to log2 3=1.585. The entire breakthrough = "kill one of the four sub-multiplications."
- Splitting is necessary (gives recursion) but not sufficient (k=4⇒n²). The cross term normally needs 2 mults; the identity recovers it from products ALREADY computed (a1b1, a2b2) + ONE new product (a1+a2)(b1+b2). Net 4→3.
- Additions O(n)/level, subdominant: extra-additions recurrence still solves to Θ(n^{log2 3}) (UChicago eq 2/3). Trading multiplications for additions is a strict asymptotic win — the assumed-unbeatable n² (for operations, not just multiplications) falls because the added O(n)/level doesn't change the order. Recursion-tree: level i has 3^i nodes, O(n/2^i) combine each ⇒ Σ n(3/2)^i over log2 n levels, ratio 3/2>1 ⇒ dominated by last level = n(3/2)^{log2 n} = n·n^{log2 3}/n = n^{log2 3}.
- Base case: n=1 single bit/digit multiply (multiplication table).

## Code (canonical recursive, grounded in UChicago handout + IACR eq 1 + standard textbook/CPython-style impl)
Two-number form, base-10 limbs:
```
def karatsuba(x, y):
    if x < 10 or y < 10:                 # base case: single digit
        return x * y
    m = max(num_digits(x), num_digits(y)) // 2
    hi_x, lo_x = divmod(x, 10**m)        # x = hi_x*10^m + lo_x
    hi_y, lo_y = divmod(y, 10**m)
    z2 = karatsuba(hi_x, hi_y)                       # a1 b1
    z0 = karatsuba(lo_x, lo_y)                       # a2 b2
    z1 = karatsuba(lo_x+hi_x, lo_y+hi_y) - z2 - z0   # cross via ONE product
    return z2 * 10**(2*m) + z1 * 10**m + z0
```
Must use integer divmod / `//` (not float division). Scaffold (pre-method) = generic divide step + the naive 4-product recombine TODO; method fills "only 3 products."
