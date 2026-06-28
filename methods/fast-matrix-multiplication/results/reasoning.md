OK, let me think this through from scratch. I want to multiply two n×n matrices and I want it to be cheap. The formula I have is c_ij = Σ_k a_ik b_kj. For each of the n² output entries I run a length-n inner product: n multiplications and n−1 additions. So n³ multiplications, about n³ additions, Θ(n³) total. The question I actually care about is whether that cubic count is forced or just the first thing anyone wrote down.

My gut says it's forced. There are n³ products a_ik b_kj floating around — three free indices i, k, j — and it feels like I have to compute each of them at least once. How could you possibly get the value of c_ij without forming the individual terms a_ik b_kj and adding them? Every entry of A pairs with every entry of B in a way the output "sees." So n³ products, end of story.

But let me not trust the gut. "Every product a_ik b_kj is needed" is a statement about a *particular way* of computing the answer — the inner-product way. The output entries are bilinear forms in the a's and b's. Nothing says I have to realize a bilinear form by literally forming each monomial and summing. Maybe I can form *combinations* of inputs, multiply those, and have the cross-terms I don't want cancel out. That's the only loophole, but it's a real one.

Is there any precedent for that loophole paying off? Yes — polynomial/integer multiplication. Take (a0 + a1 X)(b0 + b1 X) = a0b0 + (a0b1 + a1b0) X + a1b1 X². Naively that middle coefficient a0b1 + a1b0 costs two multiplications, four in total. But Karatsuba's move: compute the corner products a0b0 and a1b1, then compute (a0+a1)(b0+b1) = a0b0 + a0b1 + a1b0 + a1b1, and *subtract off the corners*: (a0+a1)(b0+b1) − a0b0 − a1b1 = a0b1 + a1b0. Now I have all three coefficients from three multiplications, not four. I spent extra additions and one combined multiplication to delete a multiplication. Recursed on n-digit numbers — split each into high and low halves, three half-size products instead of four — the recurrence is T(n) = 3T(n/2) + O(n), which solves to n^{log₂ 3} ≈ n^{1.585}, well under the schoolbook n². So the "you need every monomial" intuition is just wrong in general: combinations plus cancellation beat it.

Two things I want to extract from Karatsuba before going back to matrices. First: what controlled the final exponent? The recurrence had a 3 in front because I made 3 recursive subproblems; the additions were O(n), the cheap part, and the Master theorem buried them — log₂ 3 came purely from the *number of multiplications*, 3 instead of 4. The additions never enter the exponent. Second: the win compounds *because it recurses*. Saving one multiplication once is nothing; saving it at every level of a divide-and-conquer is everything.

So for matrices: divide and conquer. Split each n×n matrix (say n even) into four (n/2)×(n/2) blocks,

  A = [[A11,A12],[A21,A22]],  B = [[B11,B12],[B21,B22]],

and the product blocks are
  C11 = A11 B11 + A12 B21
  C12 = A11 B12 + A12 B22
  C21 = A21 B11 + A22 B21
  C22 = A21 B12 + A22 B22.

Count the block multiplications: eight. Plus four block additions. So T(n) = 8T(n/2) + O(n²). And by the Master theorem that's Θ(n^{log₂ 8}) = Θ(n³). I'm exactly back where I started. Of course I am — 8 = 2³, the recursion just re-derives cubic. Recursion by itself buys *nothing*; it only reshuffles the same n³ scalar multiplications into a tree.

But now the Karatsuba lesson sharpens into a precise target. The exponent is log₂(number of block multiplications). Eight gives 3. Seven would give log₂ 7 ≈ 2.807 — subcubic. Six would give log₂ 6 ≈ 2.585. *Any* drop below eight, recursed, breaks the cubic barrier. So the entire problem reduces to one tiny algebraic question: can I multiply two 2×2 matrices with fewer than eight multiplications?

Let me try to just do it. Karatsuba worked by reusing a combined product, so let me hunt for an algebraic identity among these eight terms — A11B11, A12B21, A11B12, A12B22, A21B11, A22B21, A21B12, A22B22 — that lets some combined product serve double duty.

There's a constraint I have to respect, and it's easy to forget. The block entries are *matrices*. Matrices don't commute: A11 B12 is not B12 A11, and I can never use A·A or B·B products either, because the recursion needs each of my seven (hopefully) products to be (something built from A's blocks) times (something built from B's blocks), with the blocks kept on their correct sides. So whatever identity I find, every product I form has to look like (linear combination of A-blocks) × (linear combination of B-blocks), in that order, never mixing two A's or two B's into one multiplication, and never relying on commuting the factors. Karatsuba's identity actually satisfies this — it only ever multiplies an a-thing by a b-thing — so there's hope a matrix analogue exists, but I have to be disciplined about non-commutativity.

Let me try the naive Karatsuba transplant first, because it's the obvious thing and I want to see it fail so I understand why. Karatsuba on 2×2: treat the columns of A and rows of B like the two "digits." Hmm. The four output blocks are four separate bilinear forms, not the three coefficients of one product, so the shapes don't line up cleanly. Let me just try to save a multiplication on the diagonal. C11 = A11B11 + A12B21. By Karatsuba's trick I'd want (A11+A12)(B11+B21) = A11B11 + A11B21 + A12B11 + A12B21, and the cross terms A11B21 + A12B11 are *not* the corners I need to subtract — they're garbage that doesn't appear in any C entry. Dead end: the four output forms don't share monomials the way the three coefficients of a single polynomial product do, so the corner-subtraction structure isn't there. I can't just copy the identity.

Let me think about what I actually have freedom over. I get to pick seven products, each P_k = u_k(A) · v_k(B) where u_k is a linear combination of the four A-blocks and v_k a linear combination of the four B-blocks. Then I get to form each output block C_ij as a linear combination of the seven P_k. So I need fixed scalar coefficients such that the formal expansion of Σ_k (coefficients in C_ij) · u_k(A) v_k(B) equals the right bilinear form for each C_ij, as a polynomial identity in the 8 indeterminate blocks. There are four output bilinear forms and 8 target monomials total, with no target monomial shared between two outputs. I want to hit all 8 target monomials and cancel all unwanted ones, using only 7 rank-one products.

This is a system. Each product u_k(A)v_k(B), when I expand it, is a sum over (A-block, B-block) pairs — up to 16 possible monomials A_pq B_rs. There are 16 such monomials in the full 2×2-by-2×2 expansion. My eight targets are a specific 8 of them (the ones with matching inner index: A11B11, A12B21 for C11, etc.), each wanted with coefficient 1, and the other 8 monomials wanted with coefficient 0. Seven products, each contributing a rank-one pattern across these 16 monomials, must linearly combine (with the C-recombination coefficients) to this target. Whether 7 suffice is not obvious at all; it's a question about decomposing a specific structure into 7 rank-one pieces.

Let me stop trying to be clever and search more systematically, the way you'd grope for Karatsuba if you'd never seen it. The thing that made Karatsuba tick was a product of *sums* on both sides — (a0+a1)(b0+b1) — that generated several wanted monomials at once, and then auxiliary products that subtracted the unwanted ones. So I should look for products where I add blocks on *both* the A side and the B side, to get multiple useful monomials per multiplication, and then fix up with cheaper one-sided products.

Try a fully symmetric one: P = (A11 + A22)(B11 + B22) = A11B11 + A11B22 + A22B11 + A22B22. Interesting — this gives me A11B11 (wanted in C11) and A22B22 (wanted in C22) in a single product, plus two unwanted cross terms A11B22 and A22B11. That's the Karatsuba flavor: one product, two useful diagonal monomials, two to kill. The two useful ones even land in *different* output blocks (C11 and C22), which is promising — it means one product can serve two outputs.

Let me build the rest around making the unwanted terms of this central product cancel, while filling in the monomials I still need. I need, across the four outputs: A11B11, A12B21 (C11); A11B12, A12B22 (C12); A21B11, A22B21 (C21); A21B12, A22B22 (C22). The central product P1 = (A11+A22)(B11+B22) gave me A11B11 and A22B22 (good) plus A11B22 and A22B11 (bad).

C12 = A11B12 + A12B22 and C21 = A21B11 + A22B21 look like they want one factor held fixed and a difference on the other side. A11(B12 − B22) = A11B12 − A11B22: this produces A11B12 (wanted in C12) and, as a bonus, −A11B22 — which is exactly one of the junk terms P1 left lying around. So call P3 = A11(B12 − B22). Symmetrically A22(B21 − B11) = A22B21 − A22B11 gives A22B21 (wanted in C21) and −A22B11, the *other* junk term from P1. Call P4 = A22(B21 − B11). The diagonal is starting to close up: the two cross terms A11B22, A22B11 polluting P1 have matching negatives waiting in P3 and P4.

Now the remaining wanted monomials: A12B21 (for C11), A12B22 (for C12), A21B11 (for C21), A21B12 (for C22). And the off-diagonal blocks A12, A21 have barely appeared. Let me get the ones I can cheaply: (A21+A22)B11 = A21B11 + A22B11 gives A21B11 (wanted in C21) — call P2 = (A21+A22)B11; it spits out an extra A22B11. (A11+A12)B22 = A11B22 + A12B22 gives A12B22 (wanted in C12) — call P5 = (A11+A12)B22; extra A11B22. Two more products of the "sum times single block" type.

Let me try to assemble C12 and C21 now and see what cancels. C12 should be A11B12 + A12B22. I have A11B12 in P3 (= A11B12 − A11B22) and A12B22 in P5 (= A11B22 + A12B22). Add them: P3 + P5 = A11B12 − A11B22 + A11B22 + A12B22 = A11B12 + A12B22 = C12. The A11B22 cancels exactly. So C12 = P3 + P5. Clean.

C21 = A21B11 + A22B21. I have A21B11 in P2 (= A21B11 + A22B11) and A22B21 in P4 (= A22B21 − A22B11). Add: P2 + P4 = A21B11 + A22B11 + A22B21 − A22B11 = A21B11 + A22B21 = C21. The A22B11 cancels. So C21 = P2 + P4. Also clean. That's five products P1..P5 and two of the four outputs done. The same stray terms keep showing up with opposite signs: A11B22 is available as −A11B22 in P3 and +A11B22 in P5, while A22B11 is available as +A22B11 in P2 and −A22B11 in P4. I should be able to use those signs again when I assemble the diagonal blocks.

Now the diagonal outputs, and I have two products left to spend (I'm allowing myself seven; this is P6, P7). C11 = A11B11 + A12B21. I have A11B11 sitting inside P1. Let me see what combination of what-I-have yields C11. Try P1 + P4 − P5: P1 = A11B11 + A11B22 + A22B11 + A22B22; P4 = A22B21 − A22B11; −P5 = −A11B22 − A12B22. Sum so far: A11B11 + A22B22 + A22B21 − A12B22. The A11B22 and A22B11 from P1 cancelled against P4 and P5 — good, the junk is dying. What's left is A11B11 (want it) + A22B22 (don't want it in C11) + A22B21 (don't want) − A12B22 (don't want). I still need to turn A22B22 + A22B21 − A12B22 into +A12B21, and clean up. That's a mess of leftover terms involving A12, A22 and B21, B22. This is exactly the kind of corner I need one more *product of two sums* to sweep up — the leftover monomials all live in the A12/A22 × B21/B22 quadrant.

The leftover I want to fix is: I have +A11B11 + A22B22 + A22B21 − A12B22 and I want +A11B11 + A12B21. So I need to add (A12B21 − A22B22 − A22B21 + A12B22) = A12(B21 + B22) − A22(B21 + B22) = (A12 − A22)(B21 + B22). That factors perfectly into one bilinear product. Call P7 = (A12 − A22)(B21 + B22). Then C11 = P1 + P4 − P5 + P7. Let me double check by expanding P7 = A12B21 + A12B22 − A22B21 − A22B22 and adding to the running A11B11 + A22B22 + A22B21 − A12B22: the A22B22 cancels, the A22B21 cancels, the A12B22 cancels (−A12B22 + A12B22), leaving A11B11 + A12B21. Exactly C11. Seven... wait, I've only used P1, P4, P5, P7 here and P2, P3 elsewhere; P6 is still free. Good, C11 = P1 + P4 − P5 + P7.

Last one: C22 = A21B12 + A22B22, with one product, P6, left. I have A22B22 in P1. By the mirror of the C11 construction, try P1 − P2 + P3 + (last product). P1 = A11B11 + A11B22 + A22B11 + A22B22; −P2 = −A21B11 − A22B11; P3 = A11B12 − A11B22. Sum: A11B11 + A22B22 − A21B11 + A11B12 (the A11B22 cancels with P3, the A22B11 cancels with −P2). I want C22 = A21B12 + A22B22, so I need to add (A21B12 + A22B22) − (A11B11 + A22B22 − A21B11 + A11B12) = A21B12 + A21B11 − A11B11 − A11B12 = A21(B12 + B11) − A11(B11 + B12) = (A21 − A11)(B11 + B12). One bilinear product. Call P6 = (A21 − A11)(B11 + B12). Then C22 = P1 − P2 + P3 + P6. Check: P6 = A21B11 + A21B12 − A11B11 − A11B12; add to A11B11 + A22B22 − A21B11 + A11B12: A11B11 cancels, A21B11 cancels, A11B12 cancels, leaving A22B22 + A21B12 = C22. 

So with seven products
  P1 = (A11 + A22)(B11 + B22)
  P2 = (A21 + A22) B11
  P3 = A11 (B12 − B22)
  P4 = A22 (B21 − B11)
  P5 = (A11 + A12) B22
  P6 = (A21 − A11)(B11 + B12)
  P7 = (A12 − A22)(B21 + B22)
I recover
  C11 = P1 + P4 − P5 + P7
  C12 = P3 + P5
  C21 = P2 + P4
  C22 = P1 − P2 + P3 + P6.

Seven multiplications instead of eight. Let me count the additions: P1 needs 2 (one sum on each side), P2 1, P3 1, P4 1, P5 1, P6 2, P7 2 — that's 10 additions to form the operands. Then the recombination: C11 needs 3, C12 1, C21 1, C22 3 — 8 more. Eighteen additions/subtractions total. Doesn't matter for the exponent; it's O(n²) glue.

Before I trust this enough to recurse, two things have to hold, and I've only hand-waved both: that the cancellations are exact for *all four* blocks (I checked them one at a time while building, which is exactly the way to fool myself), and that nothing I did secretly relied on the blocks commuting. Let me nail both.

Let me make sure I actually believe the cancellations as a formal identity, not just on the diagonal I checked. Treat all eight blocks as non-commuting symbols and expand C11 = P1 + P4 − P5 + P7:
 P1 = A11B11 + A11B22 + A22B11 + A22B22
 P4 = A22B21 − A22B11
 −P5 = −A11B22 − A12B22
 P7 = A12B21 + A12B22 − A22B21 − A22B22
Sum: A11B11 (stays) + A11B22 − A11B22 (0) + A22B11 − A22B11 (0) + A22B22 − A22B22 (0) + A22B21 − A22B21 (0) − A12B22 + A12B22 (0) + A12B21 (stays) = A11B11 + A12B21. For C12 = P3 + P5, the expansion is A11B12 − A11B22 + A11B22 + A12B22 = A11B12 + A12B22. For C21 = P2 + P4, it is A21B11 + A22B11 + A22B21 − A22B11 = A21B11 + A22B21. For C22 = P1 − P2 + P3 + P6, it is A11B11 + A11B22 + A22B11 + A22B22 − A21B11 − A22B11 + A11B12 − A11B22 + A21B11 + A21B12 − A11B11 − A11B12 = A21B12 + A22B22. In each of the four expansions I only ever concatenate an A-side symbol with a B-side symbol in that order, and every cancellation is between two identical ordered words — so the equalities don't use commuting anywhere.

Symbolic expansion is where I make sign errors, though, so let me ground it with numbers before I commit. Take the scalar 2×2 case (1×1 blocks, where I can also see the answer by hand): A = [[1,2],[3,4]], B = [[5,6],[7,8]], so AB = [[1·5+2·7, 1·6+2·8],[3·5+4·7, 3·6+4·8]] = [[19,22],[43,50]]. Now run the seven products. P1 = (1+4)(5+8) = 5·13 = 65; P2 = (3+4)·5 = 35; P3 = 1·(6−8) = −2; P4 = 4·(7−5) = 8; P5 = (1+2)·8 = 24; P6 = (3−1)(5+6) = 2·11 = 22; P7 = (2−4)(7+8) = −2·15 = −30. Recombine: C11 = P1+P4−P5+P7 = 65+8−24−30 = 19 ✓; C12 = P3+P5 = −2+24 = 22 ✓; C21 = P2+P4 = 35+8 = 43 ✓; C22 = P1−P2+P3+P6 = 65−35−2+22 = 50 ✓. All four match [[19,22],[43,50]]. The arithmetic survives a concrete instance, which catches the sign slips the symbolic pass can hide.

That instance was scalar, though, and scalars commute — so it can't tell me whether I'm secretly leaning on A11B12 = B12A11 somewhere. I need a case where the blocks genuinely don't commute. So take 2×2 blocks that are themselves 2×2 matrices: pick eight random integer 2×2 matrices for A11…B22, with (say) A11B12 ≠ B12A11 (easy to arrange and easy to check). Form the four output blocks both by the seven-product recombination and by honest 4×4 multiplication A B, and compare. When I do this the two agree entry for entry. Since the block entries here demonstrably fail to commute and the answer is still exact, the identity is using only the non-commutative algebra — multiplying A-side by B-side on the correct sides — and nothing else. That is the property the recursion depends on.

So the identity is a *polynomial identity in 8 non-commuting indeterminate matrices*: every P_k is (a combination of A-blocks) on the left times (a combination of B-blocks) on the right, no product ever multiplies two A-blocks or two B-blocks, and no factor is ever reordered. It therefore stays true when the eight entries are matrix blocks, and the recursion is legitimate.

So the recurrence is now T(n) = 7T(n/2) + O(n²). Master theorem, 7 > 4: T(n) = Θ(n^{log₂ 7}). log₂ 7 = 2.807…. Subcubic. The cubic barrier is broken, and broken by a finite algebraic identity recursed — nothing about the structure of A or B, just seven products and a recombination.

Now I want to step back, because something about this bugs me, in a productive way. I found these seven products by groping — central product, kill the junk, sweep the corners. It worked, but it feels like luck. *Why seven?* Could a cleverer person find six? And is "seven products" even the right object to be staring at, or is it an artifact of the basis I happened to write the matrices in? Karatsuba's three is not just an implementation detail; it is a shorter bilinear decomposition of the polynomial-product map. I want the analogous object here.

The invariant I need is already sitting in the bilinear formula. Every bilinear algorithm for matrix multiplication has products

  p_k = u_k(A) v_k(B)

where u_k and v_k are linear forms, and then scalar recombination weights w_{k,ij} fill the output entries by

  C_ij = Σ_k w_{k,ij} p_k.

My algorithm is exactly this with r = 7: the u_k and v_k are the bracketed sums, and the weights w_{k,ij} are read directly from C11 = P1 + P4 − P5 + P7, C12 = P3 + P5, C21 = P2 + P4, and C22 = P1 − P2 + P3 + P6. For any one algorithm, r is the length of this decomposition. Minimizing r over all such decompositions is the real invariant. The question "is 7 optimal" is "what is the shortest decomposition of this form."

If I stack the coefficients, the whole map ⟨2,2,2⟩ becomes a single 3-dimensional array — a tensor t. One axis indexes the A-entries, one the B-entries, one the C-entries; the entry of t at (A-slot, B-slot, C-slot) is 1 exactly when that A-entry times that B-entry feeds that C-entry. For ⟨2,2,2⟩ the inner-index matching c_ij = Σ_k a_ik b_kj means t lives in a 4×4×4 array with a sparse {0,1} pattern. A single product u(A)v(B) contributing to outputs through scalar weights w is precisely a *rank-one* tensor u⊗v⊗w — an outer product of three pieces: the A-side linear form, the B-side linear form, and the output-recombination pattern. So a bilinear algorithm with r multiplications is a way to write t as a sum of r rank-one tensors after only linear preprocessing and linear recombination. The minimum such r, over the field I'm working in, is the tensor rank R(t). The number of multiplications I'm fighting over is a tensor rank — completely basis-free once the bilinear map is fixed. My groping showed R(⟨2,2,2⟩) ≤ 7. Whether 6 is possible is now a precise mathematical question about this 4×4×4 tensor; for the exponent bound I only need the seven-term witness.

This also explains why a 2×2 identity gives an n×n algorithm. Matrix tensors multiply: the tensor of ⟨k,m,n⟩ times the tensor of ⟨k',m',n'⟩ — their tensor (Kronecker) product — is exactly the tensor of ⟨kk', mm', nn'⟩. Multiplying 2^L × 2^L matrices is ⟨2,2,2⟩^{⊗L}, the L-th tensor power. And rank is submultiplicative: R(s ⊗ t) ≤ R(s)·R(t). So

  R(⟨2^L,2^L,2^L⟩) = R(⟨2,2,2⟩^{⊗L}) ≤ R(⟨2,2,2⟩)^L ≤ 7^L = (2^L)^{log₂ 7}.

Setting n = 2^L, R(⟨n,n,n⟩) ≤ n^{log₂ 7}. The recursion *is* the tensor power, and the exponent log₂ 7 is the log of the seven-term rank bound for the base tensor. That's why saving one multiplication once compounds: each tensor power multiplies the available rank bound, so the base bound goes into the exponent bound. This also tells me the general principle for going further: find *any* matrix tensor ⟨n,n,n⟩ with R < n³, and I get ω ≤ log_n R. Define ω as the infimum of β with R(⟨n,n,n⟩) ≤ O(n^β); my result is ω ≤ log₂ 7 ≈ 2.807, the trivial bounds are 2 ≤ ω ≤ 3, and *any* rank bound on *any* fixed-size matrix tensor that beats cubic pushes ω down.

Let me write the witness as real, runnable code, because the whole thing has to bottom out in something I can run. A single self-contained C++17 program: read `n` and two n×n integer matrices A, B from stdin (row-major, whitespace-separated), print C = A·B to stdout. Recurse with the seven products; below a crossover size the recursion overhead and the eighteen additions aren't worth it, so switch to a cubic leaf kernel there; pad odd sizes up so the halving always works; accumulate in `long long` so the products don't overflow.

If I were not confident I could implement Strassen's seven-product block recurrence and the P1..P7 recombination correctly within the budget, I would fall back to the standard cubic matrix multiplication kernel for this problem and ship that instead.

```cpp
// Reads n, then two n x n integer matrices A and B from stdin (row-major,
// whitespace-separated); prints C = A*B (n rows, space-separated) to stdout.
// Recursive Strassen (7 block products) with a cubic leaf kernel; long long
// accumulation to avoid overflow.
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using Mat = vector<vector<ll>>;

static const int LEAF = 64;

// Cubic leaf kernel / ground truth: C = A*B for two s x s matrices.
Mat matmul_naive(const Mat& A, const Mat& B) {
    int s = (int)A.size();
    Mat C(s, vector<ll>(s, 0));
    for (int i = 0; i < s; i++)
        for (int k = 0; k < s; k++) {
            ll a = A[i][k];
            if (a == 0) continue;
            for (int j = 0; j < s; j++)
                C[i][j] += a * B[k][j];
        }
    return C;
}

Mat add(const Mat& A, const Mat& B) {
    int s = (int)A.size();
    Mat C(s, vector<ll>(s));
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) C[i][j] = A[i][j] + B[i][j];
    return C;
}

Mat sub(const Mat& A, const Mat& B) {
    int s = (int)A.size();
    Mat C(s, vector<ll>(s));
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Recursive Strassen for an s x s matrix where s is even (or <= LEAF).
Mat strassen(const Mat& X, const Mat& Y) {
    int n = (int)X.size();
    if (n <= LEAF || (n & 1)) return matmul_naive(X, Y);

    int m = n / 2;
    Mat a(m, vector<ll>(m)), b(m, vector<ll>(m)), c(m, vector<ll>(m)), d(m, vector<ll>(m));
    Mat e(m, vector<ll>(m)), f(m, vector<ll>(m)), g(m, vector<ll>(m)), h(m, vector<ll>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            a[i][j] = X[i][j];     b[i][j] = X[i][j + m];
            c[i][j] = X[i + m][j]; d[i][j] = X[i + m][j + m];
            e[i][j] = Y[i][j];     f[i][j] = Y[i][j + m];
            g[i][j] = Y[i + m][j]; h[i][j] = Y[i + m][j + m];
        }

    // Seven products: each is (A-side combination) x (B-side combination),
    // A always on the left, so the identity lifts to non-commuting blocks.
    Mat p1 = strassen(add(a, d), add(e, h));   // (A11+A22)(B11+B22)
    Mat p2 = strassen(add(c, d), e);           // (A21+A22) B11
    Mat p3 = strassen(a, sub(f, h));           // A11 (B12-B22)
    Mat p4 = strassen(d, sub(g, e));           // A22 (B21-B11)
    Mat p5 = strassen(add(a, b), h);           // (A11+A12) B22
    Mat p6 = strassen(sub(c, a), add(e, f));   // (A21-A11)(B11+B12)
    Mat p7 = strassen(sub(b, d), add(g, h));   // (A12-A22)(B21+B22)

    // Recombine; cross terms cancel exactly by the Strassen identity.
    Mat c11 = add(sub(add(p1, p4), p5), p7);   // C11 = p1 + p4 - p5 + p7
    Mat c12 = add(p3, p5);                      // C12 = p3 + p5
    Mat c21 = add(p2, p4);                      // C21 = p2 + p4
    Mat c22 = add(add(sub(p1, p2), p3), p6);    // C22 = p1 - p2 + p3 + p6

    Mat C(n, vector<ll>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            C[i][j] = c11[i][j];     C[i][j + m] = c12[i][j];
            C[i + m][j] = c21[i][j]; C[i + m][j + m] = c22[i][j];
        }
    return C;
}

// Pad to the next even size when needed so the halving always works.
Mat recursive_matmul(const Mat& A, const Mat& B) {
    int n = (int)A.size();
    if (n == 0) return Mat();
    if (n <= LEAF) return matmul_naive(A, B);

    int s = (n & 1) ? n + 1 : n;
    if (s == n) return strassen(A, B);

    Mat Ap(s, vector<ll>(s, 0)), Bp(s, vector<ll>(s, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Ap[i][j] = A[i][j]; Bp[i][j] = B[i][j]; }
    Mat Cp = strassen(Ap, Bp);
    Mat C(n, vector<ll>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i][j] = Cp[i][j];
    return C;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    Mat A(n, vector<ll>(n)), B(n, vector<ll>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> A[i][j];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> B[i][j];

    Mat C = recursive_matmul(A, B);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j) cout << ' ';
            cout << C[i][j];
        }
        cout << '\n';
    }
    return 0;
}
```

The causal chain, start to finish: the cubic count came from believing every monomial a_ik b_kj must be formed, but bilinear forms can be built from *combinations* with cancellation, as Karatsuba already showed for polynomials; recursion alone reproduces cubic because eight block products is 2³, so the only lever is the multiplication count, whose log₂ becomes the exponent; an algebraic hunt — one central product of sums to grab two diagonal monomials, auxiliary products to cancel the junk and sweep the corners — yields seven block products that reconstruct all four output blocks without ever multiplying two same-side blocks, so the identity holds for non-commuting matrix entries and recurses to Θ(n^{log₂ 7}); and finally, reading "number of multiplications" as the rank of the matrix-multiplication tensor turns the recursion into a tensor power, with rank submultiplicativity explaining why the base rank bound lands in the exponent bound.
