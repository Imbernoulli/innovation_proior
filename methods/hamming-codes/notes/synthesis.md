# Synthesis — Hamming codes

## The pain (self-account, primary)
Hamming, Model 5 relay computer (NYC, ~1947-48), low man on totem pole -> only weekend
machine time (Fri 5pm -> Mon 8am). Loads tape with many problems, promises Murray Hill
friends answers Tuesday. The machine has 2-out-of-5 codes that DETECT errors; on detecting
one it retries the step up to 3 times, then drops the problem and moves on. One weekend the
machine fails Friday night -> nothing Monday. Apologizes, reloads. SAME thing next weekend.
Frustration boils over: "Damn it, if the machine can detect an error, why can't it locate
where it is and correct it by changing the bit to the opposite state?" (verbatim across
3 sources: Art of Doing Science ch.12, Computer Pioneers bio taped interview, Wikipedia.)
This emotional frustration is, in his telling, the trigger of the whole idea.

## Prepared mind (antecedents he already knew)
- 2-out-of-5 / 3-out-of-7 codes used in relay machines & telephone offices: detection only.
- Parity check: append a bit so total #1s is even; one error flips parity -> detect single
  (odd) error. He'd "examined their fundamentals very carefully." A parity check need not
  cover all positions -- it can be over selected positions only. THIS is the lever.
- Triple modular redundancy (3 copies + majority vote) gives correction but at 200%+ cost --
  he knew it existed and rejected it as too expensive. Motivates: cheaper correction.
- Binary number system familiarity (uncommon in 1947) -- "repeatedly played a prominent role."

## The discovery path (Hamming's OWN sequence -- the backbone of reasoning.md)
1. detect->correct leap (the outburst).
2. Rectangular array: arrange message bits in a rectangle, parity on each row and each
   column. Two failing checks (one row, one column) give the COORDINATES of the single bad
   bit -- including the corner parity bit. First real idea: overlapping parity checks локate.
   Redundancy ~ (rows+cols)/(rows*cols); square is best; bigger is better but double-error
   risk grows. Double-error ambiguity noted (diagonal pairs indistinguishable).
3. Triangle: parity checks along the diagonal, each checking its row AND column -> better
   redundancy than rectangle. (A few weeks later, riding the mail car through north Jersey.)
4. 3D cube: parity over whole planes, check bit on each axis -> 3 coordinates at cost 3n-2
   checks for n^3 message bits. Better.
5. n-dim cube: even better. Then the key collapse: a 2x2x2x...x2 cube (don't physically
   arrange, just interwire) with n+1 parity checks is best -- APPARENTLY.
6. "Burnt fingers once" -> demand a PROOF it's best. Counting argument: n+1 checks => an
   (n+1)-bit syndrome => 2^(n+1) things, but only need 2^n + 1 (the 2^n points plus 'correct').
   "Off by almost a factor of 2." So the apparent best wasn't tight -- can do better.
7. THE insight: use the syndrome of the error as a BINARY NUMBER that NAMES the position of
   the error, all-0 = correct (easier test than all-1 on most computers). Familiarity with
   binary central.
8. Construction: write positions 1,2,3,... in binary. Parity check #i must cover every
   position with a 1 in column i. So check#1 -> 1,3,5,7,...; check#2 -> 2,3,6,7,...; etc.
   Put CHECK bits at positions that are powers of two (1,2,4,8): then each parity check sets
   exactly its own check bit, checks independent, easy encoding. Rest are message positions.
9. (7,4): n=7,m=4,k=3. Checks at 1,2,4; message at 3,5,6,7. Necessary cond 2^k >= m+k+1,
   equality sufficient. Worked example: position-6 error -> syndrome 110 = 6 -> flip. "Magical".
10. Linearity: correct messages form an additive group mod 2; correct msg -> all-zero
    syndrome; (msg+single error) -> syndrome = position of error, independent of message.
    The parity checks "concentrate on the error and ignore the message."
11. Equivalence: permuting columns / complementing a column gives an equivalent code; the
    "Hamming code" arrangement is just a cute (convenient) one; in practice check bits can be
    grouped at the end.
12. SEC-DED: add ONE overall parity bit over the whole codeword -> min distance 4.
    Decode table: old-syndrome 000 & new-parity 0 = right; 000 & 1 = error in new parity bit;
    xxx & 1 = single error (old check locates it); xxx & 0 = double error (detect, can't fix).
13. Geometric/Hamming-distance view (BSTJ §5; AoDSE): each n-bit string = vertex of unit
    n-cube; D(x,y) = #positions differing = L1 / least edges traversed. Metric (4 axioms).
    Sphere of radius r. Single-error-correcting <=> code points pairwise >= distance 3 <=>
    disjoint radius-1 spheres. Each radius-1 sphere holds 1 + n points; 2^n / (n+1) bound ==
    the same 2^k >= m+k+1 bound. min-dist table:
      1 unique, 2 SED, 3 SEC, 4 SEC+DED, 5 DEC, 2k+1 -> k-correct, 2k+2 -> k-correct+(k+1)-detect.
    Sphere-packing / Hamming bound. Code = subset of vertices with the required min distance.

## Primary paper structure (BSTJ 1950) -- matches book
Part I special codes: SED (parity), SEC (the power-of-two construction, Tables I-III, (7,4)
worked example with position-5 error -> syndrome 101 = 5), SEC+DED (add overall parity, 8-pos).
Part II general theory: geometric model + metric, sphere packing, equivalence, proves Part I
codes are minimum-redundancy / "best". The 2^k >= m+k+1 condition; bound B(n,d). Notes Golay
(IRE 1949) as the only prior printed work in error correction.

## Design decisions -> WHY (for reasoning depth)
- Multiple OVERLAPPING parity checks (not one global parity): a single global check only
  detects (parity flips) -- it can't say WHERE. Overlapping checks each see a different subset;
  the PATTERN of which checks fail is information about position. (rectangle row+col is the
  first instance; the cube generalizes it.)
- Check bits at POWER-OF-TWO positions: so that position 2^i is the unique position covered
  by check i alone -> each check sets its own bit independently -> encoding is trivial and
  checks are mutually independent. If check bits sat elsewhere, setting one would disturb
  another. (Hamming: "so the setting of the parity check will be easy".)
- Syndrome = binary position: each position j is covered by exactly the checks i with bit i
  set in j. So a single error at j fails exactly those checks -> the failed-check vector,
  read as binary, equals j. This is why it's optimal: n+1 distinct syndromes (n positions +
  zero) is exactly what k checks afford when 2^k = n+1, no waste. The rectangle wasted
  syndrome space (used 2 numbers per error, one per axis, ~2*sqrt(n) checks for n bits;
  optimal uses ~log2(n)).
- Why power-of-two cube beats rectangle/triangle/3D: redundancy. Rectangle ~2*sqrt(m) checks;
  d-cube ~d*m^(1/d) checks; the limit d->log: 2^(...)-cube needs only ~log2(m) checks. Fewer
  check bits per message bit as dimension rises.
- Necessary condition 2^k >= m+k+1: the k-bit syndrome must name m+k positions + 'no error'.
  Equality is sufficient (perfect/tight code) -> (7,4),(15,11),(31,26),...
- All-zero = "no error" rather than all-one: easier hardware test on most computers.
- SEC-DED via one extra overall parity: pushes min distance from 3 to 4. Distance-3 code:
  a double error lands the received word distance-1 from a WRONG codeword -> miscorrected to a
  third error. The overall parity catches odd vs even number of errors: syndrome says
  something's wrong, overall-parity-intact says it was even (>=2) -> don't "correct", flag.
- Hamming distance / sphere packing: gives the clean correctness criterion (min dist 3 <=>
  SEC) and the optimality bound (disjoint unit spheres: 2^n/(n+1)), unifying everything.

## Canonical code (grounded, tested)
code/hamming.py -- general (n) construction: check bits at powers of two, parity check i over
positions with bit i set, syndrome names the error position; optional overall-parity SEC-DED.
Verified exhaustively for (7,4) single-error correction, (8,4) SEC-DED single-correct +
double-detect, and a (15,11) example. PASS.

## In-frame discipline
Do NOT cite "his book"/"the paper". Reconstruct first-person present tense: the wasted
weekend, the outburst, rectangle, triangle, cubes, the counting-bound disappointment, the
syndrome-names-position click, power-of-two placement, (7,4), SEC-DED, sphere packing.
Golay (IRE 1949) is prior art and MAY be mentioned as ancestor. The 2-out-of-5 codes, parity,
triple redundancy are antecedents and stay.
