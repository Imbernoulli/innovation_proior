It is the second weekend in a row that the machine has thrown my work away, and I am furious. Here is the situation: I am low man on the totem pole, so I get the Model 5 relay computer only from Friday at five until Monday at eight. I load the input tape with a long string of problems, promise the people back at Murray Hill their answers on Tuesday, and go home. The machine runs unattended. It has 2-out-of-5 checking built in — exactly two of every five relays are supposed to be up, and if the count is wrong the machine knows something went wrong. When that happens it retries the step a few times, and if the trouble persists it gives up on that problem and moves to the next. Last Friday night, just after I left, a relay failed and the whole run collapsed; I got essentially nothing Monday. I apologized, reloaded, promised Tuesday. And it happened again. Two weekends gone.

And here is the thing that has me grinding my teeth: the machine *knew*. Every single time, the checking circuit caught the error in the act. It detected it. It just couldn't do anything except stop and sulk. So I say — and the actual words were stronger — damn it, if the machine can tell that there's an error, why can't it tell *where* the error is, and just change that bit to the opposite state and keep going?

Stated that baldly it sounds almost trivial, and that is exactly why it is worth chasing: detection and correction feel like they should be the same kind of thing, yet everything in service only detects. The 2-out-of-5 codes detect. The 3-out-of-7 detects. A parity bit detects. None of them locates. I want to understand whether that gap is fundamental or whether I've just never looked hard enough.

What do I actually have to work with? I know one cheap way to *correct*: build three copies of the machine, run them together, and let the majority vote outvote a single failure. That works. It also more than triples the equipment, and I find the cost obscene for the job. There has to be something better, something that pays in slow-growing redundancy rather than a flat factor of three. The other thing I know, and know to the bone because I just spent a while studying it, is the parity check. Append a bit so the number of 1s is even. One flip anywhere makes the count odd, and the check fails. That detects any single error. And here is the property of it I keep turning over: a parity check does not have to cover the whole symbol. It can be taken over *selected positions only*. That smells important. A single global parity bit tells me whether *an* error happened but says nothing about *which* bit — because every position is treated the same, so every position produces the identical "parity is now odd" signal. The signal carries no address.

So if one parity check over everything gives me one bit of "yes/no" with no address in it, what if I use *several* parity checks, each over a *different* subset of positions? Then a single flipped bit will fail some checks and not others, depending on which subsets happen to contain it. The *pattern* of which checks fail is no longer one yes/no — it's several, and different bits sit in different combinations of subsets, so different bits could produce different patterns. The address I was missing might live in the pattern.

Let me make that concrete in the simplest geometry I can draw. Lay the message bits out in a rectangle, rows and columns. Put a parity check on each row and a parity check on each column. Now flip one interior bit. Its row's parity goes bad; its column's parity goes bad; every other row and column stays good. So exactly one row check fails and exactly one column check fails, and the failing row number together with the failing column number are the *coordinates* of the broken bit. I don't just know an error happened — I know precisely where, and to fix it I flip that one bit. (If I'm careful and put a parity bit in the corner that checks the row-parities and the column-parities consistently with even parity everywhere, even an error in one of the check bits themselves shows up as a coordinate, so the scheme covers its own checking apparatus.) This is the first time the "several overlapping checks" idea pays off: two checks fail, and their identities spell the location.

Now I care about cost. For a rectangle that's r rows by c columns I'm protecting about r·c message bits with r + c check bits. To carry a fixed amount of message at the least redundancy I want r·c large with r + c small — and anyone who's done a little calculus knows that for a fixed product r·c the sum r + c is smallest when the rectangle is square, r ≈ c ≈ √(message). So a square is the best rectangle, costing on the order of 2√m checks for m message bits, and bigger blocks are more efficient — though if I make the block enormous the chance of *two* errors in it grows, and two errors break this scheme (if the two bad bits aren't in the same row or column I get two failing rows and two failing columns and can't tell which diagonal pairing caused them; if they share a row I lose that row's information entirely). So there's an engineering ceiling on block size, but within it, square rectangles, 2√m checks. That's already wildly better than tripling. But is it best?

A few weeks pass. I'm riding the company mail car through north Jersey at an unholy hour to catch the machine, and there is nothing whatsoever to look at, so out of habit I'm replaying my successes in my head to keep the style automatic — and I'm turning the rectangular code over once more. And suddenly, with no reason I can reconstruct: I don't need the full rectangle. Take a triangle. Put the parity checks along the diagonal, and let each diagonal check be responsible for *both* the row it sits in and the column it sits in. Same locating power — a single error still lights up a row-check and a column-check — but I've folded the two arms together and spend fewer check bits for the same message. Better redundancy than the rectangle.

That little win is the dangerous moment, because it tells me my "square is best" was not the end of the road, and now I distrust every apparent optimum — including this one. So push it. The rectangle was two-dimensional: two coordinates, two arms of checks. Why two? Put the message bits in a *cube*. Now run parity checks across whole planes — one family of planes per axis — with a check bit on each axis. A single error lights up one plane-check per axis, and the three failing planes give me three coordinates: the location in 3-space. The accounting: for an n×n×n cube I'm protecting n³ message bits and I need about 3n − 2 parity checks to name the three coordinates. Compare that to the rectangle, which would need about 2·n^{3/2} checks to protect the same n³ bits. The cube is cheaper. The pattern is unmistakable — each extra dimension lets me address more message bits with proportionally fewer checks, because the coordinates multiply while the check-arms only add.

So I keep climbing dimensions. Four dimensions is better than three; I don't have to physically arrange the bits in 4-space, I only have to *wire* the checks as if they were. Five better than four. And then the climb collapses to its limit, which I can see is the extreme case: make every axis have just *two* points. A 2×2×2×…×2 cube. Then a "coordinate" along each axis is a single bit — am I on the 0-side or the 1-side of that axis — and q binary axes give me 2^q positions to address. The old coordinate construction would spend about q+1 parity checks to do the job, so the redundancy is now growing like the *logarithm* of the message length instead of its square root. That feels like the answer. It feels best.

But I have just been burned — I called the square "best" and the triangle proved me wrong — so I refuse to be smug about "feels best." Can I *prove* the 2×2×…×2 cube is optimal? Let me try the obvious counting argument. With q+1 parity checks, the result of running them all is a string of q+1 bits. As a binary number that string can take 2^{q+1} distinct values. So I have 2^{q+1} possible "verdicts" available. How many distinct things do I actually need to distinguish? I need to name each possible error location, and there are 2^q vertices in the cube, plus one more verdict for "no error at all." That's 2^q + 1 things. And 2^{q+1} is almost twice 2^q + 1. I have roughly twice as many verdicts as I have situations to name. I'm wasting almost a whole bit. The thing I thought was optimal is leaving a factor of nearly two on the table — so it is *not* optimal, and there is room for a tighter code. (At which point the mail car arrives, I have to sign in and go to a meeting, and the idea has to sit.)

When I get back to it days later, that wasted factor of two pulls me to look at the counting from the other side. Forget arranging anything in a cube. I have a codeword with positions in it, some carrying message and some carrying checks. When I run the parity checks at the receiving end, each check comes out "satisfied" or "violated." Write a 0 for each satisfied check and a 1 for each violated one, in order, as a binary number. Call that number the syndrome. I want this number to *be the answer*: I want the syndrome to come out as the binary representation of the *position* of the bad bit, with the all-zero syndrome meaning "no error." (All-zero for "good," incidentally, is a happier convention than all-ones, because testing a register for zero is the easy test on most machines.)

If I demand that, the whole design falls out, because it tells me exactly which check must cover which position. The syndrome's rightmost bit is parity check #1; for the syndrome to read out the position, check #1 must come out 1 exactly when the error is at a position whose binary representation has a 1 in its rightmost place. So check #1 must cover all and only the positions 1, 3, 5, 7, 9, … — every position that is odd, i.e. has bit 0 set. Check #2 is the next bit of the syndrome, so it must cover every position with a 1 in its second binary place: 2, 3, 6, 7, 10, 11, … . Check #3 covers every position with a 1 in the third place: 4, 5, 6, 7, 12, 13, 14, 15, … . And so on, one parity check per bit-position of the address. Now flip the single bit at position e. A given check fails exactly when e is one of the positions it covers — that is, exactly when the corresponding bit of e is 1. So the set of failing checks is precisely the set of 1-bits of e, and the syndrome, read as a binary number, *is* e. The pattern of failed checks doesn't merely hint at the location; it spells it out in binary. There's the address I was hunting for from the very first parity check.

Two things to nail down: where do the check bits themselves live, and how many do I need. For placement, watch what happens if I put the check bits at the positions that are *powers of two* — positions 1, 2, 4, 8, …. Position 1 = binary 1 is covered by check #1 and no other; position 2 = binary 10 is covered by check #2 and no other; position 4 = binary 100 by check #3 alone; in general position 2^i is the unique position whose only set bit is bit i, so it belongs to check #i and to no other check. That means check bit i is the *only* check bit involved in parity check #i. So I can set each check bit independently: compute the parity that check #i demands over its message positions, and drop the answer into position 2^i to make that check come out even. No check bit ever sits inside another check's group, so setting one never disturbs another. Encoding becomes trivial, and the checks are mutually independent. If I'd scattered the check bits anywhere else, setting them would be a tangle of simultaneous constraints. Powers of two untangle it completely. Everything that isn't a power of two carries message.

For the count: I have r check bits, the syndrome is r bits, so it can name 2^r different verdicts. I need it to name every position of the n-bit codeword plus the "no error" verdict: that's n + 1 things, and n = m + r. So I need 2^r ≥ m + r + 1. That is the condition, and now I see *why* the 2×2×…×2 cube was off by a factor of two — it effectively spent q+1 checks where this construction spends only r ≈ log₂ n, because it was naming far fewer things than its syndrome could distinguish. Here there's no waste: when 2^r = m + r + 1 exactly, every one of the 2^r syndromes is doing real work — the all-zero one for "clean," the other 2^r − 1 each pointing at one of the n = 2^r − 1 positions. Tight. That equality picks out the efficient sizes: r = 3 gives 2^3 = 8 = m + 3 + 1, so m = 4, n = 7; r = 4 gives m = 11, n = 15; then 31, 26; and so on.

Let me actually build the smallest interesting one, the case r = 3, to make sure it isn't a fantasy. Seven positions. Check bits go at positions 1, 2, 4; message bits at 3, 5, 6, 7. Check #1 uses positions 1,3,5,7; check #2 uses 2,3,6,7; check #3 uses 4,5,6,7. If I call the data bits d1,d2,d3,d4 in positions 3,5,6,7, and the check bits p1,p2,p4 in positions 1,2,4, even parity forces p1 = d1 ⊕ d2 ⊕ d4, p2 = d1 ⊕ d3 ⊕ d4, and p4 = d2 ⊕ d3 ⊕ d4. Take a message, say I want bits 1,0,0,1 sitting in message positions 3,5,6,7. Those equations give p1=0, p2=0, p4=1, so the encoded word is 0,0,1,1,0,0,1. Now I deliberately damage it: flip the bit in position 6. At the receiving end I run the three checks. Check #1 covers 1,3,5,7 — position 6 isn't in it, so it passes: write 0. Check #2 covers 2,3,6,7 — position 6 *is* in it, it fails: write 1. Check #3 covers 4,5,6,7 — position 6 is in it, fails: write 1. Reading the syndrome from check #3 down to check #1 gives 110 in binary, which is 6. The syndrome literally handed me the number 6. I flip position 6 back, throw away the check positions 1, 2, 4, and recover 1,0,0,1. It works, and it feels like magic until you see why it can't fail to work.

It can't fail to work, and the reason is that the legal codewords — the messages with their checks correctly set — are exactly the bit strings on which all the parity checks come out even. Parity is addition mod 2, and the checks are linear in the bits, so the set of legal codewords is closed under bitwise XOR: the XOR of two legal codewords is legal. A clean codeword produces the all-zero syndrome by construction. Now suppose one bit flips: the received word is (a legal codeword) XOR (a single-1 error vector at position e). Because the checks are linear, the syndrome of a XOR is the XOR of the syndromes — and the legal codeword contributes zero — so the syndrome of the received word equals the syndrome of the bare error vector, which is e. The message I happened to send has *no effect* on the syndrome; the parity checks concentrate entirely on the error and ignore the message. That's why a single lookup-and-flip corrects every message identically.

I should notice, before I get attached to the exact arrangement, that nothing essential depends on it. If both ends agree to permute the columns, the code is the same code wearing different clothes; if I complement a fixed position at both ends, likewise. So "check bits at 1, 2, 4, 8" is just a convenient arrangement — in practice I might gather all the check bits at the tail of the message instead of interleaving them — and none of it changes the correcting power.

Now, what about *two* errors? My distance-3 single-error code will, faced with two flips, happily "correct" — it'll find some syndrome, point at some position, and flip a perfectly good bit, turning two errors into three. I'd rather, when I can't fix it, at least *know* I can't. The cheapest possible patch: add one more parity bit, an overall parity check over the entire codeword. Walk the cases. No errors: every original check passes and the overall parity is even — syndrome 0, overall good. One error in one of the original positions: the overall parity flips to odd, and the original syndrome points at that position; correct it. One error in the new overall-parity position: the overall parity is odd, but the original syndrome is 0; there is no message bit to flip, only the new check bit is wrong. Two errors: each one flips the overall parity, so two flips leave it *even* again — overall parity looks fine — yet the original checks are disturbed and the syndrome comes out nonzero. So the tell is: original syndrome says "something's wrong" but the overall parity says "even number of errors" — that combination can only be a double error under the two-error model, and I refuse to correct, I flag it. One extra bit buys single-error-correction *plus* double-error-detection. (The redundancy on a tiny 4-bit message with now 4 check bits is poor, but the number of check bits grows only like the log of the message, so for any reasonably long block this is cheap; too long and you risk an uncorrectable double, too short and the redundancy stings — an engineering balance.)

I still don't know *how good* these codes can possibly be — whether something tighter exists — and to get at that I go back to the picture of a length-n bit string as a vertex of the unit n-dimensional cube. A single error moves a codeword along one edge to an adjacent vertex; d errors move it d edges. So define the distance between two strings as the number of positions in which they differ — which is the same as the least number of cube edges between them. That's a genuine metric: it's non-negative, zero only for identical strings, symmetric, and it obeys the triangle inequality (to get from x to z you can't differ in fewer places than the direct count, and routing through y can only differ in more). Now a code is just a chosen subset of vertices, and its power is governed entirely by the *minimum distance* between chosen vertices. If every two codewords are at least distance 2 apart, a single error lands on a non-codeword, so it's *detectable*. If every two are at least distance 3 apart, then a single error from any codeword lands strictly closer to that codeword than to any other — so I can *correct* it by snapping to the nearest codeword. Distance 4 gives single-error-correction with double-error-detection (which is exactly what the overall parity bit bought me: it lifted the minimum distance from 3 to 4). Distance 5 corrects two errors. In general minimum distance 2k+1 corrects k errors, and 2k+2 corrects k and detects k+1.

And this same picture hands me the optimality bound and shows my counting condition was no accident. Around each codeword draw a sphere of radius 1 — the codeword plus all n strings one flip away, so 1 + n points. For single-error correction those spheres must not overlap (an overlap is a string within one flip of two codewords — undecodable). The whole space has 2^n vertices, so the number of disjoint radius-1 spheres, hence the number of codewords, is at most 2^n / (n + 1). With m message bits I have 2^m codewords, so 2^m ≤ 2^n/(n+1), i.e. 2^r ≥ n + 1 = m + r + 1 — the very inequality I derived from the syndrome count. In the usual (n,k) notation, where k is the number of message bits, this is 2^(n-k) ≥ n+1. When it holds with equality the spheres tile the cube perfectly with no vertex left over, which is why those sizes — (7,4), (15,11), (31,26) — are exactly the tight, minimum-redundancy ones. The sphere-packing view and the syndrome view are the same fact seen twice.

So the path closes: a detected-but-uncorrected error wasting my weekends made me ask why a machine that can detect can't locate; one parity check can't locate because it carries no address; several *overlapping* parity checks can, because the *pattern* of which checks fail is an address; pushing that idea — rectangle, triangle, cube, higher cubes — to its efficient limit and then demanding the syndrome literally *be* the binary position of the error forces each check to cover exactly the positions with a given bit set, forces the check bits to sit at the powers of two so they decouple, and forces the size condition 2^r ≥ m+r+1; the smallest tight case is the seven-bit code carrying four message bits, an overall parity bit upgrades it to catch double errors, and the unit-cube distance picture proves these codes pack the space as tightly as any single-error-correcting code possibly can.

Now the code, built straight off the construction — check bits at powers of two, parity check i over every position whose i-th bit is set, syndrome reads out the bad position:

```python
def _is_power_of_two(j):
    return j > 0 and j & (j - 1) == 0


def encode(data_bits, detect_double=False):
    m = len(data_bits)

    # smallest r whose r-bit syndrome can name all n positions plus "no error":
    # 2**r >= m + r + 1, with n = m + r.
    r = 0
    while (1 << r) < m + r + 1:
        r += 1
    n = m + r

    # positions 1..n (1-indexed so a position's binary form picks its checks);
    # powers of two are check positions, everything else carries the message.
    code = [0] * (n + 1)
    di = 0
    for j in range(1, n + 1):
        if not _is_power_of_two(j):
            code[j] = data_bits[di]
            di += 1

    # set check bit at position 2**i to the even parity of every position whose
    # i-th binary bit is 1 (excluding 2**i itself -- the only check it belongs to,
    # which is why the check bits decouple and encoding is trivial).
    for i in range(r):
        cpos = 1 << i
        parity = 0
        for j in range(1, n + 1):
            if j != cpos and (j >> i) & 1:
                parity ^= code[j]
        code[cpos] = parity

    codeword = code[1:]

    if detect_double:
        # one overall parity bit over the whole codeword lifts the minimum
        # distance from 3 to 4: single error corrected, double error detected.
        overall = 0
        for b in codeword:
            overall ^= b
        codeword = codeword + [overall]

    return codeword


def decode(codeword, detect_double=False):
    if detect_double:
        overall_received = codeword[-1]
        code_part = codeword[:-1]
    else:
        code_part = list(codeword)

    n = len(code_part)
    code = [0] + list(code_part)

    r = 0
    while (1 << r) < n + 1:
        r += 1

    # syndrome bit i = parity over every position whose i-th bit is set. A clean
    # codeword gives all-zero. A single flip at position e fails exactly the
    # checks whose bit is set in e, so the syndrome read as a binary number = e.
    syndrome = 0
    for i in range(r):
        parity = 0
        for j in range(1, n + 1):
            if (j >> i) & 1:
                parity ^= code[j]
        if parity:
            syndrome |= (1 << i)

    status = 'ok'

    if detect_double:
        overall_calc = 0
        for j in range(1, n + 1):
            overall_calc ^= code[j]
        overall_fail = (overall_calc ^ overall_received) != 0
        if syndrome == 0:
            # all original checks pass; an odd overall check means only the new
            # overall-parity position is bad, so the data bits are already right.
            status = 'corrected_overall_parity' if overall_fail else 'ok'
        elif overall_fail and syndrome <= n:
            # under the single-error case, the original syndrome names the bad
            # original position; flip it back.
            code[syndrome] ^= 1
            status = 'corrected'
        elif overall_fail:
            # an invalid syndrome can only come from an unpromised multi-error
            # pattern in a shortened code.
            status = 'uncorrectable'
        else:
            # overall parity intact but syndrome nonzero -> double error
            status = 'double_error'
    else:
        if 0 < syndrome <= n:
            code[syndrome] ^= 1
            status = 'corrected'
        elif syndrome != 0:
            status = 'invalid_syndrome'

    data_bits = [code[j] for j in range(1, n + 1) if not _is_power_of_two(j)]
    return data_bits, status
```
