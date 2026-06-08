Let me hold the thing I actually want in one hand before I reach for any machinery. Two strangers, on a wire that everybody can tap, and I want one of them to send the other something only the receiver can read — without their ever having met to agree on a secret. And separately, I want the sender to be able to attach something to a message that proves *he* sent it, that anyone can check but no one can forge. Diffie and Hellman have already told me, cleanly, what the missing piece is. Each user should publish an enciphering procedure E and keep a deciphering procedure D private, and the whole thing lives or dies on one property: that publishing E gives away *no feasible way* to compute D. Easy forward, infeasible backward — except trivial backward if you happen to hold a secret that was baked in when E was built. That is their trap-door one-way function, and they've shown that if I had even one of them I'd get private mail, signatures, key exchange, all of it. What they don't have is a single example. Their best public-key candidate — encipher by multiplying by an invertible matrix, decrypt by inverting it — they themselves throw out, because inverting an n×n matrix costs about n³ and the legitimate map costs n², so the attacker is slower by a factor of only n. You need the attacker slower by a factor of 10^6, 10^20, something astronomical, and a ratio of n doesn't even come close. So the real question, the only question, is: where do I find a forward map whose backward map is not merely *somewhat* harder but *exponentially* harder, unless you hold a key?

So what do I actually have on the shelf that's already known to be lopsided in exactly that way — cheap one way, apparently brutal the other? There's exactly one clean example sitting right there in the same work: exponentiation in a finite field. Fix a prime q, pick a primitive α, and Y = α^X mod q. Going forward — given X, compute Y — is nothing, repeated squaring, a couple of log₂q multiplications and you're done. Going backward — given Y, recover X — is the discrete logarithm, and the best anyone knows is on the order of √q operations, which is exponential in the number of digits. That's the asymmetry I want, and it's *number-theoretic*, not some hand-wavy "the program is confusing to read." It's solid. The trouble is what they built with it: a key *exchange*. Two people who are both online, batting α^X values back and forth, end up agreeing on α^{XY}. Lovely, but there's no published E that a stranger can grab off a shelf and use, alone, to send me mail. And there's certainly no trapdoor — discrete log mod a prime is hard for *everybody*, including me; I have no secret that makes it easy. That's the wrong shape. A one-way function with no back door is a lock with no key; nobody can open it, me included.

So let me stare at exponentiation harder, because the *form* feels right even if their use of it is wrong. What I really want isn't a one-way street, it's a one-way street with a private tunnel underneath that only I know about. Let me write down what an exponentiation cipher even looks like, forgetting the field for a second. Encrypt: C ≡ M^e (mod n). What would decrypt it? If I'm lucky, *another exponentiation*, C^d ≡ M (mod n) — because then encryption and decryption are the *same operation*, just with different exponents, and the same repeated-squaring routine does both. That symmetry is gorgeous if I can get it. So the whole game becomes: can I find a modulus n and a pair (e, d) such that raising to the e and then to the d gets me back where I started? Raising to e then to d is raising to e·d. So I need M^{e·d} ≡ M (mod n) for the messages M. The exponents *compose*: M^e then ^d is M^{ed}. I need the product ed to act like the identity exponent.

When does raising to a power act like the identity, modulo something? This is exactly the territory of Fermat and Euler. Fermat: for a prime p and M not divisible by p, M^{p−1} ≡ 1 (mod p). So modulo a prime, the exponent p−1 is "invisible" — you can stack on any multiple of p−1 for free. M^{k(p−1)+1} ≡ M·(M^{p−1})^k ≡ M. So if I worked modulo a prime p and chose ed ≡ 1 (mod p−1), then M^{ed} ≡ M and decryption works. Let me sit with that for a second, because it almost looks like I'm done… and that's exactly where it falls apart. Modulo a prime, the magic number is p−1. And p is *public* — it's the modulus, everyone sees it. So everyone can compute p−1, and everyone can solve ed ≡ 1 (mod p−1) for d given e by the same extended-Euclid I'd use. The decryption exponent is sitting in plain view. There's no secret. It's the discrete-log problem all over again: hard for nobody, because the structure controlling the exponents is exposed by the modulus itself. This is the Pohlig–Hellman setting — exponentiation mod a prime — and it's a fine *symmetric* cipher, where e is the shared secret, but it can never be public-key, because the thing that inverts e is computable by anyone who can read the modulus.

So that's the wall, and naming it precisely tells me how to climb it. The exponent arithmetic is governed by some number — call it the order of the group, the count of units — and modulo a prime that number is p−1, which the public modulus hands to everybody. I need a modulus whose "magic exponent" is *not* readable off the modulus. I need the count of invertible residues to be a secret even though the modulus is public. When is the number of units modulo n something you *can't* just read off n? Euler answers Fermat's question for composite moduli: the magic exponent is φ(n), the count of integers below n coprime to n, and M^{φ(n)} ≡ 1 (mod n) for M coprime to n. And here's the crack I was looking for — φ(n) is *not* a simple function of n that you can evaluate by looking at n. It depends on how n *factors*. If n is prime, φ(n) = n−1, trivial, public, useless to me. But if n is a product of two primes, n = p·q, then because φ is multiplicative on coprime parts, φ(n) = φ(p)·φ(q) = (p−1)(q−1). And to compute (p−1)(q−1) you need p and q. Whoever built n by multiplying p and q knows them and can compute φ(n) instantly. Whoever only sees the product n has to *factor* it first — and factoring a big number is the three-hundred-year-old problem nobody can do.

That's it. That's the trapdoor. The modulus must be *composite*, n = pq, precisely so that the exponent-arithmetic constant φ(n) is hidden behind the difficulty of factoring n. The factorization is the secret tunnel. I publish n, I keep p and q. With p and q I know φ(n) and can do the exponent arithmetic; without them you'd have to factor, and you can't.

Now let me build it carefully, because I want to be sure decryption really is the identity, not just "morally." I pick two large random primes p and q and form n = pq. I have φ(n) = (p−1)(q−1) because I know the factors. I want an encryption exponent e and a decryption exponent d with ed acting like the identity exponent modulo n — meaning, from the Euler story, ed ≡ 1 (mod φ(n)), so that ed = kφ(n) + 1 for some integer k and M^{ed} = M^{kφ(n)+1} = M·(M^{φ(n)})^k ≡ M. So the requirement on the exponents is exactly ed ≡ 1 (mod φ(n)). How do I get such a pair? I pick d first — a large random integer — and I need its inverse e modulo φ(n) to exist. An inverse modulo φ(n) exists if and only if d is coprime to φ(n); otherwise d isn't a unit in the ring of integers mod φ(n) and there's no e to be had. So the condition is gcd(d, φ(n)) = 1. Choose d coprime to φ(n) — any prime larger than both p and q does the trick, since then it can't share a factor with (p−1)(q−1) — and then run extended Euclid on φ(n) and d to produce e with ed ≡ 1 (mod φ(n)). That's cheap: the number of steps is under 2 log₂ n. (And I should keep d out of a small range — if e came out smaller than log₂ n I'd just start over with a fresh d, so that every real message actually wraps around the modulus rather than slipping through untouched.) The public key is the pair (e, n); the private key is (d, n); p, q, and φ(n) I keep or destroy.

Let me make this concrete with tiny numbers so I can watch it work. Take p = 47, q = 59, so n = 2773, and φ(n) = 46·58 = 2668. Say I pick d = 157 — and 157 is coprime to 2668, good. Run Euclid: 2668 = 157·16 + 156, then 157 = 156·1 + 1, so back-substituting, 1 = 157 − 156 = 157 − (2668 − 157·16) = 157·17 − 2668, which says 157·17 ≡ 1 (mod 2668). So e = 17. Check ed = 17·157 = 2669 = 2668 + 1 = φ(n) + 1. Exactly one more than φ(n); k = 1. Now encrypt M = 920: I want 920^17 mod 2773. By repeated squaring — 17 is 10001 in binary, so it's (((((1)²·M)²)²)²)²·M — that comes out to 948. And decryption: 948^157 mod 2773 should give back 920, and it does. The map and its inverse are both just exponentiation, both run by the same squaring routine, and they undo each other. Good.

But I've been sloppy in one spot and I have to fix it, because "it works on the example" is not "it works on every message," and a cipher that silently corrupts some messages is broken. The Euler step I leaned on — M^{φ(n)} ≡ 1 (mod n) — has a precondition: M must be coprime to n. For most messages that's true, but my message space is *all* integers from 0 to n−1, and some of those share a factor with n. If M happens to be a multiple of p (or of q), then gcd(M, n) ≠ 1, Euler's theorem doesn't apply, and my one-line proof has a hole. I can't just wave it away — I need M^{ed} ≡ M for *every* M in the range, or it isn't a permutation and signatures (where D gets applied to arbitrary M) will break too. So let me not work modulo n directly; let me work modulo p and modulo q separately, where Fermat is clean, and then glue the two back together. This is the right way to handle a composite modulus anyway: a congruence mod pq is exactly a pair of congruences mod p and mod q, by the Chinese Remainder Theorem, since p and q are distinct primes hence coprime.

So fix M and look at M^{ed} − M modulo p first. Recall ed = kφ(n) + 1 = k(p−1)(q−1) + 1. Two cases, and I have to do both honestly. If p does *not* divide M, Fermat gives M^{p−1} ≡ 1 (mod p), and then M^{ed} = M·(M^{p−1})^{k(q−1)} ≡ M·1^{k(q−1)} = M (mod p). If p *does* divide M, then M ≡ 0 (mod p), so M^{ed} ≡ 0 ≡ M (mod p) — both sides are zero, the congruence holds trivially. The case that broke Euler — M divisible by p — is precisely the case where both sides vanish mod p, so it's fine; I just had to look at it mod p instead of mod n. Now do q with the same care. If q does *not* divide M, Fermat gives M^{q−1} ≡ 1 (mod q), and M^{ed} = M·(M^{q−1})^{k(p−1)} ≡ M (mod q). If q *does* divide M, then M ≡ 0 (mod q), so again M^{ed} ≡ 0 ≡ M (mod q). Therefore for every M whatsoever, M^{ed} agrees with M modulo p and modulo q. Now CRT: p and q are coprime, so agreement modulo both primes means agreement modulo pq = n. Therefore M^{ed} ≡ M (mod n) for *all* M in [0, n), with no coprimality assumption left over. Encryption followed by decryption is the identity on the whole message space. And since the map is onto the whole space and undone by its inverse, it's a bijection — a genuine permutation of {0, …, n−1}. That last fact matters: it means every ciphertext is itself a legal message and vice versa, which is exactly the extra property I need for signatures, where I'll want to apply the decryption map to a message that was never encrypted.

Now I have to test how many apparent shortcuts collapse back to the trap door. The cleanest case: if an attacker can factor n, he gets p and q, hence φ(n) = (p−1)(q−1), hence d = e^{-1} mod φ(n) by the same Euclid I used, and he's me. So factoring breaks it — no surprise, that's the trapdoor working in reverse for whoever can open it. The real content is the converse direction: are the *other* shortcuts any easier than factoring? Suppose he skips factoring and tries to compute φ(n) directly. Then I claim he's gained nothing, because knowing φ(n) lets him factor n. Watch: φ(n) = n − (p+q) + 1, so from n and φ(n) he gets s = p + q. And (p − q)² = (p + q)² − 4pq = s² − 4n, so the integer square root gives r = |p−q|. Then the factors are (s+r)/2 and (s−r)/2, up to order. So computing φ(n) is no easier than factoring — they stand or fall together. (And this is exactly *why n has to be composite*: if n were prime, φ(n) = n−1 would be free, and there'd be nothing to hide.)

Suppose instead he goes straight for d, never bothering with φ(n). Same conclusion, by a slightly different route: once he has d, he has ed − 1, which is a *multiple* of φ(n) (since ed ≡ 1 mod φ(n) means φ(n) divides ed − 1). And factoring n from any multiple of φ(n) is a solved problem — Miller showed how. So recovering d also collapses to factoring. I should be careful that he can't sneak in some *other* exponent d′ that decrypts without equalling my d. Any exponent that works for all messages gives ed′ ≡ 1 modulo lcm(p−1, q−1), so ed′ − 1 is a multiple of that least-common-multiple version of the same hidden exponent structure; the same factoring route applies with lcm(p−1, q−1) in place of φ(n). So that escape is closed too. Three of the obvious attacks — factor n, find φ(n), find a decrypting exponent — are tied directly to factoring. There's a fourth I can't tie down: maybe someone computes e-th roots mod n by some entirely different trick that never touches the factorization. I can't prove that's as hard as factoring; "extract e-th roots mod a composite" isn't a classically named hard problem the way factoring is. I'll have to be honest that this is a conjecture — the security rests on factoring being hard *and* on there being no clever root-extraction shortcut — and the only honest test is to publish it and dare people to break it. But the main obvious doors all open only onto factoring, and factoring 200-digit numbers is, by every estimate I can find, out past 10^9 years of computer time. That's the margin I wanted, and it's the margin the matrix scheme never had.

Then the signature comes essentially for free, and it's the reason I worked so hard to make the thing a *permutation* and not just a function. Because E and D are inverse permutations, they commute: D(E(M)) = M and also E(D(M)) = M. So to sign a message M, Bob applies his *private* decryption exponent to it: S = M^{d_B} mod n_B. Deciphering an un-enciphered message "makes sense" precisely because every M is the ciphertext of *something* under the permutation — that's property (d). Anyone can then verify by applying Bob's *public* encryption exponent: (S)^{e_B} = M^{d_B e_B} ≡ M (mod n_B). For a specified message M, producing such an S without Bob's private exponent is exactly the public inversion problem again: find an e_B-th root of M modulo n_B. So the same hardness assumption that protects decryption gives the signature its force; the signed value is bound to the whole message because verification recomputes M from S. The same exponentiation machinery, run with the private exponent first, gives signatures; run with the public exponent first, gives secrecy. One construction, both of Diffie and Hellman's goals.

The only thing left is to make sure every piece is actually *computable* at the sizes I need, because an existence proof that takes a million years to set up is no cipher. Encryption and decryption: m^e mod n by repeated squaring and multiplication, at most 2 log₂ e modular multiplications, cost growing as the cube of the digit length of n — a 200-digit message goes through in seconds. Inverting e from d: extended Euclid, under 2 log₂ n steps. The one genuinely number-theoretic setup cost is *finding* the primes. I need two random ~100-digit primes, and here the asymmetry that protects me also helps me: *testing* a number for primality is far cheaper than factoring it. I generate random odd 100-digit candidates and test each with the Solovay–Strassen probabilistic test — pick a random a, check gcd(a, b) = 1 and that the Jacobi symbol J(a, b) ≡ a^{(b−1)/2} (mod b); a prime always passes, a composite fails for at least half of the a's, so 100 independent rounds leave a chance of error below 2^{-100}. By the prime number theorem a random 100-digit odd number is prime often enough that I'll hit one after testing on the order of 100-some candidates. To be safe against special-purpose factoring tricks I'll insist p and q differ in length by a few digits, that p−1 and q−1 each carry a large prime factor, and that gcd(p−1, q−1) is small — all cheap to arrange. And then I throw the primes away, or guard them as the trapdoor, and publish only (e, n).

So the whole chain, start to finish: I needed a function that's easy to compute, infeasible to invert from public data, yet trivial to invert with a secret — Diffie and Hellman's trap-door one-way function, which nobody had exhibited. Exponentiation gave me the easy-forward/hard-backward asymmetry, but modulo a prime the inverting constant p−1 is public, so there's no secret — the wall. Moving to a composite modulus n = pq hides that constant, φ(n) = (p−1)(q−1), behind the difficulty of factoring n: whoever knows the factors knows φ(n) and can invert; whoever doesn't must factor. Choosing e coprime to φ(n) and d = e^{-1} mod φ(n) makes ed = kφ(n)+1, and then m^{ed} ≡ m (mod n) — proven for *every* message, including those sharing a factor with n, by checking mod p and mod q with Fermat and gluing by CRT — so encryption m ↦ m^e mod n and decryption c ↦ c^d mod n are inverse permutations. The factorization of n is the trap door; factoring n, computing φ(n), or recovering a decrypting exponent are tied directly together; and the permutation structure hands me signatures by running the same map with the private exponent first. The algebra lets me choose either exponent first. For implementation I choose the public exponent e first, require gcd(e, φ(n)) = 1, and compute d = e^{-1} mod φ(n), because that leaves the public operation fast while preserving the same inverse relation.

```python
# RSA: a concrete trap-door one-way permutation.
# Forward map m -> m^e mod n is easy for anyone; the inverse c -> c^d mod n is
# easy ONLY for the holder of the factorization of n (the trap door).

import secrets

def modexp(base, exp, mod):
    # m^e mod n by repeated squaring & multiply: <= 2*log2(exp) modular mults.
    result, base = 1, base % mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def egcd(a, b):                      # extended Euclid: a*x + b*y = gcd(a,b)
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):                    # inverse of a mod m; needs gcd(a, m) == 1
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("not invertible")
    return x % m

def jacobi(a, n):
    # Jacobi symbol J(a, n), for odd n > 0.
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0

def is_probable_prime(b, rounds=100):
    # Solovay-Strassen style test: a composite survives one round with probability <= 1/2.
    if b in (2, 3):
        return True
    if b < 2 or b % 2 == 0:
        return False
    for _ in range(rounds):
        a = 2 + secrets.randbelow(b - 3)
        if egcd(a, b)[0] != 1:
            return False
        if modexp(a, (b - 1) // 2, b) != jacobi(a, b) % b:
            return False
    return True

def random_prime(num_bits, rounds=100):
    # Random odd candidates of the requested size; primality is much cheaper than factoring.
    if num_bits < 2:
        raise ValueError("num_bits must be at least 2")
    while True:
        candidate = secrets.randbits(num_bits)
        candidate |= (1 << (num_bits - 1)) | 1
        if is_probable_prime(candidate, rounds):
            return candidate

def generate_keys(prime_bits, public_exponent=65537, rounds=100):
    # Generate p and q, then keep resampling until e is invertible modulo phi(n).
    if public_exponent <= 1:
        raise ValueError("public exponent must be greater than 1")
    while True:
        p = random_prime(prime_bits, rounds)
        q = random_prime(prime_bits, rounds)
        if p == q:
            continue
        n = p * q
        phi = (p - 1) * (q - 1)      # = phi(n); computable ONLY with the factors
        e = public_exponent
        if e < phi and egcd(e, phi)[0] == 1:
            d = modinv(e, phi)       # d = e^{-1} mod phi  =>  e*d = k*phi + 1
            return (e, n), (d, n)    # public key (e, n); private key (d, n)

def require_residue(x, n, name):
    if not (0 <= x < n):
        raise ValueError(f"{name} must satisfy 0 <= {name} < n")

def encrypt(m, public_key):          # easy forward direction, anyone can run it
    e, n = public_key
    require_residue(m, n, "message")
    return modexp(m, e, n)           # c = m^e mod n

def decrypt(c, private_key):         # inverse, easy ONLY with the trap door (d)
    d, n = private_key
    require_residue(c, n, "ciphertext")
    return modexp(c, d, n)           # m = c^d mod n; m^{ed} = m by Euler/Fermat+CRT

def sign(m, private_key):            # signature dual: private exponent first
    d, n = private_key
    require_residue(m, n, "message")
    return modexp(m, d, n)           # S = m^d mod n

def verify(m, s, public_key):        # anyone verifies with the public exponent
    e, n = public_key
    require_residue(m, n, "message")
    require_residue(s, n, "signature")
    return modexp(s, e, n) == m      # (m^d)^e = m^{ed} = m mod n
```
