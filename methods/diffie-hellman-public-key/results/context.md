# Context: secrecy and authentication without a pre-shared secret

## Research question

Two people who have never met want to communicate privately over a channel they must assume is being recorded — a telephone line, a radio link, a packet network. Every cryptographic system in use answers this the same way: the two parties first agree on a secret key `K`, and that key must reach both of them over some *separate, secure* channel — a courier, registered mail, a trusted operator — before any private traffic can flow. The secure channel is slow and expensive; it exists only to carry the key, never the message itself.

This is tolerable for a pair of correspondents who can plan ahead. It collapses for the world that cheap digital hardware is bringing into being: remote terminals, cash dispensers, electronic mail, business-to-business teleprocessing, where the natural pattern is two strangers who want to transact *now*, with no prior contact and no time to wait for a courier. In a network of `n` subscribers there are `n(n-1)/2` pairs who might at some point wish to talk privately — about five thousand for a hundred users, half a million for a thousand, and on the order of `2×10^16` for a system the size of the telephone network. Pre-distributing a distinct key to every pair is unthinkable; postponing each new conversation until a key can be physically carried defeats the point of an instantaneous network.

A solution would have to let two parties end up holding a common secret — or, more ambitiously, let each party encrypt to the other — using *only* messages sent in the clear over the public channel, with an eavesdropper who hears every bit still unable to recover the secret at any feasible cost. A second, seemingly separate goal sits alongside it: a purely digital message can be copied bit-for-bit, so how can such a message carry a *signature* — something the recipient (and any later third party, such as a judge) can verify as coming from one specific person, yet that no one else, not even the recipient, could have produced?

## Background

**Secrecy rests on a shared key, and on computational cost.** A conventional (symmetric) cipher is a family `{S_K}` of invertible transformations indexed by a key `K`. The transmitter sends `C = S_K(P)`; the receiver, knowing the same `K`, recovers `P = S_K^{-1}(C)`. Kerckhoffs's principle (codified 1883, and recounted by Kahn's *The Codebreakers*) holds that the family `{S_K}` itself should be assumed public — standard, even — and that *all* the secrecy must live in `K`. So the entire security of the scheme reduces to: can an opponent who knows `{S_K}` and sees `C` (and perhaps some matched plaintext) recover `P` or `K`?

**Shannon's dividing line.** Shannon's 1949 theory of secrecy systems split this question in two. A system is *unconditionally secure* if it resists an opponent with unlimited computation; it is *computationally secure* if it merely makes cryptanalysis too costly to be worth it. Shannon proved that perfect secrecy — the intercepted cryptogram leaving the a-posteriori message probabilities exactly equal to their a-priori values — is attainable, but at a price: the key must carry at least as much entropy as the message, `H(K) ≥ H(M)`. If the message is generated at some rate, key must be generated at least as fast. The one-time pad realizes this and is provably unbreakable, but its key is as long as everything you will ever send, which is why almost every practical system instead settles for *computational* security with a key of a few hundred bits. Crucially, the moment you accept short keys, security becomes a statement about *the cost of computation*, not about information — and that is the door through which the theory of algorithms enters.

**The cost of computation is now a science.** Two young disciplines had begun to classify exactly that cost: computational complexity theory and the analysis of algorithms. A problem is in `P` if a deterministic Turing machine solves it in time polynomial in the input length — loosely, "easy." A problem is in `NP` if a solution can be *verified* in polynomial time (equivalently, found in polynomial time by a machine with unlimited parallelism). Karp (1972) had exhibited 21 `NP`-complete problems — including the travelling-salesman, satisfiability, graph-colouring, and *knapsack* problems — with the property that if any one is in `P` then all of `NP` is; it is widely believed `NP` is strictly larger than `P`, so these problems are believed to have no polynomial algorithm. This gives, for the first time, a principled vocabulary for "easy to do, infeasible to undo." A caveat rides along: `NP`-completeness is a *worst-case* statement, whereas a cipher needs a problem that is hard on *typical* randomly chosen instances.

**One-way functions already exist, quietly, in login systems.** A function `f` that is easy to compute but infeasible to invert was already in service. Needham's password scheme (described in Wilkes, *Time-Sharing Computer Systems*, 1972) stores not each user's password `PW` but its image `f(PW)`; at login the system computes `f` of the proffered string and compares. An intruder who steals the whole password table gains nothing usable, because the entries are not passwords and the table is full of values that the login routine itself will reject. Evans, Kantrowitz, and Weiss (1974) and Purdy (1974) had recently given constructions for such functions — Purdy proposing sparse, very-high-degree polynomials over finite fields, whose roots are far costlier to find than the polynomial is to evaluate.

**Challenge-and-response identification.** Since the 1950s, Identification-Friend-or-Foe systems (Feistel's group at the Air Force Cambridge Research Center, and later the MK XII IFF) had authenticated aircraft by having a radar issue a randomly chosen *challenge* and accept the plane only if it returns the correctly encrypted response. Because the challenge never repeats, a recorded legitimate response is worthless on the next encounter — protection against an eavesdropper who can replay. But it offers no protection against an opponent who captures the radar and learns its key; he can then fool any radar keyed the same.

These two existing mechanisms — challenge/response and the login one-way function — defend against *different* threats. The challenge/response defeats an eavesdropper but falls to capture of the secret; the one-way function survives capture of the password table (analogous to capturing the radar) but falls to anyone who intercepts the (unchanging) login message. Each guards one flank and exposes the other.

**Authentication today cannot settle a dispute.** Appending an encrypted authenticator (a function of message, secret key, and the date/time) to a message lets the receiver detect third-party forgery and tampering, because the shared key is needed to produce a valid authenticator. But the key is *shared*: the receiver holds it too, so he could have manufactured the authenticator himself. Such a system therefore cannot adjudicate a dispute between transmitter and receiver — exactly the case a written signature is meant to cover (the stockbroker who forges a client's order, the client who disavows an order that lost money). Lamport had observed that a one-way function can give a partial answer for `N` message bits: publish images `y_i, y_i'` of `2N` secret random vectors `x_i, x_i'`, and reveal `x_i` or `x_i'` per bit; the receiver checks each revealed pre-image against its stored image and cannot forge a changed bit. It works but expands the message roughly a hundred-fold.

## Baselines

**The one-time pad (Vernam; Shannon's analysis).** Combine the plaintext with a fresh random key of equal length, modulo 2. Provably unconditionally secure — the cryptogram is statistically independent of the message. *Gap:* the key is as long as the entire correspondence and must itself be pre-shared over a secure channel, which is the very key-distribution problem at issue, only worse.

**Conventional short-key ciphers (computationally secure block/stream ciphers).** A few hundred bits of key, security resting on the cost of cryptanalysis (ciphertext-only, known-plaintext, chosen-plaintext attacks define the threat ladder). Cheap and fast. *Gap:* still require the secret key to be agreed in advance over a secure channel; in an `n`-user network this is the `n(n-1)/2`-key blow-up. They also cannot produce a signature that binds the sender against the receiver, since both share the key.

**Key distribution centre (KDC).** Each subscriber pre-shares one key with a central trusted node; to talk to another subscriber, a user asks the centre, which hands out a session key encrypted under the two parties' master keys. This is the conventional answer to the `n^2` blow-up — one key per user instead of one per pair. *Gap:* it reintroduces a *trusted third party* into every conversation. The centre learns (or can learn) every session key; compromising it — by burglary, by subpoena — compromises past, present, and future traffic. The whole appeal of cryptography was that it required trust in no party outside the conversation; the KDC throws that away.

**Merkle's puzzles (secure communication over an insecure channel).** A construction that *drops* Shannon's assumption that the eavesdropper cannot read the key channel — it grants the eavesdropper `Z` perfect knowledge of every transmission and still agrees a key. The sender `X` manufactures `N` puzzles. Each puzzle is a cryptogram under a *deliberately weakened* key space (say 20 bits), with a recognizable constant embedded so that a correct trial decryption is detectable; inside each puzzle sit a puzzle-ID and a puzzle-key. `X` sends all `N` puzzles in the clear. The receiver `Y` picks *one* puzzle at random and brute-forces it — `O(N)` work for that single puzzle — then sends back its ID (not the key). `X`, who recorded which key went with which ID, now shares that key with `Y`. The eavesdropper saw the ID but not which of the `N` puzzles it came from in a way that yields the key; he must crack puzzles at random until he hits `Y`'s chosen one, on average `N/2` of them at `O(N)` each, i.e. `O(N^2)` work. So legitimate effort is `O(N)` and the attacker's is `O(N^2)`: a genuine advantage from public messages alone. *Gap:* the advantage is only *quadratic*. The honest parties must also *transmit* all `N` puzzles, so the cost is bandwidth as much as computation; practical limits cap the work ratio at around `10^4`, "too small for most applications." A determined opponent with comparable hardware is only a square-root away.

## Evaluation settings

The natural yardstick is the *work factor*: the ratio of the cryptanalyst's effort to the legitimate users' effort, measured under the best known attack, with both costs in concrete units (operations, gate delays, or dollars at then-current prices — enciphering on the order of a cent, an opponent's budget on the order of millions of dollars). A scheme is judged "computationally infeasible" to break when that effort is finite but impossibly large, with `10^100` instructions serving as a representative wall. The threat model is the standard ladder — ciphertext-only, known-plaintext, chosen-plaintext — with the strongest (chosen-plaintext, the IFF-style "submit text and observe the cryptogram") taken as the certification standard, and the system `{S_K}` itself assumed known to the opponent. Bit lengths are the scaling axis: for a candidate built on a hard number-theoretic problem of size roughly `2^b`, one tabulates legitimate cost (polynomial in `b`) against attack cost (the best known algorithm's growth in `b`) across `b = 100, 200, ..., 1000`, and asks whether the gap is merely large or *exponential* in `b`. Because no scheme whose public data uniquely determines its secret can be unconditionally secure, every candidate here is certified only relative to the *best presently known* algorithm for its underlying problem — so a standing question for each is whether a faster algorithm for that problem exists.

## Code framework

The starting scaffold is the symmetric-cipher harness plus an empty public-channel key-establishment interface. Modular arithmetic over a finite field is available as a primitive; fast exponentiation by repeated squaring is a known algorithm.

```python
# ---- primitives that already exist ----

def modexp(base, exp, mod):
    # repeated-squaring exponentiation: O(log exp) multiplications mod `mod`
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def random_int(low, high):
    # draw from a true hardware RNG (e.g. a noisy diode)
    ...

# conventional symmetric cipher: both parties already share K somehow
def encrypt_symmetric(K, plaintext): ...
def decrypt_symmetric(K, ciphertext): ...


# ---- public-channel key-establishment interface ----
# Goal: two parties holding NO prior shared secret end up with a common K,
# (or with the ability to encrypt to each other) using only public messages.

class PublicParameters:
    # TODO: whatever public, agreed-in-advance, non-secret data the scheme needs
    pass

def generate_keypair(params):
    # TODO: each user locally produces a (public, secret) pair from fresh randomness;
    #       the public part may be broadcast / listed in a directory,
    #       the secret part never leaves the terminal.
    raise NotImplementedError

def derive_shared_secret(my_secret, their_public, params):
    # TODO: combine one's own secret with the other party's PUBLIC value
    #       so that both parties compute the SAME secret,
    #       yet an eavesdropper holding only the two public values cannot.
    raise NotImplementedError

# The hardness this must rest on:
def one_way(x, params):
    # TODO: a function easy to evaluate forward, computationally infeasible to invert,
    #       with enough algebraic structure that two such outputs can be combined.
    raise NotImplementedError
```
