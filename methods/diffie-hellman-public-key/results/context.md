# Context: secrecy and authentication without a pre-shared secret

## Research question

Two people who have never met want to communicate privately over a channel they must assume is being recorded — a telephone line, a radio link, a packet network. Every cryptographic system in use answers this the same way: the two parties first agree on a secret key `K`, and that key reaches both of them over some *separate, secure* channel — a courier, registered mail, a trusted operator — before any private traffic flows. The secure channel carries only the key, never the message.

Cheap digital hardware is bringing into being a new pattern of use: remote terminals, cash dispensers, electronic mail, business-to-business teleprocessing, where two strangers want to transact now, with no prior contact. In a network of `n` subscribers there are `n(n-1)/2` pairs who might at some point wish to talk privately — about five thousand for a hundred users, half a million for a thousand, and on the order of `2×10^16` for a system the size of the telephone network.

The question is whether two parties can end up holding a common secret — or, more ambitiously, each encrypt to the other — using *only* messages sent in the clear over the public channel, with an eavesdropper who hears every bit still unable to recover the secret at any feasible cost. A second, related question sits alongside it: a purely digital message can be copied bit-for-bit, so how can such a message carry a *signature* — something the recipient (and any later third party, such as a judge) can verify as coming from one specific person, yet that no one else, not even the recipient, could have produced?

## Background

**Secrecy rests on a shared key, and on computational cost.** A conventional (symmetric) cipher is a family `{S_K}` of invertible transformations indexed by a key `K`. The transmitter sends `C = S_K(P)`; the receiver, knowing the same `K`, recovers `P = S_K^{-1}(C)`. Kerckhoffs's principle (codified 1883, and recounted by Kahn's *The Codebreakers*) holds that the family `{S_K}` itself should be assumed public — standard, even — and that *all* the secrecy lives in `K`. The security of the scheme reduces to: can an opponent who knows `{S_K}` and sees `C` (and perhaps some matched plaintext) recover `P` or `K`?

**Shannon's dividing line.** Shannon's 1949 theory of secrecy systems split this question in two. A system is *unconditionally secure* if it resists an opponent with unlimited computation; it is *computationally secure* if it merely makes cryptanalysis too costly to be worth it. Shannon proved that perfect secrecy — the intercepted cryptogram leaving the a-posteriori message probabilities exactly equal to their a-priori values — is attainable, but at a price: the key must carry at least as much entropy as the message, `H(K) ≥ H(M)`. If the message is generated at some rate, key must be generated at least as fast. The one-time pad realizes this and is provably unbreakable, with a key as long as everything that will ever be sent; almost every practical system instead settles for *computational* security with a key of a few hundred bits. With short keys, security becomes a statement about *the cost of computation* rather than about information.

**The cost of computation is now a science.** Two young disciplines had begun to classify exactly that cost: computational complexity theory and the analysis of algorithms. A problem is in `P` if a deterministic Turing machine solves it in time polynomial in the input length — loosely, "easy." A problem is in `NP` if a solution can be *verified* in polynomial time (equivalently, found in polynomial time by a machine with unlimited parallelism). Karp (1972) had exhibited 21 `NP`-complete problems — including the travelling-salesman, satisfiability, graph-colouring, and *knapsack* problems — with the property that if any one is in `P` then all of `NP` is; it is widely believed `NP` is strictly larger than `P`, so these problems are believed to have no polynomial algorithm. This gives a principled vocabulary for "easy to do, infeasible to undo." `NP`-completeness is a *worst-case* statement, whereas a cipher needs a problem that is hard on *typical* randomly chosen instances.

**One-way functions already exist, quietly, in login systems.** A function `f` that is easy to compute but infeasible to invert was already in service. Needham's password scheme (described in Wilkes, *Time-Sharing Computer Systems*, 1972) stores not each user's password `PW` but its image `f(PW)`; at login the system computes `f` of the proffered string and compares. An intruder who steals the whole password table gains values that the login routine itself will reject. Evans, Kantrowitz, and Weiss (1974) and Purdy (1974) had recently given constructions for such functions — Purdy proposing sparse, very-high-degree polynomials over finite fields, whose roots are far costlier to find than the polynomial is to evaluate.

**Challenge-and-response identification.** Since the 1950s, Identification-Friend-or-Foe systems (Feistel's group at the Air Force Cambridge Research Center, and later the MK XII IFF) had authenticated aircraft by having a radar issue a randomly chosen *challenge* and accept the plane only if it returns the correctly encrypted response. Because the challenge never repeats, a recorded legitimate response is worthless on the next encounter. An opponent who captures the radar and learns its key can then fool any radar keyed the same.

The challenge/response defeats an eavesdropper, and depends on the secret staying secret; the login one-way function survives capture of the password table (analogous to capturing the radar), and the (unchanging) login message travels over the same channel an interceptor would watch.

**Authentication via a shared key.** Appending an encrypted authenticator (a function of message, secret key, and the date/time) to a message lets the receiver detect third-party forgery and tampering, because the shared key is needed to produce a valid authenticator. The key is *shared*: the receiver holds it too, and so could have manufactured the authenticator himself. A written signature, by contrast, is meant to adjudicate a dispute between transmitter and receiver — the stockbroker who forges a client's order, the client who disavows an order that lost money. Lamport had observed that a one-way function can give a partial answer for `N` message bits: publish images `y_i, y_i'` of `2N` secret random vectors `x_i, x_i'`, and reveal `x_i` or `x_i'` per bit; the receiver checks each revealed pre-image against its stored image, and a changed bit cannot be forged. This expands the message roughly a hundred-fold.

## Baselines

**The one-time pad (Vernam; Shannon's analysis).** Combine the plaintext with a fresh random key of equal length, modulo 2. Provably unconditionally secure — the cryptogram is statistically independent of the message. The key is as long as the entire correspondence and is pre-shared over a secure channel.

**Conventional short-key ciphers (computationally secure block/stream ciphers).** A few hundred bits of key, security resting on the cost of cryptanalysis (ciphertext-only, known-plaintext, chosen-plaintext attacks define the threat ladder). Cheap and fast. The secret key is agreed in advance over a secure channel; in an `n`-user network there are `n(n-1)/2` such keys. Both parties share the key.

**Key distribution centre (KDC).** Each subscriber pre-shares one key with a central trusted node; to talk to another subscriber, a user asks the centre, which hands out a session key encrypted under the two parties' master keys. This keeps one key per user instead of one per pair. The centre is a trusted third party present in every conversation; it learns (or can learn) every session key.

**Merkle's puzzles (secure communication over an insecure channel).** A construction that *drops* Shannon's assumption that the eavesdropper cannot read the key channel — it grants the eavesdropper `Z` perfect knowledge of every transmission and still agrees a key. The sender `X` manufactures `N` puzzles. Each puzzle is a cryptogram under a *deliberately weakened* key space (say 20 bits), with a recognizable constant embedded so that a correct trial decryption is detectable; inside each puzzle sit a puzzle-ID and a puzzle-key. `X` sends all `N` puzzles in the clear. The receiver `Y` picks *one* puzzle at random and brute-forces it — `O(N)` work for that single puzzle — then sends back its ID (not the key). `X`, who recorded which key went with which ID, now shares that key with `Y`. The eavesdropper saw the ID but not which puzzle yields the key; he cracks puzzles at random until he hits `Y`'s chosen one, on average `N/2` of them at `O(N)` each, i.e. `O(N^2)` work. Legitimate effort is `O(N)` and the attacker's is `O(N^2)`, a quadratic advantage from public messages alone. The honest parties also *transmit* all `N` puzzles, so practical limits put the work ratio at around `10^4`.

## Evaluation settings

The natural yardstick is the *work factor*: the ratio of the cryptanalyst's effort to the legitimate users' effort, measured under the best known attack, with both costs in concrete units (operations, gate delays, or dollars at then-current prices — enciphering on the order of a cent, an opponent's budget on the order of millions of dollars). A scheme is judged "computationally infeasible" to break when that effort is finite but impossibly large, with `10^100` instructions serving as a representative wall. The threat model is the standard ladder — ciphertext-only, known-plaintext, chosen-plaintext — with the strongest (chosen-plaintext, the IFF-style "submit text and observe the cryptogram") taken as the certification standard, and the system `{S_K}` itself assumed known to the opponent. Bit lengths are the scaling axis: for a candidate built on some hard underlying problem of size roughly `2^b`, one tabulates legitimate cost (polynomial in `b`) against attack cost (the best known algorithm's growth in `b`) across `b = 100, 200, ..., 1000`, and asks whether the gap is merely large or *exponential* in `b`. Because no scheme whose public data uniquely determines its secret can be unconditionally secure, every candidate here is certified relative to the *best presently known* algorithm for its underlying problem.

## Code framework

The starting scaffold is the symmetric-cipher harness plus an empty public-channel key-establishment interface. A library of arithmetic over finite fields and integers is available.

```python
# ---- primitives that already exist ----

def random_int(low, high):
    # draw from a true hardware RNG (e.g. a noisy diode)
    ...

# conventional symmetric cipher: both parties already share K somehow
def encrypt_symmetric(K, plaintext): ...
def decrypt_symmetric(K, ciphertext): ...


# ---- public-channel key-establishment interface ----
# Goal: two parties holding NO prior shared secret end up with a common K,
# (or with the ability to encrypt to each other) using only public messages.

def establish(params):
    # TODO
    raise NotImplementedError
```
