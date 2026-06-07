# The Polymerase Chain Reaction (PCR)

## The problem

I need to take a single chosen stretch of DNA — specified only by its sequence — sitting at one copy
in a complex genome (a single-copy locus is one part in ~3×10⁹ in human DNA), and make so many copies
of it that it becomes the dominant species in the tube. That solves my original problem, reading a
single base such as the sickle-cell coding-strand A→T in β-globin (GAG→GTG), by enrichment: a target
too dilute to detect is raised to abundance.

## The key idea

I start from Sanger's 1977 primer-extension logic: a primer can make one chosen site readable if the
target is already abundant enough. The enrichment step is to use two short oligonucleotide primers
that flank the target, one complementary to each strand, with their 3′ ends pointing **inward** toward
each other. A DNA polymerase extends each primer across the target. Because each primer is extended
*through the binding site of the other*, every newly synthesized strand carries the other primer's
site and becomes a template for the other primer in the next round. I repeat melt / anneal / extend:
cycle 1 makes one-ended long products, cycle 2 first makes exact-length single strands, cycle 3 puts
the fully bounded duplex into the doubling pool, and later cycles make the bounded amplicon grow
exponentially, about 2ⁿ after n cycles up to the linear long-product correction. Kleppe–Khorana
repair replication (1971) refilled a duplex, but it did not close this reciprocal template-feedback
loop.

## The protocol

I assemble one reaction once, then thermally cycle it. Into one tube I put template DNA containing
the target sequence; primer P, complementary to the bottom strand with its 3′ end facing the target;
primer Q, complementary to the top strand with its 3′ end facing the target — so P and Q flank the
region and their 3′ ends point inward; the four dNTPs (dATP, dCTP, dGTP, dTTP); thermostable Taq DNA
polymerase, added once; and buffer with Mg²⁺. I cycle the tube about 25–35 times, each cycle with
three steps: (1) denature at 94–98 °C for ~20–30 s, where heat breaks the base-pairing and duplex DNA
becomes single strands; (2) anneal at 50–65 °C for ~20–40 s, where cooling lets primers P and Q bind
their complementary sites; and (3) extend at 72 °C for ~1 min, where Taq adds dNTPs to the primer 3′
ends, copying 5′→3′. I read the reaction on an agarose gel stained with ethidium bromide, looking for
one discrete band at length |P‥Q|.

The enzyme choice fixes the practical failure. Klenow fragment of *E. coli* Pol I can extend primers
at 37°C, but each 94°C melt destroys it, so I would have to re-add enzyme every cycle. **Taq
polymerase** from *Thermus aquaticus* (Brock & Freeze 1969; Chien 1976), optimum ~75–80°C, remains
active through brief melt steps, so I add it once, automate the reaction in a thermal cycler, and —
because extension runs at 72°C — reduce extension from weak primer-template matches without
eliminating every possible off-target event.

## The exponential mechanism, cycle by cycle

I let P extend rightward and Q extend leftward across the target.

- **Cycle 1.** I separate the strands; P and Q anneal to the two original strands and extend *past*
  the far primer site into the downstream sequence, making indefinite-length ("long") products.
- **Cycle 2.** The primers also anneal to the cycle-1 long products. Extending Q on the strand P made
  (and vice versa) runs to the *end* of that template — the 5′ boundary of the other primer —
  producing for the first time exact-length single strands bounded by both primers.
- **Cycle ≥3.** Those exact strands template their complements, so the fully bounded amplicon enters
  the doubling pool. The bounded product grows exponentially, about 2ⁿ after n cycles up to the
  linear correction, while one-ended long products accumulate only linearly (≈2n). The amplicon
  rapidly dominates.

The arithmetic is the reason the method works: 2¹⁰ ≈ 10³, 2²⁰ ≈ 10⁶, and 2³⁰ ≈ 1.07×10⁹. Thirty
cycles take one copy to ~a billion bounded copies of the chosen fragment, apart from the linear
one-ended byproducts made from the original genomic strands.

## Why each choice

- **Two primers, not one** → I make a *chain* reaction: one primer's product is the other's template.
  One primer gives only linear copying.
- **Opposite strands, 3′ ends inward** → each extension carries the other primer's site, enabling
  mutual templating and the discrete inter-primer product.
- **Primers define the boundaries** → the dominant product is a single fixed length ("distinction"),
  read as a clean gel band rather than a smear.
- **Active thermal cycling, not isothermal** → duplexes do not denature spontaneously at one
  temperature (the first 37°C experiment gave no product); heat has to drive denaturation each round.
- **Thermostable Taq, not Klenow** → the 94°C melt destroys a mesophilic enzyme; Taq remains active
  through brief melt steps, is added once, enables automation, and at 72°C reduces extension from
  weak primer-template matches.
- **Exponential growth** → only 2ⁿ enrichment lifts a single-copy target out of a 3×10⁹-bp genome
  into a detectable, dominant species, solving the dilution problem I began with.
