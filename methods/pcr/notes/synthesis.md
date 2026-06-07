# PCR synthesis (pre-Phase-2)

## Pain point / goal (in-frame, Mullis's actual starting problem)
Read a single base at a chosen position in genomic DNA — diagnose a point mutation (sickle-cell:
A→T in β-globin). The natural tool: hybridize an oligo to the site, extend with polymerase in the
presence of one labeled ddNTP per aliquot → which ddNTP adds tells you the next base. "Sanger
sequencing at a single base pair." WALL: on human DNA (3×10^9 bp) one oligo binds thousands of
near-matches; the single target site is far too dilute to read. Need to raise the relative
concentration of the one site. That is the real problem PCR solves; Mullis backed into it.

## The discovery chain (Mullis's own path — the backbone)
1. Two oligos instead of one (free control): one to the upper strand, one to the lower, 3' ends
   adjacent to the base in question. Originally just to get a control band.
2. Worry: contaminating dNTPs would let the polymerase add the wrong bases before the ddNTP. Fix
   considered: pre-run polymerase to burn off dNTPs, then heat to melt off the extended oligos,
   cool, let fresh oligos hybridize.
3. The realization: an oligo extended on the upper strand, if extended far enough, now CONTAINS the
   binding site for the lower-strand oligo (and vice versa). So in the next round each extended
   product is itself a template for the OTHER primer. → doubling. EUREKA.
4. Doubling compounds: 2^n. 2^10≈10^3, 2^20≈10^6, 2^30≈10^9. 30 cycles from one copy → a billion.
5. Second aha (a mile later): primers needn't flank a single base — put them any distance apart;
   after a few cycles the dominant product is a fixed-length fragment bounded exactly by the two
   primers. Abundance + a clean discrete band. "Abundance and distinction."

## The wall + the fix (lands the protocol)
- The night-drive picture is isothermal in spirit: melt, anneal, extend, repeat. Mullis first
  HOPED it would run by itself at 37°C (strands breathe open, re-prime) — first experiment Sept 9,
  no band. Reality: you must actively CYCLE the temperature — hot (~94°C) to denature, cool to
  anneal+extend — every cycle.
- WALL: the polymerase he had (Klenow fragment of E. coli Pol I) is heat-labile; the 94°C melt
  kills it. So fresh enzyme must be pipetted in after every single cycle — 30 manual additions,
  tedious, and a barrier to automation. (First success Dec 16 on pBR322 this laborious way.)
- FIX: a polymerase that survives 94°C. Thermus aquaticus (Brock & Freeze 1969, Yellowstone hot
  spring, grows ~70°C); its DNA polymerase Taq (Chien 1976) is thermostable, optimum ~75–80°C.
  Add it ONCE at the start; it survives every melt. Extension now run at 72°C, which also raises
  primer-annealing stringency → more specific, longer, higher-yield products, and full automation
  in a thermal cycler (Saiki et al. 1988).

## Final artifact (non-computational): the protocol + the exponential mechanism
Three-step cycle, repeated ~25–35×:
- Denature 94–98°C ~30s.
- Anneal 50–65°C ~30s (two primers, opposite strands, 3' ends inward, flanking target).
- Extend 72°C ~1 min (Taq; or 37°C for Klenow), polymerase copies 5'→3'.
Mechanism cycle-by-cycle:
- Cycle 1: each primer extends along its template, running PAST the far primer site → "long
  products" of indefinite length.
- Cycle 2: primers also anneal to the cycle-1 long products; extending to the END of those
  templates yields, for the first time, the FIXED-length product bounded by both primer 5' ends
  (the "amplicon").
- Cycle ≥3: amplicons template amplicons → the fixed-length product doubles each cycle (2^n),
  while long products accumulate only linearly (2n). After ~30 cycles the amplicon overwhelms
  everything: ~2^30 ≈ 10^9 copies of exactly the chosen fragment.
Numbers: 2^10≈10^3, 2^20≈10^6, 2^30≈1.07×10^9. Saiki 1985 (Klenow) measured ~220,000× (≈2^17.7).

## Design-decision → why
- TWO primers (not one): one primer copies one strand into the binding site for the other → makes
  it a *chain* reaction. One primer = linear (Kleppe/Khorana's repair replication, never doubling).
- Primers on OPPOSITE strands, 3' ends pointing INWARD: extension of each fills toward the other,
  so each product carries the other primer's site → mutual templating → exponential.
- Primers define the boundaries → the dominant product is a discrete fixed-length band ("distinction").
- THERMAL CYCLING (vs isothermal): dsDNA won't spontaneously stay open enough at 37°C; you must
  drive denaturation by heat each round (first experiment's failure proved this).
- THERMOSTABLE Taq (vs Klenow): the melt step destroys a mesophilic enzyme; Taq survives → add
  once, automate, and the high extension temp raises specificity/yield/length.
- Why exponential matters: a single-copy site in 3×10^9 bp is unreadable; 2^n lifts it to a clean
  dominant species detectable by gel — solves the original "raise the concentration" wall.

## In-frame discipline
Mullis is the narrator at the wheel and in the lab, 1983 onward. No "this paper," no citing the
Nobel lecture. Prior art (Sanger, Kleppe/Khorana, Brock/Chien Taq, Klenow) cited by name/year as
known background. End on protocol + mechanism, NOT code (a tiny arithmetic check of 2^n is the only
numeric thing, matching what Mullis literally did with pen and paper).
