# PCR — antecedents and protocol (web-grounded reference notes)

## Mullis self-account (Nobel Lecture 1993) — refs/mullis-nobel-lecture-1993.{html,txt}
The backbone of the trace. Key passages (verbatim, paraphrased here):
- Cetus, 1979: DNA-synthesis lab; Biosearch machine churning out oligonucleotides "much faster
  than the molecular biologists at Cetus could use them." Mullis "started playing with the
  oligonucleotides to find out what they could do."
- Henry Erlich's neighboring lab worked on detecting point mutations; Mullis's "oligomer
  restriction" idea worked on concentrated targets (plasmid) but NOT on a single-copy gene in
  human DNA.
- The single-base-read idea: hybridize one oligo to a specific site, extend it with DNA
  polymerase in presence of ONLY dideoxynucleoside triphosphates, one radioactive per aliquot →
  read which base is adjacent to the 3' end. "It would be like doing Sanger sequencing at a
  single base pair." Fails on human DNA: oligo binds hundreds/thousands of sites. "What I needed
  was some method of raising the relative concentration of the specific site of interest. What I
  needed was PCR, but I had not considered that possibility."
- The night drive: Friday night, Berkeley → Mendocino, girlfriend Jennifer Barnett asleep.
  Two oligos instead of one — one binds the upper strand, one the lower strand, 3' ends adjacent
  to opposing bases (a free control).
- Worry about contaminating dNTPs in the sample → idea: leave out ddNTPs, drop in polymerase,
  let it use up the dNTPs by extending the oligos; then HEAT to melt off the extended oligos,
  COOL so fresh unextended oligos hybridize; the extended ones are outnumbered.
- "But what if the oligonucleotides ... had been extended so far they could now hybridize to
  unextended oligonucleotides of the opposite polarity in this second round." → "EUREKA!!!! The
  result would be exactly the same only the signal strength would be doubled." → "I could do it
  over and over again. Every time I did it I would double the signal."
- Stopped at mile marker 46.7 on Highway 128. Paper + pen from glove compartment. "I confirmed
  that two to the tenth power was about a thousand and that two to the twentieth power was about a
  million, and that two to the thirtieth power was around a billion, close to the number of base
  pairs in the human genome." 30 cycles → immense signal, almost no background.
- A mile later: the two oligos can be at ANY arbitrary distance, not just flanking one base pair →
  arbitrarily many copies of any chosen sequence, and after a few cycles most copies are the SAME
  SIZE (defined by the primers). "Abundance and distinction." "Dear Thor!"
- The skeptic wall: Monday in the library, nothing in the literature; colleagues unimpressed.
  "There was not a single unknown in the scheme. Every step ... had been done already."
- First experiment Sept 9 (midnight): human DNA + NGF primers, boiled, cooled, added ~10 units
  DNA polymerase, 37°. He HOPED extension/dissociation/re-priming would happen on its own at some
  finite rate (no cycling) — "I did not relish the idea of heating, cooling, adding polymerase
  over and over again, and held this for a last resort."
- 12-hr sample: no 400-bp band by ethidium bromide. Slowly "succumbed" to the unavoidable:
  cycling between single-stranded (hot) and double-stranded (cool) temperatures — "This also
  meant adding the thermally unstable polymerase after every cycle."
- First success Dec 16, on a short fragment from plasmid pBR322 (retreated from human DNA).
  Celebrated with Fred Faloona.

NOTE: The Nobel lecture narrates up to the first success with the HEAT-LABILE polymerase
(Klenow fragment of E. coli Pol I, added fresh each cycle). The Taq fix came later (Saiki 1988).

## Antecedents
- **Kleppe & Khorana 1971** ("Studies on polynucleotides XCVI. Repair replication of short
  synthetic DNA's as catalyzed by DNA polymerases," J. Mol. Biol. 56:341–361). Proposed extending
  primers on both strands by DNA polymerase ("repair replication"), even noted adding "a fresh
  dose of the enzyme" — but single primer-template complex, NO thermal cycling, NO chain reaction,
  NO exponential amplification. Idea never reduced to the doubling realization. (Saiki/PMC,
  Wikipedia History of PCR.)
- **Sanger 1977 dideoxy sequencing.** A primer extended by DNA polymerase (Klenow fragment of
  E. coli Pol I) in presence of dNTPs + chain-terminating ddNTPs (lack 3'-OH → no further bond).
  Mullis's single-base-extension idea is "Sanger sequencing at a single base pair."
- **Oligonucleotide synthesis** — automated DNA synthesizers (Khorana lineage; Biosearch machine
  at Cetus) made custom short oligos cheap and fast: the enabling supply that made TWO primers,
  and arbitrary sequences, trivial.
- **Klenow fragment** of E. coli DNA Pol I: polymerase + 3'→5' proofreading, lacks 5'→3'
  exonuclease. Heat-labile — destroyed at the ~94°C melting step → must be re-added every cycle.
- **Thomas Brock & Hudson Freeze 1969**: isolated *Thermus aquaticus* from Mushroom Spring,
  Lower Geyser Basin, Yellowstone. Grows ~70°C. **Chien et al. 1976** characterized its heat-stable
  DNA polymerase (Taq), optimum ~75–80°C. Surviving 94°C denaturation → no re-adding enzyme →
  automation. (Saiki et al. 1988, Science 239:487–491.)

## Protocol / mechanism (grounded: Wikipedia PCR, whatisbiotechnology, Saiki 1985/1988)
- Three-step cycle:
  - **Denaturation** 94–98°C, ~20–30 s: melt dsDNA into single strands.
  - **Annealing** 50–65°C, ~20–40 s: two primers hybridize, each to one strand, flanking target,
    3' ends pointing inward toward each other.
  - **Extension** 72°C (Taq optimum 75–80°C) / 37°C for Klenow: polymerase extends each primer 5'→3'.
- **Exponential doubling**: copies after n cycles = 2^n (ideal). 30 cycles → 2^30 ≈
  1.07×10^9. Saiki 1985 reported ~220,000× (2^17.7) amplification of β-globin with Klenow.
- First two cycles produce variable-length products (extension runs past the far primer site);
  from cycle 3 on the **fixed-length amplicon bounded by the two primer 5' ends** is produced and
  it comes to dominate (~2^n), while the long variable products grow only linearly (~2n).

## Primary papers
- Saiki RK, Scharf S, Faloona F, Mullis KB, Horn GT, Erlich HA, Arnheim N (1985) "Enzymatic
  amplification of beta-globin genomic sequences and restriction site analysis for diagnosis of
  sickle cell anemia." Science 230:1350–1354. First PCR application; Klenow; ~220,000× amplification.
- Saiki RK, Gelfand DH, Stoffel S, Scharf SJ, Higuchi R, Horn GT, Mullis KB, Erlich HA (1988)
  "Primer-directed enzymatic amplification of DNA with a thermostable DNA polymerase." Science
  239:487–491. Taq polymerase → automation, specificity, longer products.
- Mullis KB (1990) "The Unusual Origin of the Polymerase Chain Reaction," Sci. Am. 262(4):56–65.
