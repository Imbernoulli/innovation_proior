I am reporting the mechanism behind the burst of mutable loci observed in the self-pollinated progeny of maize plants whose chromosome 9 had been carried through the breakage-fusion-bridge cycle. The phenomenon is not a collection of unstable genes but the activity of mobile, chromatin-based controlling elements that insert beside ordinary genes, regulate their expression, and move again. The canonical formulation is the Ac/Ds system of transposable controlling elements in maize.

The starting observation is that roughly forty distinct loci became mutable at once in these plants, producing sharply bounded variegated sectors in kernels and seedlings. A sector is a clonal patch, which means the change happened once in a single ancestor cell and was inherited by all descendants. The size of a sector therefore records the developmental timing of the event: large sectors arise from early events, small spots from late events. Reciprocal twin sectors, in which two adjacent daughter clones show opposite changes, cannot be explained by a point mutation of a gene, because a point mutation does not simultaneously give one daughter more and the other less. The twin pattern points to a physical component that is partitioned unequally at mitosis.

That physical component is an inserted chromatin element. The first element I identify is Dissociation, or Ds. Ds was first detected as a site of recurrent somatic breakage on the short arm of chromosome 9. An event at Ds makes sister chromatids fuse at that site, producing a dicentric chromatid and an acentric distal fragment that is lost as the cell divides. When Ds lies between Wx and the centromere, the clonal descendants lose the dominant markers C, Bz, and Wx together, and cytology confirms that the marked telomeric knob is also lost. The timing and frequency of Ds breakage match the timing and frequency of mutation events at mutable loci, so Ds is not a separate curiosity; it is a window onto the same underlying process.

Ds is also relocatable. In most kernels the breakage point lies between Wx and the centromere, but occasionally a kernel shows a different heritable pattern with a shift to a more distal breakpoint, and cytology confirms that the breakage site has moved up the arm. This means Ds has transposed to a new chromosomal address. A locus that changes its address is not a fixed gene bead; it is a mobile element.

The crucial demonstration comes when Ds inserts next to a gene. A gamete appeared carrying a chromosome 9 that looked morphologically normal but in which Ds had transposed to the C locus. At that moment the aleurone became colorless, as if C had been mutated to c. But this was not a mutation or a deficiency, because in later descendants C activity returned: colorless tissue threw colored spots, and some sectors reverted to stable, full wild-type color. The return of C action was locked to the disappearance of Ds activity at that site. Therefore the gene had remained intact throughout. Ds inserted beside C inhibited its expression, and when Ds was removed C resumed normal function unchanged. This is the mutable c-m1 allele, and it redefines what a mutable locus is: not an unstable gene, but a stable gene with a mobile inhibitor parked beside it.

Other mutable loci extend the picture. A second C-derived mutable locus, c-m2, shows a range of quantitative color expression, including intensities stronger than the original wild type, and produces twin sectors in which one daughter clone is darker than the other. A Wx-derived mutable locus, wx-m1, behaves similarly. These cases show that different inserted states can produce graded expression, reciprocal twin sectors, and different balances of phenotypic consequences.

Ds alone, however, is not sufficient. When Ac, the Activator, is bred out of the stock, Ds-controlled mutable loci produce no mutation events and Ds produces no breakage events. When Ac is crossed back in, activity returns. Ac is inherited as a single unit, segregates independently of Ds, and its map position wanders from cross to cross, which means Ac is itself mobile. This two-element architecture parallels Rhoades' earlier Dt/a1 system, but with the added insight that both elements can transpose and that the mutable allele is an intact gene under insertion control.

Ac does more than turn the system on. It sets the developmental timing through dosage. In the triploid endosperm, one, two, or three doses of a dosage-responsive Ac state produce progressively later times of mutation at Ac-controlled loci. Because late events generate small late sectors and early events generate large early sectors, Ac dosage is the dial that positions events on the sector-size clock. Heritable changes in the state of Ds and Ac further modulate the relative frequency of breakage, inhibition, and restoration, so the visible pattern on any kernel is jointly determined by the state of the element beside the gene and the dose and state of Ac.

The origin of the burst traces back to the breakage-fusion-bridge cycle. The cycle breaks and fuses preferentially at heterochromatic knobs and centromeres, and the plants under study all inherited chromosome 9 short arms that had passed through this cycle. The cycle subjects heterochromatin to mechanical stress, and heterochromatin is the material from which controlling elements appear to be mobilized or generated. A direct test uses Rhoades' Dotted, Dt, which normally resides in the heterochromatic knob at the end of chromosome 9 and makes the otherwise stable a1 allele give colored dots. In a stock homozygous for a1 with no Dt, running a knobbed chromosome 9 through the breakage-fusion-bridge cycle produces de novo a1-to-A1-like dotting indistinguishable from Dt action, with no Dt stock present to contaminate the result. Genome shock at heterochromatin can therefore create or activate controlling elements.

The resulting picture is that the genome is not a static string of beads at fixed addresses. It carries autonomous and non-autonomous mobile elements. Ac is autonomous: it can activate itself and Ds. Ds is non-autonomous: it can break, transpose, and inhibit neighboring genes, but only when Ac is present. Insertion of such an element beside a gene can switch the gene off; removal can restore it intact. The clonal sectors, the developmental clock read from sector size, the reciprocal twins from unequal mitotic partition, and the wandering map positions of the controllers all follow from this mobile-element mechanism. Genome shock, particularly the wrenching of heterochromatin by the breakage-fusion-bridge cycle, mobilizes these elements and explains why the burst appeared in the specific experimental material.

To make the timing and sector-size relationship concrete, I will run a small Python simulation of Ac-dose-dependent transposition events during kernel development. The simulation follows a single founding cell through a fixed number of divisions. At each division, every extant cell may undergo a Ds event with a probability that rises with the cell's age but is delayed by higher Ac dose. An event founds a clonal sector whose size equals the number of descendants that cell would leave if it continued dividing until the final stage. Higher Ac dose should shift events toward later divisions and therefore produce smaller sectors.

```python
import random


def simulate_sectors(ac_dose, n_divisions=10, base_rate=0.3):
    """
    Simulate Ds excision/breakage events during kernel development.
    Higher Ac dose shifts the per-cell event probability curve to later ages.
    Returns a list of (division_when_event_occurred, sector_size).
    """
    sectors = []
    # Each live cell stores the division at which it was born.
    cells = [0]
    # Later Ac doses raise the age threshold at which events become likely.
    threshold = 2 + 2 * (ac_dose - 1)

    for division in range(n_divisions):
        next_cells = []
        for birth_div in cells:
            age = division - birth_div
            # Logistic-like rise: probability is low until age exceeds threshold.
            rate = base_rate / (1 + 2.0 ** (threshold - age))
            rate = min(rate, 0.95)
            if random.random() < rate:
                # Event creates a sector; descendants continue developing,
                # but the visible sector size is fixed by the event time.
                sector_size = 2 ** (n_divisions - division - 1)
                sectors.append((division, sector_size))
            next_cells.append(birth_div)
            next_cells.append(birth_div)
        cells = next_cells

    return sectors


def report(dose, runs=200):
    mean_divisions = []
    mean_sizes = []
    for _ in range(runs):
        events = simulate_sectors(ac_dose=dose)
        if events:
            mean_divisions.append(sum(d for d, _ in events) / len(events))
            mean_sizes.append(sum(s for _, s in events) / len(events))
    return (
        sum(mean_divisions) / len(mean_divisions) if mean_divisions else 0.0,
        sum(mean_sizes) / len(mean_sizes) if mean_sizes else 0.0,
    )


if __name__ == "__main__":
    random.seed(42)
    for dose in [1, 2, 3]:
        mean_div, mean_size = report(dose)
        print(
            f"Ac dose = {dose}: "
            f"mean event division = {mean_div:.2f}, "
            f"mean sector size = {mean_size:.1f}"
        )
```

In summary, I propose that the canonical method for explaining the observed burst of mutable loci in these maize stocks is the Ac/Ds system of transposable controlling elements. Ds is a non-autonomous element that can break chromosomes, transpose, and insert beside genes to inhibit them reversibly. Ac is an autonomous, mobile activator required in trans for Ds action, and its dosage controls the developmental timing of events. The visible variegation patterns are the clonal readout of these insertion, excision, and breakage events, and the whole system can be mobilized by genome shock acting on heterochromatin.
