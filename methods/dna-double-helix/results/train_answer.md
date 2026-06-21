DNA had to be the gene. Avery's transforming principle experiment showed that the heritable trait is carried by deoxyribonucleic acid, not protein, so the molecule's three-dimensional structure had to explain two things at once: how it stores an essentially unlimited amount of arbitrary information, and how it is copied exactly when a cell divides. The available evidence was strong but partial. Franklin and Gosling's B-form fiber-diffraction photograph gave a bold helical cross, a 34 Å repeat, a 3.4 Å meridional reflection, and ten residues per turn, while the A-form crystalline data imposed C2 symmetry with a dyad axis. The diffraction pattern, however, only constrains the structure; it does not give atomic coordinates. Any candidate also had to survive stereochemistry: fixed bond lengths and angles, hard-sphere van der Waals contacts, and the correct ionization state of the phosphates.

The obvious baseline models failed. Putting the phosphate backbone inside the molecule, whether bridged by magnesium ions or packed as in Pauling and Corey's triple helix, ignored that phosphates are ionized at neutral pH and therefore charged, hydrophilic, and desperate for water and counterions. Burying them in a dry core created electrostatic repulsion that tore the model apart; Pauling's further assumption of un-ionized phosphates made the substance not even an acid. Like-with-like base pairing also collapsed: two purines are much wider than two pyrimidines, so A-A and G-G rungs would bulge while T-T and C-C rungs would pinch, destroying the regular helix the photograph demanded. Like-with-like also said nothing about Chargaff's measured 1:1 ratios of A to T and G to C. Even the drawing of the bases had to be fixed: textbook enol forms of guanine and thymine place the hydrogen on the wrong atom, scrambling which atoms are hydrogen-bond donors and which are acceptors.

The method that survives all of these constraints is the DNA double helix. DNA is built from two antiparallel sugar-phosphate chains wound as right-handed helices around a common axis. The charged phosphate backbones sit on the outside, exposed to water and cations; the flat bases turn inward and stack 3.4 Å apart, ten per turn, giving the 34 Å pitch and roughly 20 Å diameter seen in the B-form pattern. The two chains run in opposite chemical directions, which is exactly what the C2 dyad axis perpendicular to the fiber axis requires for two directed backbones to map onto each other.

The central innovation is the base-pairing rule. Each rung is one purine paired with one pyrimidine, so the overall width is one large ring plus one small ring at every level, keeping the diameter constant no matter the sequence. With the bases in their ordinary keto and amino tautomeric forms, adenine and thymine form two hydrogen bonds, and guanine and cytosine form three. The remarkable fact is that the A-T and G-C pairs have identical glycosidic-attachment geometry: the bonds that anchor each base to its sugar come off in the same relative positions and at the same distance, so the two kinds of rungs are interchangeable without disturbing the regular backbone. This means A on one chain demands T on the other, and G demands C. Chargaff's ratios stop being a coincidence and become a structural necessity, while the proportion of A-T rungs versus G-C rungs remains free to vary between species.

Because each base specifies its partner, the two strands are complementary. The sequence along one chain can be arbitrary, which is how the molecule stores information, but once it is chosen the other chain is fully determined. Separate the two strands and build a new partner against each original strand, and you obtain two daughter double helices, each conserving one parental strand and one newly synthesized strand. The structure therefore carries its own copying mechanism: the specific pairing immediately suggests semiconservative replication of the genetic material.

```python
"""
Minimal working model of the DNA double helix: generate B-form coordinates
and enforce Watson-Crick complementarity between two antiparallel strands.
"""

import math

# B-form geometric parameters from fiber diffraction
RISE = 3.4          # Å rise per base pair
TWIST = 36.0        # degrees per base pair
RADIUS = 10.0       # Å phosphate radius
BASE_RADIUS = 3.4   # Å base-pair radius from axis
PAIRS_PER_TURN = 10
PITCH = RISE * PAIRS_PER_TURN  # 34 Å

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
HBONDS = {"A": 2, "T": 2, "G": 3, "C": 3}


def generate_helix(sequence: str, antiparallel: bool = False):
    """Return list of (x, y, z, base) for one strand."""
    points = []
    n = len(sequence)
    for i, base in enumerate(sequence):
        theta = math.radians(i * TWIST)
        if antiparallel:
            theta = -theta
        x = RADIUS * math.cos(theta)
        y = RADIUS * math.sin(theta)
        z = i * RISE
        points.append((x, y, z, base))
    return points


def base_pair_geometry(i: int):
    """Coordinates of the hydrogen-bonded base pair at rung i."""
    theta = math.radians(i * TWIST)
    x1 = BASE_RADIUS * math.cos(theta)
    y1 = BASE_RADIUS * math.sin(theta)
    x2 = -x1
    y2 = -y1
    z = i * RISE
    return (x1, y1, z), (x2, y2, z)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return "".join(COMPLEMENT[b] for b in seq)[::-1]


def is_antiparallel_pair(seq1: str, seq2: str) -> bool:
    """Check that seq2 is the reverse complement of seq1."""
    return len(seq1) == len(seq2) and reverse_complement(seq1) == seq2


def melting_contribution(seq: str) -> float:
    """Approximate total hydrogen bonds per base pair in the duplex."""
    return sum(HBONDS[b] for b in seq) / len(seq)


if __name__ == "__main__":
    seq = "AGCTTAGCCGAT"
    seq_rc = reverse_complement(seq)
    # For the antiparallel strand, the bottom-to-top order is the plain
    # complement, so strand1[i] pairs with strand2[i] at the same height.
    seq2_bottom = "".join(COMPLEMENT[b] for b in seq)

    strand1 = generate_helix(seq, antiparallel=False)
    strand2 = generate_helix(seq2_bottom, antiparallel=True)

    assert is_antiparallel_pair(seq, seq_rc)
    assert len(strand1) == len(strand2)
    n = len(strand1)

    print(f"Sequence:       {seq}")
    print(f"Reverse comp:   {seq_rc}")
    print(f"Length:         {n} bp")
    print(f"Pitch:          {PITCH} Å")
    print(f"Rise per bp:    {RISE} Å")
    print(f"Twist per bp:   {TWIST}°")
    print(f"Diameter:       {2 * RADIUS} Å")
    print(f"Avg H-bonds/bp: {melting_contribution(seq):.2f}")

    print("\nFirst three base-pair rungs (backbone P coordinates):")
    for i in range(min(3, n)):
        x1, y1, z1, b1 = strand1[i]
        x2, y2, z2, b2 = strand2[i]
        assert COMPLEMENT[b1] == b2
        print(f"  rung {i}: {b1}-{b2}  P1=({x1:+.1f},{y1:+.1f},{z1:.1f})  "
              f"P2=({x2:+.1f},{y2:+.1f},{z2:.1f})")

    # Replication: each separated strand templates a new complement.
    template = seq
    new_partner = reverse_complement(template)
    print(f"\nAfter replication, one daughter duplex: 5'-{template}-3' / "
          f"3'-{new_partner}-5'")
```
