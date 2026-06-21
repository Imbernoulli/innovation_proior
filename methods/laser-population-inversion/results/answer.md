# Laser Population Inversion

Laser action is coherent optical amplification by stimulated emission in a resonant cavity. The active medium is pumped into a nonthermal population distribution so that, at the radiating transition, stimulated emission removes atoms from the upper state and adds photons to the selected optical mode faster than stimulated absorption removes photons from that mode.

For a two-level transition at frequency `nu`, Einstein's coefficients give the competing stimulated rates:

```text
absorption:           N1 B12 rho(nu)
stimulated emission:  N2 B21 rho(nu)
spontaneous emission: N2 A21
```

The coherent gain is controlled by the field-proportional difference:

```text
net stimulated drive = rho(nu) (N2 B21 - N1 B12)
```

Ordinary thermal populations make this negative: absorption dominates. Pumping must create population inversion, meaning the transition has enough upper-state population that

```text
N2 B21 > N1 B12
```

or, for equal degeneracies with `B12 = B21`,

```text
N2 > N1.
```

Spontaneous emission can provide an initial seed, but it is not the organizing mechanism. The organizing mechanism is stimulated emission: the emitted photon joins the same mode as the stimulating field, with matching frequency, phase, propagation direction, and polarization.

The resonator converts gain into oscillation. If the active length is `L`, the small-signal gain coefficient is `g`, distributed loss is `alpha_loss`, and mirror intensity reflectivities are `R1` and `R2`, the round-trip intensity factor is

```text
G_rt = R1 R2 exp(2 (g - alpha_loss) L).
```

Threshold occurs when the returning mode exactly replaces its losses:

```text
R1 R2 exp(2 (g_th - alpha_loss) L) = 1
g_th = alpha_loss + (1 / (2 L)) ln(1 / (R1 R2)).
```

The final artifact is the mechanism:

```text
external pump
  -> nonthermal upper-state population
  -> stimulated emission exceeds stimulated absorption
  -> optical resonator feeds the same mode back through the gain medium
  -> round-trip gain reaches threshold
  -> coherent self-sustained optical amplification
```

This is not a recombination account. Recombination may be one way to prepare excited carriers in some materials, but laser action itself is the dominance of stimulated emission over absorption under population inversion, stabilized and selected by cavity feedback.
