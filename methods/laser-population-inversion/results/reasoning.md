I need a source of optical radiation that is not just brighter glow. A lamp already has atoms dropping from excited states, but each atom emits on its own clock, with its own phase and direction. If I put mirrors around that, I can filter some frequencies, but filtering is not the same as making the field grow coherently. The field has to come back through the material and leave larger than it entered, with the added light in the same mode.

Start with the smallest possible light-matter bookkeeping. Take two atomic levels, `1` and `2`, separated by `h nu`, and put radiation at that frequency through them. There are three processes I have to keep separate. Upper atoms can decay without help: `N2 A21`. Radiation can push lower atoms upward: `N1 B12 rho(nu)`. Radiation can also push upper atoms downward: `N2 B21 rho(nu)`. The last two are the important pair because they both scale with the radiation already in the mode. Spontaneous emission can seed a photon, but it is not the multiplication mechanism; the field-proportional terms decide whether an existing wave is strengthened or weakened.

If the material is in ordinary thermal equilibrium, this immediately goes the wrong way. The lower level has more atoms than the upper level. For a nondegenerate transition, the Einstein balance gives the stimulated coefficients symmetrically enough that the incident field sees more absorbers than emitters. The net stimulated rate has the shape `rho(nu) (N2 B21 - N1 B12)`. In equilibrium that is negative. So the simple idea "excite some atoms and let them radiate" hits a wall: as long as most atoms available to the transition sit in the lower state, the same resonant light that could stimulate emission is more likely to be absorbed.

This is why recombination is the wrong mental model here. Recombination can populate excited states and can produce photons, but the question is not whether photons are produced. The question is whether an already present optical wave can force atoms to add photons with its phase, direction, polarization, and frequency faster than other atoms remove photons from that wave. That is a sign question on the stimulated terms.

So I need to break the equilibrium ordering of populations. I do not need every atom excited, and I do not need spontaneous emission to become orderly by itself. I need the transition to see more effective upper-state population than lower-state population, with degeneracy factors folded in if necessary. Then the sign flips:

`N2 B21 > N1 B12`.

Now an incident field at the transition frequency is not attenuated. It induces more downward events into its own mode than upward events out of it. The added photons are coherent with the stimulating field because stimulated emission copies the mode that stimulated it. That is the microscopic gain element.

But a single pass through an excited medium still feels incomplete. A wave enters, grows by some factor, exits, and then it is gone. If I want an oscillator, I need the field to regenerate itself. Put the medium in a resonant cavity. The mirrors return a selected optical mode through the same excited medium, and only fields whose round-trip phase fits the cavity keep arriving in step with themselves. The cavity is not the source of energy; the pump is. The cavity is the feedback and mode selector.

Let the intensity gain coefficient of the active medium be `g`. Over active length `L`, a pass contributes roughly `exp(g L)` in intensity when the gain is small-signal and unsaturated. Loss inside the cavity contributes `exp(-alpha_loss L)` per pass through the same length, and the two mirrors keep fractions `R1` and `R2` of the intensity after a full round trip. After one round trip the intensity is multiplied by

`R1 R2 exp(2 (g - alpha_loss) L)`.

If this factor is below one, any seed field dies away. If it is above one, the selected mode grows. The threshold is the equality:

`R1 R2 exp(2 (g_th - alpha_loss) L) = 1`,

so

`g_th = alpha_loss + (1 / (2 L)) ln(1 / (R1 R2))`.

That equation is the macroscopic version of the microscopic sign flip. The population preparation must make the stimulated-emission excess large enough that the gain coefficient reaches this threshold. Below threshold, spontaneous emission may be visible, but it is not a self-sustained coherent oscillator. Above threshold, the cavity mode repeatedly stimulates more emission into itself, until gain saturation brings the net round-trip factor back to one in steady operation.

Now the pump has a precise job. It is not "make light" directly. It must maintain the nonthermal population distribution against spontaneous decay, stimulated depletion, collisions, and any lower-state refilling. A two-level system pumped resonantly is awkward because the pump itself tends to equalize the two populations rather than maintain the upper one above the lower one. Practical media therefore use additional levels or selective preparation, so the radiating transition can have the required upper-over-lower population condition while the lower laser level is emptied or the upper level is fed.

The final mechanism is now forced. Prepare an active medium so the radiating transition has population inversion. Place it inside an optical resonator whose allowed modes pass repeatedly through the medium. A seed photon in one resonator mode, even if it begins as spontaneous emission, stimulates upper-state atoms to emit matching photons into that same mode. If the stimulated-emission gain exceeds absorption plus cavity losses, the field grows coherently. The threshold condition decides when growth starts; the cavity decides which modes survive; saturation decides the steady output.

The compact artifact is:

```text
two-level stimulated balance:
  stimulated absorption rate      = N1 B12 rho(nu)
  stimulated emission rate        = N2 B21 rho(nu)
  coherent small-signal gain sign = sign(N2 B21 - N1 B12)

population condition:
  gain requires N2 B21 > N1 B12
  for equal degeneracy and B12 = B21, this is N2 > N1

cavity feedback:
  round-trip intensity factor = R1 R2 exp(2 (g - alpha_loss) L)

threshold:
  R1 R2 exp(2 (g_th - alpha_loss) L) = 1
  g_th = alpha_loss + (1 / (2 L)) ln(1 / (R1 R2))

mechanism:
  pump maintains the nonthermal population;
  inversion makes stimulated emission dominate absorption;
  resonator feedback makes the same optical mode sample the gain repeatedly;
  threshold gain marks the onset of coherent self-sustained amplification.
```
