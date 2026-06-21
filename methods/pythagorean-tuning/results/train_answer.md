I would call the canonical construction the Pythagorean tuning, because it organizes pitch classes entirely through successive perfect fifths in the frequency ratio 3:2. To make a usable scale I start from a chosen tonic, move up or down by fifths, and whenever a pitch falls outside the target octave I transpose it by whole octaves, since doubling or halving a frequency returns what the ear hears as the same pitch class at a different register. Repeating this process twelve times around the circle of fifths produces twelve chromatic pitch classes; retaining a subset of those steps gives the familiar diatonic modes, including the Ionian major scale.

The mathematical heart of the system is the fact that (3/2)^12 is not a power of two. When I stack twelve pure fifths I reach a pitch slightly sharper than seven octaves above the tonic. The ratio between those two endpoints, (3/2)^12 / 2^7 ≈ 1.01364, is the Pythagorean comma. Because of that comma the chromatic circle cannot close perfectly: a note reached by going up twelve fifths is not exactly the same as the note reached by going up seven octaves. When I distribute the twelve pitch classes within one octave, some intervals that look enharmonically equivalent on paper, such as F-sharp and G-flat, occupy slightly different frequencies. In a twelve-tone Pythagorean tuning there is no single pitch that serves both names at once; choosing one forces a compromise elsewhere, which historically appeared as the wolf fifth or wolf third in keyboard instruments.

Another signature feature is the Pythagorean major third. If I form a major third as four stacked fifths reduced by octaves, the resulting frequency ratio is (3/2)^4 / 2^2 = 81/64 ≈ 1.2656. That is noticeably wider than the just intonation major third of 5/4 = 1.25 and wider still than the equal-tempered major third of 2^(4/12) ≈ 1.2599. The fifths themselves remain pure at 3/2, so melodic motion by fifth sounds exceptionally smooth, while triadic harmony sounds restless because the thirds beat. This trade-off is why Pythagorean tuning is valued for monophonic or strongly melodic music and why later temperament systems modified the fifths to tame the thirds.

I would implement the construction as a small numerical script. I begin by fixing a tonic frequency and a number of fifths to stack, positive for ascending fifths and negative for descending fifths. For each step I multiply by 3/2 raised to the step index and then divide or multiply by the appropriate power of two so that the result lies between the tonic and its octave. Sorting the reduced frequencies yields the chromatic collection. From there I can extract a diatonic scale by selecting the pitch classes that correspond to the standard Pythagorean sequence, and I can verify the Pythagorean comma by comparing the frequency reached after twelve fifths with the frequency seven octaves above the tonic. A brute-force check also lets me measure every fifth and every major third in the resulting scale.

The script below carries out exactly this demonstration. It builds the chromatic Pythagorean scale around a reference tonic, prints the frequency ratios of the twelve pitch classes, extracts the C-major-like diatonic subset, and compares the Pythagorean intervals with their just and equal-tempered counterparts. Running it should make the comma visible as a small but non-zero frequency difference and should confirm that the perfect fifths are exact while the major thirds are wide.

```python
import math

TONIC = 261.625565  # Middle C, close to scientific pitch reference
OCTAVE = 2.0
FIFTH = 3.0 / 2.0


def pythagorean_chromatic(tonic=TONIC, n_fifths=12):
    pitches = []
    for k in range(-n_fifths // 2, n_fifths // 2 + n_fifths % 2):
        freq = tonic * (FIFTH ** k)
        # Reduce into the interval [tonic, 2*tonic)
        while freq >= tonic * OCTAVE:
            freq /= OCTAVE
        while freq < tonic:
            freq *= OCTAVE
        pitches.append((k, freq))
    # Sort by frequency within the octave
    pitches.sort(key=lambda x: x[1])
    return pitches


def ratio_name(k):
    names = {
        0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B",
        -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb"
    }
    return names.get(k, f"step({k})")


chromatic = pythagorean_chromatic()
print("Chromatic Pythagorean scale (ratios to tonic):")
for k, f in chromatic:
    print(f"  {ratio_name(k):3s}  ratio = {f / TONIC:.6f}  freq = {f:.3f} Hz")

# Diatonic C major subset: C, D, E, F, G, A, B
diatonic_indices = [0, 2, 4, 5, 7, 9, 11]  # indices into the sorted chromatic list
diatonic = [chromatic[i] for i in diatonic_indices]
print("\nDiatonic C-major subset:")
for k, f in diatonic:
    print(f"  {ratio_name(k):3s}  ratio = {f / TONIC:.6f}")

# Verify exact fifths and wide thirds
print("\nInterval checks:")
fifth_ratio = diatonic[4][1] / diatonic[0][1]  # G/C
third_ratio = diatonic[2][1] / diatonic[0][1]  # E/C
print(f"  Perfect fifth G/C = {fifth_ratio:.6f}  (target 3/2 = {3/2:.6f})")
print(f"  Major third E/C  = {third_ratio:.6f}  (just 5/4 = {5/4:.6f}, 12-TET = {2**(4/12):.6f})")

# Pythagorean comma: twelve fifths vs seven octaves
comma = (FIFTH ** 12) / (OCTAVE ** 7)
print(f"\nPythagorean comma = {comma:.6f} (+{100*(comma-1):.3f}% or about {1200*math.log2(comma):.2f} cents)")
```

I would present this as the standard Pythagorean tuning construction. The script is only a verification aid, but it captures the defining tension of the system: pure fifths stacked around the circle of fifths fail to close by the comma, and that residual mismatch shapes every practical tuning decision. If I needed a different key center, I would transpose the same set of ratios; if I needed to play in every key, I would either add split accidentals or abandon pure fifths in favor of a temperament.
