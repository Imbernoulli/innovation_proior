I will show that the hunter cannot guarantee being within distance 100 of the invisible rabbit after 10^9 rounds, using what I call the symmetric ambiguity block strategy for the IMO 2017 hunter-rabbit problem. The crucial observation is that the rabbit's only real advantage is informational: the tracking report may be off by one unit, and the hunter must commit his own one-unit move after seeing that report. This one-unit transverse uncertainty can be preserved for many rounds and then converted into a guaranteed increase in the distance between the two players.

The natural hunter idea is to trust the reports, perhaps by following the latest report or averaging many reports to filter out noise. This intuition fails because the rules do not require the rabbit to follow a single hidden path whose reports are merely noisy observations. Instead, many distinct rabbit histories can be consistent with exactly the same sequence of reports. If two legal histories produce identical reports, the hunter, whose moves depend only on the reports, must end at the same point in both histories. The rabbit then picks whichever history leaves the hunter farther away. So the right object to analyze is not a single estimated rabbit position but the entire cloud of positions still consistent with the reports, and averaging is powerless when the cloud contains two symmetric points on opposite sides of the reported line.

The strategy works one block at a time. Within each block of 200 rounds, the rabbit will keep alive two possible endpoints that are symmetric about a line through the hunter and the rabbit, while reporting only points on that line. Because both candidate paths stay within one unit of the line, reporting the corresponding points on the line is legal. To make the argument as strong as possible, I will even reveal the rabbit's exact location to the hunter at the start of each block; this only helps the hunter, so an escape proven against an omniscient hunter also works against the real one.

Let H be the hunter, R the rabbit, and d = HR the current distance, with d < 100. If H is not equal to R, let r be the oriented line from H through R; if they coincide, choose any line r through R. Choose two points Y_1 and Y_2 that are symmetric about r, each exactly one unit away from r, and each at distance 200 from R. The rabbit can travel from R to either Y_i in 200 unit steps along the segment RY_i. Every point on either segment lies within distance 1 of r, so the tracker is allowed to report points on r throughout the block. Consequently the hunter receives the same reports in both possible worlds and therefore makes the same 200 moves. Afterward the rabbit chooses whichever endpoint Y_i gives the larger final distance.

Let H' be the point on r that is 200 units forward of H, let Z be the midpoint of Y_1Y_2 (which lies on r), and let R' be the point on r with RR' = 200. Since RY_i = 200 but Y_i is one unit off r, the midpoint Z lies slightly behind R'. The gap is

$$
\epsilon = R'Z = 200 - \sqrt{200^2 - 1} = \frac{1}{200 + \sqrt{200^2 - 1}},
$$

so 1/400 < \epsilon < 1/200. From (200 - \epsilon)^2 = 200^2 - 1 we obtain the identity \epsilon^2 + 1 = 400\epsilon, which simplifies the gain calculation.

By symmetry, any hunter endpoint off r can be projected onto r without increasing the larger of its distances to Y_1 and Y_2, so the hunter's best final point lies on r. In 200 moves the hunter cannot go farther forward than H'. There is one small endpoint case: if d < \epsilon then Z lies before H', and the hunter could reach Z and be distance exactly 1 from both candidates. But 1 > d^2 + 1/2 in this regime, so the squared distance still grows by more than 1/2. In the main case d \ge \epsilon, the point Z is at or beyond H', so the best the hunter can do is H' itself. Since H'R' = HR = d, the signed length H'Z equals d - \epsilon, and the squared distance y^2 from H' to either Y_i satisfies

$$
y^2 = (d - \epsilon)^2 + 1 = d^2 - 2\epsilon d + \epsilon^2 + 1 = d^2 + \epsilon(400 - 2d).
$$

Because d < 100, we have 400 - 2d > 200, and with \epsilon > 1/400 this yields y^2 > d^2 + 1/2. Thus in every 200-round block the squared distance increases by more than 1/2. Repeating blocks while d < 100, the squared distance exceeds 100^2 after at most 2 * 100^2 = 20000 blocks, using at most 200 * 20000 = 4 * 10^6 rounds, far fewer than 10^9. Once d > 100, the rabbit simply moves one unit directly away from the hunter each round; the distance rises by one before the hunter moves, and the hunter's one-unit response cannot pull it below its previous value. Therefore no hunter strategy can guarantee distance below 100 after 10^9 rounds.

The following Python script verifies the key inequality for representative distances and simulates the repeated block argument, confirming that the rabbit escapes past 100 well before a billion rounds.

```python
import math

BLOCK = 200                      # rounds per symmetric ambiguity block
TARGET = 100                     # hunter's desired distance bound
EPS = BLOCK - math.sqrt(BLOCK**2 - 1)  # = 1 / (BLOCK + sqrt(BLOCK^2 - 1))

def lower_squared_after_block(d2):
    """Lower bound on the rabbit's squared distance after one block."""
    d = math.sqrt(d2)
    if d < EPS:
        # Hunter can reach Z, but even then squared distance is 1 > d^2 + 1/2.
        return 1.0
    # Best hunter point is H', giving y^2 = d^2 + eps*(400 - 2d).
    return d2 + EPS * (2 * BLOCK - 2 * d)

def rounds_to_escape(target=TARGET):
    """Simulate repeated blocks until the distance exceeds target."""
    d2 = 0.0
    rounds = 0
    while d2 < target * target:
        d2 = lower_squared_after_block(d2)
        rounds += BLOCK
    return rounds, math.sqrt(d2)

def squared_gain(d):
    """Additive gain in squared distance from a single block."""
    d2 = d * d
    return lower_squared_after_block(d2) - d2

if __name__ == "__main__":
    print(f"epsilon = {EPS:.8f}")
    print(f"1/400 = {1/400:.8f}, 1/200 = {1/200:.8f}")
    for d in [0.0, 0.1, 1.0, 10.0, 50.0, 99.9]:
        gain = squared_gain(d)
        print(f"d = {d:5.1f}: squared-distance gain = {gain:.6f}, "
              f"gain > 1/2? {gain > 0.5}")
    rounds, final_d = rounds_to_escape()
    print(f"\nRounds to exceed distance {TARGET}: {rounds}")
    print(f"Final distance after escape: {final_d:.3f}")
    print(f"Escapes within 10^9 rounds? {rounds < 1_000_000_000}")
```
