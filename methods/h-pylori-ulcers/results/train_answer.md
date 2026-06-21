Peptic ulcer disease was long explained by Karl Schwarz's 1910 dictum, "no acid, no ulcer." The recurring craters in the stomach and duodenum were treated as chemical burns produced by hydrochloric acid and pepsin overwhelming a weakened mucosal defense, with stress, smoking, personality, and heredity filling in the gaps. The H2-receptor antagonists cimetidine and ranitidine seemed to confirm this: block acid secretion and the ulcer heals. But the clinical course did not fit a simple acid injury. Stop the drug and the ulcer returns, often for years. The disease was chronic, relapsing, familial, and lifelong — a pattern that looks like persistent infection, not chemistry. A second dogma made the mismatch invisible: the stomach was assumed sterile because its acid killed bacteria, so whenever spiral organisms were seen in gastric mucosa they were dismissed as post-mortem contaminants or harmless passengers.

The existing framework therefore failed twice. It could not explain why acid suppression healed lesions without curing disease, and it could not explain why ulcers followed an epidemiological pattern tied to households and poverty rather than to acid secretion alone. Bismuth salts healed ulcers without lowering acid, and a 1981 trial showed bismuth-treated patients relapsed less often than cimetidine-treated patients — a result the acid theory could not interpret. The missing move was to treat the spiral bacteria not as passengers but as pathogens, and to test that hypothesis with the standard tools of medical microbiology rather than pharmacology.

The new method is recognition that the cause of most peptic ulcer disease is infection with Helicobacter pylori, a spiral, gram-negative, microaerophilic bacterium that colonizes the gastric mucosa. The organism is not an opportunistic contaminant; it is the persistent cause of chronic active gastritis and the setting in which ulcers form. Its survival in the stomach depends on two adaptations. First, it lives beneath the mucus layer against the epithelium, where the local pH is near neutral, rather than floating in the acid lumen. Second, it produces abundant urease, which hydrolyzes urea into ammonia and creates an alkaline microenvironment that neutralizes acid at the bacterial surface. This is why a century of observers saw curved bacilli in gastric tissue yet "knew" no bacterium could live there.

The natural history follows from childhood acquisition. Infection is acquired around age two or three as a transient acute vomiting illness with achlorhydria and foul breath, then settles into lifelong asymptomatic chronic gastritis. In susceptible hosts, that smoldering inflammation creates the conditions for acid and pepsin to produce an ulcer decades later. Person-to-person spread within families, often by the fecal-oral route, explains the apparent heritability and the socioeconomic gradient. Acid-suppressing drugs never touch the organism, so healing the crater leaves the infection intact and the ulcer returns; eradicating the organism removes the cause and ends the relapsing course.

Causation was established by the classical microbiological standard, Koch's postulates, because association alone could always be dismissed as the bacteria merely colonizing inflamed tissue. A prospective, blinded study of 100 consecutive endoscopy patients found the organism in every duodenal-ulcer patient (13/13, P = 0.00044), in 77% of gastric ulcers, and in almost all active chronic gastritis. The organism was grown in pure culture on microaerophilic media with longer incubation than routine throat-swab protocols. When an animal model failed, the investigator infected himself with the pure culture after documented normal baseline histology, producing acute gastritis with the organism recoverable from the infected mucosa — satisfying the remaining postulates in the only host that reliably harbors the bacterium. The final proof was interventional: a four-arm trial showed that ulcer relapse tracked eradication of the organism, not acid suppression. Smoking and other "risk factors" lost their predictive power once the infection was cleared.

```python
import numpy as np
from scipy import stats

# Simulated evidence pipeline for H. pylori as the cause of peptic ulcer disease.

np.random.seed(42)

# 1. Prospective association study: 100 consecutive dyspepsia/endoscopy patients.
n_total = 100
# Under the null, H. pylori carriage is independent of duodenal ulcer and ~50% overall.
null_carriage_rate = 0.50
n_duodenal_ulcer = 13

# Observed: 13/13 duodenal ulcer patients carry H. pylori.
observed_positive_du = n_duodenal_ulcer

# Binomial probability of seeing 13/13 positives if carriage were unrelated.
p_association = stats.binomtest(observed_positive_du, n_duodenal_ulcer, null_carriage_rate, alternative='greater')
print(f"Association p-value (13/13 DU positive under 50% carriage): {p_association.pvalue:.2e}")

# 2. Koch's postulates demonstration on a susceptible human host.
class KochDemonstration:
    def __init__(self):
        self.baseline_normal = True
        self.baseline_organism_present = False
        self.infected = False
        self.gastritis = False
        self.recovered = False
        self.organism_recovered = False

    def document_healthy_host(self):
        return self.baseline_normal and not self.baseline_organism_present

    def inoculate(self, pure_culture, acid_suppression=True):
        # Lower acid barrier briefly to allow establishment.
        self.infected = bool(pure_culture) and acid_suppression
        return self.infected

    def observe_disease(self, days_post_infection):
        # Days 5-8: acute illness with achlorhydric vomiting and halitosis.
        # Day 10: histologic gastritis with recoverable organism.
        if 5 <= days_post_infection <= 8:
            return "acute illness: achlorhydric vomiting, halitosis"
        elif days_post_infection == 10:
            self.gastritis = True
            self.organism_recovered = True
            return "histologically proven gastritis; organism recovered"
        elif days_post_infection >= 14:
            # In this acute self-experiment, infection cleared spontaneously.
            self.recovered = True
            self.infected = False
            return "resolving gastritis; organism no longer recoverable"
        return "asymptomatic incubation"

    def recover_organism(self):
        return self.organism_recovered

demo = KochDemonstration()
assert demo.document_healthy_host(), "Baseline must show healthy, organism-free stomach"
demo.inoculate(pure_culture=True, acid_suppression=True)
for d in [7, 10, 14]:
    print(f"Day {d}: {demo.observe_disease(d)}")
assert demo.recover_organism(), "Organism must be recoverable from infected host"

# 3. Intervention trial: eradication prevents relapse, acid suppression does not.
# Simulate 2-year relapse rates for four treatment arms.
arms = {
    "cimetidine": {"heals_ulcer": True, "eradicates_hp": False, "relapse_rate": 0.90},
    "cimetidine + antibiotic": {"heals_ulcer": True, "eradicates_hp": True, "relapse_rate": 0.15},
    "bismuth": {"heals_ulcer": True, "eradicates_hp": "partial", "relapse_rate": 0.60},
    "bismuth + antibiotic": {"heals_ulcer": True, "eradicates_hp": True, "relapse_rate": 0.10},
}

n_per_arm = 40
print("\nTwo-year relapse rates:")
for arm, params in arms.items():
    relapses = np.random.binomial(n_per_arm, params["relapse_rate"])
    print(f"  {arm}: {relapses}/{n_per_arm} relapsed ({relapses/n_per_arm:.0%})")

# Key conclusion: relapse tracks eradication, not acid suppression.
print("\nConclusion: Ulcer relapse is driven by persistent H. pylori infection,")
print("not by acid secretion. Eradication cures the disease; acid suppression treats the lesion.")
```
