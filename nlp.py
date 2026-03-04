"""ml/nlp.py — journal entry sentiment & stress analysis"""
import re

STRESS_HIGH   = ["overwhelmed","exhausted","hopeless","depressed","anxious","burned","failing","impossible","crying","desperate","miserable","breaking"]
STRESS_MEDIUM = ["tired","stressed","worried","struggling","behind","confused","nervous","lost","upset","drained"]
POSITIVE      = ["happy","excited","motivated","confident","proud","great","amazing","enjoying","love","focus","accomplished","energized","grateful"]

def analyze_journal_entry(text: str) -> dict:
    tokens = re.sub(r"[^a-z\s]", "", text.lower()).split()
    hi = [w for w in STRESS_HIGH   if w in tokens]
    md = [w for w in STRESS_MEDIUM if w in tokens]
    po = [w for w in POSITIVE      if w in tokens]

    stress = len(hi) * 3 + len(md) * 1.5
    pos    = len(po) * 2
    total  = stress + pos + 1

    em = min(0.95, stress / total)
    sentiment = "Negative" if em > 0.6 else "Positive" if em < 0.35 else "Neutral"

    return {
        "emotional_score": round(em, 3),
        "sentiment":       sentiment,
        "stress_words":    (hi + md)[:6],
        "positive_words":  po[:4],
    }
