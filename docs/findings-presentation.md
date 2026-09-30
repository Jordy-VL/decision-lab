# Findings presentation preference

User preference recorded 2026-09-30: use the presentation style of Eli Stewart's
[A practical guide to extraction confidence scores](https://www.llamaindex.ai/blog/what-makes-an-extraction-confidence-score-useful)
(LlamaIndex, September 16, 2026) as a reference for Decision Lab findings.

The reference begins with an operational question, explains threshold choice,
and uses an interactive cutoff with explicit accepted-correct, accepted-wrong
and review counts. Comparisons and methodology follow. It exposes supporting
tables and explains metric denominators. These are presentation observations,
not independent validation of the article's product comparisons.

## Application to our findings

- Lead with the decision: how much can we accept at the chosen error target?
- Show our frozen calibration threshold applied to test by default. Put any
  interactive test-threshold exploration in a visibly exploratory view.
- Display correct accepts, incorrect accepts, deferred decisions, accepted
  error rate and coverage with integer counts and named denominators.
- Pair each figure with one finding and its practical implication; put formula
  detail, provenance and complete tables after the main explanation.
- Keep methods, seeds, source slices and raw/calibrated variants identifiable.
- Show uncertainty and failed targets as clearly as improvements. Our present
  finding is that calibration has not achieved the intended 5% test-risk target.
- Use measured operating points, accessible labels and downloadable values.
  Show cost comparisons only when equivalent measurement scope is available.

The [confidence evaluation protocol](confidence-evaluation-protocol.md) remains
authoritative for metrics and threshold fitting. This note guides the next
report presentation; it does not change numerical results or the manuscript.
