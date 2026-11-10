# Model card - Court Case Forecasting
**Task.** Forecast time from filing to disposal (survival curve, P(delay > 24 months), conformal interval).
**Data.** e-Courts records (Development Data Lab), planned scope Maharashtra 2012-2014 then scaled; censored at 25 Mar 2020; pandemic-affected decisions excluded. *Demo bundle in this repo is trained on synthetic data.*
**Intended use.** Research, capacity planning, backlog analysis at court/aggregate level.
**Prohibited use.** Predicting case outcomes or guilt; ranking or evaluating judges, lawyers or parties; influencing bail, sentencing or listing of an individual case; any automated decision about a person.
**Known biases/risks.** Court records reflect historical backlogs and unequal access; case-type labels are normalised imperfectly; small courts have noisier forecasts; excluded pandemic period; dashboard shows a disclaimer on every screen.
**Metrics.** See `docs/results.md` (regenerate on real data); report temporal and spatial splits, not only random.
**Maintenance.** Retrain when new extractions arrive; re-check calibration and coverage per court size.
