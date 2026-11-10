# Court Case Forecasting System

**Forecast how long an Indian district-court case will stay pending, with an honest measure of uncertainty.**

This project turns millions of historical e-Courts records into a tool that answers one question:  
*"Given a case's court, type, act and filing date, how likely is it to still be open after 24 months?"*
It never predicts who wins. It predicts **time**, nothing else.

---

## The problem

Indian courts carry a huge backlog. Millions of cases stay pending for years, and that delay is costly for
the people waiting, for court administrators who plan capacity, and for researchers studying access to justice.

Two things make delay hard to study:

1. **Most cases are not finished when the data is collected.** If you only look at decided cases, you
   throw away the slowest ones and badly underestimate how long cases take. This is the classic
   *right-censoring* problem, and ordinary regression handles it poorly.
2. **A single predicted number is misleading.** "This case will take 14 months" sounds precise but hides
   large uncertainty. A useful tool must show how sure it is.

## What we built

A complete, tested pipeline from raw court records to a working service:

| Part | What it does |
|---|---|
| **Data pipeline** | Reads the Development Data Lab e-Courts files, cleans them, marks cases still open as censored, and writes a versioned Parquet table |
| **Models** | Gradient-boosted baseline, Cox proportional hazards, Random Survival Forest (DeepHit optional) |
| **Uncertainty** | Conformal intervals around the probability of a 24-month delay |
| **Validation** | Random, **temporal** (train on earlier years, test on later) and **spatial** (hold out whole districts) splits |
| **Dynamic forecasts** | Landmark models that re-forecast at filing and at the first hearing |
| **Explainability** | Feature-group importance and ablation, plus performance by court size |
| **API** | FastAPI service returning a survival curve, P(delay > 24 months) and an interval |
| **Dashboard** | Streamlit app with a case view, a court view and a policy simulator |
| **Simulator** | A small SimPy queueing toy for capacity "what-if" questions |

## Why this adds value

- **Handles unfinished cases properly.** Survival analysis uses the information in pending cases instead
  of discarding it.
- **Gives a curve, not one number.** The output is the probability a case is still open at *every* point in
  time, which is more useful for planning than a single estimate.
- **Honest validation.** We report temporal and spatial results next to random-split ones, because
  random splits flatter a model. The temporal split re-censors the training data so no future decisions leak in.
- **Scales to real data.** Built on Polars and DuckDB, and it runs on millions of rows (we used 4.8 million
  Maharashtra cases from 2010 to 2014).
- **Responsible by design.** The tool forecasts duration only. It excludes inferred gender and religion,
  does not rank judges, and shows a disclaimer on every dashboard screen (see `docs/MODEL_CARD.md`).
- **Reproducible.** Tests, pinned dependencies, Docker files and a one-command pipeline.

### Who it is for

Court administrators and policy researchers planning capacity, academics studying delay, and students
learning survival analysis on real data. It is **not** meant to decide anything about an individual case.

## What we found

In our development run on Maharashtra (2012–2014 filings), the Cox model reached a concordance index of
about 0.71 and an AUC of about 0.77 for the 24-month question. Performance held up on districts the model
had never seen, and in small courts. **Case type and act** were by far the most informative features.
Court identity added a little, and congestion and filing season added almost nothing.

Full, regenerated numbers for every split are in [`docs/results.md`](docs/results.md).

## Quick start

```bash
git clone github.com/vaishnaviwangalwar-cpu/Court-Case-Forecast.git
cd court-case-forecast
conda create -n ccf python=3.12 -y && conda activate ccf
pip install -e ".[dev]"

# 1. try it on synthetic demo data (no download needed)
python -m src.pipeline --synthetic --n 20000 --fast
python -m pytest -q

# 2. run it
uvicorn src.api.main:app --reload        # API docs at http://localhost:8000/docs
streamlit run src/dashboard/app.py       # dashboard at http://localhost:8501
```

### Using the real data

1. Download the Development Data Lab judicial dataset (see *Data* below) into `data/raw/courtcasesdata/`.
2. Run the pipeline for one state and a range of years:
   ```bash
   python -m src.pipeline --years 2010 2011 2012 2013 2014 --max-rows 100000
   ```
   `--max-rows` models a random sample so the feature matrix fits in memory. Add `--fast` for a quicker,
   lighter run while developing.

Use a clean conda environment. Mixing pip packages into an existing Anaconda base environment caused
crashes in the math library on our Windows machine.

## How it works

```
raw CSVs ─► ingest ─► clean + censor ─► modelling table (Parquet, versioned)
                                              │
        at-filing features ◄──────────────────┘
              │
   baseline · Cox · Random Survival Forest · landmarks
              │
   conformal interval ─► validation (random / temporal / spatial)
              │
   saved model bundle ─► FastAPI  +  Streamlit dashboard
```

**Key design decisions**
- Cases with no decision date are **right-censored** at the extraction date, 25 March 2020.
- Decisions from mid-March 2020 are excluded as pandemic-affected. The exact cut-off is our assumption.
- Features use only what was known **at filing**, so there is no leakage from the outcome.
- Rare courts, case types and acts are grouped into an "other" bucket.

## Project layout

```
src/
  data/         ingest.py, clean.py, features.py, synthetic.py
  models/       baseline.py, cox.py, rsf.py, deephit.py, landmark.py
  uncertainty/  conformal.py
  evaluation/   validate.py, explain.py
  api/          FastAPI service
  dashboard/    Streamlit app (case, court, simulator pages)
  simulation/   SimPy toy simulator
tests/          pytest suite
docs/           methods, limitations, model card, auto-generated results
docker/         API and dashboard containers
```

## Limitations (please read)

- **The hearing history is thin.** The public data gives only a first listing date, not every hearing, so
  dynamic forecasts add little beyond the at-filing model. The 6- and 12-month landmark models are
  poorly calibrated and should not be relied on.
- **The conformal interval is wide.** It guarantees coverage of the yes/no outcome, so it is not very
  informative, and coverage drops under the temporal split.
- **We did not match the published baseline.** Our replicated 24-month AUC fell below the paper's 0.89.
  We have not found out why.
- **The random forest was run in a light configuration** and is not a fair comparison to Cox yet.
- **DeepHit, Docker images and MLflow** were not tested in our environment.
- **Results come from a sample** of the full table, and from one state.
- **Historical records reflect historical backlogs.** Forecasts can carry past inequities. See
  `docs/limitations.md` and `docs/MODEL_CARD.md`.

## Responsible use

Use this for research and aggregate capacity planning. **Do not** use it to predict case outcomes, rank or
evaluate judges or lawyers, or influence the handling of any individual case.

## Data

This project uses the **Development Data Lab judicial data**
(<https://www.devdatalab.org/judicial-data>). The data is **not** included in this repository. Download it
yourself and follow the provider's terms and citation requirements.

## Roadmap

- Gradient-boosted survival model to close the gap with the published baseline
- A more informative uncertainty measure than the current conformal interval
- Train on more years and states, and add the court names from the key files
- Optional: DeepHit, competing-risks (settlement vs. contested disposal), calibrated simulator

## License  

Code: MIT (see `LICENSE`). Documentation: CC BY 4.0. [![License: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)  The data keeps its own terms.

## Author

Vaishnavi Wangalwar — Department of Artificial Intelligence and Data Science, Dr. D Y Patil Institute of Technology, Pune, India.

## Acknowledgements

Development Data Lab for the e-Courts data, and the open-source libraries this builds on:
scikit-survival, lifelines, Polars, DuckDB, FastAPI, Streamlit and SimPy.
