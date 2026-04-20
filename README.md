# Humor Incongruity: Surprisal and Semantic Distance

**"The Same Punchline, Two Different Surprises: Testing Whether Surprisal and Semantic Distance Capture Distinct Dimensions of Humor Incongruity"**

---

## What this does

Tests whether GPT-2 surprisal and SBERT cosine distance predict humor ratings differently depending on joke type (frame-based vs. discourse-centered), using 2,535 jokes from the SemEval 2021 Task 7 dataset.

---

## Project structure

```
humor_incongruity/
├── README.md
├── requirements.txt
├── data/                         # Not included — see Data section below
└── src/
    ├── 01_sbert_distance.py      # Computes SBERT cosine distance per joke
    ├── 02_gpt2_surprisal.py      # Computes GPT-2 surprisal per joke
    └── 03_analysis.py            # Correlations, regression, subgroup analysis
```

Results are saved to a `results/` folder created automatically when the scripts run.

---

## Data

This project uses the **SemEval 2021 Task 7 (HaHackathon)** dataset (Meaney et al., 2021). 

Place the following two CSV files in the `data/` folder before running:

**`jokes_wide.csv`** — full dataset (2,535 jokes)

| Column | Description |
|---|---|
| `setup` | Setup of the joke |
| `punchline` | Punchline of the joke |
| `humor_rating` | Mean humor rating (0–4) |
| `joke_type` | Annotation where available, empty otherwise |

**`jokes_classified.csv`** — 100 manually annotated jokes (subset of the above)

| Column | Description |
|---|---|
| `setup` | Setup of the joke |
| `punchline` | Punchline of the joke |
| `humor_rating` | Mean humor rating (0–4) |
| `joke_type` | `frame-based` or `discourse-centered` |

---

## How to run

```bash
pip install -r requirements.txt

python src/01_sbert_distance.py
python src/02_gpt2_surprisal.py
python src/03_analysis.py
```
