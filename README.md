# Disaster Tweet Triage on CrisisMMD

Automatic triage of disaster-related tweets for humanitarian response.
DATA4001 final project, TU Dublin, 2026/27.

During a disaster, the information rescue teams need (blocked roads, collapsed buildings, people to rescue, donations) is buried under thousands of irrelevant messages. This project trains models that read a tweet, its text and its attached image, and decide whether it is useful and what kind of help it is about.

It contributes to the UN Sustainable Development Goals **11** (sustainable cities and communities) and **13** (climate action).

## Tasks

| Task | Question | Labels |
|---|---|---|
| 1 · Informative | Is the tweet useful for humanitarian aid? | `informative`, `not_informative` |
| 2 · Humanitarian | What kind of aid is it about? | 5 classes, listed below |

The five humanitarian classes are `affected_individuals`, `infrastructure_and_utility_damage`, `rescue_volunteering_or_donation_effort`, `other_relevant_information` and `not_humanitarian`.

## Data

We use [CrisisMMD v2.0](https://crisisnlp.qcri.org/crisismmd) (Alam et al., 2018): tweets and images collected during seven natural disasters of 2017 (hurricanes Harvey, Irma and Maria, the Mexico and Iraq–Iran earthquakes, the California wildfires and the Sri Lanka floods).

All models share the official "agreed labels" splits of Ofli et al. (2020), which keep only the tweet–image pairs where text and image carry the same label.

| Task | Train (tweets / pairs) | Dev | Test |
|---|---|---|---|
| Informative | 8,293 / 9,601 | 1,573 | 1,534 |
| Humanitarian | 5,263 / 6,126 | 998 | 955 |

The dataset is not stored in this repository. Download `crisismmd_datasplit_agreed_label.zip` from the CrisisMMD page and extract it so that the files sit here:

```
data/crisismmd_datasplit_agreed_label/task_{informative,humanitarian}_text_img_agreed_lab_{train,dev,test}.tsv
```

The official host was unreachable in November 2026. The same archive is available through the Internet Archive's Wayback Machine.

## Models

| Model | Input | Status |
|---|---|---|
| TF-IDF + logistic regression | text | available: `notebooks/02_baseline_tfidf.ipynb`, `notebooks/02b_tfidf_protocol_b.ipynb` |
| Fine-tuned BERTweet (`vinai/bertweet-base`) | text | available: `src/bertweet.py` |
| Fine-tuned Vision Transformer | image | planned |
| CLIP embeddings fusion + logistic regression | text + image | planned |
| Zero-shot LLM (`llama3.2:latest`) | text | available: `src/llm_zero_shot.py` |

## Evaluation

- **Protocol A, in-domain.** Train on `train`, select on `dev`, evaluate once on `test`. Scores are comparable to Ofli et al. (2020).
- **Protocol B, unseen disaster.** Hold one disaster out of training and test on it. This is the question the project focuses on: do the models still work on a disaster they have never seen?

The main metric is macro F1, because the classes are imbalanced. Weighted F1, accuracy and per-class scores are reported next to it.

## Results so far

Protocol A (in-domain test split, seed 42):

| Task | Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|---|
| 1 · Informative (1,534 tweets) | TF-IDF + logistic regression | 0.786 | 0.817 | 0.825 |
| 1 · Informative (1,534 tweets) | Fine-tuned BERTweet | 0.859 | 0.876 | 0.875 |
| 1 · Informative (1,534 tweets) | Zero-shot LLM (`llama3.2:latest`) | 0.647 | 0.655 | 0.649 |
| 2 · Humanitarian (955 tweets) | TF-IDF + logistic regression | 0.634 | 0.744 | 0.740 |
| 2 · Humanitarian (955 tweets) | Fine-tuned BERTweet | 0.697 | 0.808 | 0.802 |
| 2 · Humanitarian (955 tweets) | Zero-shot LLM (`llama3.2:latest`) | 0.554 | 0.659 | 0.686 |

For reference, the text-only model of Ofli et al. (2020) reaches weighted F1 scores of 80.9 on Task 1 and 67.7 on Task 2 on the same splits. Detailed Protocol A, Protocol B (14 unseen-disaster folds) and error-analysis reports for the TF-IDF baseline are available in `reports/`.

Protocol B for BERTweet (macro F1 on the test tweets of the held-out event; the in-domain model is the Protocol A model on the same tweets):

| Task | Held-out event | Test tweets | In-domain | Unseen event | Gap |
|---|---|---|---|---|---|
| 1 · Informative | Hurricane Maria | 359 | 0.874 | 0.876 | −0.002 |
| 1 · Informative | California wildfires | 179 | 0.700 | 0.664 | +0.035 |
| 2 · Humanitarian | Hurricane Maria | 232 | 0.860 | 0.786 | +0.074 |
| 2 · Humanitarian | California wildfires | 78 | 0.487 | 0.382 | +0.106 |

Every score, with its 95% bootstrap interval, is in `results/metrics/`.

## Getting started

Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The six cleaned CSV splits are already versioned in `splits/`. To regenerate them from the raw official TSVs and images (`data/CrisisMMD_v2.0/data_image/`):

```bash
python src/prepare_data.py
```

Run the unit test suite:

```bash
python -m unittest discover -s tests -v
```

Train BERTweet and write its predictions and metrics:

```bash
python src/bertweet.py informative                                   # Protocol A, task 1
python src/bertweet.py humanitarian                                  # Protocol A, task 2
python src/bertweet.py informative --event hurricane_maria           # Protocol B
python src/bertweet.py informative --event california_wildfires
python src/bertweet.py humanitarian --event hurricane_maria
python src/bertweet.py humanitarian --event california_wildfires
```

A Protocol A run also writes the in-domain side of the Protocol B comparison for the two target events (`results/{predictions,metrics}/in_domain/`). A Protocol B run also scores every tweet of the held-out event (`all_target/`).

Task 1 takes about four minutes on an Apple Silicon GPU. The script picks CUDA, then Apple MPS, then the CPU.

To reproduce the TF-IDF baseline and error analysis, run `notebooks/02_baseline_tfidf.ipynb` (Protocol A), `notebooks/02b_tfidf_protocol_b.ipynb` (Protocol B) and `notebooks/07_error_analysis_tfidf.ipynb` using the `.venv` kernel.

Run the zero-shot LLM classifier (`llama3.2:latest` via Ollama at temperature 0, seed 42) and cross-model error analysis:

```bash
python -m src.llm_zero_shot informative --export-protocol-b
python -m src.llm_zero_shot humanitarian --export-protocol-b
python -m src.error_analysis informative
python -m src.error_analysis humanitarian
```

Each run writes `results/predictions/{task}_{model}_{protocol}_{fold}.csv` with one row per test example:

```
tweet_id, image_id, event_name, y_true, y_pred, proba_<class 1>, ..., proba_<class N>
```

Evaluate any prediction file and compute 95% bootstrap confidence intervals (1,000 resamples, seed 42):

```bash
python -m src.evaluate results/predictions/informative_tfidf_A_official.csv
```

## Repository layout

```
.
├── splits/                  # cleaned official agreed-label CSV splits
├── src/
│   ├── prepare_data.py      # TSV + image validation and split generation
│   ├── data.py              # shared split loader for protocols A and B
│   ├── evaluate.py          # shared evaluation metrics and bootstrap CI
│   ├── bertweet.py          # BERTweet fine-tuning, Protocol A and B predictions
│   ├── llm_prompt.txt       # fixed zero-shot prompts for tasks 1 and 2
│   ├── llm_zero_shot.py     # zero-shot LLM inference and Protocol A/B export
│   └── error_analysis.py    # cross-model error analysis and comparison tables
├── notebooks/
│   ├── 02_baseline_tfidf.ipynb
│   ├── 02b_tfidf_protocol_b.ipynb
│   └── 07_error_analysis_tfidf.ipynb
├── results/
│   ├── predictions/         # one CSV per task, model and protocol
│   ├── metrics/             # JSON metrics and bootstrap CIs from src/evaluate.py
│   └── tables/              # summary tables and validation checks
├── figures/                 # confusion matrices and generalization gap plots
├── reports/                 # data preparation and TF-IDF experiment reports
├── tests/                   # unit tests for data prep, splits and evaluation
├── data/                    # raw TSVs and images (not versioned)
└── requirements.txt         # pinned versions
```

## References

1. Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. *ICWSM*.
2. Ofli, F., Alam, F., & Imran, M. (2020). Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response. *ISCRAM*.
3. Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *EMNLP*.
