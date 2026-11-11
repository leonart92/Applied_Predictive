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
| TF-IDF + logistic regression | text | planned |
| Fine-tuned BERTweet (`vinai/bertweet-base`) | text | available: `src/bertweet.py` |
| Fine-tuned Vision Transformer | image | planned |
| CLIP embeddings fusion + logistic regression | text + image | planned |
| Zero-shot LLM | text | planned |

## Evaluation

- **Protocol A, in-domain.** Train on `train`, select on `dev`, evaluate once on `test`. Scores are comparable to Ofli et al. (2020).
- **Protocol B, unseen disaster.** Hold one disaster out of training and test on it. This is the question the project focuses on: do the models still work on a disaster they have never seen?

The main metric is macro F1, because the classes are imbalanced. Weighted F1, accuracy and per-class scores are reported next to it.

## Results so far

Fine-tuned BERTweet on task 1, protocol A, single run (seed 42), raw official files:

| Split | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| Test (1,534 tweets) | 0.860 | 0.876 | 0.876 |

For reference, the text-only model of Ofli et al. (2020) reaches an F1 of 80.9 on the same split.

## Getting started

Python 3.11 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train BERTweet and write the test predictions:

```bash
python src/bertweet.py                # task 1, informative
python src/bertweet.py humanitarian   # task 2
```

Task 1 takes about four minutes on an Apple Silicon GPU. The script picks CUDA, then Apple MPS, then the CPU.

Each run writes `results/predictions/{task}_{model}_{protocol}_{fold}.csv` with one row per test example:

```
tweet_id, image_id, event_name, y_true, y_pred, proba_<class 1>, ..., proba_<class N>
```

## Repository layout

```
.
├── src/
│   └── bertweet.py          # BERTweet fine-tuning and test predictions
├── results/
│   └── predictions/         # one CSV per task, model and protocol
├── data/                    # not versioned
└── requirements.txt         # pinned versions
```

## References

1. Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. *ICWSM*.
2. Ofli, F., Alam, F., & Imran, M. (2020). Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response. *ISCRAM*.
3. Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *EMNLP*.
