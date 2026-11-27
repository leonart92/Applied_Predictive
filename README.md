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

The image models need the images themselves, under `data/CrisisMMD_v2.0/data_image/`. `python src/fetch_images.py` downloads the 1.8 GB `CrisisMMD_v2.0` archive from that snapshot with resumable parallel range requests, extracts only its `data_image` subtree, and then opens and fully decodes each of the 12,708 images the splits reference, reusing the checker of `src/prepare_data.py`. It writes `results/tables/image_fetch_report.json` and fails rather than silently dropping an unusable image.

## Models

| Model | Input | Status |
|---|---|---|
| TF-IDF + logistic regression | text | available: `notebooks/02_baseline_tfidf.ipynb`, `notebooks/02b_tfidf_protocol_b.ipynb` |
| Fine-tuned BERTweet (`vinai/bertweet-base`) | text | available: `src/bertweet.py` |
| Fine-tuned Vision Transformer (`google/vit-base-patch16-224-in21k`) | image | available: `src/vit.py` |
| CLIP embeddings fusion + logistic regression (`openai/clip-vit-base-patch32`) | text + image | available: `src/clip_fusion.py` |
| Zero-shot LLM (`llama3.2:latest`) | text | available: `src/llm_zero_shot.py` |

## Evaluation

- **Protocol A, in-domain.** Train on `train`, select on `dev`, evaluate once on `test`. Scores are comparable to Ofli et al. (2020).
- **Protocol B, unseen disaster.** Hold one disaster out of training and test on it. This is the question the project focuses on: do the models still work on a disaster they have never seen?

The main metric is macro F1, because the classes are imbalanced. Weighted F1, accuracy and per-class scores are reported next to it.

## Results so far

Protocol A (in-domain test split, seed 42):

| Task | Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|---|
| 1 · Informative (1,534 pairs) | TF-IDF + logistic regression | 0.786 | 0.817 | 0.825 |
| 1 · Informative (1,534 pairs) | Fine-tuned BERTweet | 0.859 | 0.876 | 0.875 |
| 1 · Informative (1,534 pairs) | Fine-tuned ViT (image only) | 0.849 | 0.865 | 0.864 |
| 1 · Informative (1,534 pairs) | **CLIP fusion + logistic regression** | **0.922** | 0.931 | 0.931 |
| 1 · Informative (1,534 pairs) | Zero-shot LLM (`llama3.2:latest`) | 0.647 | 0.655 | 0.649 |
| 2 · Humanitarian (955 pairs) | TF-IDF + logistic regression | 0.634 | 0.744 | 0.740 |
| 2 · Humanitarian (955 pairs) | Fine-tuned BERTweet | 0.697 | 0.808 | 0.802 |
| 2 · Humanitarian (955 pairs) | Fine-tuned ViT (image only) | 0.743 | 0.828 | 0.828 |
| 2 · Humanitarian (955 pairs) | **CLIP fusion + logistic regression** | **0.820** | 0.880 | 0.879 |
| 2 · Humanitarian (955 pairs) | Zero-shot LLM (`llama3.2:latest`) | 0.554 | 0.659 | 0.686 |

For reference, the text-only model of Ofli et al. (2020) reaches weighted F1 scores of 80.9 on Task 1 and 67.7 on Task 2 on the same splits. Detailed Protocol A, Protocol B (14 unseen-disaster folds) and error-analysis reports for the TF-IDF baseline are available in `reports/`.

Multimodal fusion is the best model on both tasks, and its bootstrap interval overlaps no text-only model: +0.063 macro F1 over BERTweet on Task 1, +0.123 on Task 2. The image alone is enough to match the text on Task 1 and to beat it on Task 2, where deciding *which kind* of aid a tweet is about reads better from the photo than from 140 characters. Ablating the fusion (`results/tables/clip_fusion_ablation.csv`) separates the two halves:

| Task | Text only (512) | Image only (512) | Fusion (1,024) |
|---|---|---|---|
| 1 · Informative | 0.864 | 0.884 | **0.922** |
| 2 · Humanitarian | 0.693 | 0.768 | **0.820** |

`reports/vit_clip_fusion_report.md` details both image models, including the leakage controls run before publishing these scores.

Protocol B (macro F1 on the test pairs of the held-out event; the in-domain column is the Protocol A model scored on those same pairs, so the gap is the cost of never having seen the disaster):

| Task | Model | Held-out event | Test pairs | In-domain | Unseen event | Gap |
|---|---|---|---|---|---|---|
| 1 · Informative | Fine-tuned BERTweet | Hurricane Maria | 359 | 0.874 | 0.876 | −0.002 |
| 1 · Informative | Fine-tuned ViT | Hurricane Maria | 359 | 0.816 | 0.807 | +0.009 |
| 1 · Informative | CLIP fusion | Hurricane Maria | 359 | 0.936 | 0.910 | +0.026 |
| 1 · Informative | Fine-tuned BERTweet | California wildfires | 179 | 0.700 | 0.664 | +0.035 |
| 1 · Informative | Fine-tuned ViT | California wildfires | 179 | 0.828 | 0.584 | **+0.244** |
| 1 · Informative | CLIP fusion | California wildfires | 179 | 0.812 | 0.755 | +0.056 |
| 2 · Humanitarian | Fine-tuned BERTweet | Hurricane Maria | 232 | 0.860 | 0.786 | +0.074 |
| 2 · Humanitarian | Fine-tuned ViT | Hurricane Maria | 232 | 0.607 | 0.583 | +0.024 |
| 2 · Humanitarian | CLIP fusion | Hurricane Maria | 232 | 0.857 | 0.893 | −0.036 |
| 2 · Humanitarian | Fine-tuned BERTweet | California wildfires | 78 | 0.487 | 0.382 | +0.106 |
| 2 · Humanitarian | Fine-tuned ViT | California wildfires | 78 | 0.517 | 0.453 | +0.065 |
| 2 · Humanitarian | CLIP fusion | California wildfires | 78 | 0.574 | 0.539 | +0.035 |

Holding out a hurricane costs almost nothing: three other hurricanes remain in training. Holding out the only wildfire is what hurts, and it hurts the image-only model most — the ViT loses 0.244 macro F1 on Task 1, the largest drop of any model in the project, because nothing in floods and rubble prepares it for smoke and flames. The fusion absorbs it (−0.056): the text still says "wildfire". The image helps most in a known domain, and the text is what rescues the model when the disaster type is new.

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

Train the image model and the fusion model. Both need the images, so run `python src/fetch_images.py` first; both follow the same CLI, protocols and output layout as BERTweet:

```bash
python src/vit.py informative                                        # Protocol A, task 1
python src/vit.py humanitarian
python src/vit.py informative --event hurricane_maria                # Protocol B
python src/vit.py informative --event california_wildfires
python src/vit.py humanitarian --event hurricane_maria
python src/vit.py humanitarian --event california_wildfires

python src/clip_fusion.py informative --ablation                     # Protocol A, task 1
python src/clip_fusion.py humanitarian --ablation
python src/clip_fusion.py informative --event hurricane_maria        # Protocol B
python src/clip_fusion.py informative --event california_wildfires
python src/clip_fusion.py humanitarian --event hurricane_maria
python src/clip_fusion.py humanitarian --event california_wildfires
```

The ViT is fine-tuned with BERTweet's recipe (batch 32, lr 2e-5, 5 epochs, patience 2, class-weighted loss, best epoch on dev macro F1), so the difference between the two models comes from the modality and not from the tuning. Each run takes about 25 to 40 minutes on an Apple Silicon GPU; `--batch-size` and `--workers` are available.

CLIP is never fine-tuned: its frozen encoders turn each tweet and each image into a 512-dimension vector, the two are L2-normalized and concatenated, and the classifier on top is the TF-IDF baseline's logistic regression, with the same C grid selected on dev. The embeddings of all 12,708 images and of every text are computed once into `cache/` (about four minutes), so each later run takes seconds. `--ablation` scores text-only, image-only and fusion features into `results/tables/clip_fusion_ablation.csv`.

Redraw the figures and confusion-matrix tables of any model from the metrics already written, without running a model again:

```bash
python src/plot_metrics.py clip_fusion vit
```

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
│   ├── fetch_images.py      # image archive download, extraction and validation
│   ├── data.py              # shared split loader for protocols A and B
│   ├── evaluate.py          # shared evaluation metrics and bootstrap CI
│   ├── images.py            # shared image loading for the image and fusion models
│   ├── bertweet.py          # BERTweet fine-tuning, Protocol A and B predictions
│   ├── vit.py               # Vision Transformer fine-tuning, image only
│   ├── clip_fusion.py       # CLIP embedding fusion of text and image
│   ├── plot_metrics.py      # figures and matrix tables from the saved metrics
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
├── reports/                 # data preparation, TF-IDF and image/fusion reports
├── tests/                   # unit tests for data prep, splits, evaluation and image models
├── data/                    # raw TSVs and images (not versioned)
├── cache/                   # CLIP embeddings, rebuilt on demand (not versioned)
└── requirements.txt         # pinned versions
```

## References

1. Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. *ICWSM*.
2. Ofli, F., Alam, F., & Imran, M. (2020). Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response. *ISCRAM*.
3. Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. *EMNLP*.
