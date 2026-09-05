# Scientific Paper Citation Link Prediction

**Python · PyTorch · Hugging Face Transformers · SPECTER · Doc2Vec · FastText · CatBoost**

An NLP and machine-learning pipeline for predicting whether one scientific paper cites another. Built for the **GammaFest Data Science Competition 2025 at Institut Pertanian Bogor (IPB)**, where the finalist team competed under the name **Manusia Pojok**.

The pipeline models **410K+ paper-reference pairs** by combining full text, titles, concepts, authors, publication attributes, and citation metadata. Its final CatBoost experiment achieved **0.5842 validation MCC** after oversampling and MCC-based decision-threshold optimization. The final private leaderboard placed Manusia Pojok **4th with 0.5688 MCC**.

[Read the project story and retrospective on LinkedIn](https://www.linkedin.com/posts/raffyzeidan_this-post-is-mainly-here-as-a-reference-for-ugcPost-7496932806653194241-q6xp/)

## Problem

Finding relevant academic references is difficult when a literature collection becomes large. The task is framed as imbalanced binary classification: given a source paper and a candidate reference paper, predict whether a citation link exists. Submissions were evaluated with Matthews Correlation Coefficient (MCC), which is suitable for imbalanced labels.

The data contains 4,354 scientific papers and more than 410,000 labeled paper-reference pairs. Each paper may contribute full text, title, concepts, authors, publication information, and citation statistics.

## Approach

<p align="center">
  <img src="assets/methodology.jpg" alt="Scientific paper citation prediction methodology: Doc2Vec, FastText, SPECTER, pairwise vector features, metadata features, and a tree-based model" width="760">
</p>

<p align="center"><em>Original problem-solving architecture shared in the project’s LinkedIn retrospective.</em></p>

The original deep-learning-heavy direction was constrained by GPU memory and a roughly ten-day competition window. The final hybrid solution combined:

- **Full-text representation:** Doc2Vec context vectors for long papers, selected for their lower compute cost and lack of a short transformer context limit.
- **Title and concept representation:** FastText plus the domain-specific `allenai/specter` transformer through Hugging Face and PyTorch.
- **Pairwise semantic relevance:** cosine similarity, Euclidean and Manhattan distance, Pearson correlation, angular relationships, directional projections, vector differences, and aggregate statistics.
- **Structured signals:** author, publication, temporal, citation, popularity, and other metadata-derived features.
- **Final feature space:** approximately 258 engineered features describing the relationship between each source and candidate reference paper.
- **Imbalanced classification:** random oversampling followed by a GPU-trained CatBoost classifier.
- **Decision optimization:** probability-threshold search using validation MCC rather than assuming a fixed 0.5 cutoff.

Several tree-based and neural classifiers were explored. The final pipeline used CatBoost after feature and model comparisons showed that Doc2Vec relationships and temporal features were among the strongest signals.

## Repository contents

| File | Description |
| --- | --- |
| [`main.ipynb`](main.ipynb) | Original Kaggle competition notebook and modeling pipeline. |
| [`DSC25125_Manusia Pojok_Laporan.pdf`](DSC25125_Manusia%20Pojok_Laporan.pdf) | Six-page technical report in Indonesian. |
| [`DSC25125_Manusia Pojok_Presentasi.pdf`](DSC25125_Manusia%20Pojok_Presentasi.pdf) | Competition presentation deck. |
| [`FINALIS - Mohammad Raffy Zeidan.pdf`](FINALIS%20-%20Mohammad%20Raffy%20Zeidan.pdf) | Finalist certificate. |
| [`PESERTA - Mohammad Raffy Zeidan.pdf`](PESERTA%20-%20Mohammad%20Raffy%20Zeidan.pdf) | Participant certificate. |

## Running the notebook

The notebook was developed for Kaggle and expects the private competition dataset at:

```text
/kaggle/input/gammax/
├── train.csv
├── test.csv
├── papers_metadata.csv
└── Paper Database/
    └── Paper Database/
```

To reproduce it:

1. Create a Kaggle notebook with the competition data mounted as `gammax`.
2. Select a GPU accelerator; the final CatBoost configuration uses `task_type="GPU"`.
3. Install the dependencies from `requirements.txt` if they are not already available.
4. Run `main.ipynb` from top to bottom.

The notebook downloads the `allenai/specter` checkpoint and NLTK resources at runtime. Exact reproducibility also depends on access to the original GammaFest data, which is not redistributed here.

## Results

| Evaluation | MCC | Notes |
| --- | ---: | --- |
| Notebook validation split | **0.5842** | Best threshold: 0.1189; recorded directly in the committed notebook output. |
| Final private leaderboard | **0.5688** | 4th place, shown in the competition leaderboard image below. |

The report also documents the following model progression:

| Configuration | Reported MCC |
| --- | ---: |
| CatBoost + raw TF-IDF document/metadata features | 0.320 |
| Tuned LightGBM + TF-IDF/cosine document features | 0.410 |
| Tuned Random Forest + TF-IDF/cosine document features | 0.470 |
| CatBoost + Doc2Vec document features | 0.500 |
| CatBoost + Doc2Vec + FastText + engineered metadata | 0.520 |
| CatBoost + Doc2Vec + FastText + SPECTER + engineered pairwise features | 0.540 |
| Tuned CatBoost with the full feature set | **0.568** |

The progression numbers are transcribed from the submitted report. Validation and leaderboard MCC are reported separately because they come from different data partitions and should not be treated as interchangeable.

<p align="center">
  <img src="assets/leaderboard.jpg" alt="GammaFest IPB 2025 private leaderboard showing Manusia Pojok in fourth place with a score of 0.5688" width="760">
</p>

<p align="center"><em>Final private leaderboard shown in the LinkedIn post: 4th place, MCC 0.5688.</em></p>

## Team

- Bryant Farrel
- Mohammad Raffy Zeidan
- Evans Kizito

## Limitations and future work

- The competition dataset is not included, so the notebook is not self-contained.
- The pipeline was produced under strict time and compute constraints and contains exploratory notebook code rather than a packaged training application.
- Validation threshold selection and oversampling should be revisited with a rigorous cross-validation protocol before production use.
- The SPECTER representation and pair construction deserve further experimentation.
- Citation graphs naturally motivate graph-based models such as GNNs, which were outside the competition-time implementation.

## Acknowledgements

Developed by team **Manusia Pojok** for the **GammaFest Data Science Competition 2025** organized by Institut Pertanian Bogor (IPB). The original project files were migrated from the author's consolidated competition repository while preserving their Git history.
