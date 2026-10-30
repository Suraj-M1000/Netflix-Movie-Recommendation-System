# Dataset Information

This project uses the [Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data), available on Kaggle.

The project uses two files:

| File | Description |
|---|---|
| `combined_data_1.csv` | Movie ratings containing user IDs, ratings, and dates. This is the first of four combined rating files in the Netflix Prize dataset. |
| `movie_titles.csv` | Movie ID, release year, and movie title lookup used to display recommendations. |

## Dataset Download and Setup

The raw dataset files are not included in this GitHub repository because of their size.

To run the project:

1. Download the Netflix Prize dataset from [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
2. Locate `combined_data_1.csv` and the movie titles file.
3. Place both files inside this `data/` directory.
4. Ensure the movie titles file is named `movie_titles.csv`.

Your directory should look like this:

```text
data/
├── combined_data_1.csv
├── movie_titles.csv
└── README.md
```

## Pretrained Model

A pretrained SVD model is included in the repository:

```text
models/Final_Svd_Model.pkl
```

The model can be loaded for recommendation inference without retraining. However, generating recommendations for arbitrary users also requires the relevant user-rating data and movie-title lookup.

**Important:** Only load the pickle model from this trusted project repository. Python pickle files can execute code during deserialization.