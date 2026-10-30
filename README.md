# Netflix Movie Recommendation System
### Collaborative Filtering using Singular Value Decomposition (SVD)

## 1. Project Overview

This project builds a movie recommendation system using collaborative filtering. It predicts how a user might rate movies they have not watched, based on historical rating patterns from other users.

The underlying assumption is that users who have expressed similar preferences in the past may also have similar preferences for other movies.

**Workflow:**

User ratings → Data preprocessing → Collaborative filtering → Rating prediction → Top-N recommendations

Rating prediction is treated as a **regression problem**, rather than classification, because the numerical distance between predicted and actual ratings matters. RMSE is therefore used as the primary evaluation metric.

## 2. Why This Project?

Recommendation systems help platforms with large content catalogs provide personalized suggestions.

This project uses data from the Netflix Prize competition to explore collaborative filtering, matrix factorization, model evaluation, and personalized recommendations.

## 3. Dataset

**Source:** [Netflix Prize Dataset – Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

This project uses the first of the four combined rating files.

| File | Description |
|---|---|
| `combined_data_1.csv` | User ratings, movie IDs, and rating dates |
| `movie_titles.csv` | Movie ID, release year, and movie title |

**Dataset statistics:**

- Raw rows: 24,058,263 (including movie-ID marker rows)
- Actual ratings: 24,053,764
- Unique movies: 4,499
- Unique users: 470,758

**Scope limitation:** The dataset does not contain movie genres, cast, plots, or other descriptive metadata. Consequently, this project implements collaborative filtering rather than content-based filtering.

Dataset download and setup instructions are available in [`data/README.md`](data/README.md).

## 4. Methodology

### 4.1 Data Cleaning

Implemented in `src/data_prep.py`.

The raw ratings file contains movie-ID marker rows (for example, `1:`), followed by the ratings associated with that movie.

The preprocessing pipeline uses vectorized operations:

1. Identify movie-ID marker rows using `str.contains(':')`.
2. Extract movie IDs using `np.where`.
3. Propagate movie IDs to subsequent rating rows using forward filling (`ffill()`).
4. Convert IDs into integer format.
5. Remove marker rows and reset the index.

### 4.2 Data Filtering

To reduce sparsity, the project applies quantile-based filtering to movies and users.

- Movies retained: at least 908 ratings (60th percentile).
- Users retained: at least 36 ratings (60th percentile).

**After filtering:**

| Metric | Before | After |
|---|---:|---:|
| Movies | 4,499 | 1,800 |
| Users | 470,758 | 188,716 |
| Ratings retained | — | 81.88% |

This reduces the catalog and user population while retaining most of the original rating interactions.

### 4.3 Model Development

The project uses the [Surprise library](http://surpriselib.com/) for collaborative filtering.

Models were evaluated on a fixed random sample of 1,000,000 rows from the filtered dataset (`random_state=42`) to make experimentation more manageable.

Three approaches were compared:

| Model | Description | RMSE |
|---|---|---:|
| BaselineOnly | Global mean with user and item biases | 0.968 |
| SVD (untuned) | Matrix factorization using default parameters | 0.985 |
| SVD (tuned) | Matrix factorization with tuned hyperparameters | **0.961** |

The tuned SVD model achieved the lowest RMSE.

**Best hyperparameters:**

```python
{
    "n_factors": 20,
    "n_epochs": 30,
    "lr_all": 0.005,
    "reg_all": 0.1
}
```

Hyperparameter tuning was performed using `GridSearchCV`.

### 4.4 Ranking Evaluation

RMSE measures rating prediction accuracy, but it does not directly measure the quality of a ranked recommendation list.

Therefore, Precision@K and Recall@K were also evaluated using per-user predictions from a 75:25 train-test split (`random_state=42`).

**Relevance definition:** A movie is considered relevant when its actual test-set rating is greater than or equal to 4.

The evaluation procedure:

1. Generate predictions for the held-out test set.
2. Group predictions by user.
3. Rank each user's test movies by predicted rating in descending order.
4. Select the top K movies.
5. Compare their actual ratings against the relevance threshold of 4.

Only users with at least K test items are included in each evaluation. This ensures that every evaluated user has enough test items to form a complete top-K list.

| K | Users Evaluated | Precision@K | Recall@K |
|---|---:|---:|---:|
| 5 | 8,052 | 0.5482 | 0.9127 |
| 10 | 371 | 0.4706 | 0.9336 |

**Interpretation:**

- At K=5, approximately 54.82% of the top-five test movies were relevant.
- At K=10, approximately 47.06% of the top-ten test movies were relevant.
- Recall@K measures how many of the relevant movies in each user's test set were captured in the top-K predictions.

**Evaluation limitation:** These metrics evaluate ranking performance among held-out test movies, not against the complete catalog of unseen movies. Furthermore, the K=5 and K=10 results use different user populations because of the minimum test-item requirement.

### 4.5 Recommendation Inference

Implemented in `src/recommend.py`.

The function:

```python
recommend_for_user(
    user_id,
    model,
    movie_rating_df,
    movie_title_df,
    keep_movie_ids,
    n=10
)
```

generates top-N movie recommendations for a given user.

It:
- Checks whether the user exists in the available rating data.
- Excludes movies the user has already rated.
- Predicts ratings for eligible movies.
- Sorts predictions and returns the highest-scoring recommendations.

## 5. Results and Observations

- Tuned SVD achieved an RMSE of 0.961, outperforming the bias-only baseline of 0.968.
- The improvement is modest, suggesting that user and movie rating tendencies already explain a substantial amount of predictable variation.
- At K=5, Precision@K was approximately 0.55 and Recall@K was approximately 0.91.
- Predictions are clipped to the dataset's 1–5 rating scale. Consequently, some top recommendations may have tied predicted ratings of 5.0.

## 6. Visualizations

The `reports/` directory contains visualizations generated during the analysis.

- Rating distribution
- Filtering comparison
- Precision and recall comparison

## 7. Project Structure

```text
Netflix Project/
│
├── data/
│   └── README.md
│
├── models/
│   └── Final_Svd_Model.pkl
│
├── notebooks/
│   ├── Netflix_recommender.ipynb
│   └── Model_Testing.ipynb
│
├── reports/
│   ├── filtering_comparison.png
│   ├── precision_recall_comparison.png
│   └── rating_distribution.png
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   └── recommend.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Installation and Usage

### Prerequisites

- Python 3.10 or compatible version
- Jupyter Notebook
- Required Python packages

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
cd "Netflix Project"
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the dataset

Follow the instructions in [`data/README.md`](data/README.md).

### Step 5: Run the notebook

Start Jupyter:

```bash
jupyter notebook
```

Open `notebooks/Netflix_recommender.ipynb` and execute the cells in order.

For pretrained model loading and recommendation testing, see `notebooks/Model_Testing.ipynb`.

## 9. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Surprise
- Jupyter Notebook
- Collaborative Filtering
- Singular Value Decomposition (SVD)

## 10. Future Improvements

- Experiment with additional Netflix Prize rating files.
- Explore alternative matrix factorization approaches.
- Improve ranking evaluation and recommendation diversity.
- Develop an interactive recommendation interface.

## 11. Disclaimer

This is an educational and portfolio project developed to demonstrate data preprocessing, collaborative filtering, model evaluation, and recommendation inference. It is not affiliated with or endorsed by Netflix.