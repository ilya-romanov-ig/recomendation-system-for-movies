import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def create_rating_matrix(ratings_df: pd.DataFrame):
    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()

    user_to_idx = {id: idx for idx, id in enumerate(user_ids)}
    movie_to_idx = {id: idx for idx, id in enumerate(movie_ids)}

    user_indicies = ratings_df['userId'].map(user_to_idx)
    movies_indicies = ratings_df['movieId'].map(movie_to_idx)
    ratings = ratings_df['rating'].values

    sparse_matrix = csr_matrix(
        (ratings, (user_indicies, movies_indicies)),
        shape=(len(user_ids), len(movie_ids))
    )

    return sparse_matrix, user_to_idx, movie_to_idx, user_ids, movie_ids

def train_test_split_by_user(rating_matrix : csr_matrix, test_size=0.2, random_state=42):
    train_data = []
    test_data = []

    coo_matrix = rating_matrix.tocoo()

    for user_id in range(rating_matrix.shape[0]):
        mask = coo_matrix.row == user_id
        indicies = np.where(mask)[0]

        if len(indicies) > 1:
            train_idx, test_idx = train_test_split(
                indicies,
                random_state=random_state,
                test_size=test_size
            )

            for idx in train_idx:
                train_data.append((
                    coo_matrix.row[idx],
                    coo_matrix.col[idx],
                    coo_matrix.data[idx]
                ))
        
            for idx in test_idx:
                test_data.append((
                    coo_matrix.row[idx],
                    coo_matrix.col[idx],
                    coo_matrix.data[idx]
                ))

        if train_data:
            row, col, data = zip(*train_data)
            train_matrix = csr_matrix(
                (data, (row, col)),
                shape=rating_matrix.shape
            )
        else:
            train_matrix = csr_matrix(rating_matrix.shape)
            
        if test_data:
            row, col, data = zip(*test_data)
            test_matrix = csr_matrix(
                (data, (row, col)),
                shape=rating_matrix.shape
            )
        else:
            test_matrix = csr_matrix(rating_matrix.shape)

    return train_matrix, test_matrix

def eval_model(model, test_matrix):
    actual_rating = []
    predicted_rating = []

    rows, cols = test_matrix.shape

    for row in range(rows):
        for col in range(cols):
            rating = test_matrix[row, col]
            if rating > 0:
                pred_rating = model._predict_rating(row, col)
                predicted_rating.append(pred_rating)
                actual_rating.append(rating)

    metrics = {
        'rmse': root_mean_squared_error(actual_rating, predicted_rating),
        'mae': mean_absolute_error(actual_rating, predicted_rating)
    }

    return metrics, actual_rating, predicted_rating
