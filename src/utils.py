from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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
