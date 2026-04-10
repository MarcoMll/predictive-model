from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def print_search_results(search_obj: GridSearchCV | RandomizedSearchCV, splits: list[ tuple[str, tuple] ]):
    """
    Prints best params / CV score and metrics using search_obj.best_estimator_
    """

    best_model = search_obj.best_estimator_

    print("Best params:", search_obj.best_params_)
    print("Best CV score:", round(search_obj.best_score_, 4))

    for split_name, split_data in splits:
        x_data, y_data = split_data
        prediction = best_model.predict(x_data)

        print(f"\n=== {split_name} metrics ===")
        print("Accuracy:", round(accuracy_score(y_data, prediction), 4))
        print("Confusion matrix:\n", confusion_matrix(y_data, prediction))
        print("Report:\n", classification_report(y_data, prediction, zero_division=0))