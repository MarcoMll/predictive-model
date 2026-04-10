from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils.debug.model_report_debugger import print_search_results
from utils.serialization.model_serializer import save_model

def train_classification_tree(train_split: tuple, test_split: tuple, validation_split: tuple):
    x_train, y_train = train_split

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 14],
        "min_samples_leaf": [5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    save_model(grid_search.best_estimator_, x_train, "classification_tree")
    print_search_results(grid_search,[("Train", train_split), ("Validation", validation_split),
                                      ("Test", test_split)])