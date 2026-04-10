from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform, loguniform
from utils.debug.model_report_debugger import print_search_results
from utils.serialization.model_serializer import save_model

def train_xgboost_classifier(train_split: tuple, test_split: tuple, validation_split: tuple):
    x_train, y_train = train_split

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    param_grid = {
        "n_estimators": randint(100, 800),
        "max_depth": randint(2, 9),
        "learning_rate": loguniform(1e-3, 2e-1),
        "subsample": uniform(0.6, 0.4),  # 0.6..1.0
        "colsample_bytree": uniform(0.6, 0.4),  # 0.6..1.0
        "min_child_weight": randint(1, 12),
        "gamma": uniform(0.0, 5.0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(x_train, y_train)

    save_model(random_search.best_estimator_, x_train, "xgboost")
    print_search_results(random_search,[("Validation", validation_split), ("Test", test_split)])