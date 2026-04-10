from utils.data_management.data_analysis import *
from utils.data_management.metric_calculator import *
from predictive_models.decision_tree import train_classification_tree
from predictive_models.xgboost_model import train_xgboost_classifier
from predictive_models.model_runner import run_model

import pandas as pd

if __name__ == '__main__':
    # config
    df = pd.read_csv("Titanic-Dataset.csv")
    prediction_target = "Survived"

    # populating age with median
    populate_nan_columns(df, "Age", find_column_median(df, "Age"))

    # populating embarked location with mode
    populate_nan_columns(df, "Embarked", find_column_mode(df, "Embarked"))

    # removing redundant columns
    remove_columns(df, ["PassengerId", "Ticket", "Cabin", "Name"])

    # converting every categorical variables to dummy
    # changing df's structure on every iteration so it better to store the categorical columns from the beginning
    categorical_columns = [column for column in df.columns if df[column].dtype == "str"]

    for column in categorical_columns:
        if df[column].dtype == "str":
            df = convert_to_dummy(df, column)

    #run_eda(df, prediction_target)

    # splitting the data set into three
    train_set, test_set, validation_set = split_dataset(df, 0.2, 0.2, prediction_target)

    # splitting features and target for each set
    train_split = split_features_and_target(train_set, prediction_target)
    test_split = split_features_and_target(test_set, prediction_target)
    validation_split = split_features_and_target(validation_set, prediction_target)

    '''
    # making sure every x split doesn't have null or categorical values
    # and every y split is only 0/1
    print(f"========== Train X ==========\n")
    run_eda(train_split[0], sections=["missing_values", "dtypes"])
    print(f"Train Y\n")
    run_eda(train_split[1], sections=["dtypes"])
    print(f"The dimensions match: {len(train_split[0]) == len(train_split[1])}")

    print(f"========== Test X ==========\n")
    run_eda(test_split[0], sections=["missing_values", "dtypes"])
    print(f"Test Y\n")
    run_eda(test_split[1], sections=["dtypes"])
    print(f"The dimensions match: {len(test_split[0]) == len(test_split[1])}")

    print(f"========== Validation X ==========\n")
    run_eda(validation_split[0], sections=["missing_values", "dtypes"])
    print(f"Validation Y\n")
    run_eda(validation_split[1], sections=["dtypes"])
    print(f"The dimensions match: {len(validation_split[0]) == len(validation_split[1])}")
    '''

    # training models
    # train_classification_tree(train_split, test_split, validation_split)
    # train_xgboost_classifier(train_split, test_split, validation_split)

    # running models
    random_passenger = get_random_row(test_split[0]) # getting the same passenger for both

    print("XGBoost")
    run_model("xgboost", test_split, random_passenger)
    print(f"\n\nClassification Tree")
    run_model("classification_tree", test_split, random_passenger)