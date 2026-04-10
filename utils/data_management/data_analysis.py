import pandas as pd

from sklearn.model_selection import train_test_split
from typing import Literal

EdaSection = Literal["shape", "columns", "target_distribution", "target_ratio", "dtypes", "missing_values", "duplicates"]

def run_eda(df: pd.DataFrame, target:str = None, sections: list[EdaSection] = None):
    if sections is None:
        sections = ["shape", "columns", "target_distribution", "target_ratio", "dtypes", "missing_values", "duplicates"]

    if target is not None:
        y = df[target]
    else:
        y = None

    output: str = ""

    if "shape" in sections:
        output += f"Shape: {df.shape}\n\n"

    if "columns" in sections:
        output += f"Columns: {df.columns.tolist()}\n\n"

    if "target_distribution" in sections and y is not None:
        output += f"Target distribution:\n{y.value_counts()}\n\n"

    if "target_ratio" in sections and y is not None:
        output += f"Target ratio (%):\n{(y.value_counts(normalize=True) * 100).round(2)}\n\n"

    if "dtypes" in sections:
        output += f"Dtypes:\n{df.dtypes}\n\n"

    if "missing_values" in sections:
        output += "Missing values:\n"
        missing = df.isna().sum().to_frame("missing_count")
        missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
        output += f"{missing.sort_values('missing_count', ascending=False)}\n\n"

    if "duplicates" in sections:
        output += f"Duplicates: {df.duplicated().sum()}\n\n"

    print(output)

def populate_nan_columns(df: pd.DataFrame, column: str, value: str | float | int):
    df[column] = df[column].fillna(value)

def remove_columns(df: pd.DataFrame, columns: list[str]):
    df.drop(columns=columns, inplace=True)

def convert_to_dummy(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df[column].dtype != "str":
        raise TypeError("only categorical columns can be converted to dummies")
    return pd.get_dummies(df, columns=[column], dtype="int")

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.2, target: str = "Survived"):
    """Split dataset into train, test and validation sets. The ratio for training set is calculated via:
    1 - (test_size + validation_size)"""

    remaining_data, testing_set = train_test_split(df, test_size=test_size,
                                                   random_state=42, stratify=df[target])

    # After the first split, the remaining data = (1 - test_size) of the data left
    # We need to select a portion of that remaining data, so that it represents 0.2 of the entire dataset
    validation_relative_size = validation_size / (1 - test_size)

    training_set, validation_set = train_test_split(remaining_data, test_size=validation_relative_size,
                                                    random_state=42, stratify=remaining_data[target])

    return training_set, testing_set, validation_set

def get_random_row(df):
    return df.sample(n=1)

def split_features_and_target(df: pd.DataFrame, target: str):
    x = df.drop(columns=[target])
    y = df[target]
    return x, y