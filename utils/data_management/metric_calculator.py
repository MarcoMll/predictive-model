import statistics
import pandas as pd


def find_column_median(df: pd.DataFrame, column: str):
    """Only for numeric columns"""
    if df[column].dtype != "int64" and df[column].dtype != "float64":
        raise TypeError("find_column_median() function works only with numeric columns")

    data = df[column].dropna() # getting all non-null values from this column
    return statistics.median(data)

def find_column_mode(df: pd.DataFrame, column: str):
    """Only for string columns"""
    if df[column].dtype != "str":
        raise TypeError("find_column_mode() function works only with string columns")

    data = df[column].dropna()
    return statistics.mode(data)