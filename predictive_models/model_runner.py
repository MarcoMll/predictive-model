from utils.serialization.model_serializer import load_model
from utils.data_management.data_analysis import get_random_row
from pandas import DataFrame, Series

def run_model(model_name: str, split: tuple[DataFrame, Series], random_passenger: DataFrame = None):
    model, features_names = load_model(model_name)
    x_data, y_data = split

    if random_passenger is None:
        random_passenger = get_random_row(x_data)
    actual_value = y_data.loc[random_passenger.index[0]]

    random_passenger = random_passenger[features_names]
    prediction = model.predict(random_passenger)

    print("Prediction:")
    if prediction[0] == 0:
        print("Survived")
    else:
        print("Not survived")

    print("Actual:")
    if actual_value == 0:
        print("Survived")
    else:
        print("Not survived")
