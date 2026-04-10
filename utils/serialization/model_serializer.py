import os
import joblib

from utils.project_paths.path_initializer import get_paths

Paths = get_paths()

def save_model(best_model, training_features, model_name: str):
    os.makedirs(f"{Paths.MODELS_DIR}", exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "feature_names": training_features.columns.tolist(),
        },
        f"{Paths.MODELS_DIR}/{model_name}.joblib"
    )

def load_model(model_name: str):
    model_data = joblib.load(f"{Paths.MODELS_DIR}/{model_name}.joblib")
    return model_data["model"], model_data["feature_names"]