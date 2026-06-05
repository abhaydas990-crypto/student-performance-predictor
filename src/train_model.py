# src/train_model.py

import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import (
    load_data,
    create_preprocessor
)


DATA_PATH = "../data/StudentsPerformance.csv"

MODEL_PATH = "../models/student_score_model.pkl"


def train():

    X, y = load_data(DATA_PATH)

    preprocessor = create_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\nModel Evaluation")
    print("-" * 30)

    print(
        "MAE:",
        mean_absolute_error(y_test, y_pred)
    )

    mse = mean_squared_error(y_test, y_pred)

    print("RMSE:", np.sqrt(mse))

    print(
        "R2 Score:",
        r2_score(y_test, y_pred)
    )

    os.makedirs("../models", exist_ok=True)

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print("\nModel saved successfully.")


if __name__ == "__main__":
    train()