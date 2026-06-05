# src/predict.py

import os
import joblib
import pandas as pd


# Get project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Build absolute path to model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_score_model.pkl"
)

# Load model once
model = joblib.load(MODEL_PATH)


def predict_math_score(
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
):

    sample = pd.DataFrame({

        "gender": [gender],

        "race/ethnicity": [race_ethnicity],

        "parental level of education":
            [parental_level_of_education],

        "lunch": [lunch],

        "test preparation course":
            [test_preparation_course],

        "reading score": [reading_score],

        "writing score": [writing_score]

    })

    prediction = model.predict(sample)

    return round(float(prediction[0]), 2)


if __name__ == "__main__":

    result = predict_math_score(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="completed",
        reading_score=85,
        writing_score=88
    )

    print(f"Predicted Math Score: {result}")