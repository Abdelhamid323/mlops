import pandas as pd

from model_pipeline import FEATURES, prepare_data


def test_prepare_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # le scaler.joblib du test est écrit ici

    df = pd.DataFrame(
        {
            "RowNumber": range(1, 11),
            "CustomerId": range(101, 111),
            "Surname": ["A"] * 10,
            "CreditScore": [600, 700, 650, 800, 550, 620, 710, 690, 580, 640],
            "Geography": ["France"] * 10,
            "Gender": ["Male", "Female"] * 5,
            "Age": [30, 40, 35, 50, 45, 28, 60, 33, 41, 38],
            "Tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
            "Balance": [0.0, 1000.0, 2000.0, 0.0, 500.0, 0.0, 300.0, 0.0, 10.0, 0.0],
            "NumOfProducts": [1, 2, 1, 2, 1, 1, 2, 1, 3, 1],
            "HasCrCard": [1, 0] * 5,
            "IsActiveMember": [0, 1] * 5,
            "EstimatedSalary": [50000.0] * 10,
            "Exited": [0, 1] * 5,
        }
    )

    x_train, x_test, y_train, y_test = prepare_data(df)

    assert list(x_train.columns) == FEATURES
    assert len(x_train) == 8 and len(x_test) == 2
    assert not x_train.isnull().values.any()
    assert abs(x_train["Age"].mean()) < 1e-6