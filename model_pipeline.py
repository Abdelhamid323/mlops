"""
model_pipeline.py — étapes du pipeline churn (extraites du notebook).

Préparation : load_data() importe le CSV, prepare_data() nettoie,
encode, découpe et standardise (via scale_data()).
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_FILE = "Churn_Modelling.csv"
MODEL_FILE = "classifier.joblib"
SCALER_FILE = "scaler.joblib"

TARGET = "Exited"
COLUMNS_TO_DROP = ["Surname", "Geography"]  # comme dans le notebook
ID_COLUMNS = ["RowNumber", "CustomerId"]
# Ordre des features (= ordre de new_client dans main.py)
FEATURES = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


# ======================================================================
# 1) PRÉPARATION — une fonction par étape
# ======================================================================
def load_data(path=DATA_FILE):
    """Importe le fichier CSV brut."""
    df = pd.read_csv(path)
    print(f"Données importées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def scale_data(x_train, x_test):
    """Standardise : le scaler est ajusté sur le TRAIN uniquement.

    (Dans le notebook, il était ré-ajusté sur le test : fuite de données.)
    Le scaler est sauvegardé pour pouvoir transformer de nouveaux clients.
    """
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )
    joblib.dump(scaler, SCALER_FILE)
    return x_train_scaled, x_test_scaled


def prepare_data(df=None):
    """Nettoie, encode, découpe et standardise les données.

    Si df n'est pas fourni, les données sont importées avec load_data().
    Retourne x_train, x_test, y_train, y_test.
    """
    if df is None:
        df = load_data()

    # Nettoyage : valeurs manquantes et doublons
    df = df.dropna().drop_duplicates()

    # Encodage : Gender (Female=0, Male=1), suppression Surname / Geography
    df = df.copy()
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Séparation features / cible
    x = df.drop(columns=ID_COLUMNS + [TARGET])[FEATURES]
    y = df[TARGET]

    # Découpage train / test (80 % / 20 %)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    # Standardisation (scaler ajusté sur le train uniquement)
    x_train, x_test = scale_data(x_train, x_test)
    return x_train, x_test, y_train, y_test


# ======================================================================
# 2) ENTRAÎNEMENT
# ======================================================================
def train_model(x_train, y_train, n_estimators=100):
    """Entraîne un RandomForest (comme dans le notebook)."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(x_train, y_train)
    return model


# ======================================================================
# 3) ÉVALUATION
# ======================================================================
def evaluate_model(model, x_test, y_test):
    """Affiche accuracy, rapport de classification et matrice de confusion."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f} %")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :")
    print(matrix)
    return {"accuracy": accuracy, "confusion_matrix": matrix}


# ======================================================================
# 4) SAUVEGARDE / CHARGEMENT
# ======================================================================
def save_model(model, path=MODEL_FILE):
    joblib.dump(model, path)


def load_model(path=MODEL_FILE):
    return joblib.load(path)


# ======================================================================
# 5) PRÉDICTION
# ======================================================================
def predict(x, model=None):
    """Prédit sur des données DÉJÀ préparées (ex : x_test)."""
    if model is None:
        model = load_model()
    return model.predict(x)


def predict_raw(rows, model=None):
    """Prédit sur de nouveaux clients bruts (non standardisés).

    rows : liste de listes dans l'ordre de FEATURES, ou DataFrame.
    """
    if model is None:
        model = load_model()
    scaler = joblib.load(SCALER_FILE)
    x = pd.DataFrame(rows, columns=FEATURES)
    x_scaled = pd.DataFrame(scaler.transform(x), columns=FEATURES)
    return model.predict(x_scaled)
