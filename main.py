"""
main.py — exécute les étapes du pipeline via des arguments CLI.

Exemples :
    python3 main.py --prepare
    python3 main.py --train
    python3 main.py --evaluate
    python3 main.py --predict
    python3 main.py --all        # tout le pipeline d'un coup
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    predict_raw,
    prepare_data,
    save_model,
    train_model,
)

# Exemple : nouveau client (même ligne que dans le notebook)
# CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
# HasCrCard, IsActiveMember, EstimatedSalary
NEW_CLIENT = [[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]]


def title(text):
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)


def show_predictions(x_test, y_test):
    print("Predictions (5 premiers clients test):", predict(x_test[:5]))
    print("Vraies valeurs                       :", y_test[:5].values)
    print("Nouveau client -> churn =", predict_raw(NEW_CLIENT)[0])


def main():
    parser = argparse.ArgumentParser(description="Churn ML pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--save", action="store_true", help="Train and save the model")
    parser.add_argument("--load", action="store_true", help="Load the model")
    parser.add_argument("--predict", action="store_true", help="Predict with the model")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run everything: prepare, train, save, load, evaluate, predict",
    )
    args = parser.parse_args()

    # Aucun argument : afficher l'aide
    if not any(vars(args).values()):
        parser.print_help()
        return

    # --all : exécute tout le pipeline dans l'ordre
    if args.all:
        title("1) PRÉPARATION DES DONNÉES")
        x_train, x_test, y_train, y_test = prepare_data()
        print("Data prepared.")
        print(f"Train: {x_train.shape}, Test: {x_test.shape}")

        title("2) ENTRAÎNEMENT + SAUVEGARDE")
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

        title("3) CHARGEMENT DU MODÈLE")
        model = load_model()
        print(model)

        title("4) ÉVALUATION")
        evaluate_model(model, x_test, y_test)

        title("5) PRÉDICTION")
        show_predictions(x_test, y_test)

        print("\n>>> Pipeline complet terminé.")
        return

    if args.prepare:
        x_train, x_test, y_train, y_test = prepare_data()
        print("Data prepared.")
        print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    if args.train or args.save:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.predict:
        x_train, x_test, y_train, y_test = prepare_data()
        show_predictions(x_test, y_test)

    if args.load:
        model = load_model()
        print(model)


if __name__ == "__main__":
    main()
