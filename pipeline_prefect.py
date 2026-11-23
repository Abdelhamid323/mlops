"""
pipeline_prefect.py — pipeline churn orchestré avec Prefect (Atelier 3).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)

PYTHON = sys.executable  # le python du venv activé
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


# -------------fonction _run-------------


def _run(cmd):
    """Lance une commande, écrit sa sortie dans les logs Prefect,
    et fait échouer la task si la commande échoue."""
    logger = get_run_logger()
    logger.info("$ %s", " ".join(cmd))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        logger.info(output)
    if result.returncode != 0:
        raise RuntimeError(f"Échec de la commande : {' '.join(cmd)}")


# ------------------tasks mta3 el projet w el code----------------


@task(name="get-code")
def get_code():
    """Récupère la dernière version du projet depuis le repo Git (étape 06)."""
    logger = get_run_logger()
    if not shutil.which("git") or not Path(".git").exists():
        logger.warning("Git absent ou pas de dépôt : étape ignorée.")
        return
    remotes = subprocess.run(["git", "remote"], capture_output=True, text=True)
    if not remotes.stdout.strip():
        logger.warning("Pas de dépôt remote : étape ignorée.")
        return
    _run(["git", "pull"])


@task(name="install-dependencies", retries=2, retry_delay_seconds=10)
def install_dependencies():
    _run([PYTHON, "-m", "pip", "install", "-q", "-r", "requirements.txt"])


@task(name="format-code")
def format_code():
    _run([PYTHON, "-m", "black", *CODE_FILES])


@task(name="check-quality")
def check_quality():
    _run([PYTHON, "-m", "flake8", "--max-line-length=88", *CODE_FILES])


@task(name="check-security")
def check_security():
    _run([PYTHON, "-m", "bandit", "-ll", *CODE_FILES])


@task(name="run-tests")
def run_tests():
    _run([PYTHON, "-m", "pytest", "tests", "-q"])


# -----------tasks mta3 el data w el modèle----------------


@task(name="prepare-data", cache_policy=NO_CACHE)
def prepare():
    x_train, x_test, y_train, y_test = prepare_data()
    get_run_logger().info("Train: %s, Test: %s", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


@task(name="train-model", cache_policy=NO_CACHE)
def train(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="save-model", cache_policy=NO_CACHE)
def save(model):
    save_model(model)
    get_run_logger().info("Modèle sauvegardé.")


@task(name="load-model", cache_policy=NO_CACHE)
def load():
    return load_model()


@task(name="evaluate-model", cache_policy=NO_CACHE)
def evaluate(model, x_test, y_test):
    results = evaluate_model(model, x_test, y_test)
    get_run_logger().info("Accuracy : %.2f %%", results["accuracy"] * 100)
    return results["accuracy"]


@task(name="predict", cache_policy=NO_CACHE)
def make_predictions(model, x_test, y_test):
    get_run_logger().info("Prédictions : %s", predict(x_test[:5], model))
    get_run_logger().info("Vraies valeurs : %s", y_test[:5].values)


# ------------les flows---------------


@flow(name="code", log_prints=True)
def flow_code():
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train", log_prints=True)
def flow_train():
    x_train, x_test, y_train, y_test = prepare()
    model = train(x_train, y_train)
    save(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    _, x_test, _, y_test = prepare()
    model = load()
    return evaluate(model, x_test, y_test)


@flow(name="all", log_prints=True)
def flow_all():
    get_code()
    install_dependencies()
    flow_code()
    x_train, x_test, y_train, y_test = prepare()
    model = train(x_train, y_train)
    save(model)
    accuracy = evaluate(model, x_test, y_test)
    make_predictions(model, x_test, y_test)
    return accuracy


# ------------le script------------

FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "code": flow_code,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline churn avec Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
