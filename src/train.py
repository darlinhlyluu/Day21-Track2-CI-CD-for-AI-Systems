import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from pathlib import Path
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EVAL_THRESHOLD = 0.70
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("models")
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
REPORT_PATH = OUTPUTS_DIR / "report.txt"
MODEL_PATH = MODELS_DIR / "model.pkl"
TARGET_LABELS = [0, 1, 2]
TARGET_NAMES = {0: "thap", 1: "trung_binh", 2: "cao"}


def configure_mlflow() -> str:
    """Dat cau hinh MLflow mac dinh an toan cho local run neu user chua export env."""
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality")
    explicit_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    fallback_file_uri = (Path("mlruns_local").resolve()).as_uri()
    candidate_uris = (
        [explicit_tracking_uri] if explicit_tracking_uri else ["sqlite:///mlflow.db", fallback_file_uri]
    )
    last_error = None

    for tracking_uri in candidate_uris:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()

            if client.get_experiment_by_name(experiment_name) is None:
                artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")
                artifact_location = None

                if artifact_root:
                    artifact_location = (Path(artifact_root) / experiment_name).resolve().as_uri()

                client.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location,
                )

            if tracking_uri == fallback_file_uri:
                print(
                    "Canh bao: sqlite:///mlflow.db loi, da fallback sang file store mlruns_local."
                )
            return experiment_name
        except Exception as exc:
            last_error = exc

    raise last_error


def build_model(params: dict):
    """Khoi tao model dua tren model_type va cac tham so hop le."""
    model_type = params.get("model_type", "random_forest")

    if model_type == "random_forest":
        allowed_keys = {"n_estimators", "max_depth", "min_samples_split"}
        model_params = {key: value for key, value in params.items() if key in allowed_keys}
        return RandomForestClassifier(**model_params, random_state=42)

    if model_type == "gradient_boosting":
        allowed_keys = {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
        }
        model_params = {key: value for key, value in params.items() if key in allowed_keys}
        return GradientBoostingClassifier(**model_params, random_state=42)

    if model_type == "logistic_regression":
        allowed_keys = {"C", "max_iter"}
        model_params = {key: value for key, value in params.items() if key in allowed_keys}
        model_params.setdefault("max_iter", 3000)
        classifier = LogisticRegression(**model_params, random_state=42)
        return make_pipeline(StandardScaler(), classifier)

    raise ValueError(
        "Unsupported model_type. Use one of: random_forest, gradient_boosting, "
        "logistic_regression."
    )


def compute_label_distribution(y_train: pd.Series) -> tuple[dict[str, float], list[str]]:
    """Tinh ty le nhan trong tap train va canh bao neu lop nao qua it mau."""
    distribution = (
        y_train.value_counts(normalize=True)
        .reindex(TARGET_LABELS, fill_value=0.0)
        .sort_index()
    )
    label_distribution = {str(label): float(ratio) for label, ratio in distribution.items()}
    warnings = []

    for label, ratio in distribution.items():
        if ratio < 0.10:
            warnings.append(
                f"Canh bao: lop {label} ({TARGET_NAMES[label]}) chi chiem {ratio:.2%} tap train."
            )

    return label_distribution, warnings


def write_report(
    acc: float,
    f1: float,
    confusion: list[list[int]],
    per_class_metrics: dict[str, dict[str, float | int]],
    label_distribution: dict[str, float],
    warnings: list[str],
):
    """Ghi bao cao danh gia ra outputs/report.txt."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    lines = [
        "Wine Quality Training Report",
        "============================",
        f"accuracy: {acc:.4f}",
        f"f1_score(weighted): {f1:.4f}",
        "",
        "label_distribution:",
    ]

    for label in TARGET_LABELS:
        lines.append(
            f"- class {label} ({TARGET_NAMES[label]}): {label_distribution[str(label)]:.4f}"
        )

    lines.extend(["", "confusion_matrix:", "rows=true, cols=pred"])
    for row in confusion:
        lines.append(" ".join(str(value) for value in row))

    lines.extend(["", "per_class_metrics:"])
    for label in TARGET_LABELS:
        class_metrics = per_class_metrics[str(label)]
        lines.append(
            f"- class {label} ({TARGET_NAMES[label]}): "
            f"precision={class_metrics['precision']:.4f}, "
            f"recall={class_metrics['recall']:.4f}, "
            f"support={class_metrics['support']}"
        )

    if warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    label_distribution, warnings = compute_label_distribution(y_train)
    for warning in warnings:
        print(warning)

    experiment_name = configure_mlflow()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        run_params = {"model_type": params.get("model_type", "random_forest"), **params}
        mlflow.log_params(run_params)

        model = build_model(run_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))
        confusion = confusion_matrix(y_eval, preds, labels=TARGET_LABELS)
        precision, recall, _, support = precision_recall_fscore_support(
            y_eval,
            preds,
            labels=TARGET_LABELS,
            zero_division=0,
        )
        per_class_metrics = {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(TARGET_LABELS)
        }

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(
            f"Model: {run_params['model_type']} | Accuracy: {acc:.4f} | "
            f"F1: {f1:.4f}"
        )

        OUTPUTS_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": run_params["model_type"],
            "eval_threshold": EVAL_THRESHOLD,
            "label_distribution": label_distribution,
            "warnings": warnings,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion.tolist(),
        }
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_report(
            acc=acc,
            f1=f1,
            confusion=confusion.tolist(),
            per_class_metrics=per_class_metrics,
            label_distribution=label_distribution,
            warnings=warnings,
        )

        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(REPORT_PATH))
        joblib.dump(model, MODEL_PATH)

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
