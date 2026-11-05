"""Huấn luyện MLP/baseline, lưu dữ liệu đánh giá thực để tái lập và demo."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import os
from datetime import datetime, timezone
import importlib.metadata
from uuid import uuid4

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, brier_score_loss, average_precision_score, balanced_accuracy_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import encode_features, load_dataset, make_preprocessor, split_dataset
from .config import setting
from .dataset_repository import load_dataset_from_database
from .database import EvaluationMetric, ModelVersion, TrainingRun
from .storage import DemoStore
from .model import DiabetesMLP
from .paths import ML, RAW
from .schema import FEATURES
from .data import DATASET_REF

def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Ngưỡng 0.5 cố định trước khi xem test; positive class = nhãn Positive."""
    predicted = (scores >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }

def run_training(seed: int = 42, max_epochs: int = 100, patience: int = 12,
                 data_source: str | None = None) -> dict:
    """Chọn checkpoint bằng validation BCE; chỉ đánh giá test ở cuối pipeline."""
    ML.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data_source = data_source or setting("TRAINING_DATA_SOURCE", "csv")
    store = None
    if data_source == "database":
        store = DemoStore(create_schema=False)
        frame, stats = load_dataset_from_database(store)
    elif data_source == "csv":
        frame, stats = load_dataset()
    else:
        raise ValueError("data_source phải là 'database' hoặc 'csv'")
    training_run_id = str(uuid4())
    if store:
        with store.sessions.begin() as session:
            session.add(TrainingRun(
                id=training_run_id, dataset_version_id=int(stats["dataset_version_id"]), status="running",
                seed=seed, config_json={"max_epochs": max_epochs, "patience": patience, "data_source": data_source},
            ))
    splits = split_dataset(frame, seed)
    features = encode_features(frame)
    preprocessor = make_preprocessor()
    # Cả imputer và scaler chỉ nhìn thấy train.
    arrays = {"train": preprocessor.fit_transform(features.iloc[splits["train"]]).astype(np.float32)}
    for name in ["validation", "test"]:
        arrays[name] = preprocessor.transform(features.iloc[splits[name]]).astype(np.float32)
    targets = {name: frame.iloc[index].target.to_numpy(dtype=np.float32) for name, index in splits.items()}
    for left, right in [("train", "test"), ("train", "validation"), ("validation", "test")]:
        if set(frame.iloc[splits[left]].group) & set(frame.iloc[splits[right]].group):
            raise RuntimeError("Phát hiện group leakage!")

    input_dim = arrays["train"].shape[1]
    model = DiabetesMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_data = TensorDataset(torch.from_numpy(arrays["train"]), torch.from_numpy(targets["train"]))
    loader = DataLoader(train_data, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_x, val_y = torch.from_numpy(arrays["validation"]), torch.from_numpy(targets["validation"])
    history, best_loss, best_state, best_epoch, stale = [], float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(val_x), val_y).item()
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_data), "validation_loss": val_loss})
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train BCE={history[-1]['train_loss']:.5f}, validation BCE={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    baseline = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    baseline.fit(arrays["train"], targets["train"])
    with torch.inference_mode():
        mlp_scores = torch.sigmoid(model(torch.from_numpy(arrays["test"]))).numpy()
    baseline_scores = baseline.predict_proba(arrays["test"])[:, 1]
    metrics = {"MLP": classification_metrics(targets["test"], mlp_scores),
               "Logistic Regression": classification_metrics(targets["test"], baseline_scores)}
    majority = DummyClassifier(strategy="most_frequent").fit(arrays["train"], targets["train"])
    metrics["Majority baseline"] = classification_metrics(targets["test"], majority.predict_proba(arrays["test"])[:, 1])
    torch.save(model.state_dict(), ML / "mlp.pt")
    joblib.dump(preprocessor, ML / "preprocessor.joblib")
    joblib.dump(baseline, ML / "logistic.joblib")
    pd.DataFrame(history).to_csv(ML / "training_history.csv", index=False)
    pd.concat([frame.iloc[index][["source_row", "group", "target"]].assign(split=name)
               for name, index in splits.items()]).to_csv(ML / "splits.csv", index=False)
    pd.DataFrame({"source_row": frame.iloc[splits["test"]].source_row.to_numpy(),
                  "y_true": targets["test"].astype(int), "mlp_score": mlp_scores,
                  "logistic_score": baseline_scores}).to_csv(ML / "test_predictions.csv", index=False)
    pd.DataFrame([{"model": name, **{k: v for k, v in result.items() if k != "confusion_matrix"}}
                  for name, result in metrics.items()]).to_csv(ML / "model_comparison.csv", index=False)

    model_hash = hashlib.sha256((ML / "mlp.pt").read_bytes()).hexdigest()
    model_version_id = str(uuid4())
    model_version = f"mlp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{model_hash[:8]}"
    metadata = {
        "seed": seed, "dataset_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "generated_utc": datetime.now(timezone.utc).isoformat(), "features": FEATURES,
        "dataset_ref": DATASET_REF, "input_dim": input_dim,
        "transformed_feature_names": list(preprocessor.get_feature_names_out()),
        "inference_numeric_ranges": {name: {"min": float(features.iloc[splits["train"]][name].min()), "max": float(features.iloc[splits["train"]][name].max())} for name in ["age", "bmi", "hba1c_level", "blood_glucose_level"]},
        "architecture": f"{input_dim} -> 16 ReLU Dropout(0.1) -> 8 ReLU -> 1 logit",
        "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
        "batch_size": 512, "max_epochs": max_epochs, "epochs_run": len(history),
        "patience": patience, "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "threshold": 0.5, "preprocessing": "numeric/binary median+StandardScaler; gender/smoking one-hot; fit train only",
        "split_strategy": "deduplicate selected 8 X + y; StratifiedGroupKFold 5 outer / 4 inner; group = all selected X",
        "dataset": stats,
        "splits": {name: {"rows": len(index), "positive": int(targets[name].sum()),
                           "negative": int(len(index) - targets[name].sum())} for name, index in splits.items()},
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas", "scikit-learn"]},
        "data_source": data_source, "dataset_version_id": stats.get("dataset_version_id"),
        "model_version": model_version, "model_artifact_sha256": model_hash,
    }
    summary = {"metadata": metadata, "metrics": metrics}
    (ML / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if store:
        with store.sessions.begin() as session:
            run = session.get(TrainingRun, training_run_id)
            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)
            session.query(ModelVersion).filter(ModelVersion.active.is_(True)).update({"active": False})
            session.add(ModelVersion(
                id=model_version_id, training_run_id=training_run_id, name="DiabetesMLP",
                version=model_version, artifact_path=str(ML / "mlp.pt"), artifact_hash=model_hash,
                metadata_json=metadata, active=True,
            ))
            for model_name, model_metrics in metrics.items():
                for metric_name, metric_value in model_metrics.items():
                    session.add(EvaluationMetric(
                        model_version_id=model_version_id, split_name=f"test:{model_name}",
                        metric_name=metric_name,
                        metric_value=float(metric_value) if isinstance(metric_value, (int, float)) else None,
                        metric_json={"value": metric_value} if not isinstance(metric_value, (int, float)) else None,
                    ))
    create_plots(history, targets["test"], mlp_scores, baseline_scores, metrics)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

def create_plots(history: list[dict], y: np.ndarray, mlp: np.ndarray, baseline: np.ndarray, metrics: dict) -> None:
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train BCE")
    axis.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], label="Validation BCE")
    axis.set(xlabel="Epoch", ylabel="BCE loss", title="MLP training (checkpoint selected on validation)")
    axis.legend(); fig.tight_layout(); fig.savefig(ML / "training_loss.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    for axis, (name, result) in zip(axes, metrics.items()):
        ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=["Negative", "Positive"]).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(name)
    fig.tight_layout(); fig.savefig(ML / "confusion_matrices.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, mlp, name="MLP", ax=axis)
    RocCurveDisplay.from_predictions(y, baseline, name="Logistic Regression", ax=axis)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    fig.tight_layout(); fig.savefig(ML / "roc_curves.png", dpi=160); plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["database", "csv"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(seed=args.seed, data_source=args.source)
