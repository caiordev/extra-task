"""End-to-end training script for Option B (shared LSTM) using the existing
classes in `codes/`.

Run with:
    python -m codes.train_lstm --dataset_dir ../dataset --window 240 --step 1 \
        --batch 128 --epochs 100 --exp_dir experiments/exp01
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import losses

from codes.data_utils import load_csv_parts, window_generator, train_val_test_split
from codes.TensorflowDataPreprocessor import TensorflowDataPreprocessor
from codes.LSTM import LSTMModel, F1Score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared-sensor LSTM trainer")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--window", type=int, default=240)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)  # global seed
    p.add_argument("--focal", action="store_true", help="Use BinaryFocalCrossentropy loss")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--exp_dir", type=str, default="experiments/default")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # ------------------
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV parts …")
    df = load_csv_parts(args.dataset_dir)
    print(f"Loaded {len(df):,} rows.")

    print("Generating windows …")
    X, y = window_generator(df, window_size=args.window, step_size=args.step)
    print(f"Windows: {X.shape}, positives: {y.sum()} ({y.mean():.4%})")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    # Expand dims for LSTM (samples, window, 1)
    X_train_e = np.expand_dims(X_train, -1)
    X_val_e = np.expand_dims(X_val, -1)
    X_test_e = np.expand_dims(X_test, -1)

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weight_vals = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {0: class_weight_vals[0], 1: class_weight_vals[1]}
    print("Class weights:", class_weights)

    print("Building LSTM model …")
    loss_fn = losses.BinaryFocalCrossentropy() if args.focal else None

    model = LSTMModel(
        window_size=args.window,
        metrics=[Precision(name="precision"), Recall(name="recall"), F1Score()],
        class_weights=class_weights,
        learning_rate=args.lr,
        debug=args.debug,
    )

    callbacks = model.setup_callbacks(model_name=str(exp_dir / "chkpt"))
    history = model.model.fit(
        X_train_e,
        y_train,
        validation_data=(X_val_e, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        shuffle=True,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluation using ModelEvaluator for richer metrics
    print("Evaluating on test set …")
    test_probs = model.model.predict(X_test_e, batch_size=args.batch)
    from codes.ModelEvaluator import ModelEvaluator
    evaluator = ModelEvaluator(test_probs, y_test, threshold=-1, minPrecision=0.7)
    evaluator.execute()

    # Save metrics and figures
    metrics = evaluator.metrics
    with open(exp_dir / "metrics.json", "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    import matplotlib.pyplot as plt
    params_fig = {
        'outputDir': str(exp_dir),
        'DatasetType': 'test',
        'WindowSize': args.window,
        'stepSize': args.step
    }
    evaluator.plotCurve(evaluator.ROCCurve['falsePositiveRates'], evaluator.ROCCurve['truePositiveRates'], evaluator.ROCCurve['AUROC'], 'FPR', 'TPR', 'ROC Curve', saveFigure=True, params=params_fig)
    evaluator.plotCurve(evaluator.PRCurve['recalls'], evaluator.PRCurve['precisions'], evaluator.PRCurve['AUPRC'], 'Recall', 'Precision', 'PR Curve', saveFigure=True, params=params_fig)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    preds_thr = (test_probs >= evaluator.estimatedThreshold).astype(int)
    cm = confusion_matrix(y_test, preds_thr)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(exp_dir / "confusion_matrix.png")
    plt.close()
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

    y_pred = (test_probs.ravel() > 0.5).astype(int)
    precision, recall_, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc_val = roc_auc_score(y_test, test_probs)
    print(f"Test Precision: {precision:.4f} | Recall: {recall_:.4f} | F1: {f1:.4f} | AUROC: {auc_val:.4f}")

    # Save final model
    model_path = exp_dir / "final_model"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # Save training history to CSV for later analysis
    import pandas as pd
    pd.DataFrame(history.history).to_csv(exp_dir / "history.csv", index=False)


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
