"""Improved training script for extreme class imbalance scenarios.

This script includes optimizations specifically for datasets with extreme class imbalance
like the vibration sensor dataset where positive samples are ~0.003% of the data.

Run with:
    python -m codes.train_lstm_improved --dataset_dir ../dataset --window 240 --step 10 \
        --batch 64 --epochs 150 --exp_dir experiments/improved_run --focal --lr 0.005
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras import losses
import json

from codes.data_utils import load_csv_parts, window_generator, train_val_test_split
from codes.TensorflowDataPreprocessor import TensorflowDataPreprocessor
from codes.LSTM import LSTMModel, F1Score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Improved shared-sensor LSTM trainer for extreme imbalance")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--window", type=int, default=240)
    p.add_argument("--step", type=int, default=10)  # Larger step to reduce redundancy
    p.add_argument("--batch", type=int, default=64)  # Smaller batch for better gradient updates
    p.add_argument("--epochs", type=int, default=150)  # More epochs for rare events
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focal", action="store_true", help="Use BinaryFocalCrossentropy loss (recommended)")
    p.add_argument("--lr", type=float, default=0.005)  # Higher learning rate for rare events
    p.add_argument("--exp_dir", type=str, default="experiments/improved")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--weight_amplify", type=float, default=3.0, help="Amplification factor for minority class weight")
    return p.parse_args()


def analyze_data_distribution(y_train, y_val, y_test):
    """Analyze and print data distribution statistics."""
    total_samples = len(y_train) + len(y_val) + len(y_test)
    
    print("\n" + "="*60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pos_count = y_split.sum()
        neg_count = len(y_split) - pos_count
        pos_pct = (pos_count / len(y_split)) * 100
        
        print(f"{split_name:5s}: {len(y_split):6,} samples | "
              f"Positive: {pos_count:4d} ({pos_pct:.4f}%) | "
              f"Negative: {neg_count:6,}")
    
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Overall positive rate: {(y_train.sum() + y_val.sum() + y_test.sum()) / total_samples:.6f}")
    print("="*60)


def create_balanced_dataset(X, y, batch_size=64, oversample_factor=10):
    """Create a more balanced dataset using oversampling of minority class."""
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # Oversample positive class
    pos_oversampled = np.tile(pos_indices, oversample_factor)
    
    # Combine with negative samples (sample to match oversampled positive)
    n_pos_oversampled = len(pos_oversampled)
    neg_sampled = np.random.choice(neg_indices, size=min(n_pos_oversampled * 2, len(neg_indices)), replace=False)
    
    # Combine indices
    combined_indices = np.concatenate([pos_oversampled, neg_sampled])
    np.random.shuffle(combined_indices)
    
    return X[combined_indices], y[combined_indices]


def main() -> None:
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV parts …")
    df = load_csv_parts(args.dataset_dir)
    print(f"Loaded {len(df):,} rows.")

    print("Generating windows …")
    X, y = window_generator(df, window_size=args.window, step_size=args.step)
    print(f"Windows: {X.shape}, positives: {y.sum()} ({y.mean():.6%})")

    # Check if we have any positive samples
    if y.sum() == 0:
        print("ERROR: No positive samples found in the dataset!")
        print("This suggests a problem with label processing.")
        return

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    
    # Analyze data distribution
    analyze_data_distribution(y_train, y_val, y_test)

    # Expand dims for LSTM (samples, window, 1)
    X_train_e = np.expand_dims(X_train, -1)
    X_val_e = np.expand_dims(X_val, -1)
    X_test_e = np.expand_dims(X_test, -1)

    # Compute enhanced class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weight_vals = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    
    # Apply amplification factor for extreme imbalance
    class_weights = {
        0: class_weight_vals[0], 
        1: class_weight_vals[1] * args.weight_amplify
    }
    print(f"\nClass weights: {class_weights}")
    print(f"Minority class amplification: {class_weight_vals[1] * args.weight_amplify:.2f}")

    # Create balanced training dataset
    if y_train.sum() > 0:  # Only if we have positive samples
        print("\nCreating balanced training dataset...")
        X_train_balanced, y_train_balanced = create_balanced_dataset(X_train_e, y_train)
        print(f"Balanced training set: {X_train_balanced.shape}, positives: {y_train_balanced.sum()} ({y_train_balanced.mean():.4%})")
    else:
        X_train_balanced, y_train_balanced = X_train_e, y_train

    # Build tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_balanced, y_train_balanced))
        .shuffle(buffer_size=10000)
        .batch(args.batch)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_e, y_val))
        .batch(args.batch)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Use focal loss for extreme imbalance
    loss_fn = losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25) if args.focal else 'binary_crossentropy'
    print(f"\nUsing loss function: {'Focal Loss' if args.focal else 'Binary Crossentropy'}")

    # Build model with comprehensive metrics
    model = LSTMModel(
        window_size=args.window,
        metrics=[
            Precision(name="precision"), 
            Recall(name="recall"), 
            F1Score(),
            AUC(name='auc_roc'),
            AUC(name='auc_pr', curve='PR')
        ],
        class_weights=class_weights,
        learning_rate=args.lr,
        debug=args.debug,
    )

    # Override loss function if using focal loss
    if args.focal:
        model.model.compile(
            optimizer=model.model.optimizer,
            loss=loss_fn,
            metrics=model.model.metrics
        )

    print(f"\nModel architecture:")
    model.model.summary()

    # Setup callbacks and train
    callbacks = model.setup_callbacks(model_name=str(exp_dir / "chkpt"))
    
    print(f"\nStarting training...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch}, Learning rate: {args.lr}")
    
    history = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save training history
    history_df = tf.keras.utils.get_file.__globals__['pd'].DataFrame(history.history)
    history_df.to_csv(exp_dir / "history.csv", index=False)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_e, y_test)).batch(args.batch)
    test_metrics = model.model.evaluate(test_ds, verbose=1)
    
    # Save test metrics
    test_results = dict(zip(model.model.metrics_names, test_metrics))
    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nTest Results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.6f}")

    # Generate predictions for analysis
    y_pred_proba = model.model.predict(test_ds)
    y_pred = (y_pred_proba > 0.01).astype(int)  # Lower threshold for extreme imbalance
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save final model
    model.save_model(str(exp_dir / "final_model"))
    print(f"\nModel saved to: {exp_dir / 'final_model'}")
    
    print(f"\nExperiment completed! Results saved to: {exp_dir}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
