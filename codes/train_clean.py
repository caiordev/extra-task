"""Clean training script that avoids compatibility issues.

This script uses only standard Keras components and avoids custom metrics
that cause sample_weight conflicts.

Run with:
    python train_clean.py --epochs 10 --step 10
"""
import sys
sys.path.append('codes')

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras import losses
import json

from codes.data_utils import load_csv_parts, window_generator, train_val_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def parse_args():
    p = argparse.ArgumentParser(description="Clean LSTM trainer for extreme imbalance")
    p.add_argument("--dataset_dir", type=str, default="dataset")
    p.add_argument("--window", type=int, default=240)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--exp_dir", type=str, default="experiments/clean_run")
    p.add_argument("--focal", action="store_true", help="Use focal loss")
    return p.parse_args()

def plot_training_history(history, exp_dir):
    """Plot training history with loss and metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC-PR (most important for imbalanced data)
    axes[0, 1].plot(history['auc_pr'], label='Training AUC-PR', color='blue')
    axes[0, 1].plot(history['val_auc_pr'], label='Validation AUC-PR', color='red')
    axes[0, 1].set_title('AUC-PR (Area Under Precision-Recall Curve)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-PR')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC-ROC
    axes[1, 0].plot(history['auc_roc'], label='Training AUC-ROC', color='blue')
    axes[1, 0].plot(history['val_auc_roc'], label='Validation AUC-ROC', color='red')
    axes[1, 0].set_title('AUC-ROC (Area Under ROC Curve)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall (important for finding positive cases)
    axes[1, 1].plot(history['recall'], label='Training Recall', color='blue')
    axes[1, 1].plot(history['val_recall'], label='Validation Recall', color='red')
    axes[1, 1].set_title('Recall (Sensitivity)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Training history plot saved to: {exp_dir / 'training_history.png'}")

def plot_confusion_matrix(y_true, y_pred, exp_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fault'], 
                yticklabels=['Normal', 'Fault'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(exp_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Confusion matrix saved to: {exp_dir / 'confusion_matrix.png'}")

def build_clean_model(window_size=240, learning_rate=0.005):
    """Build model using only standard Keras components."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(window_size, 1)),
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(64, dropout=0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 CLEAN TRAINING SCRIPT")
    print("="*50)
    
    # Load data
    print("📊 Loading data...")
    df = load_csv_parts(args.dataset_dir)
    print(f"Dataset: {len(df):,} rows")
    
    # Generate windows
    print("🔄 Generating windows...")
    X, y = window_generator(df, window_size=args.window, step_size=args.step)
    print(f"Windows: {X.shape}, Positives: {y.sum()} ({y.mean():.6%})")
    
    if y.sum() == 0:
        print("❌ No positive samples found!")
        return
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    
    # Expand dims
    X_train_e = np.expand_dims(X_train, -1)
    X_val_e = np.expand_dims(X_val, -1)
    X_test_e = np.expand_dims(X_test, -1)
    
    print(f"Train: {X_train_e.shape} (pos: {y_train.sum()})")
    print(f"Val: {X_val_e.shape} (pos: {y_val.sum()})")
    print(f"Test: {X_test_e.shape} (pos: {y_test.sum()})")
    
    # Class weights
    if len(np.unique(y_train)) > 1:
        class_weight_vals = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = {0: class_weight_vals[0], 1: class_weight_vals[1] * 3.0}
        print(f"Class weights: {class_weights}")
    else:
        print("❌ Only one class in training data")
        return
    
    # Build model
    print("🧠 Building model...")
    model = build_clean_model(window_size=args.window, learning_rate=args.lr)
    
    # Compile with standard metrics only
    loss_fn = losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25) if args.focal else 'binary_crossentropy'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc_roc'),
            AUC(name='auc_pr', curve='PR')
        ]
    )
    
    print("Model compiled successfully!")
    model.summary()
    
    # Create datasets
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_e, y_train))
        .shuffle(buffer_size=10000)
        .batch(args.batch)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_e, y_val))
        .batch(args.batch)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc_pr', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(exp_dir / "best_model.keras"),
            monitor='val_auc_pr',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"🏃‍♂️ Training for {args.epochs} epochs...")
    print(f"Loss: {'Focal Loss' if args.focal else 'Binary Crossentropy'}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save history
    import pandas as pd
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(exp_dir / "history.csv", index=False)
    
    # Plot training history
    print("\n📊 Generating training plots...")
    plot_training_history(history.history, exp_dir)
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_e, y_test)).batch(args.batch)
    test_metrics = model.evaluate(test_ds, verbose=1)
    
    # Save test results
    test_results = dict(zip(model.metrics_names, test_metrics))
    with open(exp_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n🎯 FINAL RESULTS:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.6f}")
    
    # Calculate F1 manually
    y_pred_proba = model.predict(test_ds, verbose=0)
    y_pred = (y_pred_proba > 0.01).astype(int)  # Low threshold for extreme imbalance
    
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    
    if y_test.sum() > 0:  # Only if we have positive samples in test
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"\n📊 MANUAL METRICS (threshold=0.01):")
        print(f"  Precision: {precision:.6f}")
        print(f"  Recall: {recall:.6f}")
        print(f"  F1-Score: {f1:.6f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        print("\n📊 Generating confusion matrix...")
        plot_confusion_matrix(y_test, y_pred, exp_dir)
    
    # Success indicators
    val_auc_pr = history.history.get('val_auc_pr', [0])[-1]
    val_auc_roc = history.history.get('val_auc_roc', [0])[-1]
    val_recall = history.history.get('val_recall', [0])[-1]
    
    print(f"\n🎉 SUCCESS INDICATORS:")
    if val_auc_pr > 0.01:
        print(f"✅ AUC-PR: {val_auc_pr:.4f} > 0.01 (model learning)")
    else:
        print(f"❌ AUC-PR: {val_auc_pr:.4f} ≤ 0.01")
    
    if val_auc_roc > 0.5:
        print(f"✅ AUC-ROC: {val_auc_roc:.4f} > 0.5 (better than random)")
    else:
        print(f"❌ AUC-ROC: {val_auc_roc:.4f} ≤ 0.5")
    
    if val_recall > 0.0:
        print(f"✅ Recall: {val_recall:.4f} > 0 (detecting positives)")
    else:
        print(f"❌ Recall: {val_recall:.4f} = 0")
    
    print(f"\n💾 Results saved to: {exp_dir}")

if __name__ == "__main__":
    main()
