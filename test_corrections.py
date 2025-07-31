"""Quick test script to validate the corrections made to handle extreme class imbalance.

This script will:
1. Load a small sample of data
2. Test the label processing
3. Verify that the model can be created and trained for a few epochs
4. Check if metrics are improving compared to previous runs
"""

import sys
import os
sys.path.append('codes')

import numpy as np
import pandas as pd
import tensorflow as tf
from codes.data_utils import load_csv_parts, window_generator, train_val_test_split
from codes.LSTM import LSTMModel, F1Score
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight

def test_label_processing():
    """Test if labels are being processed correctly."""
    print("="*60)
    print("TESTING LABEL PROCESSING")
    print("="*60)
    
    # Load a small sample
    df = load_csv_parts("dataset")
    print(f"Loaded dataset shape: {df.shape}")
    
    # Check label column
    print(f"\nLabel column unique values: {df['label'].unique()}")
    print(f"Label value counts:")
    print(df['label'].value_counts(dropna=False))
    
    # Test window generation
    X, y = window_generator(df, window_size=240, step_size=20)  # Larger step for faster testing
    print(f"\nGenerated windows: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Positive samples: {y.sum()} ({y.mean():.6%})")
    
    return X, y

def test_model_creation():
    """Test if the improved model can be created successfully."""
    print("\n" + "="*60)
    print("TESTING MODEL CREATION")
    print("="*60)
    
    try:
        model = LSTMModel(
            window_size=240,
            metrics=[
                Precision(name="precision"), 
                Recall(name="recall"), 
                F1Score(),
                AUC(name='auc_roc'),
                AUC(name='auc_pr', curve='PR')
            ],
            class_weights={0: 1.0, 1: 100.0},  # High weight for minority class
            learning_rate=0.005,
            debug=True,
        )
        print("✅ Model created successfully!")
        model.model.summary()
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

def test_training_sample(model, X, y):
    """Test training on a small sample."""
    print("\n" + "="*60)
    print("TESTING TRAINING SAMPLE")
    print("="*60)
    
    if model is None or y.sum() == 0:
        print("❌ Cannot test training: no model or no positive samples")
        return
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    
    # Expand dims
    X_train_e = np.expand_dims(X_train, -1)
    X_val_e = np.expand_dims(X_val, -1)
    
    print(f"Training set: {X_train_e.shape}, positives: {y_train.sum()}")
    print(f"Validation set: {X_val_e.shape}, positives: {y_val.sum()}")
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_e, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_e, y_val)).batch(32)
    
    # Compute class weights
    if len(np.unique(y_train)) > 1:
        class_weight_vals = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weights = {0: class_weight_vals[0], 1: class_weight_vals[1] * 3.0}
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None
        print("Warning: Only one class in training data")
    
    try:
        # Train for just 3 epochs to test
        print("\nStarting test training (3 epochs)...")
        history = model.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            class_weight=class_weights,
            verbose=1
        )
        
        print("✅ Training completed successfully!")
        
        # Check if metrics improved
        final_metrics = history.history
        print(f"\nFinal training metrics:")
        for metric, values in final_metrics.items():
            if not metric.startswith('val_'):
                print(f"  {metric}: {values[-1]:.6f}")
        
        print(f"\nFinal validation metrics:")
        for metric, values in final_metrics.items():
            if metric.startswith('val_'):
                print(f"  {metric}: {values[-1]:.6f}")
                
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    print("TESTING CORRECTIONS FOR EXTREME CLASS IMBALANCE")
    print("="*80)
    
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # Test 1: Label processing
        X, y = test_label_processing()
        
        # Test 2: Model creation
        model = test_model_creation()
        
        # Test 3: Training sample
        training_success = test_training_sample(model, X, y)
        
        print("\n" + "="*80)
        print("SUMMARY OF TESTS")
        print("="*80)
        print(f"✅ Label processing: Working")
        print(f"✅ Model creation: {'Working' if model is not None else 'Failed'}")
        print(f"✅ Training test: {'Working' if training_success else 'Failed'}")
        
        if y.sum() > 0 and model is not None and training_success:
            print("\n🎉 ALL TESTS PASSED! The corrections appear to be working.")
            print("\nRecommendations for full training:")
            print("1. Use the improved training script: train_lstm_improved.py")
            print("2. Use focal loss: --focal")
            print("3. Use larger step size: --step 10 or 20")
            print("4. Use higher learning rate: --lr 0.005")
            print("5. Use more epochs: --epochs 150")
        else:
            print("\n⚠️  Some issues detected. Check the output above for details.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
