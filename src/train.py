# src/train.py
import argparse
import os
import joblib
import wandb
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils import load_dataset, save_model

def main(args):
    # W&B init
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY must be set in environment")
    
    # Trim whitespace from API key to prevent common copy-paste issues
    wandb_api_key = wandb_api_key.strip()
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY is empty after trimming whitespace")
    
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    if wandb_entity:
        wandb_entity = wandb_entity.strip()
    wandb_project = os.getenv("WANDB_PROJECT", "mlops-capstone").strip()
    
    print(f"Initializing W&B with project: {wandb_project}, entity: {wandb_entity}")

    # Initialize W&B run with comprehensive configuration
    run = wandb.init(
        project=wandb_project, 
        entity=wandb_entity, 
        job_type="train",
        tags=["iris", "logistic-regression", "classification"]
    )
    
    # Log hyperparameters and configuration
    config = {
        **vars(args),
        "model_type": "LogisticRegression",
        "preprocessing": "StandardScaler",
        "dataset": "iris",
        "timestamp": datetime.now().isoformat(),
        "python_version": os.sys.version,
        "sklearn_version": joblib.__version__
    }
    run.config.update(config)

    # Load and validate dataset
    print("Loading and validating dataset...")
    df = load_dataset()
    
    # Dataset validation
    dataset_info = {
        "total_samples": len(df),
        "features": list(df.columns[:-1]),
        "target_classes": sorted(df["target"].unique()),
        "class_distribution": df["target"].value_counts().to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }
    
    print(f"Dataset loaded: {dataset_info['total_samples']} samples, {len(dataset_info['features'])} features")
    print(f"Class distribution: {dataset_info['class_distribution']}")
    
    # Log dataset information
    run.log({"dataset_info": dataset_info})
    
    # Validate dataset quality
    if dataset_info["missing_values"] > 0:
        print(f"WARNING: Found {dataset_info['missing_values']} missing values")
    if dataset_info["duplicate_rows"] > 0:
        print(f"WARNING: Found {dataset_info['duplicate_rows']} duplicate rows")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Create and train model
    print("Training model...")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=args.seed))
    
    # Cross-validation for model validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    model.fit(X_train, y_train)

    # Comprehensive evaluation
    print("Evaluating model...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Get prediction probabilities for ROC-AUC
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    metrics = {
        # Basic metrics
        "train_accuracy": accuracy_score(y_train, train_preds),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "train_precision": precision_score(y_train, train_preds, average='weighted'),
        "test_precision": precision_score(y_test, test_preds, average='weighted'),
        "train_recall": recall_score(y_train, train_preds, average='weighted'),
        "test_recall": recall_score(y_test, test_preds, average='weighted'),
        "train_f1": f1_score(y_train, train_preds, average='weighted'),
        "test_f1": f1_score(y_test, test_preds, average='weighted'),
        
        # Cross-validation metrics
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_min": cv_scores.min(),
        "cv_max": cv_scores.max(),
        
        # Model complexity
        "n_features": len(X.columns),
        "n_classes": len(np.unique(y))
    }
    
    # Add serializable model parameters only
    model_params = model.get_params()
    serializable_params = {}
    for key, value in model_params.items():
        try:
            # Test if the value is JSON serializable
            import json
            json.dumps(value)
            serializable_params[key] = value
        except (TypeError, ValueError):
            # Skip non-serializable objects (like sklearn estimators)
            if isinstance(value, str) or isinstance(value, (int, float, bool)) or value is None:
                serializable_params[key] = value
            else:
                serializable_params[key] = str(type(value).__name__)
    
    metrics["model_params"] = serializable_params
    
    # ROC-AUC for multiclass (one-vs-rest)
    try:
        metrics["train_roc_auc"] = roc_auc_score(y_train, train_proba, multi_class='ovr')
        metrics["test_roc_auc"] = roc_auc_score(y_test, test_proba, multi_class='ovr')
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")
    
    # Log all metrics
    run.log(metrics)
    
    # Log detailed classification report
    class_report = classification_report(y_test, test_preds, output_dict=True)
    run.log({"classification_report": class_report})
    
    # Log confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    run.log({"confusion_matrix": cm.tolist()})
    
    # Create confusion matrix visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y))
    plt.yticks(tick_marks, np.unique(y))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    run.log({"confusion_matrix_plot": wandb.Image(plt)})
    plt.close()

    # Save model locally and create comprehensive artifact
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    # Create model artifact with metadata
    artifact = wandb.Artifact(
        name="iris-logreg-model",
        type="model",
        description=f"Iris classification model trained on {len(df)} samples",
        metadata={
            "model_type": "LogisticRegression",
            "preprocessing": "StandardScaler",
            "dataset": "iris",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": metrics["test_accuracy"],
            "test_f1": metrics["test_f1"],
            "cv_mean": metrics["cv_mean"],
            "cv_std": metrics["cv_std"],
            "features": dataset_info["features"],
            "target_classes": dataset_info["target_classes"],
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {
                "test_size": args.test_size,
                "seed": args.seed,
                "max_iter": 200
            }
        }
    )
    
    # Add model file
    artifact.add_file(model_path)
    
    # Add dataset summary
    dataset_summary = pd.DataFrame({
        "metric": ["total_samples", "train_samples", "test_samples", "n_features", "n_classes"],
        "value": [len(df), len(X_train), len(X_test), len(X.columns), len(np.unique(y))]
    })
    dataset_summary_path = os.path.join(output_dir, "dataset_summary.csv")
    dataset_summary.to_csv(dataset_summary_path, index=False)
    artifact.add_file(dataset_summary_path)
    
    # Add performance summary
    performance_summary = pd.DataFrame({
        "metric": ["test_accuracy", "test_precision", "test_recall", "test_f1", "cv_mean", "cv_std"],
        "value": [
            metrics["test_accuracy"], 
            metrics["test_precision"], 
            metrics["test_recall"], 
            metrics["test_f1"],
            metrics["cv_mean"],
            metrics["cv_std"]
        ]
    })
    performance_summary_path = os.path.join(output_dir, "performance_summary.csv")
    performance_summary.to_csv(performance_summary_path, index=False)
    artifact.add_file(performance_summary_path)
    
    # Log artifact with versioning
    run.log_artifact(artifact, aliases=["latest", f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"])
    
    # Model validation checks
    validation_results = {
        "accuracy_threshold_passed": metrics["test_accuracy"] >= 0.9,  # 90% accuracy threshold
        "cv_stability_passed": metrics["cv_std"] <= 0.05,  # Low variance in CV
        "overfitting_check_passed": abs(metrics["train_accuracy"] - metrics["test_accuracy"]) <= 0.1,  # No significant overfitting
        "dataset_quality_passed": dataset_info["missing_values"] == 0 and dataset_info["duplicate_rows"] == 0
    }
    
    run.log({"validation_results": validation_results})
    
    # Overall validation status
    all_validations_passed = all(validation_results.values())
    run.log({"model_validation_passed": all_validations_passed})
    
    if all_validations_passed:
        print("✅ All model validations passed!")
    else:
        print("⚠️ Some model validations failed:")
        for check, passed in validation_results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {passed}")

    run.finish()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"📊 Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"📊 CV Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    print(f"🔍 Model validation: {'PASSED' if all_validations_passed else 'FAILED'}")
    print(f"💾 Model saved to: {model_path}")
    print(f"📈 Artifact logged to W&B with comprehensive metadata")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    args = parser.parse_args()
    main(args)
