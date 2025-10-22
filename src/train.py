# src/train.py
import argparse
import os
import joblib
import wandb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils import load_dataset, save_model

def main(args):
    # W&B init
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY must be set in environment")
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    wandb_project = os.getenv("WANDB_PROJECT", "mlops-capstone")
    
    print(f"Initializing W&B with project: {wandb_project}, entity: {wandb_entity}")

    run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="train")
    run.config.update(vars(args))

    df = load_dataset()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Simple pipeline
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    run.log({"accuracy": acc})

    # Save model locally and log as artifact
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)

    artifact = wandb.Artifact("iris-logreg-model", type="model")
    artifact.add_file(model_path)
    # Log artifact as :latest so inference can use "model:latest"
    run.log_artifact(artifact, aliases=["latest"])

    run.finish()
    print(f"Training finished. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    args = parser.parse_args()
    main(args)
