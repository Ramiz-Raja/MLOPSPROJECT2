# backend/app/model_manager.py
import os
import joblib
import pathlib
import wandb

MODEL_NAME = "iris-logreg-model"

def download_latest_model(wandb_entity, wandb_project, dest_dir="/tmp/model"):
    """
    Downloads the latest model artifact from W&B and returns local path to model file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    api = wandb.Api()
    # Format: {entity}/{project}/{artifact_name}:alias
    if wandb_entity:
        artifact_ref = f"{wandb_entity}/{wandb_project}/{MODEL_NAME}:latest"
    else:
        # If no entity, Api still can reference project within default
        artifact_ref = f"{wandb_project}/{MODEL_NAME}:latest"
    try:
        artifact = api.artifact(artifact_ref)
    except Exception as e:
        raise RuntimeError(f"Could not fetch artifact {artifact_ref}: {e}")
    local_path = artifact.download(root=dest_dir)
    # the artifact dir contains the uploaded file; find joblib file
    for p in pathlib.Path(local_path).rglob("*.joblib"):
        return str(p)
    # fallback: try .pkl or model.joblib
    for p in pathlib.Path(local_path).rglob("*model*"):
        if p.suffix in (".pkl", ".joblib"):
            return str(p)
    raise FileNotFoundError("Model file not found in artifact")

def load_model_from_wandb(wandb_entity, wandb_project):
    model_file = download_latest_model(wandb_entity, wandb_project)
    model = joblib.load(model_file)
    return model
