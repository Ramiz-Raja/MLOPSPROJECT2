import os
import traceback
from pathlib import Path

def main():
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    api_key = os.environ.get("WANDB_API_KEY")
    print("WANDB_ENTITY:", entity)
    print("WANDB_PROJECT:", project)
    print("WANDB_API_KEY set:", bool(api_key))

    if not project:
        print("WANDB_PROJECT not set; aborting")
        return

    artifact_name = "iris-logreg-model:latest"
    if entity:
        ref = f"{entity}/{project}/{artifact_name}"
    else:
        ref = f"{project}/{artifact_name}"

    print("Attempting to fetch artifact:", ref)

    try:
        import wandb
        api = wandb.Api()
        art = api.artifact(ref)
        print("Artifact fetched:", getattr(art, 'id', None))
        d = art.download()
        print("Downloaded to:", d)
        for p in Path(d).rglob("*"):
            print(p)
    except Exception as e:
        print("Exception while fetching artifact:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
