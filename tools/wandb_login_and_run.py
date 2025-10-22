import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Ensure WANDB_API_KEY is present in environment
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        print("WANDB_API_KEY not found in environment. Aborting.")
        sys.exit(1)

    # Import wandb and login
    try:
        import wandb
    except Exception as e:
        print("Failed to import wandb:", e)
        sys.exit(1)

    print("Logging in to W&B using WANDB_API_KEY from environment...")
    login_res = wandb.login(key=wandb_key)
    print("wandb.login returned:", login_res)

    # Run the training script as a subprocess so it runs with the same interpreter
    train_script = os.path.join(os.getcwd(), "src", "train.py")
    cmd = [sys.executable, train_script, "--output-dir", args.output_dir, "--test-size", str(args.test_size), "--seed", str(args.seed)]
    print("Running training command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
