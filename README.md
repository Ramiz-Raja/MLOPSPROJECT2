# mlops-capstone

End-to-end MLOps demo:
- Training: `src/train.py` (Iris dataset, sklearn).
- Tracking & artifacts: Weights & Biases (W&B) Artifacts and run metrics.
- CI: GitHub Actions triggers training and model registration.
- Serving: `backend/app/main.py` (FastAPI) loads latest W&B model artifact and exposes `/predict` and `/health`.
- Frontend: `frontend/streamlit_app.py` calls backend's `/predict`.

## Quick start (local)

1. Copy secrets:
   - `export WANDB_API_KEY=<your_wandb_api_key>`
   - `export WANDB_ENTITY=<your_wandb_entity>`
   - `export WANDB_PROJECT=mlops-capstone`

2. Install:
   - `python -m venv venv && source venv/bin/activate`
   - `pip install -r src/requirements.txt`

3. Train:
   - `python src/train.py --epochs 1 --output-dir ./artifacts`

4. Run server:
   - `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`

5. Run frontend:
   - `streamlit run frontend/streamlit_app.py --server.port 8501`

## Deploy
We provide a Dockerfile that runs both backend and streamlit via supervisord. You can use Render (Docker service) or any container host.

## GitHub Actions
- `.github/workflows/train-and-register.yml` runs training and logs model artifact to W&B on `push` to `main`.
- `.github/workflows/build-and-deploy.yml` builds container image and optionally triggers Render.

## W&B / Secrets
Add these GitHub secrets:
- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `WANDB_PROJECT`
- `RENDER_API_KEY` (optional, only if you want Actions to trigger Render)
- `RENDER_SERVICE_ID` (optional)

See the code for more instructions.

