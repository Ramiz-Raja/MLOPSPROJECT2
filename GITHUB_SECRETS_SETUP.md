# GitHub Secrets Setup Guide

This guide explains how to set up the required secrets in your GitHub repository for the MLOps pipeline to work correctly.

## Required Secrets

### 1. W&B (Weights & Biases) Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions → Repository secrets

#### `WANDB_API_KEY`
- **Description**: Your Weights & Biases API key
- **How to get**: 
  1. Go to https://wandb.ai/settings
  2. Scroll down to "API keys" section
  3. Copy your API key
- **Value**: Your W&B API key (starts with something like `a1b2c3d4...`)

#### `WANDB_ENTITY` (Optional)
- **Description**: Your W&B username or team name
- **How to get**: 
  1. Go to https://wandb.ai/settings
  2. Look at your username in the profile section
- **Value**: Your W&B username (e.g., `your-username`)

#### `WANDB_PROJECT` (Optional)
- **Description**: The W&B project name for your experiments
- **Default**: `mlops-capstone` (if not set)
- **Value**: `mlops-capstone` or your preferred project name

### 2. Docker Hub Secrets (Optional)

#### `DOCKERHUB_USERNAME`
- **Description**: Your Docker Hub username
- **How to get**: Sign up at https://hub.docker.com
- **Value**: Your Docker Hub username

#### `DOCKERHUB_TOKEN`
- **Description**: Your Docker Hub access token
- **How to get**: 
  1. Go to https://hub.docker.com/settings/security
  2. Create a new access token
  3. Copy the token
- **Value**: Your Docker Hub access token

#### `DOCKERHUB_REPO` (Optional)
- **Description**: Full Docker Hub repository name
- **Format**: `username/repository-name`
- **Value**: e.g., `your-username/mlops-capstone`

### 3. Render Deployment Secrets (Optional)

#### `RENDER_API_KEY`
- **Description**: Your Render API key for automatic deployments
- **How to get**: 
  1. Go to https://dashboard.render.com/account/settings
  2. Generate an API key
- **Value**: Your Render API key

#### `RENDER_SERVICE_ID`
- **Description**: Your Render service ID
- **How to get**: 
  1. Go to your service on Render dashboard
  2. Copy the service ID from the URL or settings
- **Value**: Your Render service ID (e.g., `srv-abc123def456`)

## How to Add Secrets

1. Go to your GitHub repository
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Enter the secret name and value
6. Click **Add secret**

## Testing Your Setup

After adding the secrets, you can test your setup by:

1. **Manual trigger**: Go to Actions tab → Select workflow → Click "Run workflow"
2. **Push trigger**: Make a small change to trigger the workflow automatically

## Troubleshooting

### Common Issues

1. **"WANDB_API_KEY not found"**
   - Make sure you've added the `WANDB_API_KEY` secret
   - Check that the API key is valid and not expired

2. **"W&B login failed"**
   - Verify your API key is correct
   - Check that your W&B account is active

3. **"Docker build failed"**
   - Check Docker Hub credentials if using Docker Hub
   - Verify Dockerfile syntax

4. **"Render deploy failed"**
   - Verify Render API key and service ID
   - Check that your Render service is properly configured

### Debug Steps

1. Check the Actions logs for specific error messages
2. Verify all required secrets are set
3. Test locally first: `python src/train.py --output-dir ./artifacts`
4. Check W&B dashboard for logged experiments

## Security Notes

- Never commit API keys or secrets to your repository
- Use GitHub Secrets for all sensitive information
- Regularly rotate your API keys
- Use least-privilege access tokens when possible
