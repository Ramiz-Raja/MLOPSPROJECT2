# 🌸 Iris Species Classifier - MLOps Pipeline

A comprehensive end-to-end MLOps pipeline for Iris flower species classification, featuring automated training, model registry, deployment, and a beautiful web interface.

## 🚀 Features

### 🤖 Machine Learning
- **Enhanced Dataset**: Original Iris dataset augmented with synthetic data for better training
- **Advanced Model**: Logistic Regression with StandardScaler preprocessing
- **Comprehensive Validation**: Cross-validation, accuracy thresholds, overfitting detection
- **Rich Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

### 📊 Experiment Tracking
- **Weights & Biases Integration**: Complete experiment tracking and model registry
- **Rich Artifacts**: Model files, performance summaries, dataset statistics
- **Visualizations**: Confusion matrix plots, performance charts
- **Versioning**: Automatic model versioning with timestamps

### 🔄 CI/CD Pipeline
- **GitHub Actions**: Automated training and deployment workflows
- **Path-based Triggers**: Smart workflow triggering based on file changes
- **Docker Support**: Containerized backend and frontend
- **Multi-registry**: GitHub Container Registry and Docker Hub support

### 🌐 Web Application
- **Modern UI**: Beautiful Streamlit interface with custom styling
- **Real-time Predictions**: Interactive prediction with confidence scores
- **Data Visualization**: Plotly charts for measurements and probabilities
- **Model Information**: Live model performance metrics and validation status

## 📁 Project Structure

```
mlops-capstone/
├── .github/workflows/          # GitHub Actions workflows
│   ├── train-and-register.yml  # Model training and W&B logging
│   └── build-and-deploy.yml    # Docker build and deployment
├── backend/                    # FastAPI backend service
│   ├── app/
│   │   ├── main.py            # API endpoints and routes
│   │   └── model_manager.py   # W&B model loading and validation
│   └── requirements-backend.txt
├── frontend/                   # Streamlit frontend
│   ├── streamlit_app.py       # Main application interface
│   └── requirements-frontend.txt
├── src/                        # Training scripts
│   ├── train.py               # Main training script
│   ├── utils.py               # Dataset utilities and helpers
│   └── requirements.txt       # Training dependencies
├── tools/                      # Utility scripts
│   └── wandb_login_and_run.py # W&B authentication helper
├── artifacts/                  # Local model artifacts
├── Dockerfile                  # Multi-service container
├── supervisord.conf           # Process management
└── README.md                  # This file
```

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11+
- Docker (optional)
- GitHub account
- Weights & Biases account

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/MLOPSPROJECT2.git
   cd MLOPSPROJECT2/mlops-capstone
   ```

2. **Set up environment variables**
   ```bash
   export WANDB_API_KEY="your_wandb_api_key"
   export WANDB_ENTITY="your_wandb_username"
   export WANDB_PROJECT="mlops-capstone"
   ```

3. **Install dependencies**
   ```bash
   # Training environment
   pip install -r src/requirements.txt
   
   # Backend environment
   pip install -r backend/requirements-backend.txt
   
   # Frontend environment
   pip install -r frontend/requirements-frontend.txt
   ```

4. **Run the application**
   ```bash
   # Terminal 1: Start backend
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   
   # Terminal 2: Start frontend
   cd frontend
   streamlit run streamlit_app.py --server.port 8501
   ```

### Docker Deployment

1. **Build and run with Docker**
   ```bash
   docker build -t iris-classifier .
   docker run -p 8000:8000 -p 8501:8501 \
     -e WANDB_API_KEY="your_key" \
     -e WANDB_ENTITY="your_username" \
     -e WANDB_PROJECT="mlops-capstone" \
     iris-classifier
   ```

2. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 🔧 Configuration

### GitHub Secrets Setup

Configure these secrets in your GitHub repository:

| Secret | Description | Required |
|--------|-------------|----------|
| `WANDB_API_KEY` | Weights & Biases API key | ✅ |
| `WANDB_ENTITY` | Your W&B username | ⚠️ Optional |
| `WANDB_PROJECT` | W&B project name | ⚠️ Optional |
| `DOCKERHUB_USERNAME` | Docker Hub username | ⚠️ Optional |
| `DOCKERHUB_TOKEN` | Docker Hub access token | ⚠️ Optional |
| `RENDER_API_KEY` | Render deployment API key | ⚠️ Optional |
| `RENDER_SERVICE_ID` | Render service ID | ⚠️ Optional |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | - | W&B API key for authentication |
| `WANDB_ENTITY` | - | W&B entity (username/team) |
| `WANDB_PROJECT` | `mlops-capstone` | W&B project name |
| `BACKEND_URL` | `http://localhost:8000` | Backend API URL |

## 📊 API Endpoints

### Backend API (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check with model validation |
| `/predict` | POST | Iris species prediction |
| `/model/info` | GET | Model metadata and information |
| `/model/validation` | GET | Model validation results |

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
payload = {
    "features": [5.1, 3.5, 1.4, 0.2]  # sepal_length, sepal_width, petal_length, petal_width
}
response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()
print(f"Predicted species: {result['predicted_class']}")
print(f"Confidence: {max(result['probability'][0]):.2%}")
```

## 🎯 Model Performance

The model achieves excellent performance on the enhanced Iris dataset:

- **Accuracy**: >95% on test set
- **Cross-validation**: Stable performance across folds
- **Validation**: Automatic quality checks and thresholds
- **Robustness**: Handles edge cases and provides confidence scores

## 🔄 CI/CD Pipeline

### Training Workflow (`train-and-register.yml`)
- Triggers on changes to `src/` files
- Installs dependencies and runs training
- Logs comprehensive metrics to W&B
- Creates model artifacts with metadata
- Performs validation checks

### Deployment Workflow (`build-and-deploy.yml`)
- Triggers on changes to backend/frontend files
- Builds Docker images for multiple registries
- Supports GitHub Container Registry and Docker Hub
- Optional Render deployment integration

## 📈 Monitoring and Observability

### Weights & Biases Dashboard
- **Experiments**: Track all training runs with metrics
- **Artifacts**: Model registry with versioning
- **Visualizations**: Confusion matrices and performance charts
- **Metadata**: Rich model and dataset information

### Application Monitoring
- **Health Checks**: Backend service status
- **Model Validation**: Real-time model quality assessment
- **Error Handling**: Comprehensive error reporting
- **Performance Metrics**: Response times and accuracy

## 🚀 Deployment Options

### Cloud Platforms
- **Google Cloud Run**: Serverless container deployment
- **AWS ECS/Fargate**: Container orchestration
- **Azure Container Instances**: Simple container hosting
- **Render**: Easy deployment with GitHub integration

### Local Deployment
- **Docker Compose**: Multi-service orchestration
- **Kubernetes**: Production-grade container management
- **Standalone**: Direct Python execution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Iris Dataset**: Classic machine learning dataset by R.A. Fisher
- **Weights & Biases**: Experiment tracking and model registry
- **Streamlit**: Beautiful web application framework
- **FastAPI**: Modern Python web framework
- **scikit-learn**: Machine learning library

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the API documentation at `/docs` endpoint

---

**Built with ❤️ for MLOps excellence**