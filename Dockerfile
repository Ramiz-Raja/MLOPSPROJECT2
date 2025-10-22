# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create workdir
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install python deps (backend + frontend)
RUN pip install --upgrade pip
RUN pip install -r src/requirements.txt
RUN pip install -r backend/requirements-backend.txt
RUN pip install -r frontend/requirements-frontend.txt
# Install supervisord to run both processes
RUN pip install supervisor

# Supervisord conf to run uvicorn (FastAPI) and streamlit
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 8000 8501

# Entrypoint
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
