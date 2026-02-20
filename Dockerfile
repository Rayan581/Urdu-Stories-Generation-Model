# ─────────────────────────────────────────────────────────────────────────────
# Urdu Story Generator - Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file into the container
COPY requirements.txt .

# Install the dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy your backend code into the container
COPY ./backend /app/backend

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# The command that runs the FastAPI server when the container starts
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]