# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file into the container
COPY requirements.txt .

# Install the dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy your backend code into the container
COPY ./backend /app/backend

# The command that runs the FastAPI server when the container starts
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]