FROM python:3.11-slim

# Install uv package manager
RUN pip install uv

WORKDIR /app

COPY requirements.txt .

# Use uv instead of pip
RUN uv pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
