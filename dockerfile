# Base image (Python 3.9 slim for lightweight container)
FROM python:3.13

# Working directory set karo
WORKDIR /app

# uv install karo
RUN pip install --no-cache-dir uv

# Project dependencies ke liye pyproject.toml aur uv.lock (agar hai) copy karo
COPY pyproject.toml uv.lock* ./

# Dependencies install karo using uv
RUN uv sync --frozen --no-install-project

# Baki project files copy karo
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# FastAPI app run karo using uv
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]