# =========================
# 1️⃣ Base Image
# =========================
FROM python:3.10.13-slim-bookworm AS builder

# Environment variables for clean Python behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# =========================
# 2️⃣ Install system dependencies
# =========================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc build-essential \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# =========================
# 3️⃣ Create virtual environment
# =========================
RUN python -m venv /opt/venv

# =========================
# 4️⃣ Install Python dependencies
# =========================
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# 5️⃣ Copy application code, model, vectorstore
# =========================
COPY app ./app

# =========================
# 6️⃣ Expose port
# =========================
EXPOSE 8000

# =========================
# 7️⃣ Default command to run FastAPI
# =========================
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
