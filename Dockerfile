FROM python:3.13-slim

# Install library sistem yang dibutuhkan LightGBM & XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face mewajibkan user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy requirements dulu agar build lebih cepat
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy semua sisa file
COPY --chown=user . .

# Jalankan FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
