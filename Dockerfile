# Gunakan Python versi yang sama dengan Colab kamu
FROM python:3.11-slim

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

# Copy semua sisa file (main.py, folder model, dll)
COPY --chown=user . .

# Hugging Face menggunakan port 7860 secara default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
