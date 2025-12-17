# Usa l'immagine NVIDIA "devel" che contiene il compilatore NVCC e le librerie CUDA complete
# Necessario per compilare llama-cpp-python e usare i driver della RTX 5070
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Evita domande durante l'installazione dei pacchetti
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Imposta variabili per forzare la compilazione GPU di Llama.cpp
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

WORKDIR /app

# 1. Installa Python 3.10 e i tool di compilazione (build-essential è CRUCIALE)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Installa CMake aggiornato (Ubuntu 22.04 ha una versione vecchia che rompe llama.cpp)
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.sh -o cmake.sh && \
    sh cmake.sh --prefix=/usr/local --skip-license && \
    rm cmake.sh

# Link simbolico per usare 'python' invece di 'python3'
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 3. Aggiorna pip
RUN python -m pip install --upgrade pip

# 4. Installa PyTorch Nightly (Compatibile con RTX 5070 / CUDA 12.8)
# Installiamo solo 'torch' per mantenere l'immagine leggera e ridurre conflitti
RUN pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 5. Installa llama-cpp-python compilandolo DA SORGENTE
# Ora che abbiamo 'build-essential' e 'nvcc', questo passaggio funzionerà e abiliterà la GPU
RUN pip install --no-cache-dir llama-cpp-python

# 6. Copia requirements e installa le altre librerie
COPY requirements.txt .
# Filtriamo via torch e llama-cpp se presenti per non sovrascriverli
RUN grep -v "torch" requirements.txt | grep -v "llama-cpp-python" > requirements_clean.txt && \
    pip install --no-cache-dir -r requirements_clean.txt

# 7. Copia tutto il codice (incluso llm_handler.py aggiornato con il warmup)
COPY . .

# Crea cache folder
RUN mkdir -p /app/hf_cache && chmod 777 /app/hf_cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]