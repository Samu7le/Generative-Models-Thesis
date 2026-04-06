# Usa un'immagine base con PyTorch e supporto CUDA (adatta la versione CUDA in base al tuo hardware se necessario)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Evita che Python scriva file .pyc e forza l'output standard senza buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Imposta la directory di lavoro dentro il container
WORKDIR /app

# Installa eventuali dipendenze di sistema necessarie per OpenCV o Scikit-Image
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia prima il file dei requisiti per sfruttare la cache di Docker
COPY requirements.txt /app/

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice sorgente
COPY . /app/

# Comando di default all'avvio del container
# Modifica il percorso se il tuo file principale si chiama diversamente
CMD ["python", "src/main.py"]