FROM nvcr.io/nvidia/pytorch:26.01-py3

WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install dependencies (PyTorch 2.10 already included and OPTIMIZED!)
RUN pip install --upgrade pip && \
    pip install transformers==5.0.0 accelerate==1.12.0 && \
    pip install pillow flask flask-cors pydicom requests "numpy<2.0.0" huggingface-hub

# Copy application file
COPY medgemma_series_server.py /app/

EXPOSE 8080

CMD ["python3", "medgemma_series_server.py"]
