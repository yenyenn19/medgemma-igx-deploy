FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY medgemma_series_server.py .
EXPOSE 8080
CMD ["python3", "medgemma_series_server.py"]
