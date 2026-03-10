## 🚀 Quick Start

### Prerequisites

- NVIDIA IGX Orin or similar ARM64 device with NVIDIA GPU
- Docker with NVIDIA Container Toolkit
- Python 3.10+
- CUDA 12.2+
- Hugging Face account and token

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/medgemma-igx-deploy.git
cd medgemma-igx-deploy
```

2. **Set up Hugging Face token**
```bash
mkdir -p ~/.cache/huggingface
echo "YOUR_HF_TOKEN" > ~/.cache/huggingface/token
```

3. **Start Orthanc PACS**
```bash
docker network create medgemma-net

docker run -d \
  --network medgemma-net \
  --name orthanc \
  -p 4242:4242 \
  -p 8042:8042 \
  -v $(pwd)/data/orthanc-db:/var/lib/orthanc/db \
  -v $(pwd)/orthanc.json:/etc/orthanc/orthanc.json:ro \
  orthancteam/orthanc
```

4. **Build and start MedGemma server**
```bash
docker build -t medgemma-gpu .

docker run --gpus all -d \
  --network medgemma-net \
  --ipc=host \
  --name medgemma-server \
  -p 8080:8080 \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  medgemma-gpu
```

5. **Start CORS proxy and Web UI**
```bash
# Terminal 1: CORS Proxy
python3 cors_proxy.py

# Terminal 2: Web UI
python3 -m http.server 8888
```

6. **Access the interface**
```
http://localhost:8888/medgemma_ui.html
```

## 🔧 Configuration

### Environment Variables

```bash
# Hugging Face token for model access
HF_TOKEN=your_token_here

# Model configuration
MODEL_NAME=google/medgemma-4b-it
ORTHANC_URL=http://orthanc:8042

# Server ports
MEDGEMMA_PORT=8080
CORS_PROXY_PORT=5000
WEB_UI_PORT=8888
ORTHANC_WEB_PORT=8042
ORTHANC_DICOM_PORT=4242
```

### Orthanc Configuration

Edit `orthanc.json` to configure:
- Storage location
- DICOM network settings
- Web authentication
- Plugins

## 🐳 Docker Images

### Base Images Used

- **Orthanc:** `orthancteam/orthanc` (ARM64 compatible)
- **MedGemma:** `nvcr.io/nvidia/pytorch:26.01-py3` (PyTorch 2.10, CUDA 13.1)

### Building Custom Image

```bash
docker build -t medgemma-gpu -f Dockerfile .
```
