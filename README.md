## MedGemma DICOM Analyzer — IGX Deployment
AI-powered medical image analysis pipeline deployed on NVIDIA IGX Orin.  
Analyzes DICOM series using MedGemma multimodal model with a fully on-premises setup — patient data never leaves the server.

## Architecture
 
```
Browser (Web UI)
    ↓
CORS Proxy (port 5001)          MedGemma Server (port 8080)
    ↓                                   ↓
Orthanc PACS (port 8042)        Gemma4 API Server (port 5000)
                                        ↓
                                  GPU Inference (RTX 6000 Ada)
```
| Service | Port | Description |
|---|---|---|
| Orthanc PACS | 8042 | DICOM storage and REST API |
| MedGemma Server | 8080 | DICOM preprocessing + API proxy |
| Gemma4 API Server | 5000 | vLLM / HF inference engine |
| CORS Proxy | 5001 | Browser-to-Orthanc bridge |

## Hardware

| Component | Specification |
|---|---|
| Platform | NVIDIA IGX Orin |
| GPU | NVIDIA RTX 6000 Ada Generation (48 GB VRAM) |
| CPU | 12-core Arm Cortex-A78AE |
| RAM | 64 GB LPDDR5 |
| Storage | NVMe SSD |
| Architecture | ARM64 |
---

## Prerequisites
 
### 1. Gemma4 API Server
This pipeline requires the Gemma4 API Server running on port 5000.  
Set it up first: https://github.com/Kaiwei0323/gemma4-api-server
 
Once running, note the server IP address before proceeding.

### 2. Model Checkpoint
Download MedGemma 4B from HuggingFace (requires account and model access approval):
 
```bash
docker run --rm \
  -v $(pwd)/medgemma-4b-it:/model \
  python:3.11-slim \
  bash -c "pip install huggingface_hub -q && python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/medgemma-4b-it',
    local_dir='/model',
    token='YOUR_HF_TOKEN'
)
\""
```
 
---

## Setup
 
### Step 1: Clone this repo
 
```bash
git clone https://github.com/yenyenn19/medgemma-igx-deploy.git
cd medgemma-igx-deploy
```

### Step 2: Prepare DICOM Data & Start Orthanc
 
Create the data directory:
 
```bash
mkdir -p data/orthanc-db
```
 
> Place your DICOM files (`.dcm`) inside `data/orthanc-db/` before starting Orthanc. Orthanc will automatically index them on startup. 
 
```bash
docker run -d \
  --name orthanc \
  -p 8042:8042 \
  -p 4242:4242 \
  -v ~/Documents/medgemma_deploy/data/orthanc-db:/var/lib/orthanc/db \
  orthancteam/orthanc:latest
```
 
Verify Orthanc is running:
```bash
curl http://orthanc:orthanc@localhost:8042/system
```
 
Verify DICOM data is loaded:
```bash
curl http://orthanc:orthanc@localhost:8042/series
```
 
 
### Step 3: Start MedGemma Server
 
```bash
docker build -t medgemma-server .
 
docker run -d \
  --name medgemma-server \
  -p 8080:8080 \
  -e GEMMA_API_URL=http://<gemma4-api-server-ip>:5000 \
  -e PYTHONUNBUFFERED=1 \
  medgemma-server
```
Verify:
```bash
curl http://localhost:8080/health
```

### Step 4: Start CORS Proxy
 
```bash
docker build -t cors-proxy -f Dockerfile.cors .
 
docker run -d \
  --name cors-proxy \
  -p 5001:5000 \
  -e ORTHANC_URL=http://orthanc:orthanc@<orthanc-ip>:8042 \
  cors-proxy
```
 
Verify:
```bash
curl http://localhost:5001/orthanc/studies
```
 
### Step 5: Configure Web UI
 
Edit `medgemma_ui.html` and update the CONFIG section with your IGX IP:
 
```javascript
const CONFIG = {
    ORTHANC_URL: 'http://<IGX-IP>:5001/orthanc',
    MEDGEMMA_URL: 'http://<IGX-IP>:8080',
};
```
 
Also update the `orthanc_url` in the `analyzeStudy()` function:
 
```javascript
body: JSON.stringify({
    series_id: state.currentSeries,
    prompt: prompt,
    num_slices: numSlices,
    orthanc_url: 'http://orthanc:orthanc@<orthanc-ip>:8042'
})
```
### Step 6: Open Web UI
 
```bash
python3 -m http.server 8888
```
 
Open browser: `http://<IGX-IP>:8888/medgemma_ui.html`
 
---

 
## API Usage
 
Analyze a DICOM series directly via curl:
 
```bash
curl -X POST http://localhost:8080/series/predict \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "your-series-uuid",
    "prompt": "Describe any abnormalities.",
    "num_slices": 5,
    "orthanc_url": "http://orthanc:orthanc@<orthanc-ip>:8042"
  }'
```
 
Get a list of available series from Orthanc:
 
```bash
curl http://orthanc:orthanc@localhost:8042/series
```
 
---
 
