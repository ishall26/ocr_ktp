---
title: KTP OCR Model API
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
short_description: OCR API untuk identifikasi dan pembacaan KTP menggunakan model deep learning
---

# KTP OCR Model API

API untuk menjalankan model OCR KTP yang di-host di Hugging Face Hub.

## 🚀 Setup Cepat

### Instalasi Lokal

```bash
# 1. Clone dan masuk direktori
cd PCVK/ocr_ktp

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env dengan Hugging Face token dan model repository Anda

# 5. Jalankan
python app.py
```

## 📋 Konfigurasi Environment Variables

Buat file `.env`:

```env
HF_TOKEN=your_huggingface_token_here
MODEL_REPO=your-username/ktp-ocr-model
MODEL_NAME=ktp_fraud_cnn_tampering_v1
FLASK_ENV=production
PORT=5000
```

**Variabel:**
- `HF_TOKEN`: Token dari [Hugging Face Settings](https://huggingface.co/settings/tokens)
- `MODEL_REPO`: Repository ID (format: `username/repo-name`)
- `MODEL_NAME`: Nama file model yang akan didownload
- `FLASK_ENV`: `development` atau `production`
- `PORT`: Port untuk API (default: 5000)

## 🔌 API Endpoints

### 1. Health Check
```
GET /health
```
Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_repo": "your-username/ktp-ocr-model"
}
```

### 2. Model Info
```
GET /model-info
```
Response:
```json
{
  "model_name": "ktp_fraud_cnn_tampering_v1",
  "model_loaded": true,
  "device": "cuda",
  "pytorch_version": "2.1.0"
}
```

### 3. Predict (OCR)
```
POST /predict
```

**Option 1: Upload File**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

**Option 2: Send Base64**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"iVBORw0KGgoAAAANSUhEUgAAAAUA..."}'
```

Response:
```json
{
  "success": true,
  "predictions": [[...]],
  "model": "ktp_fraud_cnn_tampering_v1"
}
```

## 🐳 Docker Deployment

### Build & Run
```bash
docker build -t ktp-ocr-api .

docker run -p 5000:5000 \
  -e HF_TOKEN=your_token \
  -e MODEL_REPO=your-username/ktp-ocr-model \
  -e MODEL_NAME=ktp_fraud_cnn_tampering_v1 \
  ktp-ocr-api
```

### Dengan Docker Compose
Buat `docker-compose.yml`:
```yaml
version: '3.8'
services:
  ocr-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      HF_TOKEN: ${HF_TOKEN}
      MODEL_REPO: ${MODEL_REPO}
      MODEL_NAME: ${MODEL_NAME}
    volumes:
      - ./models:/app/models
```

```bash
docker-compose up
```

## 📤 Upload Model ke Hugging Face

```bash
# 1. Login
huggingface-cli login

# 2. Create repository
huggingface-cli repo create ktp-ocr-model

# 3. Clone & push model
git clone https://huggingface.co/your-username/ktp-ocr-model
cd ktp-ocr-model
git lfs install
cp ../ktp_fraud_cnn_tampering_v1.pt .
git add .
git commit -m "Add KTP OCR model"
git push
```

## 🧪 Testing

```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/model-info

# Predict
curl -X POST -F "file=@ktp_image.jpg" http://localhost:5000/predict
```

## 📚 Requirements

- Python 3.10+
- PyTorch 2.1.0
- Flask 3.0.0
- OpenCV 4.8.1
- Transformers 4.35.0
- Hugging Face Hub

## 📖 Dokumentasi Tambahan

- [Hugging Face Hub Docs](https://huggingface.co/docs/hub/security-tokens)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Flask Docs](https://flask.palletsprojects.com/)

## ⚠️ Troubleshooting

**Model tidak ditemukan:**
- Verify `HF_TOKEN` is valid
- Check `MODEL_REPO` and `MODEL_NAME` are correct
- Ensure model exists on Hugging Face

**Out of Memory:**
- Use GPU if available
- Reduce image size in preprocessing

**Import errors:**
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

---

Untuk dokumentasi lebih lengkap tentang Space Config, kunjungi [Hugging Face Spaces Config Reference](https://huggingface.co/docs/hub/spaces-config-reference)
