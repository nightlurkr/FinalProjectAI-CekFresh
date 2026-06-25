# 🚀 Deployment Guide — VigorScan

> Cara deploy VigorScan ke public URL supaya bisa diakses tanpa setup lokal.
> Rekomendasi: **Streamlit Community Cloud** (gratis, GitHub-integrated).

---

## ⚠️ Prerequisite

- Repository sudah di GitHub (✅ done: `nightlurkr/FinalProjectAI-CekFresh`)
- File `vigorscan_model.keras` (~117 MB) ter-commit di `03_App/`
  - Cek dengan: `git ls-files 03_App/*.keras`
- `requirements.txt` sudah lengkap

---

## 🌐 Opsi 1 — Streamlit Community Cloud (Recommended)

### Kelebihan

- ✅ Gratis (untuk public repo)
- ✅ Auto-deploy: setiap push ke `main` otomatis re-deploy
- ✅ HTTPS bawaan
- ✅ Custom subdomain `vigorscan.streamlit.app`
- ✅ Logs accessible via dashboard

### Limitasi

- Public repo only (private repo butuh paid plan)
- Resource: 1 GB RAM, ~1 vCPU
- File limit per push: 100 MB (kalau model >100 MB harus pakai Git LFS atau external storage)

### Step-by-Step

#### 1. Cek ukuran model

```cmd
cd "C:\Users\Ryan\Documents\final project AI\CekFresh"
dir 03_App\vigorscan_model.keras
```

Kalau **>100 MB**, ada 2 opsi:

**A. Git LFS** (paling clean):

```cmd
git lfs install
git lfs track "*.keras"
git add .gitattributes 03_App/vigorscan_model.keras
git commit -m "chore: track model with Git LFS"
git push origin main
```

**B. Download dari Drive saat startup** (lebih hassle, tapi tidak butuh LFS):

Tambahkan di awal `app.py`:

```python
import gdown, os
if not os.path.exists("vigorscan_model.keras"):
    gdown.download(
        "https://drive.google.com/uc?id=YOUR_FILE_ID",
        "vigorscan_model.keras", quiet=False
    )
```

Lalu add `gdown` ke `requirements.txt`.

#### 2. Daftar ke Streamlit Cloud

1. Buka <https://share.streamlit.io/>
2. Klik **Sign in with GitHub**
3. Authorize Streamlit untuk akses repo

#### 3. Deploy aplikasi

1. Klik **New app** (pojok kanan atas)
2. Isi form:
   - **Repository**: `nightlurkr/FinalProjectAI-CekFresh`
   - **Branch**: `main`
   - **Main file path**: `03_App/app.py`
   - **App URL** (opsional): `vigorscan` → akan jadi `vigorscan.streamlit.app`
3. Klik **Advanced settings**:
   - **Python version**: 3.10 atau 3.11
   - Tambahkan secrets jika perlu (untuk app ini tidak perlu)
4. Klik **Deploy!**

Tunggu ~3–5 menit untuk first build. Logs muncul real-time.

#### 4. Verifikasi

Setelah deploy:

- Test upload foto → harus jalan seperti lokal
- Test toggle dark/light
- Test camera (perlu HTTPS, Streamlit Cloud sudah HTTPS ✓)

#### 5. Troubleshooting

| Error | Solusi |
|-------|--------|
| `ModuleNotFoundError` | Tambah package ke `requirements.txt`, push |
| `Out of memory` | Model 117 MB + TF runtime bisa habis 1 GB. Coba downgrade ke quantized model atau pakai paid tier |
| `protobuf incompatible` | Pin `protobuf>=6.31.1` di requirements.txt |
| `Model file not found` | Cek model ter-commit. Lihat juga path: app.py expects `vigorscan_model.keras` di working dir, jadi pastikan `Main file path = 03_App/app.py` |
| Build hang di "Installing requirements" | Edit requirements.txt buang yang tidak perlu, push ulang |

---

## 🐳 Opsi 2 — Docker + Cloud Run / Railway / Fly.io

Lebih powerful tapi lebih effort. Bagus untuk production.

### Dockerfile (taruh di root repo)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps untuk TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY 03_App/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 03_App/ .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Deploy ke Railway

1. <https://railway.app> → Sign in with GitHub
2. New Project → Deploy from GitHub repo
3. Railway auto-detect Dockerfile
4. Set environment variable: `PORT=8501`
5. Deploy

### Deploy ke Google Cloud Run

```cmd
gcloud builds submit --tag gcr.io/PROJECT_ID/vigorscan
gcloud run deploy vigorscan --image gcr.io/PROJECT_ID/vigorscan --port 8501 --memory 2Gi --allow-unauthenticated
```

---

## 🏠 Opsi 3 — ngrok (Quick Demo Sharing)

Untuk share localhost ke audience demo tanpa deploy permanent.

### Setup

```cmd
# Install ngrok dari https://ngrok.com/download
ngrok config add-authtoken YOUR_TOKEN

# Jalankan streamlit dulu
cd "C:\Users\Ryan\Documents\final project AI\CekFresh\03_App"
python -m streamlit run app.py
```

Lalu buka terminal baru:

```cmd
ngrok http 8501
```

ngrok akan kasih URL public seperti `https://abc123.ngrok-free.app`. Bagikan ke audience.

**Catatan:** URL berubah tiap session free tier. Untuk URL tetap perlu paid.

---

## 📌 Rekomendasi untuk Tim Kalian

**Pasca-demo (Senin sore):**

1. **Jika model ≤100 MB:** langsung deploy ke Streamlit Cloud (1 jam max). Submit URL ke spreadsheet asisten.
2. **Jika model >100 MB:** pakai Git LFS atau pindahkan model ke Google Drive + auto-download.

**Untuk submission GitHub (deadline 26 Juni):**

- Tambah link Streamlit Cloud URL di `README.md` bagian top
- Update `CHECKLIST.md` Fase 5 dengan deployment status

---

## 🔗 Useful Links

- Streamlit Cloud: <https://share.streamlit.io>
- Streamlit Docs (Deploy): <https://docs.streamlit.io/deploy/streamlit-community-cloud>
- Git LFS: <https://git-lfs.com>
- ngrok: <https://ngrok.com>
- Railway: <https://railway.app>

---

*Setelah deploy, jangan lupa update README dengan badge "Live Demo" di GitHub!*
