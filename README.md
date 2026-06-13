# 🍃 VigorScan — Freshness Detector & Food Waste Solution

> Aplikasi web berbasis AI untuk mendeteksi kelayakan jual buah dan sayuran dari foto.  
> Final Project Mata Kuliah Artificial Intelligence / Machine Learning

---

## 📌 Deskripsi Proyek

**VigorScan** adalah aplikasi web yang menggunakan *Transfer Learning* dengan arsitektur **MobileNet V2** untuk mengklasifikasikan kondisi buah dan sayur secara otomatis hanya dari foto. Aplikasi ini bertujuan membantu mengurangi food waste dengan memberikan rekomendasi tindakan berdasarkan tingkat kesegaran produk.

### Buah & Sayur yang Didukung
| Produk | Label Output |
|--------|-------------|
| 🍌 Pisang | Segar / Hampir Busuk / Busuk |
| 🍊 Jeruk | Segar / Hampir Busuk / Busuk |
| 🍅 Tomat | Segar / Hampir Busuk / Busuk |

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3.x |
| Framework UI | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Model Base | MobileNet V2 (Transfer Learning) |
| Dataset | Kaggle — sriramr (Pisang & Jeruk) + raghavrpotdar (Tomat) |
| Training | Google Colab (GPU T4) |

---

## 📁 Struktur Folder

```
VigorScan/
├── 01_Dataset/
│   ├── raw/             # Dataset mentah dari Kaggle
│   └── processed/       # Dataset setelah preprocessing
├── 02_Model/
│   ├── training/        # Notebook training (fp_ai_kel8_final.ipynb)
│   └── saved/           # Model tersimpan (.keras) & class_names.json
├── 03_App/
│   ├── app.py               # Aplikasi Streamlit
│   ├── requirements.txt     # Daftar dependensi
│   ├── vigorscan_model.keras # Model hasil training
│   └── class_names.json     # Label kelas
├── 04_Evaluasi/
│   ├── grafik/          # Plot accuracy, loss, confusion matrix
│   └── hasil/           # Classification report & metrik evaluasi
├── 05_UserGuide/
│   ├── draft/           # Draft user guide
│   └── final/           # User guide final (PDF)
├── 06_HKI/
│   ├── dokumen/         # Formulir & persyaratan HKI
│   └── submitted/       # Bukti pendaftaran HKI
├── CHECKLIST.md         # Daftar tugas & progres
└── README.md            # File ini
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/nightlurkr/FinalProjectAI-VigorScan.git
cd FinalProjectAI-VigorScan/03_App
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Pastikan Model Tersedia

Pastikan file berikut ada di folder `03_App/`:
- `vigorscan_model.keras`
- `class_names.json`

### 4. Jalankan Aplikasi Streamlit

```bash
python -m streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser: `http://localhost:8501`

### 5. Cara Pakai

1. Upload foto buah/sayur (format: JPG, PNG, JPEG)
2. Klik tombol **"Deteksi Sekarang"**
3. Lihat hasil prediksi: **Segar / Hampir Busuk / Busuk**
4. Baca rekomendasi tindakan yang ditampilkan

---

## 🧠 Arsitektur Model

```
Input Gambar (224×224×3)
        ↓
MobileNet V2 (pre-trained ImageNet) — Feature Extraction
        ↓
Global Average Pooling
        ↓
Dense (128, ReLU) + Dropout (0.4)
        ↓
Dense (1, Sigmoid) — Binary classification
        ↓
Output: Fresh (0) / Rotten (1)
```

### Sistem Label 3 Tingkat
Model menghasilkan nilai probabilitas *rotten* (0.0–1.0), kemudian dikonversi ke 3 label menggunakan threshold:

| Probabilitas Rotten | Label Tampil | Status |
|---------------------|--------------|--------|
| < 30% | 🟢 Segar | ✅ Layak Jual |
| 30% – 65% | 🟡 Hampir Busuk | ⚠️ Segera Jual |
| > 65% | 🔴 Busuk | ❌ Tidak Layak Jual |

### Proses Training
- **Dataset**: 10.531 gambar (Train: 7.211 / Val: 1.324 / Test: 1.996)
- **Phase 1**: Base model di-*freeze*, hanya melatih layer classifier baru (10 epoch)
- **Phase 2**: Fine-tuning 30 layer terakhir MobileNet V2 dengan learning rate 1e-5 (15 epoch)

---

## 📊 Dataset

| Dataset | Sumber | Isi |
|---------|--------|-----|
| Fruits Fresh and Rotten | [sriramr/Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) | Pisang & Jeruk |
| Fresh and Stale Images | [raghavrpotdar/Kaggle](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) | Tomat |

---

## 📊 Evaluasi Model

Hasil evaluasi tersimpan di `04_Evaluasi/`:
- `grafik/` — Training/Validation Accuracy & Loss curve, Confusion Matrix
- `hasil/` — Classification Report (Precision, Recall, F1-Score)

---

## 📖 User Guide

Panduan penggunaan tersedia di `05_UserGuide/final/` mencakup:
- Deskripsi singkat aplikasi VigorScan
- Cara instalasi & menjalankan aplikasi
- Panduan penggunaan step-by-step (dengan screenshot)
- Penjelasan hasil deteksi & rekomendasi
- FAQ / Troubleshooting umum

---

## 👥 Tim Pengembang

| Nama | Peran |
|------|-------|
| *(Nama Anggota 1)* | Dataset & Preprocessing |
| *(Nama Anggota 2)* | Model Training & Evaluasi |
| *(Nama Anggota 3)* | Pengembangan Aplikasi Streamlit |

---

## 📅 Timeline Proyek

| Fase | Kegiatan | Status |
|------|----------|--------|
| Fase 1 | Persiapan Dataset | ✅ Selesai |
| Fase 2 | Training Model | ✅ Selesai |
| Fase 3 | Pengembangan Aplikasi | ✅ Selesai |
| Fase 4 | Penulisan Laporan | 🔄 Dalam Proses |
| Fase 5 | Pendaftaran HKI | ⏳ Menunggu |

**Deadline**: Pertengahan Juni 2026

---

*Dibuat dengan ❤️ untuk Final Project AI/ML*
