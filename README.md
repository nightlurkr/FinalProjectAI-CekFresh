# 🍃 VigorScan — Freshness Detector & Food Waste Solution

> Aplikasi web berbasis AI untuk mendeteksi kesegaran buah dan sayuran dari foto.
> Final Project Mata Kuliah Artificial Intelligence / Machine Learning — Kelompok 8

---

## 📌 Deskripsi Proyek

**VigorScan** adalah aplikasi web yang menggunakan *Transfer Learning* dengan arsitektur **EfficientNetV2-S** untuk mengklasifikasikan jenis dan kondisi buah secara otomatis dari foto. Aplikasi ini membantu pedagang & distributor mengurangi food waste dengan memberikan rekomendasi tindakan berdasarkan tingkat kesegaran produk.

### Buah & Sayur yang Didukung

| Produk | Label Output |
|--------|-------------|
| 🍌 Pisang | Segar / Hampir Busuk / Busuk |
| 🍊 Jeruk | Segar / Hampir Busuk / Busuk |
| 🍅 Tomat | Segar / Hampir Busuk / Busuk |

Model dilatih sebagai **multiclass 6 kelas**: kombinasi 3 jenis × 2 kondisi (`freshbanana`, `rottenbanana`, `freshorange`, `rottenorange`, `freshtomato`, `rottentomato`).

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3.10+ |
| Framework UI | Streamlit (dark/light mode, multi-upload, camera input) |
| Deep Learning | TensorFlow / Keras |
| Model Base | **EfficientNetV2-S** (Transfer Learning) |
| Post-processing | HSV color rescue (orange↔tomato disambiguation) |
| Dataset | Kaggle — sriramr (Pisang & Jeruk) + raghavrpotdar (Tomat busuk) + fruits-360 (Tomat segar) |
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
│   ├── app.py                  # Aplikasi Streamlit
│   ├── requirements.txt        # Daftar dependensi
│   ├── vigorscan_model.keras   # Model hasil training
│   └── class_names.json        # Label kelas (6 multiclass)
├── 04_Evaluasi/
│   ├── grafik/          # Plot accuracy, loss, confusion matrix
│   └── hasil/           # Classification report & metrik evaluasi
├── 05_UserGuide/
│   ├── draft/           # Draft user guide
│   └── final/           # VigorScan_UserGuide.docx
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
git clone https://github.com/nightlurkr/FinalProjectAI-CekFresh.git
cd FinalProjectAI-CekFresh/03_App
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

1. Toggle **Dark / Light** mode di pojok kanan atas sesuai preferensi
2. Pilih tab **Upload Foto** (multi-file) atau **Kamera Langsung**
3. Upload satu atau beberapa foto buah/sayur (JPG, PNG, JPEG)
4. Klik **"Deteksi Sekarang"**
5. Lihat hasil: jenis buah, badge **Layak Jual / Segera Jual / Tidak Layak**, skor kesegaran & pembusukan, rekomendasi tindakan + tips penanganan
6. Buka expander **"Detail probabilitas semua kelas"** untuk inspect prediksi tiap kelas

---

## 🧠 Arsitektur Model & Pipeline

```
Input Gambar (224×224×3, range [0, 255])
        ↓
EfficientNetV2-S (pre-trained ImageNet) — Feature Extraction
   • Internal Rescaling layer (otomatis [0,255] → normalized)
        ↓
Global Average Pooling
        ↓
Dense (128, ReLU) + Dropout (0.4)
        ↓
Dense (6, Softmax) — Multiclass classification
        ↓
Output: 6 probabilitas kelas
```

### Sistem Klasifikasi 2 Tahap

**Tahap 1 — Deteksi Jenis Buah:**
Aggregate per-buah: `score_pisang = p(freshbanana) + p(rottenbanana)`, dan seterusnya untuk jeruk & tomat. Jenis buah dengan skor agregat tertinggi dipilih sebagai prediksi.

**Tahap 2 — Hitung Skor Pembusukan (untuk buah terpilih):**

```
rotten_score = p(rotten_X) / (p(fresh_X) + p(rotten_X))
```

### Sistem Label 3 Tingkat

| Rotten Score | Label | Status |
|-------------|-------|--------|
| < 50% | 🟢 Segar | ✅ Layak Jual |
| 50% – 65% | 🟡 Hampir Busuk | ⚠️ Segera Jual / Gunakan |
| > 65% | 🔴 Busuk | ❌ Tidak Layak Jual |

### 🎨 Color Rescue Post-processing

Untuk handle dataset distribution shift (training data Western single-fruit-on-white-bg vs use case Indonesia multi-buah dengan daun), VigorScan menerapkan **targeted HSV color verification** khusus untuk konflik **orange↔tomato** (karena hue red & orange sangat dekat di color space):

- Analisis warna dominan di antara pixel **berwarna buah** saja (red/orange/yellow); pixel hijau (daun) di-exclude dari hitungan rasio
- Override hanya aktif jika `dominant_ratio > 1.4×` warna lawan AND di-trigger spesifik untuk konflik orange/tomato
- Saat override aktif, freshness signal di-derive dari brightness warna:
  - Merah terang & saturasi tinggi → Segar
  - Merah gelap (value rendah) → Busuk
- **Pisang (banana) tidak disentuh** oleh color rescue — prediksi murni dari model

### Proses Training

- **Dataset**: 10.531 gambar (Train: 7.211 / Val: 1.324 / Test: 1.996)
- **Augmentasi**: rotation, zoom, horizontal flip
- **Class weighting**: `compute_class_weight('balanced')` untuk handle imbalance
- **Phase 1**: Base model di-*freeze*, hanya melatih layer classifier baru (10 epoch)
- **Phase 2**: Fine-tuning 50 layer terakhir EfficientNetV2-S, learning rate 1e-5 (20 epoch)

---

## 📊 Dataset

| Dataset | Sumber | Isi |
|---------|--------|-----|
| Fruits Fresh and Rotten | [sriramr/Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) | Pisang & Jeruk (fresh + rotten) |
| Fresh and Stale Images | [raghavrpotdar/Kaggle](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables) | Tomat (stale) + tambahan pisang & jeruk |
| Fruits-360 | [moltean/Kaggle](https://www.kaggle.com/datasets/moltean/fruits) | Tomat segar |

---

## 📊 Evaluasi Model

Hasil evaluasi tersimpan di `04_Evaluasi/`:

- `grafik/` — Training/Validation Accuracy & Loss curve, Confusion Matrix
- `hasil/` — Classification Report (Precision, Recall, F1-Score per kelas)

---

## 📖 User Guide

Panduan penggunaan tersedia di `05_UserGuide/final/VigorScan_UserGuide.docx` mencakup:

- Deskripsi singkat aplikasi VigorScan
- Cara instalasi & menjalankan aplikasi
- Panduan penggunaan step-by-step (dengan screenshot)
- Penjelasan hasil deteksi & rekomendasi
- FAQ / Troubleshooting umum

---

## 👥 Tim Pengembang — Kelompok 8

| Nama | Peran |
|------|-------|
| *Mey Rosalina/5027241004* | Dataset & Preprocessing |
| *Nadia Fauziaazahra Kusumastuti/5027241094* | Model Training & Evaluasi |
| *Ryan Adya Purwanto/5027231046* | Pengembangan Aplikasi Streamlit |

---

## 📅 Timeline Proyek

| Fase | Kegiatan | Status |
|------|----------|--------|
| Fase 1 | Persiapan Dataset | ✅ Selesai |
| Fase 2 | Training Model (EfficientNetV2-S Multiclass) | ✅ Selesai |
| Fase 3 | Pengembangan Aplikasi (UI/UX, Multi-upload, Camera, Color Rescue) | ✅ Selesai |
| Fase 4 | Penulisan User Guide | ✅ Selesai |
| Fase 5 | Pendaftaran HKI | ⏳ Setelah Demo |


---

*Dibuat dengan ❤️ untuk Final Project AI/ML — Kelompok 8*
