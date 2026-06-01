# 🍎 CekFresh — Freshness Detector & Food Waste Solution

> Aplikasi web berbasis AI untuk mendeteksi kelayakan jual buah dan sayuran dari foto.  
> Final Project Mata Kuliah Artificial Intelligence / Machine Learning

---

## 📌 Deskripsi Proyek

**CekFresh** adalah aplikasi web yang menggunakan *Transfer Learning* dengan arsitektur **MobileNet V2** untuk mengklasifikasikan kondisi buah dan sayur secara otomatis hanya dari foto. Aplikasi ini bertujuan membantu mengurangi food waste dengan memberikan rekomendasi tindakan berdasarkan tingkat kesegaran produk.

### Buah & Sayur yang Didukung
| Produk | Label Output |
|--------|-------------|
| 🍎 Apel | Segar / Hampir Busuk / Busuk |
| 🍌 Pisang | Segar / Hampir Busuk / Busuk |
| 🍅 Tomat | Segar / Hampir Busuk / Busuk |

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python 3.x |
| Framework UI | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Model Base | MobileNet V2 (Transfer Learning) |
| Dataset | Kaggle — Fruits Fresh and Rotten Classification |
| Training | Google Colab |

---

## 📁 Struktur Folder

```
CekFresh/
├── 01_Dataset/
│   ├── raw/             # Dataset mentah dari Kaggle
│   └── processed/       # Dataset setelah preprocessing
├── 02_Model/
│   ├── training/        # Notebook training (.ipynb)
│   └── saved/           # Model tersimpan (.h5) & class_names.json
├── 03_App/
│   ├── app.py           # Aplikasi Streamlit
│   └── requirements.txt # Daftar dependensi
├── 04_Evaluasi/
│   ├── grafik/          # Plot accuracy, loss, confusion matrix
│   └── hasil/           # Classification report & metrik evaluasi
├── 05_Laporan/
│   ├── draft/           # Draft laporan
│   └── final/           # Laporan final
├── 06_HKI/
│   ├── dokumen/         # Formulir & persyaratan HKI
│   └── submitted/       # Bukti pendaftaran HKI
├── CHECKLIST.md         # Daftar tugas & progres
└── README.md            # File ini
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Persiapan Environment

```bash
# Clone / buka folder proyek
cd CekFresh/03_App

# Install dependensi
pip install -r requirements.txt
```

### 2. Pastikan Model Tersedia

Pastikan file berikut sudah ada di `02_Model/saved/`:
- `cekfresh_model.h5`
- `class_names.json`

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser: `http://localhost:8501`

### 4. Cara Pakai

1. Upload foto buah/sayur (format: JPG, PNG, JPEG)
2. Klik tombol **"Deteksi"**
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
Dense (128, ReLU) + Dropout (0.5)
        ↓
Dense (9, Softmax) — 9 kelas output
        ↓
Label: [Apel Segar, Apel Hampir Busuk, Apel Busuk,
        Pisang Segar, Pisang Hampir Busuk, Pisang Busuk,
        Tomat Segar, Tomat Hampir Busuk, Tomat Busuk]
```

### Proses Training
- **Phase 1**: Base model di-*freeze*, hanya melatih layer classifier baru
- **Phase 2**: Fine-tuning 30 layer terakhir MobileNet V2 dengan learning rate kecil
- **Target Akurasi**: ≥ 85%

---

## 📊 Evaluasi Model

Hasil evaluasi disimpan di `04_Evaluasi/`:
- `grafik/` — Training/Validation Accuracy & Loss curve, Confusion Matrix
- `hasil/` — Classification Report (Precision, Recall, F1-Score per kelas)

---

## 📄 Laporan

Laporan lengkap tersedia di `05_Laporan/final/` mencakup:
- Latar Belakang & Rumusan Masalah
- Tinjauan Pustaka (MobileNet V2, Transfer Learning)
- Metodologi & Implementasi
- Hasil Evaluasi & Analisis
- Kesimpulan & Saran

---

## ©️ Hak Kekayaan Intelektual (HKI)

Proyek ini didaftarkan sebagai **Hak Cipta Program Komputer** ke:  
**DJKI — Direktorat Jenderal Kekayaan Intelektual**  
Website: [djki.kemenkumham.go.id](https://djki.kemenkumham.go.id)

Dokumen HKI tersimpan di `06_HKI/`.

---

## 👥 Tim Pengembang

| Nama | Peran |
|------|-------|
| *(Nama Anggota 1)* | Model Training & Evaluasi |
| *(Nama Anggota 2)* | Pengembangan Aplikasi Streamlit |
| *(Nama Anggota 3)* | Dataset & Laporan |

---

## 📅 Timeline Proyek

| Fase | Kegiatan | Target |
|------|----------|--------|
| Fase 1 | Persiapan Dataset | Minggu 1 |
| Fase 2 | Training Model | Minggu 1–2 |
| Fase 3 | Pengembangan Aplikasi | Minggu 2 |
| Fase 4 | Penulisan Laporan | Minggu 2 |
| Fase 5 | Pendaftaran HKI | Setelah Deadline |

**Deadline**: Pertengahan Juni 2026

---

*Dibuat dengan ❤️ untuk Final Project AI/ML*
