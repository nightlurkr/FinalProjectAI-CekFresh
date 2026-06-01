# ✅ CHECKLIST PROYEK CEKFRESH

> **Proyek**: CekFresh — Freshness Detector & Food Waste Solution  
> **Deadline**: Pertengahan Juni 2026  
> **Update terakhir**: 26 Mei 2026

---

## 📊 Progres Keseluruhan

| Fase | Status | Selesai |
|------|--------|---------|
| Fase 1 — Persiapan | 🔴 Belum Mulai | 0 / 4 |
| Fase 2 — Training Model | 🔴 Belum Mulai | 0 / 6 |
| Fase 3 — Aplikasi | 🔴 Belum Mulai | 0 / 4 |
| Fase 4 — Laporan | 🔴 Belum Mulai | 0 / 4 |
| Fase 5 — HKI | 🔴 Belum Mulai | 0 / 4 |

**Total**: 0 / 22 tugas selesai

---

## 🗓️ FASE 1 — PERSIAPAN DATASET (Minggu 1)

**Target selesai**: Akhir minggu pertama

- [ ] Download dataset dari Kaggle
  - Keyword pencarian: `fruits fresh rotten classification`
  - Link: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
  - Simpan ke: `01_Dataset/raw/`

- [ ] Extract dan susun dataset ke folder `01_Dataset/raw/`
  - Struktur: `raw/Train/` dan `raw/Test/`
  - Pastikan ada subfolder untuk setiap kelas

- [ ] Preprocessing dataset
  - Resize semua gambar ke 224×224 piksel
  - Split data: 80% train / 20% validation (jika belum tersplit)
  - Simpan ke `01_Dataset/processed/`

- [ ] Verifikasi jumlah gambar per kelas
  - Catat jumlah gambar: Apel Segar, Apel Busuk
  - Catat jumlah gambar: Pisang Segar, Pisang Busuk
  - Catat jumlah gambar: Tomat Segar, Tomat Busuk
  - Pastikan tidak ada ketidakseimbangan kelas yang ekstrem

---

## 🤖 FASE 2 — TRAINING MODEL (Minggu 1–2)

**Target selesai**: Pertengahan minggu kedua

- [ ] Jalankan `CekFresh_train.ipynb` di Google Colab
  - Upload notebook ke Colab atau buka via Google Drive
  - Pastikan GPU aktif: Runtime → Change runtime type → GPU

- [ ] Training Phase 1: frozen base model
  - Base MobileNet V2 di-freeze
  - Latih hanya layer classifier baru
  - Minimal 10 epoch
  - Catat accuracy dan loss

- [ ] Training Phase 2: fine-tuning 30 layer terakhir
  - Unfreeze 30 layer terakhir MobileNet V2
  - Gunakan learning rate kecil (1e-5)
  - Minimal 10 epoch tambahan
  - Target: validation accuracy ≥ 85%

- [ ] Simpan model ke `02_Model/saved/`
  - File: `cekfresh_model.h5`
  - File: `class_names.json`
  - Download dari Colab ke lokal

- [ ] Simpan grafik evaluasi ke `04_Evaluasi/grafik/`
  - `accuracy_curve.png` — Training vs Validation Accuracy
  - `loss_curve.png` — Training vs Validation Loss
  - `confusion_matrix.png` — Confusion Matrix

- [ ] Catat hasil akurasi di `04_Evaluasi/hasil/`
  - Buat file `classification_report.txt`
  - Catat: Precision, Recall, F1-Score per kelas
  - Catat: Overall Accuracy, val_accuracy final

---

## 💻 FASE 3 — APLIKASI STREAMLIT (Minggu 2)

**Target selesai**: Akhir minggu kedua

- [ ] Letakkan file aplikasi di `03_App/`
  - Salin `CekFresh_app.py` → `03_App/app.py`
  - Salin `CekFresh_requirements.txt` → `03_App/requirements.txt`
  - Pastikan path model di `app.py` sudah benar

- [ ] Test jalankan Streamlit
  ```bash
  cd 03_App
  pip install -r requirements.txt
  streamlit run app.py
  ```
  - Pastikan aplikasi terbuka di browser tanpa error

- [ ] Verifikasi semua fitur berjalan
  - [ ] Upload gambar berhasil
  - [ ] Prediksi tampil dengan label yang benar
  - [ ] Rekomendasi muncul sesuai label (Segar/Hampir Busuk/Busuk)
  - [ ] Tampilan UI rapi dan informatif

- [ ] Perbaiki bug jika ada
  - Dokumentasikan bug yang ditemukan
  - Catat solusi yang diterapkan

---

## 📝 FASE 4 — LAPORAN (Minggu 2)

**Target selesai**: Akhir minggu kedua

- [ ] Buat draft laporan di `05_Laporan/draft/`
  - Nama file: `CekFresh_Laporan_Draft.docx`
  - Gunakan template laporan dari kampus jika ada

- [ ] Isi semua bagian laporan:
  - [ ] **BAB 1** — Latar belakang, rumusan masalah, tujuan
  - [ ] **BAB 2** — Tinjauan pustaka (MobileNet V2, Transfer Learning, food waste)
  - [ ] **BAB 3** — Metodologi (dataset, preprocessing, arsitektur, training)
  - [ ] **BAB 4** — Hasil evaluasi + grafik dari `04_Evaluasi/grafik/`
  - [ ] **BAB 5** — Kesimpulan dan saran

- [ ] Masukkan grafik dan tabel evaluasi
  - Accuracy & Loss curve
  - Confusion Matrix
  - Tabel Classification Report

- [ ] Review dan finalisasi laporan di `05_Laporan/final/`
  - Cek penulisan dan tata bahasa
  - Cek format sitasi
  - Simpan versi final: `CekFresh_Laporan_Final.docx` + `.pdf`

---

## 📋 FASE 5 — PENDAFTARAN HKI (Setelah Deadline)

**Target selesai**: Setelah deadline Juni 2026

- [ ] Cari persyaratan pendaftaran HKI Program Komputer di DJKI
  - Website: https://djki.kemenkumham.go.id
  - Pilih: Hak Cipta → Program Komputer
  - Catat dokumen yang dibutuhkan

- [ ] Siapkan dokumen pendaftaran di `06_HKI/dokumen/`
  - [ ] Deskripsi program (nama, fungsi, teknologi)
  - [ ] Source code (print/PDF dari file utama)
  - [ ] Data pencipta (nama lengkap, NIK, alamat semua anggota tim)
  - [ ] Surat pernyataan kepemilikan (jika diperlukan)
  - [ ] Formulir permohonan yang sudah diisi

- [ ] Isi formulir pendaftaran online di sistem DJKI
  - Buat akun di e-hakcipta.dgip.go.id
  - Upload semua dokumen yang dibutuhkan
  - Bayar biaya pendaftaran (jika ada)

- [ ] Simpan bukti pendaftaran di `06_HKI/submitted/`
  - Screenshot konfirmasi pendaftaran
  - Nomor permohonan HKI
  - Simpan semua bukti pembayaran

---

## 📌 CATATAN PENTING

### Tips Training di Google Colab
- Pastikan gunakan **GPU** (bukan CPU) untuk mempercepat training
- Simpan model ke **Google Drive** agar tidak hilang jika sesi Colab berakhir
- Gunakan `model.save('/content/drive/MyDrive/cekfresh_model.h5')`

### Target Akurasi
- **Minimum**: 85% overall accuracy
- **Ideal**: 90%+ per kelas

### Sistem Label & Rekomendasi
```
Segar        → "Produk layak jual, simpan di kondisi optimal"
Hampir Busuk → "Segera jual/gunakan, pertimbangkan diskon"  
Busuk        → "Tidak layak jual, pisahkan untuk kompos"
```

### Link Berguna
- Dataset Kaggle: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
- Google Colab: https://colab.research.google.com
- Streamlit Docs: https://docs.streamlit.io
- DJKI HKI: https://djki.kemenkumham.go.id
- e-HKI Portal: https://e-hakcipta.dgip.go.id

---

*Update checklist ini setiap kali menyelesaikan tugas dengan mengganti `[ ]` menjadi `[x]`*
