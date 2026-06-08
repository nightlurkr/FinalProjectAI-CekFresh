# ✅ CHECKLIST PROYEK CEKFRESH

> **Proyek**: CekFresh — Freshness Detector & Food Waste Solution  
> **Deadline**: Pertengahan Juni 2026  
> **Update terakhir**: 2 Juni 2026

---

## 📊 Progres Keseluruhan

| Fase | Status | Selesai |
|------|--------|---------|
| Fase 1 — Persiapan | ✅ Selesai | 4 / 4 |
| Fase 2 — Training Model | ✅ Selesai | 6 / 6 |
| Fase 3 — Aplikasi | ✅ Selesai | 4 / 4 |
| Fase 4 — User Guide | 🔄 Dalam Proses | 0 / 3 |
| Fase 5 — HKI | ⏳ Menunggu | 0 / 4 |

**Total**: 14 / 21 tugas selesai

---

## 🗓️ FASE 1 — PERSIAPAN DATASET ✅

- [x] Download dataset dari Kaggle
  - sriramr (Pisang & Jeruk) + raghavrpotdar (Tomat)

- [x] Extract dan susun dataset ke folder final
  - Struktur: `train/fresh`, `train/rotten`, `val/`, `test/`

- [x] Preprocessing dataset
  - Resize 224×224, split 70/15/15 (train/val/test)
  - Total: 10.531 gambar

- [x] Verifikasi jumlah gambar per kelas
  - Pisang: 1.581 fresh / 2.224 rotten
  - Jeruk: 1.466 fresh / 1.595 rotten
  - Tomat: 981 fresh / 982 rotten

---

## 🤖 FASE 2 — TRAINING MODEL ✅

- [x] Jalankan `fp_ai_kel8_final.ipynb` di Google Colab (GPU T4)

- [x] Training Phase 1: frozen base model
  - Best val_accuracy: 99.58%

- [x] Training Phase 2: fine-tuning 30 layer terakhir
  - Best val_accuracy: 99.81%

- [x] Simpan model ke `02_Model/saved/`
  - `cekfresh_model.keras` ✅
  - `class_names.json` ✅

- [x] Simpan grafik evaluasi ke `04_Evaluasi/grafik/`
  - `accuracy_loss_curve.png` ✅
  - `confusion_matrix.png` ✅

- [x] Catat hasil akurasi di `04_Evaluasi/hasil/`
  - Test Accuracy: **99.70%**
  - `classification_report.txt` ✅

---

## 💻 FASE 3 — APLIKASI STREAMLIT ✅

- [x] `app.py` dan `requirements.txt` tersedia di `03_App/`

- [x] Aplikasi berhasil jalan di `http://localhost:8501`
  ```bash
  python -m streamlit run app.py
  ```

- [x] Semua fitur berjalan
  - Upload gambar ✅
  - Prediksi dengan 3 label (Segar/Hampir Busuk/Busuk) ✅
  - Rekomendasi dan tips ✅

- [x] Demo ke dosen ✅ — 2 Juni 2026

> ⚠️ **Bug tercatat**: Model bias pada Tomat busuk (terdeteksi Segar)
> → Perlu re-train dengan dataset tomat lebih banyak

---

### 🔧 Pengembangan Lanjutan (To-Do)

- [ ] Fix bias tomat — tambah dataset & re-train model multiclass
- [ ] Tambah deteksi jenis buah otomatis (multiclass: 6 kelas)
- [ ] Fitur kamera langsung + toggle upload/kamera
- [ ] UI/UX improvement
- [ ] Push update ke GitHub setelah semua fix

---

## 📖 FASE 4 — USER GUIDE 🔄

> Dosen meminta User Guide (bukan laporan akademik)
> Simpan di `05_UserGuide/`

- [ ] Buat draft User Guide di `05_UserGuide/draft/`
  - Nama file: `CekFresh_UserGuide_Draft.pdf` / `.docx`

- [ ] Isi User Guide:
  - [ ] Deskripsi singkat aplikasi CekFresh
  - [ ] Cara instalasi & menjalankan aplikasi
  - [ ] Panduan penggunaan step-by-step (dengan screenshot)
  - [ ] Penjelasan hasil deteksi & rekomendasi
  - [ ] FAQ / Troubleshooting umum

- [ ] Finalisasi di `05_UserGuide/final/`
  - Simpan versi final: `CekFresh_UserGuide.pdf`

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
- Gunakan `model.save('/content/drive/MyDrive/CekFresh_Model/cekfresh_model.keras')`

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
