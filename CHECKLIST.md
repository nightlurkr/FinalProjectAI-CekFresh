# ✅ CHECKLIST PROYEK VIGORSCAN

> **Proyek**: VigorScan — Freshness Detector & Food Waste Solution
> **Deadline**: Pertengahan Juni 2026
> **Update terakhir**: 19 Juni 2026

---

## 📊 Progres Keseluruhan

| Fase | Status | Selesai |
|------|--------|---------|
| Fase 1 — Persiapan Dataset | ✅ Selesai | 4 / 4 |
| Fase 2 — Training Model | ✅ Selesai | 6 / 6 |
| Fase 3 — Aplikasi Streamlit | ✅ Selesai | 9 / 9 |
| Fase 4 — User Guide | ✅ Selesai | 3 / 3 |
| Fase 5 — HKI | ⏳ Menunggu | 0 / 4 |

**Total**: 22 / 26 tugas selesai (fase HKI dikerjakan setelah demo)

---

## 🗓️ FASE 1 — PERSIAPAN DATASET ✅

- [x] Download dataset dari Kaggle
  - sriramr (Pisang & Jeruk) + raghavrpotdar (Tomat) + fruits-360 (Tomat segar)

- [x] Extract dan susun dataset ke folder final
  - Struktur multiclass: `train/freshbanana`, `train/rottenbanana`, ..., `val/`, `test/`

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
  - **Arsitektur**: EfficientNetV2-S (upgrade dari MobileNetV2 untuk mengatasi class imbalance)

- [x] Training Phase 1: frozen base model + class_weight balanced
  - 10 epoch dengan classifier baru

- [x] Training Phase 2: fine-tuning 50 layer terakhir
  - 20 epoch, learning rate 1e-5

- [x] Simpan model ke `02_Model/saved/` dan `03_App/`
  - `vigorscan_model.keras` ✅
  - `class_names.json` (6 kelas multiclass) ✅

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

- [x] Multiclass detection (6 kelas, deteksi jenis buah otomatis)

- [x] Fitur multi-upload (batch detection)

- [x] Fitur kamera langsung (camera input)

- [x] Toggle Dark / Light mode

- [x] UI/UX redesign — glassmorphism hero, custom colored bars (hijau/merah),
      Plus Jakarta Sans typography, pill-style tabs, animated transitions,
      session state persistence (hasil tidak hilang saat toggle tema)

- [x] Color rescue post-processing
      Targeted HSV color verification untuk konflik orange↔tomato.
      Daun hijau di-exclude dari hitungan rasio. Banana tidak disentuh.

- [x] Push ke GitHub & demo internal

---

## 📖 FASE 4 — USER GUIDE ✅

> Dosen meminta User Guide (bukan laporan akademik)
> Simpan di `05_UserGuide/`

- [x] Buat draft User Guide di `05_UserGuide/draft/`
  - `CekFresh_UserGuide_Draft.docx` (legacy)

- [x] Finalisasi di `05_UserGuide/final/`
  - `VigorScan_UserGuide.docx` ✅

- [x] Konten User Guide
  - Deskripsi aplikasi VigorScan ✅
  - Cara instalasi & menjalankan aplikasi ✅
  - Panduan penggunaan step-by-step ✅
  - Penjelasan hasil deteksi & rekomendasi ✅
  - FAQ / Troubleshooting ✅

---

## 📋 FASE 5 — PENDAFTARAN HKI (Setelah Demo)

**Target selesai**: Setelah deadline Juni 2026

- [ ] Cari persyaratan pendaftaran HKI Program Komputer di DJKI
  - Website: <https://djki.kemenkumham.go.id>
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

### Engineering Decisions yang Bisa Dijelaskan di Q&A

**1. Kenapa upgrade ke EfficientNetV2-S dari MobileNetV2?**
MobileNetV2 lemah pada class imbalance kami (tomat 1.963 vs pisang 3.805). EfficientNetV2-S kapasitas lebih besar + compound scaling, plus dipasang `class_weight='balanced'` untuk handle imbalance secara eksplisit.

**2. Kenapa preprocessing pakai range [0,255], bukan normalisasi?**
EfficientNetV2-S punya internal `Rescaling` layer di arsitekturnya, sehingga `tf.keras.applications.efficientnet_v2.preprocess_input` adalah no-op. Normalisasi manual akan menyebabkan double-rescaling dan input out-of-distribution.

**3. Kenapa pakai color rescue?**
Dataset training (Western sources) tidak mencakup tomat di tangkai dengan daun lebat — model rentan misclassify ke orange karena hue red-orange berdekatan. Color rescue HSV memverifikasi hue dominant di antara pixel buah (daun di-exclude), khusus untuk konflik orange↔tomato.

**4. Kenapa color rescue tidak menyentuh banana?**
Pisang busuk punya bintik coklat yang bisa salah dianalisis sebagai orange (brown ≈ low-saturation orange). Demi safety, banana branch dibiarkan murni dari model.

### Threshold Klasifikasi

```
Segar         → rotten_score < 0.50  (Layak Jual)
Hampir Busuk  → 0.50 ≤ rotten_score < 0.65  (Segera Jual/Gunakan)
Busuk         → rotten_score ≥ 0.65  (Tidak Layak Jual)
```

### Link Berguna

- Dataset Kaggle: <https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification>
- Google Colab: <https://colab.research.google.com>
- Streamlit Docs: <https://docs.streamlit.io>
- DJKI HKI: <https://djki.kemenkumham.go.id>
- e-HKI Portal: <https://e-hakcipta.dgip.go.id>

---

*Update checklist ini setiap kali menyelesaikan tugas dengan mengganti `[ ]` menjadi `[x]`*
