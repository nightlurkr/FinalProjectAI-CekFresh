# 🎯 Demo Pre-Flight Checklist — VigorScan

> **Demo:** Senin, 20 Juni 2026 — 16:00 WIB via Discord
> **Aturan kunci dari asisten:** "Pastikan projek sudah siap sebelum demo dimulai (jangan run saat demo ya)"

---

## ✅ 1. Environment Cold Start (lakukan H-1)

Tutup semua terminal/streamlit yang masih jalan, lalu buka terminal baru:

```cmd
cd "C:\Users\Ryan\Documents\final project AI\CekFresh\03_App"
python -m streamlit run app.py
```

**Yang harus terjadi:**

- [ ] Terminal print `Local URL: http://localhost:8501` tanpa error
- [ ] Browser otomatis buka tab baru ke `localhost:8501`
- [ ] Hero `🍃 VigorScan` muncul dengan gradient title + animated dot
- [ ] Theme toggle (🌙 Dark / ☀️ Light) di pojok kanan atas berfungsi
- [ ] Hasil scan sebelumnya hilang (cache fresh)

**Jika error muncul:**

| Error | Solusi |
|-------|--------|
| `protobuf incompatible versions` | `pip install --upgrade "protobuf>=6.31.1"` |
| `Model file not found` | Pastikan `vigorscan_model.keras` ada di `03_App/` |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Port 8501 sudah dipakai | `python -m streamlit run app.py --server.port 8502` |

---

## ✅ 2. Test Stress dengan Foto Demo (lakukan H-1)

Upload satu-per-satu, dokumentasikan hasil di kolom **Actual**:

### Tab Upload

| # | Foto | Expected | Actual |
|---|------|----------|--------|
| 1 | Pisang segar tunggal | 🍌 Pisang **Segar** | _____ |
| 2 | Pisang busuk (brown spots) | 🍌 Pisang **Busuk** | _____ |
| 3 | Jeruk Western (oren penuh) | 🍊 Jeruk **Segar** | _____ |
| 4 | Jeruk keprok / siam Indonesia | 🍊 Jeruk **Segar** | _____ |
| 5 | Jeruk busuk (berjamur/keriput) | 🍊 Jeruk **Busuk** | _____ |
| 6 | Tomat single close-up segar | 🍅 Tomat **Segar** | _____ |
| 7 | Tomat busuk (gelap/lembek) | 🍅 Tomat **Busuk** | _____ |
| 8 | **Tomat di tangkai + daun lebat** (foto wajib) | 🍅 Tomat **Segar** | _____ |

### Test Multi-Upload

- [ ] Upload 3-4 foto sekaligus → semua di-detect dengan benar
- [ ] Tombol **"🗑️ Bersihkan"** hapus semua hasil
- [ ] Toggle dark/light → hasil deteksi TIDAK hilang

### Tab Kamera

- [ ] Klik **"📷 Kamera Langsung"** → izinkan akses kamera
- [ ] Arahkan ke buah, ambil foto → tombol Deteksi muncul
- [ ] Klik Deteksi → hasil muncul dengan benar

---

## ✅ 3. Pre-Demo Setup (30 menit sebelum)

- [ ] Tutup aplikasi lain yang berat (Discord call sudah aktif)
- [ ] Charge laptop, pastikan power adapter terhubung
- [ ] Buka 2 window berdampingan:
  - **Browser** dengan `localhost:8501` sudah load
  - **VS Code** dengan `app.py` di tab terpisah (untuk tunjuk code saat ditanya)
- [ ] Buka file `DEMO_TALKING_POINTS.md` (kalau sudah dibuat) di tab Notepad/Notes
- [ ] Folder foto demo dibuka di File Explorer, siap drag-drop
- [ ] Mute notifikasi Windows (Focus Assist on)
- [ ] Test mic + speaker via Discord voice test channel

---

## ✅ 4. Live Demo Flow (5–10 menit)

### Opening (30 detik)

> "VigorScan adalah aplikasi deteksi kesegaran buah berbasis AI. Target user: pedagang & distributor untuk mengurangi food waste. Mendukung 3 buah utama: pisang, jeruk, tomat."

### Demo Inti (3–5 menit)

1. **Tunjuk Hero & Theme Toggle** → klik dark/light untuk show smooth transition
2. **Upload Foto Tomat di Tangkai** → tunggu hasil → **Tomat Segar 100%**
3. **Buka expander "Detail probabilitas semua kelas"** → tunjukkan raw model output
4. **Upload Multi-foto** (3 buah berbeda) → batch processing
5. **Toggle theme saat hasil sudah tampil** → tunjukkan state persistence
6. **Switch ke Tab Kamera** (kalau waktu cukup)

### Closing (1 menit)

> "Untuk model, kami pakai EfficientNetV2-S dengan transfer learning + class weight balanced. Test accuracy 99.7%. Plus targeted color rescue untuk handle dataset distribution shift pada foto buah lokal Indonesia."

---

## ✅ 5. Q&A Cheat Sheet

### "Kenapa pakai EfficientNetV2-S, bukan MobileNet?"

> Awalnya MobileNetV2 tapi val_acc plateau di 95% karena class imbalance. Upgrade ke EfficientNetV2-S yang punya compound scaling + class_weight balanced → 99.7%.

### "Bagaimana handle foto buah lokal yang beda sama training data?"

> Kami implementasi **targeted color rescue** post-processing. Pakai HSV analysis untuk verifikasi hue dominan di antara pixel berwarna buah (daun hijau di-exclude). Khusus untuk konflik orange↔tomato karena hue red dan orange berdekatan. Pisang tidak disentuh.

### "Threshold-nya kenapa 50% dan 65%?"

> Empirical. Threshold 30% awalnya bikin banyak false-positive "Hampir Busuk" pada jeruk lokal yang naturally agak hijau. Setelah testing di berbagai foto, 50/65 paling balance.

### "Preprocessing-nya pakai apa?"

> Resize 224×224, tidak ada normalisasi manual. EfficientNetV2-S punya Rescaling layer internal. Normalisasi manual akan menyebabkan double-rescaling. Ini ada di komentar code di `preprocess_image()`.

### "Kalau salah deteksi, bagaimana cara debug?"

> Buka expander "Detail probabilitas semua kelas" di hasil deteksi. Akan tampil probabilitas raw untuk 6 kelas. Bisa dilihat apakah model confused antara dua kelas atau confident tapi salah.

### "Bisa deteksi buah lain selain 3 itu?"

> Tidak. Model dilatih hanya untuk 3 kelas. Foto buah lain akan dipaksa diklasifikasi ke salah satu dari 3 (no "unknown" class). Untuk future work, bisa ditambah class "other" atau threshold confidence untuk reject.

### "Dataset dari mana?"

> Tiga sumber Kaggle: sriramr (Pisang/Jeruk), raghavrpotdar (Tomat busuk + tambahan), fruits-360 (Tomat segar). Total 10.531 gambar, split 70/15/15.

### "Kenapa pakai color rescue, bukan retrain saja?"

> Sebenarnya retrain dengan dataset lokal Indonesia lebih ideal. Tapi waktu terbatas dan color rescue cukup robust untuk fix dataset distribution shift pada konflik orange↔tomato spesifik. Untuk production, kami plan retrain dengan dataset Indonesia.

### "Bisa dipakai untuk real-time video?"

> Saat ini hanya foto single (upload atau snapshot kamera). Untuk real-time, model 21M parameter perlu quantization / TFLite conversion untuk inference cepat di mobile.

---

## ✅ 6. Fallback Plan

**Kalau internet putus di tengah demo:**

- Aplikasi 100% offline (model di-load lokal)
- Hanya Discord yang butuh internet — switch ke phone hotspot

**Kalau Streamlit crash:**

- Refresh browser dulu (`F5`)
- Kalau persist: matikan terminal (`Ctrl+C`), run ulang `streamlit run app.py`
- Hasil deteksi sebelumnya akan hilang (session state reset), tapi app jalan lagi

**Kalau salah satu foto miss-detect:**

- Tetap tenang, buka expander Detail untuk show raw probabilities
- Akui ini limitation: "Foto ini OOD dari training distribution, model confused antara X dan Y"
- Tunjukkan foto lain yang work untuk recovery

**Kalau ditanya hal yang tidak dijawab di cheat sheet:**

- Tarik nafas, ulangi pertanyaan untuk konfirmasi
- Akui kalau tidak tahu: "Kami belum eksplor itu, tapi pendekatan yang akan kami coba adalah..."
- Lebih baik jujur tidak tahu daripada bullshit

---

## 📝 Post-Demo

Setelah demo selesai:

- [ ] Catat feedback / pertanyaan dosen di file terpisah
- [ ] Update CHECKLIST.md status demo: ✅ Selesai
- [ ] Push update terakhir ke GitHub (jika ada)
- [ ] Bagi tugas final submission deadline 26 Juni

---

*Semoga lancar! 🚀*
