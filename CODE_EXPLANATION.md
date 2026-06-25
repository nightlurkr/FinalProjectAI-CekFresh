# 📖 Penjelasan Code — VigorScan

> Dokumentasi teknis source code untuk Final Project AI/ML Kelompok 8.
> Dokumen ini menjelaskan struktur, alur, dan keputusan desain pada setiap komponen utama proyek.

---

## 🗂️ Struktur Source Code

| Path | Deskripsi |
|------|-----------|
| `02_Model/training/fp_ai_kel8_final.ipynb` | Notebook training di Google Colab |
| `03_App/app.py` | Aplikasi Streamlit (entry point) |
| `03_App/vigorscan_model.keras` | Model EfficientNetV2-S hasil training |
| `03_App/class_names.json` | Mapping index ke nama kelas (6 kelas multiclass) |
| `03_App/requirements.txt` | Daftar dependensi Python |

---

## 🧠 PART 1 — Notebook Training (`fp_ai_kel8_final.ipynb`)

Notebook ini menjalankan pipeline end-to-end dari dataset preparation sampai model serialization.

### Cell 1–10: Setup Environment

- Mount Google Drive untuk persistent storage
- Install dependencies (TensorFlow, scikit-learn, kagglehub)
- Set random seeds untuk reproducibility

### Cell 11–20: Dataset Download & Organization

Tiga sumber dataset di-download via `kagglehub`:

```python
sriramr_path     = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")
raghav_path      = kagglehub.dataset_download("raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables")
fruits360_path   = kagglehub.dataset_download("moltean/fruits")
```

Lalu dataset di-organize ke struktur multiclass:

```text
/content/datasets/organized/
├── train/
│   ├── freshbanana/
│   ├── rottenbanana/
│   ├── freshorange/
│   ├── rottenorange/
│   ├── freshtomato/   ← dari fruits-360 (lebih bersih)
│   └── rottentomato/  ← dari raghavrpotdar (stale_tomato)
├── val/    (sama struktur)
└── test/   (sama struktur)
```

### Cell 21 — RESET Cell ⚠️ (CRITICAL)

```python
# WAJIB DIJALANKAN sebelum split ulang. Tanpa reset, data leakage bisa terjadi
# karena cell split akan menambah ke folder yang sudah ada → val/test bocor ke train.
if os.path.exists(base_output):
    shutil.rmtree(base_output)
```

**Kenapa penting?** Tanpa reset, multiple runs cell split akan duplikasi data dan menyebabkan `val_accuracy` mencapai 99%+ tapi gagal di data nyata (data leakage).

### Cell 22–32: Data Generators

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
)
```

> **Important:** `efficientnet_v2.preprocess_input` adalah **no-op** untuk EfficientNetV2. Model punya `Rescaling` layer internal. Ini sengaja dibiarkan supaya kode portable kalau model di-swap.

### Cell 33–44: Model Architecture

```python
base_model = tf.keras.applications.EfficientNetV2S(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Phase 1: freeze

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(6, activation='softmax')  # 6 kelas multiclass
])
```

### Cell 45 — Phase 1 Training (Frozen Base)

```python
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)
```

**Kenapa `class_weight`?** Dataset imbalanced (Pisang ada 3.805 vs Tomat 1.963). Tanpa weighting, model akan bias ke pisang.

### Cell 46–47 — Phase 2 Fine-tuning

```python
base_model.trainable = True
# Hanya unfreeze 50 layer terakhir EfficientNetV2-S
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)
```

Learning rate diturunkan ke `1e-5` (dari `1e-3`) untuk fine-tuning supaya pretrained weights tidak rusak.

### Cell 48–50 — Evaluasi & Serialization

```python
# Save model + class names
SAVE_DIR = '/content/drive/MyDrive/CekFresh_Model'
model.save(f'{SAVE_DIR}/vigorscan_model.keras')

class_names = {v: k for k, v in train_generator.class_indices.items()}
with open(f'{SAVE_DIR}/class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)

# Test set evaluation
test_loss, test_acc = model.evaluate(test_generator)
# Test Accuracy: ~99.70%
```

---

## PART 2 — Aplikasi Streamlit (`03_App/app.py`)

Aplikasi 600 baris, single-file. Berikut walkthrough fungsi-fungsi utamanya.

### Section 1: Konfigurasi & State (line 1–36)

```python
st.set_page_config(page_title="VigorScan...", layout="wide", initial_sidebar_state="collapsed")

# Path resolve relatif ke lokasi script — penting untuk Streamlit Cloud
# yang CWD-nya = repo root, BUKAN folder app.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_NEW  = os.path.join(_SCRIPT_DIR, "vigorscan_model.keras")
_MODEL_OLD  = os.path.join(_SCRIPT_DIR, "cekfresh_model.keras")
MODEL_PATH       = _MODEL_NEW if os.path.exists(_MODEL_NEW) else _MODEL_OLD
CLASS_NAMES_PATH = os.path.join(_SCRIPT_DIR, "class_names.json")
```

`MODEL_PATH` punya fallback ke nama lama untuk backward compat. Path resolve via `__file__` (bukan CWD) supaya kompatibel di Streamlit Cloud yang run script dari root repo.

Session state untuk persistensi data lintas-rerun:

```python
st.session_state.upload_results  = []    # Hasil deteksi upload (list of dict)
st.session_state.camera_result   = None  # Hasil deteksi kamera (single dict)
st.session_state.dark_mode       = True  # Default dark mode
st.session_state.uploader_nonce  = 0     # Key rotation untuk reset file_uploader
st.session_state.camera_nonce    = 0     # Key rotation untuk reset camera_input
```

**Nonce pattern:** file_uploader / camera_input pakai `key=f"uploader_main_{nonce}"`. Saat user klik tombol "🗑️ Bersihkan", nonce di-increment → Streamlit treat widget sebagai instance baru → foto yang sudah di-upload juga ter-clear.

### Section 2: `apply_theme(dark)` (line 29–243)

Single function yang menginject ~200 baris CSS via `st.markdown(..., unsafe_allow_html=True)`. Pakai f-string substitution untuk swap variable warna antara dark/light mode tanpa duplikasi CSS.

Key features:

- **Glassmorphism cards** (`backdrop-filter: blur(24px)`, semi-transparent bg)
- **Gradient hero title** dengan animasi shimmer
- **Material Symbols font fix** — global `*` selector di-scope ke `body, p, h1..h6, ...` saja supaya icon Streamlit (expander arrow, dll) tetap render dengan font Material Icons
- **Pill-style tabs** dengan gradient bg saat active
- **Custom colored progress bars** via `.vs-bar-track` + `.vs-bar-fill.fresh` / `.rotten`

### Section 3: `load_model()` & `load_class_names()` (line 245–256)

```python
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
```

`@st.cache_resource` (untuk model) dan `@st.cache_data` (untuk dict) supaya tidak reload tiap rerun. Penting karena rerun terjadi setiap interaksi UI (toggle, button, dll).

### Section 4: `preprocess_image(img)` (line 257–260)

```python
def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)  # range [0, 255], NO normalization
    return np.expand_dims(arr, axis=0)
```

**JANGAN normalisasi** ke [0,1] atau [-1,1]. EfficientNetV2-S punya internal Rescaling layer; normalisasi manual akan menyebabkan double-rescaling.

### Section 5: `color_signature(img)` — HSV Analysis dengan Skin Filter

Fungsi analisis warna HSV untuk mendukung color rescue. Konversi RGB → HSV secara manual dengan numpy (lebih cepat dari loop colorsys).

```python
def color_signature(img):
    arr = np.array(img.convert("RGB").resize((140, 140))).astype(np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # ... hitung Hue, Saturation, Value ...

    # Step 1: pixel "colorful" yang berada di range warna buah
    colorful = (sat > 0.40) & (val > 0.30)
    fruit_mask = colorful & ((hue < 70) | (hue > 340))

    # Step 2: SKIN TONE FILTER — exclude pixel telapak tangan / kulit
    # Telapak tangan/wajah punya hue mirip oranye-merah tapi saturasi sedang.
    skin_mask = (
        (hue >= 5) & (hue < 40)
        & (sat >= 0.15) & (sat < 0.50)
        & (val >= 0.45) & (val < 0.85)
    )
    fruit_mask = fruit_mask & (~skin_mask)
    fruit_n = int(fruit_mask.sum())

    return {
        "red_ratio":        red_pixels / fruit_n,
        "orange_ratio":     orange_pixels / fruit_n,
        "bright_red_ratio": (red & high_sat & high_val) / fruit_n,
        "dark_red_ratio":   (red & low_val) / fruit_n,
        "fruit_n":          fruit_n,
    }
```

**Key insights:**

1. **Rasio dihitung di antara pixel berwarna buah saja**, bukan total pixel gambar. Pixel hijau (daun) tidak masuk hitungan supaya tidak mendilusi.
2. **Skin tone filter** mencegah foto buah dipegang tangan dari mendapat color rescue yang salah. Tanpa filter ini, palm pixels akan masuk hitungan `orange_ratio` (karena hue skin ~20-30°) dan bisa trigger override salah.

### Section 5b: `banana_decay_signal(img)` — White Mold Detector

Untuk handle gap training data — model tidak terlatih pada pisang dengan **white mold (jamur putih)**, hanya brown spots. Heuristic ini deteksi area pisang dengan white mold:

```python
def banana_decay_signal(img):
    # ... hitung HSV ...
    # Yellow pisang area (hue 40-70°, saturasi cukup)
    yellow_n = (sat > 0.40) & (val > 0.45) & (hue >= 40) & (hue < 70)

    # WHITE MOLD ONLY: saturasi sangat rendah, value sedang (bukan pure white BG)
    white_mold = (sat < 0.12) & (val > 0.60) & (val < 0.85)
    decay_n = white_mold.sum()

    # Safety: kalau decay >> yellow, kemungkinan BG bukan mold real
    if decay_n > yellow_n * 1.5:
        return {"decay_ratio": 0.0}

    return {"decay_ratio": decay_n / (yellow_n + decay_n)}
```

**Dark rot signal SENGAJA TIDAK dipakai** — terlalu sering false-positive pada background gelap (meja kayu/dinding). Trade-off: kehilangan deteksi pure-black banana rot, tapi model EfficientNetV2-S sudah handle brown spots dengan baik (training data utama).

### Section 6: `predict_freshness(img, model, class_names)` — Pipeline Lengkap

Pipeline 5 tahap dengan multiple defensive heuristics:

**Tahap A — Model Inference:**

```python
arr = preprocess_image(img)
probs = model.predict(arr, verbose=0)[0]  # shape (6,)
```

**Tahap B — Deteksi Jenis Buah (aggregate per fruit):**

```python
fruit_scores = {}
for fk in ["banana", "orange", "tomato"]:
    fruit_scores[fk] = p_fresh + p_rotten  # score per fruit type
detected = max(fruit_scores, key=fruit_scores.get)
```

**Tahap C — Color Rescue (targeted, hanya orange↔tomato):**

```python
sig = color_signature(img)  # sudah dengan skin tone filter
if detected == "orange":
    # Strict thresholds: red > 0.55 AND red > 1.8x orange
    # supaya skin tone tidak mendilusi rasio dan trigger override salah
    if sig["fruit_n"] >= 200 and sig["red_ratio"] > 0.55 \
       and sig["red_ratio"] > sig["orange_ratio"] * 1.8:
        detected = "tomato"
        color_override = "red_to_tomato"
        # Freshness via color brightness
        if sig["bright_red_ratio"] > 0.25 and sig["dark_red_ratio"] < 0.15:
            rotten_score = 0.12  # confident fresh
        elif sig["dark_red_ratio"] > 0.30:
            rotten_score = 0.85  # confident rotten
```

**Tahap D — Banana Decay Rescue:**

```python
if detected == "banana":
    decay = banana_decay_signal(img)
    if decay["decay_ratio"] > 0.25:
        rotten_score = max(rotten_score, 0.75)  # white mold → busuk
    elif decay["decay_ratio"] > 0.15:
        rotten_score = max(rotten_score, 0.55)  # → hampir busuk
```

**Tahap E — Reject Fallback (non-fruit detection):**

```python
# Multi-signal: low confidence OR no fruit color OR ambiguous top1↔top2
sorted_scores = sorted(fruit_scores.values(), reverse=True)
top1, top2 = sorted_scores[0], sorted_scores[1]
margin = top1 - top2

reject_signals = []
if top1 < 0.40:                          reject_signals.append("low_confidence")
if sig["fruit_n"] < 150:                 reject_signals.append("no_fruit_color")
if margin < 0.10 and top1 < 0.55:        reject_signals.append("ambiguous_top")

if (not color_override) and reject_signals:
    # Return special sentinel: fruit_key="unknown", "Tidak Terdeteksi"
    return (...)
```

**Tahap F — Rotten Score Normalisasi:**

```python
rotten_score = p_rotten / (p_fresh + p_rotten)
```

Rasio relatif, bukan probabilitas absolut, supaya threshold konsisten.

Return tuple **11 elemen** (terakhir = `fruit_key`):

```python
return (emoji, label, f_label, f_color, badge, rek, tips, rotten_score, probs_dict, conf, fruit_key)
```

`fruit_key` diperlukan render_result untuk apply jeruk visual bar calibration & detect "unknown" untuk reject UI.

### Section 7: `rotten_score_to_label(score, label)` (line 370–386)

Threshold tunggal untuk semua buah:

```python
if score < 0.50:        return "Segar"          # LAYAK JUAL
elif score < 0.65:      return "Hampir Busuk"   # SEGERA JUAL
else:                   return "Busuk"          # TIDAK LAYAK
```

Returns tuple: (label, color, badge, rekomendasi, tips).

### Section 8: `compute_predictions(files)` & `render_result(...)`

Pisahan **compute** vs **render** supaya hasil bisa disimpan di session state dan tetap muncul saat user toggle theme (yang trigger `st.rerun()`).

```python
def compute_predictions(files):
    """Hitung sekali, simpan hasil + JPEG bytes ke session_state."""
    for uf in files:
        img = Image.open(uf).convert("RGB")
        pred = predict_freshness(img, model, class_names)
        results.append({
            "name": uf.name,
            "img_bytes": img_to_bytes(img),
            "pred": pred,
        })
    return results

def render_result(img_bytes, filename, pred, show_image=True):
    """Render dari precomputed pred — idempotent, aman dipanggil ulang saat rerun."""
    (emoji, label, f_label, f_color, badge, rek, tips,
     rotten, probs_dict, conf, fruit_key) = pred

    # Branch A: reject UI (foto bukan buah/sayur target)
    if fruit_key == "unknown":
        # Render different UI: ❓ icon, "Tidak Terdeteksi" badge, no bars
        return

    # Branch B: visual calibration KHUSUS jeruk
    # Jeruk lokal (hijau-oren) sering dapat rotten_score ~0.5 dari model.
    # Visual 50/50 unconvincing meski label "Segar" benar.
    # Remap visual saja, label & threshold klasifikasi tetap original.
    if fruit_key == "orange" and rotten < 0.50:
        rotten_display = rotten * 0.80  # [0, 0.50] → [0, 0.40]
    else:
        rotten_display = rotten
    fresh_display = 1.0 - rotten_display

    # Render bars pakai *_display, label/badge/tips pakai *original* rotten
```

### Section 9: UI Layout

```python
apply_theme(st.session_state.dark_mode)

# Theme toggle pojok kanan atas — pakai st.toggle (native BaseWeb switch)
# bukan st.checkbox, supaya render sebagai pill switch yang lebih clean
col_spacer, col_toggle = st.columns([8, 1.2])
with col_toggle:
    toggled = st.toggle(
        "🌙 Dark" if is_dark else "☀️ Light",
        value=is_dark, key="theme_cb"
    )

# Hero card glassmorphism dengan gradient title shimmer
st.markdown(<hero>, unsafe_allow_html=True)

# Tabs pill-style
tab_upload, tab_camera = st.tabs(["📁 Upload Foto", "📷 Kamera Langsung"])

with tab_upload:
    # Key dengan nonce — increment saat "Bersihkan" diklik supaya widget reset
    files = st.file_uploader(
        ..., accept_multiple_files=True,
        key=f"uploader_main_{st.session_state.uploader_nonce}"
    )
    if files and st.button("Deteksi Sekarang"):
        st.session_state.upload_results = compute_predictions(files)

    if st.session_state.upload_results:
        if st.button("🗑️ Bersihkan"):
            st.session_state.upload_results = []
            st.session_state.uploader_nonce += 1   # rotate key → reset widget
            st.rerun()

    # Render dari session state — tetap muncul saat rerun (toggle theme, dll)
    for r in st.session_state.upload_results:
        render_result(r["img_bytes"], r["name"], r["pred"])
```

---

## 🎯 PART 3 — Key Design Decisions

### 3.1 Kenapa EfficientNetV2-S, bukan MobileNetV2?

| | MobileNetV2 | EfficientNetV2-S |
|---|---|---|
| Parameter | 3.5M | 21M |
| Compound scaling | ❌ | ✅ (depth/width/resolution) |
| Robust pada class imbalance | Buruk | Lebih baik |

Awal proyek pakai MobileNetV2, hasil val_acc 95% tapi tomat dengan visual ambigu sering misclassified. Upgrade ke EfficientNetV2-S + `class_weight='balanced'` → test_acc 99.70%.

### 3.2 Preprocessing tanpa normalisasi

EfficientNetV2-S punya `tf.keras.layers.Rescaling(scale=1./255)` (atau equivalent) sebagai layer pertama. Memanggil `efficientnet_v2.preprocess_input()` adalah no-op. Normalisasi manual `(arr - 127.5) / 127.5` akan menyebabkan input out-of-distribution.

**Bug history:** sempat ada normalisasi `(arr - 127.5) / 127.5` di `app.py` yang menyebabkan semua prediksi jadi rotten. Fixed dengan menghapus normalisasi.

### 3.3 Threshold 50% / 65%, bukan 30% / 65%

Threshold awal 30% terlalu agresif — banyak jeruk lokal (kulit hijau-oren) dianggap "Hampir Busuk". Setelah testing di berbagai foto, threshold 50% memberikan balance terbaik antara false-positive busuk vs missed-busuk.

### 3.4 Color Rescue: kenapa hanya orange↔tomato?

Dalam HSV color space:

- Red hue: 0–18°
- Orange hue: 18–45°
- Yellow hue: 45–70°

Red dan Orange berdekatan, sehingga model rentan keliru. Yellow berjarak cukup jauh, jadi banana classification umumnya stabil. Color rescue dibatasi pada konflik orange↔tomato karena:

1. Itu mode kegagalan paling sering
2. Pisang busuk punya brown patches (low-saturation orange) yang bisa salah trigger override

**Banana branch dibiarkan murni model** sebagai safety measure.

### 3.5 Color brightness sebagai freshness signal (saat override)

Saat color rescue mengubah `detected` dari orange ke tomato, probabilitas `freshtomato` di model tidak reliable (model awalnya tidak pilih tomato). Solusi: derive freshness dari color signature:

- Bright red (sat > 0.55, val > 0.45) → fresh (rotten_score = 0.12)
- Dark red (val < 0.35) → rotten (rotten_score = 0.85)

Ini scientifically defensible — tomat segar memang merah cerah, tomat busuk warnanya kusam/kecoklatan.

### 3.6 Session State Persistence

Streamlit men-trigger full script rerun pada setiap interaksi. Tanpa caching, toggle dark/light mode akan menghilangkan hasil deteksi (user harus upload ulang).

Solusi: pisahkan compute (`compute_predictions`) dari render (`render_result`). Hasil disimpan sebagai serializable dict (JPEG bytes + pred tuple) di `st.session_state`. Render tinggal loop dari session state setiap rerun.

### 3.7 Reject Fallback (Out-of-Distribution Detection)

Model softmax tidak punya konsep "tidak tahu" — selalu pick salah satu dari 6 kelas. Foto orang/mobil/objek random tetap akan diklasifikasi sebagai pisang/jeruk/tomat dengan rotten score acak.

Solusi: multi-signal reject check sebelum return prediksi:

1. **Low confidence** (top1 < 0.40) — model unsure
2. **No fruit color** (`fruit_n < 150` di color_signature) — image tidak ada pixel warna buah dominan
3. **Ambiguous top** (margin top1↔top2 < 0.10 AND top1 < 0.55) — model bingung antara 2 kelas

Trigger ANY → return special "unknown" tuple → render UI khusus "Tidak Terdeteksi".

### 3.8 Banana White Mold Detector

Training data pisang busuk (sriramr) hanya cover brown spots, tidak ada fungal/white mold. Untuk gap ini, dedicated heuristic deteksi white mold via color signature:

- **White mold signature:** sat < 0.12, val 0.60–0.85 (bukan pure white BG, bukan kayu)
- **Threshold:** decay > 25% area pisang → boost rotten ke 0.75 (Busuk)
- **Safety:** kalau decay >> yellow (1.5×), assume background pollution → no boost

Dark rot signal sengaja DIHAPUS karena terlalu sering false-positive di background gelap.

### 3.9 Skin Tone Filter

Pixel telapak tangan/kulit punya hue 5–40° (mirip orange-red) dengan saturasi sedang (0.15–0.50). Tanpa filter, foto buah dipegang tangan akan trigger color rescue override yang salah.

Filter ini exclude skin range dari `fruit_mask` di `color_signature()`. Test verifikasi: tomat di telapak tangan → red_ratio = 1.00 (skin filtered, hanya pixel tomat yang dihitung).

### 3.10 Jeruk Visual Bar Calibration

Model sering memberi `rotten_score ~0.49` untuk jeruk lokal Indonesia (kulit hijau-oren). Label "Segar" benar (di bawah threshold 0.50), tapi bar visual 50.6/49.4 unconvincing.

Solusi: visual-only remap untuk fruit_key="orange" dengan `rotten < 0.50`:

```python
rotten_display = rotten * 0.80  # [0, 0.50] → [0, 0.40]
```

Label, badge, threshold klasifikasi TIDAK berubah — hanya tampilan bar yang dikalibrasi. Academic integrity preserved.

---

## 🧪 PART 4 — Testing & Validation

### Test Cases yang Sudah Diverifikasi

| Foto | Expected | Hasil | Heuristic yang Aktif |
|------|----------|--------|---------------------|
| Pisang segar tunggal | Pisang Segar | ✅ | model only |
| Pisang busuk (brown spots) | Pisang Busuk | ✅ | model only |
| Pisang dengan **white mold** | Pisang Busuk | ✅ | banana decay detector |
| Pisang segar di latar **kayu gelap** | Pisang Segar | ✅ | safety check (decay > yellow×1.5) |
| Jeruk Western (pure orange) | Jeruk Segar | ✅ | model only |
| Jeruk keprok Indonesia (oren+hijau) | Jeruk Segar | ✅ | jeruk visual calibration |
| Jeruk **di telapak tangan** | Jeruk Segar | ✅ | skin tone filter |
| Tomat single close-up segar | Tomat Segar | ✅ | model only |
| Tomat busuk (gelap) | Tomat Busuk | ✅ | model only |
| **Tomat di tangkai + daun lebat** | Tomat Segar | ✅ | color rescue (orange→tomato) |
| Tomat **di telapak tangan** | Tomat Segar | ✅ | skin tone filter |
| Foto random (orang/mobil/objek) | Tidak Terdeteksi | ✅ | reject fallback |

### Edge Cases & Limitations

- Foto dengan background sangat bervariasi (multi-color, lampu, dll) mungkin menurunkan akurasi
- Model bekerja optimal dengan **buah di background netral**; foto handheld kemungkinan kurang akurat karena training data tidak ada konteks ini (mitigasi: skin tone filter, tapi tidak fix masalah OOD model)
- Color rescue hanya menangani konflik orange↔tomato; banana classification tidak disentuh
- Banana decay detector hanya cover white mold, bukan dark rot (trade-off untuk safety vs background pollution)
- Reject fallback bisa false-reject foto buah valid dengan confidence rendah (jarang, biasanya foto buram/over-exposed)

### Future Work

- Retraining dengan dataset Indonesia (jeruk keprok, tomat tangkai, pisang varietas lokal, buah dipegang tangan)
- Tambah class "other" supaya model bisa native reject foto non-buah
- Quantization model untuk deployment mobile / Edge device
- Data augmentation: random skin-tone overlay supaya model robust pada hand-held context

---

## 🔧 PART 5 — Cara Reproduce / Debug

### Re-train Model dari Awal

1. Buka `02_Model/training/fp_ai_kel8_final.ipynb` di Google Colab
2. Set runtime ke GPU T4
3. Run semua cell secara berurutan (jangan skip RESET cell sebelum split!)
4. Setelah selesai, model & class_names akan tersimpan di `/content/drive/MyDrive/CekFresh_Model/`
5. Download dan ganti file di `03_App/`

### Debug Prediksi Salah

Buka expander **"Detail probabilitas semua kelas"** di hasil deteksi. Akan tampil probabilitas mentah model untuk 6 kelas. Dari sini bisa lihat:

- Apakah model confident pada kelas yang salah, atau confusion antara 2 kelas
- Apakah `top1 < 0.40` → reject fallback hampir trigger
- Apakah jeruk vs tomat probability dekat → color rescue mungkin perlu kalibrasi

Untuk debugging lebih dalam, tambahkan logging di `predict_freshness`:

```python
print(f"DEBUG: fruit_scores  = {fruit_scores}")
print(f"DEBUG: color_sig     = {sig}")
print(f"DEBUG: color_override= {color_override}")
print(f"DEBUG: banana_decay  = {banana_decay_signal(img)}")  # untuk pisang
print(f"DEBUG: reject_signals= {reject_signals}")
```

Setiap heuristic punya signal dan threshold yang trace-able — penting untuk Q&A "kenapa model salah classify foto X".

---

## 📚 PART 6 — Glossary

| Term | Definisi |
|------|----------|
| Transfer Learning | Reuse model pretrained pada ImageNet, swap classifier head untuk task baru |
| Fine-tuning | Unfreeze sebagian layer base model + train dengan learning rate kecil |
| Class weight | Bobot loss per kelas untuk handle imbalance |
| Softmax | Activation function yang menormalisasi output ke probabilitas (sum = 1) |
| HSV | Color space: Hue (warna 0–360°), Saturation (kepekatan), Value (kecerahan) |
| Distribution Shift | Perbedaan distribusi data train vs production data |
| Test-time augmentation | Augmentasi gambar input saat inference untuk meningkatkan robustness |

---

*Dokumen ini disusun untuk Final Project AI/ML Kelompok 8 — VigorScan*
