import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CekFresh — Freshness Detector",
    page_icon="🍃",
    layout="centered"
)

# ─────────────────────────────────────────────
# PATH MODEL
# ─────────────────────────────────────────────
MODEL_PATH      = "cekfresh_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            data = json.load(f)
        # class_names.json: {"0": "fresh", "1": "rotten"}
        return {int(k): v for k, v in data.items()}
    return {0: "fresh", 1: "rotten"}

# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING GAMBAR
# ─────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)

# ─────────────────────────────────────────────
# FUNGSI KONVERSI CONFIDENCE → 3 LABEL
# Model binary: sigmoid output
#   0.0 → sangat fresh
#   1.0 → sangat rotten
# Threshold:
#   < 0.30  → Segar
#   0.30–0.65 → Hampir Busuk
#   > 0.65  → Busuk
# ─────────────────────────────────────────────
def get_label_and_recommendation(prob_rotten: float):
    if prob_rotten < 0.30:
        label       = "🟢 Segar"
        color       = "green"
        badge       = "✅ LAYAK JUAL"
        rekomendasi = (
            "Produk dalam kondisi prima dan layak jual. "
            "Simpan di tempat sejuk atau kulkas untuk menjaga kesegaran lebih lama."
        )
        tips = "💡 Tip: Simpan terpisah dari buah yang sudah matang untuk menghindari pematangan dini."

    elif prob_rotten < 0.65:
        label       = "🟡 Hampir Busuk"
        color       = "orange"
        badge       = "⚠️ SEGERA JUAL / GUNAKAN"
        rekomendasi = (
            "Produk mulai menunjukkan tanda-tanda penurunan kualitas. "
            "Segera jual atau gunakan dalam waktu dekat. "
            "Pertimbangkan memberikan diskon untuk mempercepat penjualan."
        )
        tips = "💡 Tip: Pisahkan dari produk segar agar tidak mempercepat pembusukan produk lainnya."

    else:
        label       = "🔴 Busuk"
        color       = "red"
        badge       = "❌ TIDAK LAYAK JUAL"
        rekomendasi = (
            "Produk sudah tidak layak untuk dijual atau dikonsumsi. "
            "Pisahkan segera dari produk lainnya untuk mencegah kontaminasi."
        )
        tips = "💡 Tip: Manfaatkan sebagai kompos atau pupuk organik untuk mengurangi food waste."

    return label, color, badge, rekomendasi, tips

# ─────────────────────────────────────────────
# TAMPILAN UTAMA
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#16a34a;'>🍃 CekFresh</h1>
<p style='text-align:center; color:#555; font-size:16px;'>
    Freshness Detector & Food Waste Solution<br>
    <small>Deteksi kelayakan jual buah & sayur dari foto menggunakan AI</small>
</p>
<hr style='margin: 0.5rem 0 1.5rem 0;'>
""", unsafe_allow_html=True)

# Kolom info produk yang didukung
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🍌 Pisang**\nFresh & Rotten")
with col2:
    st.markdown("**🍊 Jeruk**\nFresh & Rotten")
with col3:
    st.markdown("**🍅 Tomat**\nFresh & Rotten")

st.markdown("---")

# ─────────────────────────────────────────────
# UPLOAD GAMBAR
# ─────────────────────────────────────────────
st.subheader("📷 Upload Foto Produk")
uploaded_file = st.file_uploader(
    "Pilih gambar buah atau sayur (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Upload satu foto produk yang ingin diperiksa kesegarannya"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Tampilkan gambar yang diupload
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Foto yang diupload", use_container_width=True)

    with col_info:
        st.markdown("**📋 Info Gambar**")
        st.write(f"• Nama file : `{uploaded_file.name}`")
        st.write(f"• Ukuran    : {image.size[0]} × {image.size[1]} px")
        st.write(f"• Format    : {image.format or 'JPEG/PNG'}")
        st.markdown("---")
        st.info("Klik **Deteksi Sekarang** untuk menganalisis kondisi produk.", icon="👇")

    st.markdown("")

    # ─────────────────────────────────────────────
    # TOMBOL DETEKSI
    # ─────────────────────────────────────────────
    if st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True):
        with st.spinner("Menganalisis gambar dengan AI..."):
            try:
                model      = load_model()
                class_names = load_class_names()

                # Preprocess & predict
                img_array   = preprocess_image(image)
                prediction  = model.predict(img_array, verbose=0)
                prob_rotten = float(prediction[0][0])
                prob_fresh  = 1.0 - prob_rotten

                # Tentukan label & rekomendasi
                label, color, badge, rekomendasi, tips = get_label_and_recommendation(prob_rotten)

                # ─── HASIL DETEKSI ───
                st.markdown("---")
                st.subheader("📊 Hasil Deteksi")

                # Badge status
                if color == "green":
                    st.success(f"**{badge}**")
                elif color == "orange":
                    st.warning(f"**{badge}**")
                else:
                    st.error(f"**{badge}**")

                # Label kondisi
                st.markdown(f"### Kondisi Produk: {label}")

                # Progress bar confidence
                st.markdown("**Tingkat Kesegaran:**")
                st.progress(prob_fresh, text=f"Fresh: {prob_fresh*100:.1f}%")
                st.markdown("**Tingkat Pembusukan:**")
                st.progress(prob_rotten, text=f"Rotten: {prob_rotten*100:.1f}%")

                # Detail angka
                with st.expander("🔢 Detail Confidence Score"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Segar (Fresh)", f"{prob_fresh*100:.1f}%")
                    with col_b:
                        st.metric("Busuk (Rotten)", f"{prob_rotten*100:.1f}%")
                    st.caption(
                        "Threshold: < 30% rotten = Segar | 30–65% = Hampir Busuk | > 65% = Busuk"
                    )

                # Rekomendasi
                st.markdown("---")
                st.subheader("💬 Rekomendasi")
                if color == "green":
                    st.success(rekomendasi)
                elif color == "orange":
                    st.warning(rekomendasi)
                else:
                    st.error(rekomendasi)

                st.info(tips)

            except Exception as e:
                st.error(f"❌ Terjadi error saat memproses gambar: {e}")
                st.caption("Pastikan file model `cekfresh_model.keras` ada di folder yang sama dengan app.py")

else:
    # Placeholder saat belum ada upload
    st.markdown("""
    <div style='text-align:center; padding: 40px; background:#f0fdf4;
                border-radius:12px; border: 2px dashed #86efac;'>
        <p style='font-size:48px; margin:0;'>📸</p>
        <p style='color:#15803d; font-size:16px; margin-top:10px;'>
            Upload foto buah atau sayur untuk memulai deteksi
        </p>
        <p style='color:#888; font-size:13px;'>
            Mendukung: Pisang 🍌 | Jeruk 🍊 | Tomat 🍅
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#aaa; font-size:12px;'>
    CekFresh v1.0 — Final Project AI/ML |
    Model: MobileNetV2 Transfer Learning |
    Dataset: 10.531 gambar (Pisang, Jeruk, Tomat)
</p>
""", unsafe_allow_html=True)
