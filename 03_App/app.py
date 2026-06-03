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
MODEL_PATH       = "cekfresh_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# ─────────────────────────────────────────────
# THEME TOGGLE — disimpan di session_state
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def apply_theme(dark: bool):
    """Inject CSS variabel berdasarkan tema yang dipilih."""
    if dark:
        bg         = "#101a0d"
        bg2        = "#162012"
        card       = "#1e2d18"
        text       = "#e0f0d8"
        text2      = "#8fbe7a"
        text3      = "#5a8050"
        accent     = "#5fc877"
        accent2    = "#1a3020"
        border     = "rgba(255,255,255,0.09)"
        prog_bg    = "#1a2e16"
        badge_ok   = "#1a3825"; badge_ok_t   = "#7fe09e"
        badge_warn = "#2e2200"; badge_warn_t = "#f0c040"
        badge_err  = "#2e1010"; badge_err_t  = "#f08080"
        tip_bg     = "#162012"
    else:
        bg         = "#f7faf5"
        bg2        = "#eef5ea"
        card       = "#ffffff"
        text       = "#1a2e14"
        text2      = "#4a6640"
        text3      = "#7a9470"
        accent     = "#2d7a45"
        accent2    = "#e8f5e0"
        border     = "rgba(0,0,0,0.09)"
        prog_bg    = "#e2eede"
        badge_ok   = "#d4edda"; badge_ok_t   = "#1a5c30"
        badge_warn = "#fff3cd"; badge_warn_t = "#7d4e00"
        badge_err  = "#fde8e8"; badge_err_t  = "#7a1a1a"
        tip_bg     = "#eef5ea"

    st.markdown(f"""
    <style>
        /* ── Smooth transitions semua elemen ── */
        *, *::before, *::after {{
            transition:
                background-color 0.35s ease,
                color 0.35s ease,
                border-color 0.35s ease,
                box-shadow 0.35s ease !important;
        }}

        /* ── Reset & Root ── */
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stApp"] {{
            background-color: {bg} !important;
            color: {text} !important;
        }}

        /* Sidebar bawaan Streamlit kalau ada */
        [data-testid="stSidebar"] {{
            background-color: {card} !important;
        }}

        /* Main block container */
        .main .block-container {{
            background-color: {bg};
            padding-top: 1.5rem;
        }}

        /* ── Typography umum ── */
        h1, h2, h3, h4, p, label,
        [data-testid="stMarkdownContainer"],
        .stText, .element-container {{
            color: {text} !important;
        }}

        /* ── Upload area ── */
        [data-testid="stFileUploader"] {{
            background: {bg2} !important;
            border: 2px dashed {border} !important;
            border-radius: 12px !important;
            padding: 16px !important;
            color: {text} !important;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: {accent} !important;
        }}

        /* ── Tombol PRIMARY (Deteksi Sekarang) ── */
        .stButton > button[kind="primary"] {{
            background: {accent} !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.2rem !important;
            font-size: 15px !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            opacity: 0.88 !important;
        }}

        /* ── Tombol TOGGLE (secondary) — invisible overlay ── */
        div[data-testid="stButton"]:has(button[kind="secondary"]) > button {{
            opacity: 0 !important;
            position: absolute !important;
            width: 80px !important;
            height: 36px !important;
            cursor: pointer !important;
            z-index: 10 !important;
            margin: 0 !important;
        }}

        /* ── Toggle visual ── */
        .toggle-visual {{
            display: flex;
            align-items: center;
            gap: 8px;
            justify-content: flex-end;
            padding-top: 4px;
            pointer-events: none;
        }}
        .toggle-label-txt {{
            font-size: 12px;
            color: {text2};
            font-weight: 500;
        }}
        .toggle-track {{
            position: relative;
            width: 52px;
            height: 28px;
            background: {'#2a4020' if dark else '#d0e8c8'};
            border-radius: 14px;
            display: flex;
            align-items: center;
            padding: 0 5px;
            box-sizing: border-box;
            flex-shrink: 0;
        }}
        .toggle-icon-sun {{
            position: absolute;
            left: 6px;
            font-size: 11px;
            opacity: {'0.3' if dark else '1'};
        }}
        .toggle-icon-moon {{
            position: absolute;
            right: 5px;
            font-size: 11px;
            opacity: {'1' if dark else '0.3'};
        }}
        .toggle-knob {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: {'#5fc877' if dark else '#2d7a45'};
            margin-left: {'24px' if dark else '0px'};
            flex-shrink: 0;
        }}

        /* ── Progress bar ── */
        .stProgress > div > div {{
            background-color: {prog_bg} !important;
            border-radius: 6px !important;
        }}

        /* ── Alert boxes ── */
        .stSuccess {{
            background: {badge_ok} !important;
            color: {badge_ok_t} !important;
            border-left: 4px solid {accent} !important;
            border-radius: 10px !important;
        }}
        .stWarning {{
            background: {badge_warn} !important;
            color: {badge_warn_t} !important;
            border-radius: 10px !important;
        }}
        .stError {{
            background: {badge_err} !important;
            color: {badge_err_t} !important;
            border-radius: 10px !important;
        }}
        .stInfo {{
            background: {tip_bg} !important;
            border-left: 3px solid {accent} !important;
            border-radius: 10px !important;
            color: {text2} !important;
        }}

        /* ── Expander ── */
        [data-testid="stExpander"] {{
            background: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}

        /* ── Metric ── */
        [data-testid="stMetric"] {{
            background: {bg2} !important;
            border-radius: 10px !important;
            padding: 12px !important;
            border: 1px solid {border} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {accent} !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {text2} !important;
        }}

        /* ── Divider ── */
        hr {{
            border-color: {border} !important;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {bg2}; }}
        ::-webkit-scrollbar-thumb {{
            background: {accent};
            border-radius: 3px;
        }}

        /* ── Caption & small text ── */
        .stCaption, small {{
            color: {text3} !important;
        }}

        /* ── Image border ── */
        img {{
            border-radius: 10px !important;
        }}
    </style>
    """, unsafe_allow_html=True)


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
# TERAPKAN TEMA
# ─────────────────────────────────────────────
apply_theme(st.session_state.dark_mode)

# ─────────────────────────────────────────────
# TOGGLE DARK/LIGHT — Visual switch di kanan atas
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# TOGGLE DARK/LIGHT — pakai checkbox native
# ─────────────────────────────────────────────
is_dark = st.session_state.dark_mode

col_space, col_tog = st.columns([6, 2])

with col_tog:
    st.markdown(f"""
    <style>
        /* Sembunyikan label default checkbox */
        div[data-testid="stCheckbox"] label span:last-child {{
            display: none;
        }}
        /* Style checkbox jadi toggle switch */
        div[data-testid="stCheckbox"] label {{
            display: flex;
            align-items: center;
            gap: 8px;
            justify-content: flex-end;
            padding-top: 6px;
            cursor: pointer;
        }}
        div[data-testid="stCheckbox"] label::before {{
            content: '{"🌙" if is_dark else "☀️"}';
            font-size: 13px;
        }}
        div[data-testid="stCheckbox"] input[type="checkbox"] {{
            appearance: none !important;
            -webkit-appearance: none !important;
            width: 52px !important;
            height: 28px !important;
            border-radius: 14px !important;
            background: {"#2a4020" if is_dark else "#d0e8c8"} !important;
            position: relative !important;
            cursor: pointer !important;
            border: none !important;
            outline: none !important;
            flex-shrink: 0;
        }}
        div[data-testid="stCheckbox"] input[type="checkbox"]::after {{
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: {"#5fc877" if is_dark else "#2d7a45"};
            top: 4px;
            left: {"28px" if is_dark else "4px"};
        }}
    </style>
    """, unsafe_allow_html=True)

    toggled = st.checkbox("tema", value=is_dark, key="theme_cb", label_visibility="collapsed")
    if toggled != is_dark:
        st.session_state.dark_mode = toggled
        st.rerun()

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
                model       = load_model()
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
    ph_bg     = "#162012" if st.session_state.dark_mode else "#f0fdf4"
    ph_border = "#3a6040" if st.session_state.dark_mode else "#86efac"
    ph_text   = "#8fbe7a" if st.session_state.dark_mode else "#15803d"
    ph_sub    = "#5a8050" if st.session_state.dark_mode else "#888888"

    st.markdown(f"""
    <div style='text-align:center; padding: 40px; background:{ph_bg};
                border-radius:12px; border: 2px dashed {ph_border};'>
        <p style='font-size:48px; margin:0;'>📸</p>
        <p style='color:{ph_text}; font-size:16px; margin-top:10px;'>
            Upload foto buah atau sayur untuk memulai deteksi
        </p>
        <p style='color:{ph_sub}; font-size:13px;'>
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