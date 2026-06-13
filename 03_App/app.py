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
    page_title="VigorScan — Freshness Detector",
    page_icon="🍃",
    layout="centered"
)

# ─────────────────────────────────────────────
# PATH MODEL
# ─────────────────────────────────────────────
# Support rename lama (cekfresh) → baru (vigorscan)
MODEL_PATH       = "vigorscan_model.keras" if os.path.exists("vigorscan_model.keras") else "cekfresh_model.keras"
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

        /* ── Camera input ── */
        [data-testid="stCameraInput"] {{
            background: {bg2} !important;
            border-radius: 12px !important;
        }}

        /* ── Tabs ── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            background: {bg2} !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab"] {{
            background: transparent !important;
            border-radius: 8px !important;
            color: {text2} !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 8px 20px !important;
            border: none !important;
        }}
        [data-testid="stTabs"] [aria-selected="true"] {{
            background: {accent} !important;
            color: #fff !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            display: none !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab-border"] {{
            display: none !important;
        }}

        /* ── Result card ── */
        .result-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }}

        /* ── Produk badge chips ── */
        .produk-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: {accent2};
            border: 1px solid {border};
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 600;
            color: {text};
            margin: 4px;
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
    # Default fallback (binary) — diganti setelah retrain multiclass
    return {0: "fresh", 1: "rotten"}


# ─────────────────────────────────────────────
# MAPPING KELAS MULTICLASS
# ─────────────────────────────────────────────
FRUIT_INFO = {
    "banana": ("🍌", "Pisang"),
    "orange": ("🍊", "Jeruk"),
    "tomato": ("🍅", "Tomat"),
}

def parse_class_name(class_name: str):
    """
    Dari nama kelas seperti 'freshbanana' atau 'rottentomato',
    kembalikan (fruit_key, is_fresh).
    """
    if class_name.startswith("fresh"):
        return class_name[5:], True    # "banana", True
    elif class_name.startswith("rotten"):
        return class_name[6:], False   # "tomato", False
    # Fallback binary
    return "unknown", class_name == "fresh"


# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING GAMBAR
# ─────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


# ─────────────────────────────────────────────
# FUNGSI PREDIKSI UTAMA (MULTICLASS)
# ─────────────────────────────────────────────
def predict_freshness(img: Image.Image, model, class_names: dict):
    """
    Prediksi kesegaran buah/sayur dari gambar PIL.
    Mendukung model multiclass (6 kelas) maupun binary (fallback).

    Returns:
        fruit_emoji, fruit_label, freshness_label, freshness_color,
        badge, rekomendasi, tips, rotten_score, probs_dict
    """
    arr   = preprocess_image(img)
    probs = model.predict(arr, verbose=0)[0]  # shape (N,)
    n_classes = len(probs)

    # ── MULTICLASS (6 kelas) ──────────────────────────────────────────────
    if n_classes == 6:
        name_to_idx = {v: k for k, v in class_names.items()}

        # Cari fruit type dengan total probability tertinggi
        fruit_scores = {}
        for fruit_key in ["banana", "orange", "tomato"]:
            fresh_name  = f"fresh{fruit_key}"
            rotten_name = f"rotten{fruit_key}"
            p_f = probs[name_to_idx[fresh_name]]  if fresh_name  in name_to_idx else 0
            p_r = probs[name_to_idx[rotten_name]] if rotten_name in name_to_idx else 0
            fruit_scores[fruit_key] = p_f + p_r

        detected_fruit = max(fruit_scores, key=fruit_scores.get)
        fruit_emoji, fruit_label = FRUIT_INFO.get(detected_fruit, ("🌿", "Produk"))

        # Hitung rotten score per buah yang terdeteksi
        fresh_name  = f"fresh{detected_fruit}"
        rotten_name = f"rotten{detected_fruit}"
        p_fresh  = probs[name_to_idx.get(fresh_name,  0)]
        p_rotten = probs[name_to_idx.get(rotten_name, 1)]
        total_fruit = p_fresh + p_rotten
        rotten_score = float(p_rotten / total_fruit) if total_fruit > 1e-9 else 0.5

        # Buat dict semua probabilitas untuk ditampilkan
        probs_dict = {class_names[i]: float(probs[i]) for i in range(n_classes)}

    # ── BINARY FALLBACK (sebelum retrain) ────────────────────────────────
    else:
        fruit_emoji, fruit_label = "🌿", "Produk"
        rotten_score = float(probs[0]) if n_classes == 1 else float(probs[1])
        p_fresh  = 1.0 - rotten_score
        p_rotten = rotten_score
        probs_dict = {"fresh": float(p_fresh), "rotten": float(p_rotten)}

    # ── KONVERSI ROTTEN SCORE → 3 LABEL ──────────────────────────────────
    freshness_label, freshness_color, badge, rekomendasi, tips = \
        rotten_score_to_label(rotten_score, fruit_label)

    return (fruit_emoji, fruit_label, freshness_label, freshness_color,
            badge, rekomendasi, tips, rotten_score, probs_dict)


def rotten_score_to_label(rotten_score: float, fruit_label: str = "Produk"):
    """Konversi skor busuk (0.0–1.0) ke label, warna, badge, dan rekomendasi."""
    if rotten_score < 0.30:
        label  = "🟢 Segar"
        color  = "green"
        badge  = "✅ LAYAK JUAL"
        rekmd  = (
            f"{fruit_label} dalam kondisi prima dan layak jual. "
            "Simpan di tempat sejuk atau kulkas untuk menjaga kesegaran lebih lama."
        )
        tips   = "💡 Simpan terpisah dari buah yang sudah matang untuk menghindari pematangan dini."

    elif rotten_score < 0.65:
        label  = "🟡 Hampir Busuk"
        color  = "orange"
        badge  = "⚠️ SEGERA JUAL / GUNAKAN"
        rekmd  = (
            f"{fruit_label} mulai menunjukkan tanda-tanda penurunan kualitas. "
            "Segera jual atau gunakan dalam waktu dekat. "
            "Pertimbangkan memberikan diskon untuk mempercepat penjualan."
        )
        tips   = "💡 Pisahkan dari produk segar agar tidak mempercepat pembusukan produk lainnya."

    else:
        label  = "🔴 Busuk"
        color  = "red"
        badge  = "❌ TIDAK LAYAK JUAL"
        rekmd  = (
            f"{fruit_label} sudah tidak layak untuk dijual atau dikonsumsi. "
            "Pisahkan segera dari produk lainnya untuk mencegah kontaminasi."
        )
        tips   = "💡 Manfaatkan sebagai kompos atau pupuk organik untuk mengurangi food waste."

    return label, color, badge, rekmd, tips


# ─────────────────────────────────────────────
# FUNGSI TAMPILKAN HASIL
# ─────────────────────────────────────────────
def show_result(img: Image.Image, filename: str, model, class_names: dict, show_image=True):
    """Tampilkan gambar + hasil prediksi dalam satu blok."""
    try:
        (fruit_emoji, fruit_label, freshness_label, freshness_color,
         badge, rekomendasi, tips, rotten_score, probs_dict) = \
            predict_freshness(img, model, class_names)

        fresh_score = 1.0 - rotten_score

        # ── Layout: gambar (kiri) + ringkasan (kanan) ──
        if show_image:
            col_img, col_res = st.columns([1, 1.3])
            with col_img:
                st.image(img, use_container_width=True)
        else:
            # Kamera: gambar sudah tampil di camera_input, skip duplikasi
            col_res = st.container()

        with col_res:
            # Jenis buah
            st.markdown(
                f"<p style='font-size:18px; font-weight:700; margin:0 0 8px 0;'>"
                f"{fruit_emoji} {fruit_label}</p>",
                unsafe_allow_html=True
            )

            # Badge status
            if freshness_color == "green":
                st.success(f"**{badge}** &nbsp; {freshness_label}")
            elif freshness_color == "orange":
                st.warning(f"**{badge}** &nbsp; {freshness_label}")
            else:
                st.error(f"**{badge}** &nbsp; {freshness_label}")

            # Progress bars ringkas (cast ke float agar kompatibel dengan Streamlit)
            st.progress(float(fresh_score),  text=f"🟢 Segar:  {fresh_score*100:.0f}%")
            st.progress(float(rotten_score), text=f"🔴 Busuk:  {rotten_score*100:.0f}%")

        # ── Detail confidence (collapsed) ──
        with st.expander("🔢 Detail probabilitas semua kelas"):
            n_col = min(len(probs_dict), 3)
            cols  = st.columns(n_col)
            for i, (cls_name, prob) in enumerate(
                sorted(probs_dict.items(), key=lambda x: -x[1])
            ):
                with cols[i % n_col]:
                    st.metric(cls_name, f"{prob*100:.1f}%")

        # ── Rekomendasi ──
        if freshness_color == "green":
            st.success(f"💬 **Rekomendasi:** {rekomendasi}")
        elif freshness_color == "orange":
            st.warning(f"💬 **Rekomendasi:** {rekomendasi}")
        else:
            st.error(f"💬 **Rekomendasi:** {rekomendasi}")
        st.info(tips)

    except Exception as e:
        st.error(f"❌ Error memproses `{filename}`: {e}")
        st.caption("Pastikan `vigorscan_model.keras` dan `class_names.json` ada di folder 03_App/")


# ─────────────────────────────────────────────
# TERAPKAN TEMA
# ─────────────────────────────────────────────
apply_theme(st.session_state.dark_mode)

# ─────────────────────────────────────────────
# TOGGLE DARK/LIGHT — pakai checkbox native
# ─────────────────────────────────────────────
is_dark = st.session_state.dark_mode

col_space, col_tog = st.columns([6, 2])
with col_tog:
    st.markdown(f"""
    <style>
        div[data-testid="stCheckbox"] label span:last-child {{
            display: none;
        }}
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
# HEADER
# ─────────────────────────────────────────────
is_dark = st.session_state.dark_mode
head_sub = "#8fbe7a" if is_dark else "#4a6640"
chip_bg  = "#1e2d18" if is_dark else "#e8f5e0"
chip_bd  = "rgba(255,255,255,0.09)" if is_dark else "rgba(0,0,0,0.09)"
chip_tx  = "#e0f0d8" if is_dark else "#1a2e14"

st.markdown(f"""
<div style='text-align:center; padding: 8px 0 4px 0;'>
  <h1 style='color:#16a34a; margin:0; font-size:2.2rem;'>🍃 VigorScan</h1>
  <p style='color:{head_sub}; font-size:14px; margin:4px 0 12px 0;'>
    Freshness Detector &amp; Food Waste Solution &nbsp;·&nbsp;
    Deteksi kelayakan jual buah &amp; sayur dari foto menggunakan AI
  </p>
  <div>
    <span class='produk-chip' style='background:{chip_bg}; border-color:{chip_bd}; color:{chip_tx};'>🍌 Pisang</span>
    <span class='produk-chip' style='background:{chip_bg}; border-color:{chip_bd}; color:{chip_tx};'>🍊 Jeruk</span>
    <span class='produk-chip' style='background:{chip_bg}; border-color:{chip_bd}; color:{chip_tx};'>🍅 Tomat</span>
  </div>
</div>
<hr style='margin: 12px 0;'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT MODE — pakai Tabs (active/inactive otomatis)
# ─────────────────────────────────────────────
ph_bg     = "#162012" if is_dark else "#f0fdf4"
ph_border = "#3a6040" if is_dark else "#86efac"
ph_text   = "#8fbe7a" if is_dark else "#15803d"
ph_sub    = "#5a8050" if is_dark else "#888888"

tab_upload, tab_camera = st.tabs(["📁  Upload Foto", "📷  Kamera"])

# ─────────────────────────────────────────────
# TAB 1: UPLOAD (multi-file)
# ─────────────────────────────────────────────
with tab_upload:
    st.markdown("##### Upload satu atau beberapa foto sekaligus")
    uploaded_files = st.file_uploader(
        "Pilih foto buah atau sayur (JPG / PNG / JPEG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Bisa upload banyak foto sekaligus untuk deteksi batch",
        label_visibility="collapsed"
    )

    if uploaded_files:
        n = len(uploaded_files)
        st.caption(f"{n} foto dipilih — klik Deteksi untuk menganalisis semua.")
        st.markdown("")

        if st.button("🔍 Deteksi Sekarang", type="primary",
                     use_container_width=True, key="btn_upload"):
            model       = load_model()
            class_names = load_class_names()

            for i, uf in enumerate(uploaded_files):
                img = Image.open(uf)
                st.markdown(f"---")
                if n > 1:
                    st.markdown(f"**Foto {i+1} / {n}** — `{uf.name}`")
                with st.spinner(f"Menganalisis {uf.name}..."):
                    show_result(img, uf.name, model, class_names, show_image=True)
    else:
        st.markdown(f"""
        <div style='text-align:center; padding:40px; background:{ph_bg};
                    border-radius:12px; border:2px dashed {ph_border}; margin-top:8px;'>
            <p style='font-size:44px; margin:0;'>📸</p>
            <p style='color:{ph_text}; font-size:15px; margin:10px 0 4px 0;'>
                Drag & drop atau klik untuk memilih foto
            </p>
            <p style='color:{ph_sub}; font-size:12px; margin:0;'>
                Pisang 🍌 · Jeruk 🍊 · Tomat 🍅 &nbsp;—&nbsp; bisa multi-foto sekaligus
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2: KAMERA LANGSUNG
# ─────────────────────────────────────────────
with tab_camera:
    st.markdown("##### Ambil foto langsung dengan kamera")
    st.caption("Arahkan kamera ke buah/sayur, lalu klik ikon kamera untuk mengambil foto.")

    camera_image = st.camera_input("foto_kamera", label_visibility="collapsed")

    if camera_image is not None:
        img = Image.open(camera_image)
        st.markdown("")
        if st.button("🔍 Deteksi Sekarang", type="primary",
                     use_container_width=True, key="btn_camera"):
            model       = load_model()
            class_names = load_class_names()
            with st.spinner("Menganalisis foto..."):
                show_result(img, "foto_kamera.jpg", model, class_names, show_image=False)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:11px; margin:0;'>"
    "VigorScan v2.0 &nbsp;·&nbsp; Final Project AI/ML &nbsp;·&nbsp; "
    "MobileNetV2 Transfer Learning &nbsp;·&nbsp; "
    "🍌 Pisang &nbsp; 🍊 Jeruk &nbsp; 🍅 Tomat"
    "</p>",
    unsafe_allow_html=True
)
