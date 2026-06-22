import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import io

st.set_page_config(
    page_title="VigorScan — AI Freshness Detector",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Resolve path relative ke lokasi script (bukan CWD), karena di Streamlit Cloud
# CWD = repo root, bukan folder app.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_NEW  = os.path.join(_SCRIPT_DIR, "vigorscan_model.keras")
_MODEL_OLD  = os.path.join(_SCRIPT_DIR, "cekfresh_model.keras")
MODEL_PATH       = _MODEL_NEW if os.path.exists(_MODEL_NEW) else _MODEL_OLD
CLASS_NAMES_PATH = os.path.join(_SCRIPT_DIR, "class_names.json")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "upload_results" not in st.session_state:
    st.session_state.upload_results = []
if "camera_result" not in st.session_state:
    st.session_state.camera_result = None
# Key rotation: increment ini untuk reset file_uploader & camera_input widget
if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0
if "camera_nonce" not in st.session_state:
    st.session_state.camera_nonce = 0


def apply_theme(dark):
    if dark:
        bg_a="#0a1410"; bg_b="#0f1f17"; bg_c="#0a1a14"
        card_bg="rgba(22,36,28,0.65)"; card_brdr="rgba(95,200,119,0.15)"
        text="#e8f5e8"; text2="#a8c9a8"; text3="#6a8a6a"
        accent="#5fc877"; accent2="#3da85f"
        accent_glow="rgba(95,200,119,0.4)"; accent_soft="rgba(95,200,119,0.08)"
        ok_bg="rgba(95,200,119,0.15)"; ok_brdr="#5fc877"; ok_txt="#a8e8b8"
        warn_bg="rgba(240,180,60,0.15)"; warn_brdr="#f0b43c"; warn_txt="#f5d896"
        err_bg="rgba(240,100,100,0.15)"; err_brdr="#f06464"; err_txt="#f5b0b0"
        track_bg="rgba(255,255,255,0.06)"; shadow="0 8px 32px rgba(0,0,0,0.4)"
    else:
        bg_a="#f0f9f0"; bg_b="#e8f5e6"; bg_c="#f5fbf2"
        card_bg="rgba(255,255,255,0.75)"; card_brdr="rgba(45,122,69,0.15)"
        text="#0d2818"; text2="#4a6640"; text3="#7a9470"
        accent="#16a34a"; accent2="#15803d"
        accent_glow="rgba(22,163,74,0.25)"; accent_soft="rgba(22,163,74,0.06)"
        ok_bg="rgba(22,163,74,0.10)"; ok_brdr="#16a34a"; ok_txt="#14532d"
        warn_bg="rgba(202,138,4,0.10)"; warn_brdr="#ca8a04"; warn_txt="#713f12"
        err_bg="rgba(220,38,38,0.10)"; err_brdr="#dc2626"; err_txt="#7f1d1d"
        track_bg="rgba(0,0,0,0.06)"; shadow="0 8px 32px rgba(22,163,74,0.08)"
    knob_left = "22px" if dark else "2px"

    st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        body, p, h1, h2, h3, h4, h5, h6, label, div, span, button, input, textarea, select {{
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        }}
        .material-icons, .material-icons-outlined,
        .material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp,
        [class*="material-symbols"], [class*="material-icons"],
        [data-testid="stIconMaterial"], [data-testid*="MaterialIcon"] {{
            font-family: 'Material Symbols Outlined','Material Symbols Rounded','Material Icons','Material Icons Outlined' !important;
            font-feature-settings: 'liga'; -webkit-font-feature-settings: 'liga';
            font-weight: normal !important; font-style: normal !important;
            letter-spacing: normal !important; text-transform: none !important;
            display: inline-block; white-space: nowrap; word-wrap: normal;
            direction: ltr; -webkit-font-smoothing: antialiased;
        }}
        * {{ transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease; }}
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; }}
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display: none !important; }}
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(at 20% 10%, {accent_soft} 0px, transparent 50%),
                radial-gradient(at 80% 90%, {accent_soft} 0px, transparent 50%),
                linear-gradient(135deg, {bg_a} 0%, {bg_b} 50%, {bg_c} 100%) !important;
            color: {text} !important;
        }}
        [data-testid="stApp"] {{ background: transparent !important; }}
        .main .block-container {{ padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1200px !important; }}
        h1,h2,h3,h4,h5,h6,p,label,span,div {{ color: {text}; }}
        [data-testid="stMarkdownContainer"] p {{ color: {text2}; line-height: 1.6; }}

        .hero-wrap {{ position: relative; text-align: center; padding: 32px 20px 24px; margin-bottom: 24px;
            border-radius: 24px; background: {card_bg}; backdrop-filter: blur(24px);
            border: 1px solid {card_brdr}; box-shadow: {shadow}; overflow: hidden; }}
        .hero-wrap::before {{ content:''; position: absolute; top: -50%; left: -10%; width: 120%; height: 200%;
            background: radial-gradient(circle, {accent_glow} 0%, transparent 60%);
            opacity: 0.3; pointer-events: none; animation: pulse 6s ease-in-out infinite; }}
        @keyframes pulse {{ 0%,100% {{ opacity: 0.2; transform: scale(1); }} 50% {{ opacity: 0.35; transform: scale(1.05); }} }}
        .hero-badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px;
            background: {accent_soft}; border: 1px solid {card_brdr}; border-radius: 20px;
            font-size: 11px; font-weight: 600; color: {accent}; letter-spacing: 1px;
            text-transform: uppercase; margin-bottom: 16px; position: relative; z-index: 1; }}
        .hero-badge .dot {{ width: 6px; height: 6px; border-radius: 50%; background: {accent};
            box-shadow: 0 0 8px {accent}; animation: blink 1.5s ease-in-out infinite; }}
        @keyframes blink {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
        .hero-title {{ font-size: 3.2rem !important; font-weight: 800 !important; margin: 0 !important;
            background: linear-gradient(135deg, {accent} 0%, {accent2} 50%, {accent} 100%);
            background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent; letter-spacing: -1px;
            animation: shimmer 4s linear infinite; position: relative; z-index: 1; }}
        @keyframes shimmer {{ 0% {{ background-position: 0% center; }} 100% {{ background-position: 200% center; }} }}
        .hero-sub {{ color: {text2} !important; font-size: 15px !important; font-weight: 500;
            margin: 8px 0 18px 0 !important; position: relative; z-index: 1; }}
        .hero-chips {{ display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; position: relative; z-index: 1; }}
        .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 7px 16px;
            background: {accent_soft}; border: 1px solid {card_brdr}; border-radius: 100px;
            font-size: 13px; font-weight: 600; color: {text}; backdrop-filter: blur(8px); }}
        .chip:hover {{ transform: translateY(-2px); border-color: {accent}; box-shadow: 0 4px 12px {accent_glow}; }}

        [data-testid="stFileUploader"] {{ background: {card_bg} !important; backdrop-filter: blur(16px);
            border: 2px dashed {card_brdr} !important; border-radius: 18px !important; padding: 14px !important; }}
        [data-testid="stFileUploader"]:hover {{ border-color: {accent} !important; background: {accent_soft} !important; }}
        [data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] {{
            background: transparent !important; border: none !important; }}
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] div {{ color: {text2} !important; }}
        [data-testid="stFileUploader"] small {{ color: {text3} !important; }}
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
            background: {accent_soft} !important; color: {accent} !important;
            border: 1px solid {accent} !important; font-weight: 600 !important; }}
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {{
            background: {accent} !important; color: #ffffff !important; }}

        .stButton > button[kind="primary"] {{ background: linear-gradient(135deg, {accent} 0%, {accent2} 100%) !important;
            color: #ffffff !important; border: none !important; border-radius: 12px !important;
            font-weight: 700 !important; padding: 0.75rem 1.5rem !important; font-size: 15px !important;
            box-shadow: 0 4px 16px {accent_glow}, inset 0 1px 0 rgba(255,255,255,0.2) !important; letter-spacing: 0.3px; }}
        .stButton > button[kind="primary"]:hover {{ transform: translateY(-2px);
            box-shadow: 0 8px 24px {accent_glow}, inset 0 1px 0 rgba(255,255,255,0.2) !important; }}
        .stButton > button[kind="secondary"] {{ background: {accent_soft} !important; color: {text2} !important;
            border: 1px solid {card_brdr} !important; border-radius: 10px !important; font-weight: 600 !important; }}
        .stButton > button[kind="secondary"]:hover {{ border-color: {accent} !important; color: {accent} !important; }}

        .stSuccess, .stWarning, .stError, .stInfo {{ border-radius: 14px !important; border: 1px solid !important;
            backdrop-filter: blur(12px); padding: 14px 18px !important; }}
        .stSuccess {{ background: {ok_bg} !important; border-color: {ok_brdr} !important; color: {ok_txt} !important; }}
        .stWarning {{ background: {warn_bg} !important; border-color: {warn_brdr} !important; color: {warn_txt} !important; }}
        .stError {{ background: {err_bg} !important; border-color: {err_brdr} !important; color: {err_txt} !important; }}
        .stInfo {{ background: {accent_soft} !important; border-color: {card_brdr} !important; color: {text2} !important; }}

        [data-testid="stExpander"] {{ background: {card_bg} !important; backdrop-filter: blur(16px);
            border: 1px solid {card_brdr} !important; border-radius: 14px !important; }}
        [data-testid="stExpander"] summary {{ color: {text} !important; font-weight: 600 !important; padding: 8px 4px !important; }}
        [data-testid="stExpander"] summary:hover {{ color: {accent} !important; }}

        [data-testid="stMetric"] {{ background: {accent_soft} !important; border: 1px solid {card_brdr} !important;
            border-radius: 12px !important; padding: 14px !important; }}
        [data-testid="stMetricValue"] {{ color: {accent} !important; font-weight: 800 !important; font-size: 1.3rem !important; }}
        [data-testid="stMetricLabel"] {{ color: {text2} !important; font-size: 11px !important;
            text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600 !important; }}

        [data-testid="stImage"] img {{ border-radius: 16px !important; box-shadow: {shadow}; border: 1px solid {card_brdr}; }}
        [data-testid="stCameraInput"] {{ background: {card_bg} !important; backdrop-filter: blur(16px);
            border-radius: 18px !important; border: 1px solid {card_brdr}; padding: 12px !important; }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{ background: {card_bg} !important; backdrop-filter: blur(16px);
            border: 1px solid {card_brdr} !important; border-radius: 14px !important; padding: 6px !important; gap: 4px !important; }}
        [data-testid="stTabs"] [data-baseweb="tab"] {{ background: transparent !important; border-radius: 10px !important;
            color: {text2} !important; font-weight: 600 !important; font-size: 14px !important;
            padding: 10px 24px !important; border: none !important; }}
        [data-testid="stTabs"] [data-baseweb="tab"]:hover {{ color: {accent} !important; background: {accent_soft} !important; }}
        [data-testid="stTabs"] [aria-selected="true"] {{ background: linear-gradient(135deg, {accent} 0%, {accent2} 100%) !important;
            color: #ffffff !important; box-shadow: 0 4px 12px {accent_glow}; }}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none !important; }}

        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: {accent_soft}; border-radius: 100px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {accent}; }}

        .result-header {{ display: flex; align-items: center; gap: 14px; padding: 14px 18px;
            background: {accent_soft}; border: 1px solid {card_brdr}; border-radius: 14px; margin-bottom: 16px; }}
        .fruit-emoji-big {{ font-size: 2.4rem; line-height: 1; }}
        .fruit-label {{ font-size: 1.25rem; font-weight: 700; color: {text}; margin: 0; }}
        .fruit-sub {{ font-size: 12px; color: {text3}; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }}

        .status-badge {{ display: flex; align-items: center; gap: 12px; padding: 16px 20px; border-radius: 14px;
            border-left: 4px solid; margin-bottom: 14px; backdrop-filter: blur(12px); font-weight: 600; }}
        .status-badge.ok    {{ background: {ok_bg};   border-color: {ok_brdr};   color: {ok_txt}; }}
        .status-badge.warn  {{ background: {warn_bg}; border-color: {warn_brdr}; color: {warn_txt}; }}
        .status-badge.err   {{ background: {err_bg};  border-color: {err_brdr};  color: {err_txt}; }}
        .status-icon {{ font-size: 1.6rem; }}
        .status-text-main {{ font-size: 0.95rem; font-weight: 700; line-height: 1.3; }}
        .status-text-sub  {{ font-size: 0.78rem; opacity: 0.85; margin-top: 2px; }}

        .prob-row {{ display: flex; justify-content: space-between; align-items: center;
            font-size: 13px; font-weight: 600; margin-bottom: 6px; }}
        .prob-label {{ color: {text2}; }}
        .prob-value {{ color: {text}; font-family: 'JetBrains Mono', monospace; }}

        .vs-bar-track {{ width: 100%; height: 12px; background: {track_bg};
            border-radius: 100px; overflow: hidden; margin-bottom: 14px; border: 1px solid {card_brdr}; }}
        .vs-bar-fill {{ height: 100%; border-radius: 100px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }}
        .vs-bar-fill.fresh {{ background: linear-gradient(90deg, #2d9c47 0%, #5fc877 100%);
            box-shadow: 0 0 14px rgba(95,200,119,0.55), inset 0 1px 0 rgba(255,255,255,0.3); }}
        .vs-bar-fill.rotten {{ background: linear-gradient(90deg, #b03030 0%, #f06464 100%);
            box-shadow: 0 0 14px rgba(240,100,100,0.55), inset 0 1px 0 rgba(255,255,255,0.3); }}

        .empty-state {{ text-align: center; padding: 60px 30px; background: {accent_soft};
            border: 2px dashed {card_brdr}; border-radius: 18px; margin-top: 12px; }}
        .empty-icon {{ font-size: 4rem; margin-bottom: 12px; opacity: 0.7; }}
        .empty-title {{ font-size: 16px; font-weight: 700; color: {text}; margin: 0 0 6px 0; }}
        .empty-sub {{ font-size: 13px; color: {text3}; margin: 0; }}

        .footer-wrap {{ text-align: center; padding: 24px 0 8px; margin-top: 32px; border-top: 1px solid {card_brdr}; }}
        .footer-text {{ font-size: 11px; color: {text3}; margin: 0; letter-spacing: 0.3px; }}
        .footer-tech {{ margin-top: 8px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {text3}; }}

        hr {{ border: none !important; border-top: 1px solid {card_brdr} !important; margin: 16px 0 !important; }}

        /* ── st.toggle (BaseWeb switch) — native toggle look ── */
        [data-testid="stCheckbox"] label,
        [data-testid="stToggle"] label,
        div[data-testid="stCheckbox"] > label,
        div[data-testid="stToggle"] > label {{
            display: flex !important; align-items: center !important; gap: 10px !important;
            justify-content: flex-end !important; cursor: pointer !important;
        }}
        [data-testid="stToggle"] label > div:last-child,
        [data-testid="stCheckbox"] label > div:last-child,
        div[data-testid="stMarkdownContainer"] p {{ color: {text2} !important; }}
        [data-testid="stToggle"] [data-baseweb="checkbox"] > div:first-child,
        [data-testid="stToggle"] [role="switch"],
        [data-baseweb="switch"] > div:first-child {{
            background: {accent_soft} !important;
            border: 1px solid {card_brdr} !important;
            border-radius: 100px !important;
        }}
        [data-testid="stToggle"] [aria-checked="true"] [data-baseweb="checkbox"] > div:first-child,
        [data-testid="stToggle"] [aria-checked="true"] > div:first-child,
        [data-baseweb="switch"][aria-checked="true"] > div:first-child {{
            background: linear-gradient(135deg, {accent} 0%, {accent2} 100%) !important;
            border-color: {accent} !important;
            box-shadow: 0 0 12px {accent_glow} !important;
        }}
        /* Knob bulat di dalam switch */
        [data-baseweb="switch"] [role="presentation"],
        [data-testid="stToggle"] [data-baseweb="checkbox"] [role="presentation"] {{
            background: #ffffff !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important;
        }}
        /* Fallback styling lama st.checkbox untuk safety */
        div[data-testid="stCheckbox"] input[type="checkbox"] {{ appearance: none !important;
            -webkit-appearance: none !important; width: 44px !important; height: 24px !important;
            border-radius: 100px !important; background: {accent_soft} !important;
            border: 1px solid {card_brdr} !important; position: relative !important;
            cursor: pointer !important; outline: none !important; flex-shrink: 0; }}
        div[data-testid="stCheckbox"] input[type="checkbox"]::after {{ content: ''; position: absolute;
            width: 18px; height: 18px; border-radius: 50%;
            background: linear-gradient(135deg, {accent} 0%, {accent2} 100%);
            top: 2px; left: {knob_left}; transition: left 0.25s ease;
            box-shadow: 0 2px 6px {accent_glow}; }}

        .stSpinner > div {{ border-top-color: {accent} !important; }}
        [data-testid="stCaptionContainer"] {{ color: {text3} !important; }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {0: "fresh", 1: "rotten"}


FRUIT_INFO = {
    "banana": ("🍌", "Pisang"),
    "orange": ("🍊", "Jeruk"),
    "tomato": ("🍅", "Tomat"),
}

def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def color_signature(img):
    """
    Analisis warna fokus pada pixel berwarna buah (red/orange/yellow saja).
    Hijau (daun) di-exclude supaya tidak mendilusi rasio.
    Return:
        red_ratio, orange_ratio   — proporsi di antara pixel buah
        bright_red_ratio          — pixel red yang terang/saturasi tinggi (= tomat segar)
        dark_red_ratio            — pixel red yang gelap/value rendah (= tomat busuk)
        fruit_n                   — total pixel berwarna buah
    """
    arr = np.array(img.convert("RGB").resize((140, 140))).astype(np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    delta = maxc - minc
    sat = np.where(maxc > 0, delta / np.maximum(maxc, 1.0), 0)
    val = maxc / 255.0
    safe = np.where(delta == 0, 1.0, delta)
    hue = np.zeros_like(R)
    mr = (maxc == R) & (delta > 0)
    mg = (maxc == G) & (delta > 0)
    mb = (maxc == B) & (delta > 0)
    hue[mr] = ((G[mr] - B[mr]) / safe[mr]) % 6
    hue[mg] = ((B[mg] - R[mg]) / safe[mg]) + 2
    hue[mb] = ((R[mb] - G[mb]) / safe[mb]) + 4
    hue = hue * 60.0

    colorful = (sat > 0.40) & (val > 0.30)
    # Hanya hitung pixel di rentang warna buah: red+orange+yellow (0-70° atau >340°)
    fruit_mask = colorful & ((hue < 70) | (hue > 340))
    fruit_n = int(fruit_mask.sum())
    if fruit_n < 200:
        return {"red_ratio": 0.0, "orange_ratio": 0.0,
                "bright_red_ratio": 0.0, "dark_red_ratio": 0.0, "fruit_n": fruit_n}

    fh = hue[fruit_mask]; fs = sat[fruit_mask]; fv = val[fruit_mask]
    red_mask    = (fh < 18) | (fh > 340)
    orange_mask = (fh >= 18) & (fh < 45)

    return {
        "red_ratio":        float(red_mask.sum()) / fruit_n,
        "orange_ratio":     float(orange_mask.sum()) / fruit_n,
        "bright_red_ratio": float((red_mask & (fs > 0.55) & (fv > 0.45)).sum()) / fruit_n,
        "dark_red_ratio":   float((red_mask & (fv < 0.35)).sum()) / fruit_n,
        "fruit_n":          fruit_n,
    }


def banana_decay_signal(img):
    """
    Detect WHITE MOLD pada pisang yang model tidak catch (fungal infection).
    Dark rot signal SENGAJA TIDAK dipakai karena terlalu sering false-positive
    pada background gelap (meja kayu/dinding). White mold lebih reliable karena
    saturasi rendah JARANG ada pada wood/skin/object normal.
    """
    arr = np.array(img.convert("RGB").resize((140, 140))).astype(np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(R, G), B)
    minc = np.minimum(np.minimum(R, G), B)
    delta = maxc - minc
    sat = np.where(maxc > 0, delta / np.maximum(maxc, 1.0), 0)
    val = maxc / 255.0
    safe = np.where(delta == 0, 1.0, delta)
    hue = np.zeros_like(R)
    mr = (maxc == R) & (delta > 0)
    mg = (maxc == G) & (delta > 0)
    mb = (maxc == B) & (delta > 0)
    hue[mr] = ((G[mr] - B[mr]) / safe[mr]) % 6
    hue[mg] = ((B[mg] - R[mg]) / safe[mg]) + 2
    hue[mb] = ((R[mb] - G[mb]) / safe[mb]) + 4
    hue = hue * 60.0

    # Yellow pisang area (hue 40-70°, saturasi cukup, value cukup)
    yellow_n = int(((sat > 0.40) & (val > 0.45) & (hue >= 40) & (hue < 70)).sum())
    if yellow_n < 400:
        return {"decay_ratio": 0.0, "yellow_n": yellow_n, "decay_n": 0}

    # WHITE MOLD ONLY: saturasi sangat rendah, value sedang (bukan pure white BG, bukan kayu)
    # Wood: sat > 0.20 → ter-exclude
    # Pure white BG: val > 0.88 → ter-exclude
    # Skin tone: sat > 0.20 → ter-exclude
    white_mold = (sat < 0.12) & (val > 0.60) & (val < 0.85)
    decay_n = int(white_mold.sum())

    # Safety: kalau "decay" >> yellow, kemungkinan deteksi BG bukan mold real
    if decay_n > yellow_n * 1.5:
        return {"decay_ratio": 0.0, "yellow_n": yellow_n, "decay_n": decay_n}

    ratio = decay_n / max(yellow_n + decay_n, 1)
    return {"decay_ratio": float(ratio), "yellow_n": yellow_n, "decay_n": decay_n}


def predict_freshness(img, model, class_names):
    arr = preprocess_image(img)
    probs = model.predict(arr, verbose=0)[0]
    n = len(probs)
    if n == 6:
        name_to_idx = {v: k for k, v in class_names.items()}
        fruit_scores = {}
        for fk in ["banana", "orange", "tomato"]:
            fn, rn = f"fresh{fk}", f"rotten{fk}"
            pf = probs[name_to_idx[fn]] if fn in name_to_idx else 0
            pr = probs[name_to_idx[rn]] if rn in name_to_idx else 0
            fruit_scores[fk] = pf + pr
        detected = max(fruit_scores, key=fruit_scores.get)

        # ── TARGETED COLOR RESCUE: hanya untuk konflik orange↔tomato ──
        # Hue red (0-18°) vs orange (18-45°) sangat dekat sehingga model
        # rentan keliru antara tomat & jeruk. Kita check rasio warna di
        # antara pixel buah (daun di-exclude) untuk koreksi.
        # Banana TIDAK disentuh — supaya pisang busuk tetap akurat.
        sig = color_signature(img)
        color_override = False

        if detected == "orange":
            # Model bilang Jeruk tapi gambar dominan MERAH → kemungkinan tomat
            if (sig["fruit_n"] >= 200
                and sig["red_ratio"] > 0.45
                and sig["red_ratio"] > sig["orange_ratio"] * 1.4):
                detected = "tomato"
                color_override = "red_to_tomato"
        elif detected == "tomato":
            # Model bilang Tomat tapi gambar dominan ORANGE → kemungkinan jeruk
            if (sig["fruit_n"] >= 200
                and sig["orange_ratio"] > 0.45
                and sig["orange_ratio"] > sig["red_ratio"] * 1.4):
                detected = "orange"
                color_override = "orange_to_orange"

        emoji, label = FRUIT_INFO.get(detected, ("🌿", "Produk"))
        fn, rn = f"fresh{detected}", f"rotten{detected}"
        p_fresh = probs[name_to_idx.get(fn, 0)]
        p_rotten = probs[name_to_idx.get(rn, 1)]
        total = p_fresh + p_rotten
        rotten_score = float(p_rotten / total) if total > 1e-9 else 0.5

        # ── BANANA DECAY RESCUE ──
        # Model training tidak punya white-mold / fungal banana, hanya brown spots.
        # Detect signal mold/dark rot via color heuristic dan override rotten_score.
        if detected == "banana":
            decay = banana_decay_signal(img)
            if decay["decay_ratio"] > 0.25:
                # Pisang dengan >25% area decay → boost ke busuk
                rotten_score = max(rotten_score, 0.75)
            elif decay["decay_ratio"] > 0.15:
                # Pisang dengan 15-25% area decay → minimal hampir busuk
                rotten_score = max(rotten_score, 0.55)

        # Saat override aktif, model's freshness signal untuk target fruit
        # tidak reliable (karena model awalnya tidak pilih buah itu).
        # Pakai signal warna sebagai gantinya.
        if color_override == "red_to_tomato":
            if sig["bright_red_ratio"] > 0.25 and sig["dark_red_ratio"] < 0.15:
                rotten_score = 0.12   # tomat segar — merah terang dominan
            elif sig["dark_red_ratio"] > 0.30:
                rotten_score = 0.85   # tomat busuk — merah gelap/coklat dominan
            # else: keep model's rotten_score for tomato (mixed signal)
            conf = float(sig["red_ratio"])
        else:
            conf = float(fruit_scores[detected]) if detected in fruit_scores else 1.0

        probs_dict = {class_names[i]: float(probs[i]) for i in range(n)}

        # ── REJECT FALLBACK: foto bukan buah/sayur target ──
        # Kombinasi 3 signal: low confidence, no fruit-color pixels, ambigu top1↔top2
        # Hanya aktif kalau color_override TIDAK fire (kalau warna sudah kuat trust itu)
        sorted_scores = sorted(fruit_scores.values(), reverse=True)
        top1 = sorted_scores[0] if sorted_scores else 0.0
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = top1 - top2

        reject_signals = []
        if top1 < 0.40:
            reject_signals.append("low_confidence")
        if sig["fruit_n"] < 150:
            reject_signals.append("no_fruit_color")
        if margin < 0.10 and top1 < 0.55:
            reject_signals.append("ambiguous_top")

        if (not color_override) and reject_signals:
            return (
                "❓", "Tidak Terdeteksi", "-", "gray",
                "BUKAN BUAH/SAYUR TARGET",
                "Foto tidak mengandung Pisang, Jeruk, atau Tomat yang dapat terdeteksi dengan jelas. "
                "Coba foto dengan pencahayaan cukup, buah di tengah, dan background sederhana.",
                "Sistem ini dilatih spesifik untuk 3 jenis buah/sayur. "
                "Foto lain (orang, objek, makanan lain) akan dianggap tidak terdeteksi.",
                0.0, probs_dict, float(top1), "unknown"
            )
    else:
        emoji, label = "🌿", "Produk"
        rotten_score = float(probs[0]) if n == 1 else float(probs[1])
        probs_dict = {"fresh": 1.0 - rotten_score, "rotten": rotten_score}
        conf = 1.0
    # Track fruit_key untuk render_result (visual calibration jeruk)
    fruit_key = detected if n == 6 else "default"
    f_label, f_color, badge, rek, tips = rotten_score_to_label(rotten_score, label)
    return (emoji, label, f_label, f_color, badge, rek, tips, rotten_score, probs_dict, conf, fruit_key)



def rotten_score_to_label(score, label="Produk"):
    if score < 0.50:
        return ("Segar", "green", "LAYAK JUAL",
                f"{label} dalam kondisi prima dan layak jual. Simpan di tempat sejuk atau kulkas untuk menjaga kesegaran lebih lama.",
                "Simpan terpisah dari buah yang sudah matang untuk menghindari pematangan dini.")
    elif score < 0.65:
        return ("Hampir Busuk", "orange", "SEGERA JUAL / GUNAKAN",
                f"{label} mulai menunjukkan tanda-tanda penurunan kualitas. Segera jual atau gunakan dalam waktu dekat. Pertimbangkan memberikan diskon untuk mempercepat penjualan.",
                "Pisahkan dari produk segar agar tidak mempercepat pembusukan produk lainnya.")
    else:
        return ("Busuk", "red", "TIDAK LAYAK JUAL",
                f"{label} sudah tidak layak untuk dijual atau dikonsumsi. Pisahkan segera dari produk lainnya untuk mencegah kontaminasi.",
                "Manfaatkan sebagai kompos atau pupuk organik untuk mengurangi food waste.")


def img_to_bytes(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def compute_predictions(files):
    model = load_model()
    class_names = load_class_names()
    out = []
    for uf in files:
        try:
            img = Image.open(uf).convert("RGB")
            pred = predict_freshness(img, model, class_names)
            out.append({
                "name": getattr(uf, "name", "foto_kamera.jpg"),
                "img_bytes": img_to_bytes(img),
                "pred": pred,
            })
            st.session_state.scan_count += 1
        except Exception as e:
            st.error(f"Error memproses gambar: {e}")
    return out


def render_result(img_bytes, filename, pred, show_image=True):
    try:
        # Support pred 10 elemen (legacy) maupun 11 elemen (dengan fruit_key)
        if len(pred) == 11:
            (emoji, label, f_label, f_color, badge, rek, tips, rotten, probs_dict, conf, fruit_key) = pred
        else:
            (emoji, label, f_label, f_color, badge, rek, tips, rotten, probs_dict, conf) = pred
            fruit_key = "default"

        img = Image.open(io.BytesIO(img_bytes))

        # ── REJECT FALLBACK: foto bukan buah/sayur target ──
        if fruit_key == "unknown":
            if show_image:
                col_img, col_res = st.columns([1, 1.2], gap="medium")
                with col_img:
                    st.image(img, use_container_width=True)
            else:
                col_res = st.container()
            with col_res:
                st.markdown(f"""
                <div class='result-header' style='border-color: rgba(180,180,180,0.3);'>
                    <div class='fruit-emoji-big'>{emoji}</div>
                    <div>
                        <p class='fruit-label'>{label}</p>
                        <p class='fruit-sub'>Top-1 model: {conf*100:.1f}% (di bawah threshold)</p>
                    </div>
                </div>
                <div class='status-badge warn'>
                    <div class='status-icon'>!</div>
                    <div>
                        <div class='status-text-main'>{badge}</div>
                        <div class='status-text-sub'>Sistem hanya mendukung Pisang, Jeruk, dan Tomat</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with st.expander("Detail probabilitas model (untuk transparansi)"):
                n_col = min(len(probs_dict), 3)
                cols = st.columns(n_col)
                for i, (cls_name, prob) in enumerate(sorted(probs_dict.items(), key=lambda x: -x[1])):
                    with cols[i % n_col]:
                        st.metric(cls_name, f"{prob*100:.1f}%")
            st.warning(f"**🔍 Info —** {rek}")
            st.info(f"💡 **Catatan —** {tips}")
            return

        # Visual calibration KHUSUS jeruk: model rentan memberi rotten_score
        # ~0.5 untuk jeruk lokal (kulit hijau-oren). Visual 50/50 unconvincing
        # meski label "Segar" benar. Remap visual saja, label & threshold tetap.
        if fruit_key == "orange" and rotten < 0.50:
            # Map [0, 0.50] → [0, 0.40] → bar tampil lebih confident sebagai Segar
            rotten_display = rotten * 0.80
        else:
            rotten_display = rotten
        fresh_display = 1.0 - rotten_display

        icon = {"green": "✓", "orange": "⚠", "red": "✕"}[f_color]
        cls  = {"green": "ok", "orange": "warn", "red": "err"}[f_color]
        if show_image:
            col_img, col_res = st.columns([1, 1.2], gap="medium")
            with col_img:
                st.image(img, use_container_width=True)
        else:
            col_res = st.container()
        with col_res:
            st.markdown(f"""
            <div class='result-header'>
                <div class='fruit-emoji-big'>{emoji}</div>
                <div>
                    <p class='fruit-label'>{label}</p>
                    <p class='fruit-sub'>Confidence: {conf*100:.1f}%</p>
                </div>
            </div>
            <div class='status-badge {cls}'>
                <div class='status-icon'>{icon}</div>
                <div>
                    <div class='status-text-main'>{badge}</div>
                    <div class='status-text-sub'>Status: {f_label}</div>
                </div>
            </div>
            <div class='prob-row'>
                <span class='prob-label'>🟢 Skor Kesegaran</span>
                <span class='prob-value'>{fresh_display*100:.1f}%</span>
            </div>
            <div class='vs-bar-track'><div class='vs-bar-fill fresh' style='width: {fresh_display*100:.2f}%;'></div></div>
            <div class='prob-row'>
                <span class='prob-label'>🔴 Skor Pembusukan</span>
                <span class='prob-value'>{rotten_display*100:.1f}%</span>
            </div>
            <div class='vs-bar-track'><div class='vs-bar-fill rotten' style='width: {rotten_display*100:.2f}%;'></div></div>
            """, unsafe_allow_html=True)
        with st.expander("Detail probabilitas semua kelas"):
            n_col = min(len(probs_dict), 3)
            cols = st.columns(n_col)
            for i, (cls_name, prob) in enumerate(sorted(probs_dict.items(), key=lambda x: -x[1])):
                with cols[i % n_col]:
                    st.metric(cls_name, f"{prob*100:.1f}%")
        if f_color == "green":
            st.success(f"**💬 Rekomendasi —** {rek}")
        elif f_color == "orange":
            st.warning(f"**💬 Rekomendasi —** {rek}")
        else:
            st.error(f"**💬 Rekomendasi —** {rek}")
        st.info(f"💡 **Tips —** {tips}")
    except Exception as e:
        st.error(f"Error menampilkan {filename}: {e}")


apply_theme(st.session_state.dark_mode)
is_dark = st.session_state.dark_mode

# Theme toggle — pakai st.toggle (native) supaya tampil sebagai pill switch
col_spacer, col_toggle = st.columns([8, 1.2])
with col_toggle:
    toggled = st.toggle(
        ("🌙 Dark" if is_dark else "☀️ Light"),
        value=is_dark, key="theme_cb"
    )
    if toggled != is_dark:
        st.session_state.dark_mode = toggled
        st.rerun()

st.markdown(f"""
<div class='hero-wrap'>
    <div class='hero-badge'>
        <span class='dot'></span>
        <span>POWERED BY DEEP LEARNING</span>
    </div>
    <h1 class='hero-title'>🍃 VigorScan</h1>
    <p class='hero-sub'>
        Deteksi kesegaran buah &amp; sayur dari foto menggunakan AI<br>
        <span style='font-size:13px; opacity:0.7;'>Solusi mengurangi food waste untuk pedagang &amp; distributor</span>
    </p>
    <div class='hero-chips'>
        <span class='chip'>🍌 Pisang</span>
        <span class='chip'>🍊 Jeruk</span>
        <span class='chip'>🍅 Tomat</span>
    </div>
    <div style='margin-top:18px; display:flex; justify-content:center; gap:24px; flex-wrap:wrap;
                font-size:12px; opacity:0.7; font-family: "JetBrains Mono", monospace;'>
        <span>📊 {st.session_state.scan_count} scan sesi ini</span>
        <span>·</span>
        <span>🎯 6 kelas terlatih</span>
        <span>·</span>
        <span>⚡ EfficientNetV2S</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_camera = st.tabs(["📁  Upload Foto", "📷  Kamera Langsung"])

with tab_upload:
    st.markdown("##### Upload satu atau beberapa foto sekaligus")
    uploaded_files = st.file_uploader(
        "Pilih foto buah atau sayur (JPG / PNG / JPEG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Bisa upload banyak foto sekaligus untuk deteksi batch",
        label_visibility="collapsed",
        key=f"uploader_main_{st.session_state.uploader_nonce}"
    )
    if uploaded_files:
        n = len(uploaded_files)
        col_info, col_btn = st.columns([2, 1])
        with col_info:
            st.caption(f"📸 **{n} foto** dipilih — siap untuk dianalisis")
        with col_btn:
            run_btn = st.button("🔍 Deteksi Sekarang", type="primary",
                                use_container_width=True, key="btn_upload")
        if run_btn:
            with st.spinner(f"Menganalisis {n} foto..."):
                st.session_state.upload_results = compute_predictions(uploaded_files)
    elif not st.session_state.upload_results:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>📸</div>
            <p class='empty-title'>Drag &amp; drop atau klik untuk memilih foto</p>
            <p class='empty-sub'>Pisang 🍌 · Jeruk 🍊 · Tomat 🍅 &nbsp;—&nbsp; bisa multi-foto sekaligus</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.upload_results:
        n_res = len(st.session_state.upload_results)
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.markdown(
                f"<p style='font-size:13px; opacity:0.7; margin:14px 0 0 0;'>"
                f"📊 <b>Hasil deteksi:</b> {n_res} foto</p>",
                unsafe_allow_html=True
            )
        with col_h2:
            if st.button("🗑️ Bersihkan", use_container_width=True, key="clear_upload"):
                st.session_state.upload_results = []
                st.session_state.uploader_nonce += 1
                st.rerun()
        for i, r in enumerate(st.session_state.upload_results):
            st.markdown("<hr>", unsafe_allow_html=True)
            if n_res > 1:
                st.markdown(
                    f"<p style='font-size:13px; opacity:0.7; margin:0 0 8px 0;'>"
                    f"<b>Foto {i+1} / {n_res}</b> &nbsp;·&nbsp; <code>{r['name']}</code></p>",
                    unsafe_allow_html=True
                )
            render_result(r["img_bytes"], r["name"], r["pred"], show_image=True)

with tab_camera:
    st.markdown("##### Ambil foto langsung dengan kamera")
    st.caption("Arahkan kamera ke buah/sayur, lalu klik ikon kamera untuk mengambil foto.")
    camera_image = st.camera_input(
        "foto_kamera",
        label_visibility="collapsed",
        key=f"cam_main_{st.session_state.camera_nonce}"
    )
    if camera_image is not None:
        st.markdown("")
        if st.button("🔍 Deteksi Sekarang", type="primary",
                     use_container_width=True, key="btn_camera"):
            with st.spinner("Menganalisis foto..."):
                res = compute_predictions([camera_image])
                st.session_state.camera_result = res[0] if res else None
    if st.session_state.camera_result is not None:
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.markdown(
                "<p style='font-size:13px; opacity:0.7; margin:14px 0 0 0;'>"
                "📊 <b>Hasil deteksi terakhir</b></p>",
                unsafe_allow_html=True
            )
        with col_h2:
            if st.button("🗑️ Bersihkan", use_container_width=True, key="clear_camera"):
                st.session_state.camera_result = None
                st.session_state.camera_nonce += 1
                st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        r = st.session_state.camera_result
        render_result(r["img_bytes"], r["name"], r["pred"], show_image=False)

st.markdown("""
<div class='footer-wrap'>
    <p class='footer-text'>
        <b>VigorScan v2.1</b> &nbsp;·&nbsp; Final Project AI/ML Kelompok 8 &nbsp;·&nbsp;
        Reducing food waste with computer vision
    </p>
    <p class='footer-tech'>
        EfficientNetV2S · Transfer Learning · TensorFlow · Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
