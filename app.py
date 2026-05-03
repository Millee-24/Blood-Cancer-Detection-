import streamlit as st
import io
import warnings
warnings.filterwarnings("ignore")
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.patches as mpatches
from PIL import Image


try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


st.set_page_config(
    page_title="Blood Cancer Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
/* ── Main header ── */
.main-header {
    background: linear-gradient(135deg, #1a2a4a 0%, #0d3b6e 100%);
    padding: 1.4rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    border-left: 5px solid #4fc3f7;
}
.main-header h1 {
    color: #e3f2fd;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
}
.main-header p {
    color: #90caf9;
    font-size: 0.88rem;
    margin: 4px 0 0;
}

/* ── Section cards ── */
.section-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Result boxes ── */
.result-leukemia {
    background: #fff5f5;
    border: 2px solid #e53e3e;
    border-radius: 10px;
    padding: 1.4rem;
    text-align: center;
}
.result-leukemia h2 { color: #c53030; font-size: 1.6rem; margin: 0; }
.result-leukemia p  { color: #742a2a; font-size: 1rem; margin: 6px 0 0; }

.result-healthy {
    background: #f0fff4;
    border: 2px solid #38a169;
    border-radius: 10px;
    padding: 1.4rem;
    text-align: center;
}
.result-healthy h2 { color: #276749; font-size: 1.6rem; margin: 0; }
.result-healthy p  { color: #1c4532; font-size: 1rem; margin: 6px 0 0; }

/* ── Warning banner ── */
.warning-banner {
    background: #fffbeb;
    border: 1px solid #f6ad55;
    border-left: 4px solid #ed8936;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #744210;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #ebf8ff;
    border: 1px solid #bee3f8;
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
}
.metric-card .label { color: #2b6cb0; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; }
.metric-card .value { color: #1a365d; font-size: 1.5rem; font-weight: 700; }

/* ── Model selector ── */
.best-model-tag {
    display: inline-block;
    background: #ebf8ff;
    color: #2b6cb0;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid #bee3f8;
    border-radius: 12px;
    padding: 2px 10px;
    margin-left: 8px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1a2a4a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #90caf9 !important; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="main-header">
    <h1>🔬 Blood Cancer Detection</h1>
    <p>
        Automated Leukaemia Detection via Comparative Deep Learning &nbsp;|&nbsp;
        CNMC 2019 Dataset &nbsp;
    </p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.markdown("---")
    page = st.radio(
        "Go to section",
        options=[
            "📊 Model Performance Analytics",
            "🖼️ Image Preprocessing Pipeline",
            "🩺 Live Patient Diagnosis",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This system compares four CNN transfer-learning architectures
    trained on the **CNMC 2019** leukaemia dataset under
    **3-Fold Cross-Validation**.

    **Models evaluated:**
    - EfficientNetB0 ⭐
    - VGG16
    - ResNet50
    - InceptionV3

    **Evaluation metrics:**
    - Overall Accuracy
    - Leukemia-class Recall
    """)
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#718096;'>v1.0 · For academic use only</p>",
        unsafe_allow_html=True,
    )



TARGET_MEAN = np.array([70.27, 30.36, 80.48], dtype=np.float32)
TARGET_STD  = np.array([11.07,  8.77,  9.15], dtype=np.float32)

def stain_normalize(img_rgb: np.ndarray) -> np.ndarray:
    img_f = img_rgb.astype(np.float32)
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask  = gray > 15
    for c in range(3):
        ch = img_f[:, :, c]
        pix = ch[mask]
        if len(pix) < 50:
            continue
        src_mean = pix.mean()
        src_std  = pix.std() + 1e-6
        ch_norm  = (ch - src_mean) / src_std
        img_f[:, :, c] = ch_norm * TARGET_STD[c] + TARGET_MEAN[c]
    img_f = np.clip(img_f, 0, 255).astype(np.uint8)
    img_f[~mask] = [0, 0, 0]
    return img_f

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def unsharp_mask(img_rgb: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=1.0)
    sharp   = cv2.addWeighted(img_rgb, 1.4, blurred, -0.4, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def bilateral_denoise(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img_rgb, d=5, sigmaColor=30, sigmaSpace=30)


def full_preprocess(pil_image: Image.Image, size: int = 224) -> tuple:
    img_rgb  = np.array(pil_image.convert("RGB"))
    original = img_rgb.copy()


    img = stain_normalize(img_rgb)


    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


    img = apply_clahe(img)


    img = unsharp_mask(img)

    img = bilateral_denoise(img)


    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[gray <= 15] = [0, 0, 0]

    processed = img.copy()


    model_input = img.astype(np.float32)

    return original, processed, model_input

@st.cache_resource
def load_efficientnet(model_path: str = "efficientnetb0_fold_1.h5"):
    if not TF_AVAILABLE:
        return None, "TensorFlow is not installed."
    if not os.path.exists(model_path):
        return None, f"Model file not found: '{model_path}'"
    try:
        model = load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


SAMPLES_DIR = "samples"
def get_sample_files() -> dict:
    """Return {label: path} for images inside /samples folder."""
    if not os.path.isdir(SAMPLES_DIR):
        return {}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([
        f for f in os.listdir(SAMPLES_DIR)
        if os.path.splitext(f)[1].lower() in exts
    ])
    return {f"Sample {i+1} — {f}": os.path.join(SAMPLES_DIR, f)
            for i, f in enumerate(files)}


if page == "📊 Model Performance Analytics":

    st.markdown("## 📊 Model Performance Analytics")
    st.markdown(
        "Comparison of four CNN transfer-learning architectures evaluated "
        "under **3-Fold Cross-Validation** on the CNMC 2019 leukaemia dataset."
    )


    fold_data = {
        "EfficientNetB0": {
            "acc":    [84.41, 84.61, 80.92],
            "recall": [90.0,  93.0,  96.0],
            "prec":   [88.0,  86.0,  80.0],
            "f1":     [89.0,  89.0,  87.0],
        },
        "VGG16": {
            "acc":    [80.86, 83.66, 77.82],
            "recall": [83.0,  91.0,  86.0],
            "prec":   [88.0,  86.0,  83.0],
            "f1":     [86.0,  88.0,  84.0],
        },
        "ResNet50": {
            "acc":    [78.23, 81.88, 79.12],
            "recall": [75.0,  81.0,  89.0],
            "prec":   [91.0,  91.0,  82.0],
            "f1":     [82.0,  86.0,  85.0],
        },
        "InceptionV3": {
            "acc":    [77.29, 83.19, 75.77],
            "recall": [76.0,  87.0,  83.0],
            "prec":   [89.0,  88.0,  82.0],
            "f1":     [82.0,  87.0,  83.0],
        },
    }

    model_names = ["EfficientNetB0", "VGG16", "ResNet50", "InceptionV3"]

    def avg(lst):
        return round(sum(lst) / len(lst), 2)

    def sd(lst):
        m = avg(lst)
        return round((sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5, 2)


    col1, col2, col3, col4 = st.columns(4)
    tags = ["⭐ Best", "", "", ""]

    for col, name, tag in zip([col1, col2, col3, col4], model_names, tags):
        d = fold_data[name]
        mean_acc = avg(d["acc"])
        mean_rec = avg(d["recall"])
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{name} {tag}</div>
                <div class="value">{mean_acc}%</div>
                <div class="label" style="margin-top:4px;">Avg Accuracy</div>
                <div class="value" style="font-size:1.1rem;color:#276749;">{mean_rec}%</div>
                <div class="label">Avg Leukemia Recall</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("### 📂 Fold-by-Fold Results (All Models)")

    import pandas as pd

    fold_rows = []
    for name in model_names:
        d = fold_data[name]
        for i in range(3):
            fold_rows.append({
                "Model":          name,
                "Fold":           f"Fold {i + 1}",
                "Accuracy (%)":   d["acc"][i],
                "Recall (%)":     d["recall"][i],
                "Precision (%)":  d["prec"][i],
                "F1-Score (%)":   d["f1"][i],
            })

    fold_df = pd.DataFrame(fold_rows)
    st.dataframe(fold_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("### 📈 Fold-wise Accuracy Trend")

    colors_line = {
        "EfficientNetB0": "#1a56db",
        "VGG16":          "#e53e3e",
        "ResNet50":       "#d97706",
        "InceptionV3":    "#7c3aed",
    }

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    fig2.patch.set_facecolor("#f8fafc")
    ax2.set_facecolor("#f8fafc")

    for name in model_names:
        vals = fold_data[name]["acc"]
        ax2.plot([1, 2, 3], vals,
                 marker="o", linewidth=2.2, markersize=7,
                 label=name, color=colors_line[name])
        for xi, yi in zip([1, 2, 3], vals):
            ax2.annotate(f"{yi}%", (xi, yi),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color=colors_line[name])

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["Fold 1", "Fold 2", "Fold 3"], fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="600", color="#2d3748")
    ax2.set_title("Per-Fold Accuracy — All Models",
                  fontsize=11, fontweight="700", color="#1a202c", pad=12)
    ax2.set_ylim(70, 92)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("### 📈 Fold-wise Leukemia Recall Trend")

    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    fig3.patch.set_facecolor("#f8fafc")
    ax3.set_facecolor("#f8fafc")

    for name in model_names:
        vals = fold_data[name]["recall"]
        ax3.plot([1, 2, 3], vals,
                 marker="s", linewidth=2.2, markersize=7,
                 label=name, color=colors_line[name])
        for xi, yi in zip([1, 2, 3], vals):
            ax3.annotate(f"{yi}%", (xi, yi),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color=colors_line[name])

    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(["Fold 1", "Fold 2", "Fold 3"], fontsize=10)
    ax3.set_ylabel("Leukemia Recall (%)", fontsize=10, fontweight="600", color="#2d3748")
    ax3.set_title("Per-Fold Leukemia Recall — All Models",
                  fontsize=11, fontweight="700", color="#1a202c", pad=12)
    ax3.set_ylim(65, 102)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax3.set_axisbelow(True)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


    st.markdown("### 📊 Average Performance Comparison")

    accuracy = [avg(fold_data[m]["acc"])    for m in model_names]
    recall   = [avg(fold_data[m]["recall"]) for m in model_names]
    x        = np.arange(len(model_names))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    bars_acc = ax.bar(
        x - width / 2, accuracy, width,
        color=["#1a56db" if m == "EfficientNetB0" else "#93c5fd" for m in model_names],
        edgecolor="white", linewidth=0.8, zorder=3
    )
    bars_rec = ax.bar(
        x + width / 2, recall, width,
        color=["#15803d" if m == "EfficientNetB0" else "#86efac" for m in model_names],
        edgecolor="white", linewidth=0.8, zorder=3
    )

    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=9,
                fontweight="600", color="#1e3a5f")

    for bar in bars_rec:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=9,
                fontweight="600", color="#14532d")

    ax.set_xlabel("CNN Architecture", fontsize=11, fontweight="600", color="#2d3748")
    ax.set_ylabel("Score (%)",        fontsize=11, fontweight="600", color="#2d3748")
    ax.set_title(
        "Average Accuracy vs. Leukemia Recall (3-Fold CV)",
        fontsize=12, fontweight="700", color="#1a202c", pad=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, color="#2d3748")
    ax.set_ylim(60, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4a5568")
    for spine in ax.spines.values():
        spine.set_visible(False)

    patch_acc = mpatches.Patch(color="#1a56db", label="Avg Accuracy (%)")
    patch_rec = mpatches.Patch(color="#15803d", label="Avg Leukemia Recall (%)")
    ax.legend(handles=[patch_acc, patch_rec], loc="lower right", fontsize=9, framealpha=0.9)
    ax.annotate("★ Best Model",
                xy=(x[0], accuracy[0] + 2.5),
                fontsize=9, ha="center", color="#1a56db", fontweight="700")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    st.markdown("""
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #1a56db;
            border-radius:8px;padding:1rem 1.2rem;margin-top:0.5rem;">
    <strong style="color:#1e40af;">🩺 Why EfficientNetB0 was selected as the deployment model</strong><br><br>
    <span style="color:#1e3a5f;">
    In leukemia screening, a <strong style="color:#1e3a5f;">False Negative</strong> — failing to
    detect a leukemia cell in a patient who actually has the disease — is a clinically fatal
    error that delays treatment and worsens prognosis significantly. A
    <strong style="color:#1e3a5f;">False Positive</strong>, while causing unnecessary anxiety,
    can be resolved through follow-up tests.<br><br>
    Because of this clinical asymmetry, <strong style="color:#1e3a5f;">Recall for the leukemia
    class</strong> is the primary selection criterion — not overall accuracy. EfficientNetB0
    achieves an average leukemia recall of <strong style="color:#1e3a5f;">93.0%</strong> across
    all three folds (Fold 1: 90%, Fold 2: 93%, Fold 3: 96%), while also delivering the highest
    average accuracy of <strong style="color:#1e3a5f;">83.31%</strong>. Its compact architecture
    (~5.3M parameters) makes it additionally suitable for deployment in resource-limited clinical
    environments such as district hospitals or point-of-care settings.
    </span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("### 📋 Final Aggregate Results")

    summary_df = pd.DataFrame({
        "Architecture":      [f"{m} ⭐" if m == "EfficientNetB0" else m for m in model_names],
        "Parameters (M)":    ["5.3", "138", "25.6", "23.9"],
        "Avg Accuracy (%)":  [avg(fold_data[m]["acc"])    for m in model_names],
        "Std Dev (Acc)":     [sd(fold_data[m]["acc"])     for m in model_names],
        "Avg Recall (%)":    [avg(fold_data[m]["recall"]) for m in model_names],
        "Avg Precision (%)": [avg(fold_data[m]["prec"])   for m in model_names],
        "Avg F1-Score (%)":  [avg(fold_data[m]["f1"])     for m in model_names],
        "Validation":        ["3-Fold CV"] * 4,
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


elif page == "🖼️ Image Preprocessing Pipeline":

    st.markdown("## 🖼️ Image Preprocessing Pipeline")
    st.markdown(
        "Upload a blood smear image to view the complete preprocessing pipeline "
        "applied before CNN classification. Each step is shown separately."
    )

    uploaded = st.file_uploader(
        "Upload a blood smear image (JPG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
        key="preprocess_uploader",
    )


    samples = get_sample_files()
    if not uploaded and samples:
        st.markdown("**Or choose a sample image:**")
        chosen_label = st.selectbox(
            "Sample files", options=["— Select —"] + list(samples.keys()),
            key="preprocess_sample"
        )
        if chosen_label != "— Select —":
            with open(samples[chosen_label], "rb") as f:
                uploaded = io.BytesIO(f.read())

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_np  = np.array(pil_img)

        st.markdown("---")
        st.markdown("### Step-by-step visualisation")


        step1 = stain_normalize(img_np)
        step2 = cv2.resize(step1, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        lab   = cv2.cvtColor(step2, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(lab)
        l     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        step3 = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
        blurred = cv2.GaussianBlur(step3, (3,3), sigmaX=1.0)
        step4 = np.clip(cv2.addWeighted(step3, 1.4, blurred, -0.4, 0), 0, 255).astype(np.uint8)
        step5 = cv2.bilateralFilter(step4, d=5, sigmaColor=30, sigmaSpace=30)
        step6 = step5.copy()
        gray  = cv2.cvtColor(step5, cv2.COLOR_RGB2GRAY)
        step6[gray <= 15] = [0, 0, 0]

        steps = [
            ("1. Original input",          img_np, "#e2e8f0"),
            ("2. Stain normalised",        step1,  "#bee3f8"),
            ("3. Resized (224×224)",       step2,  "#c6f6d5"),
            ("4. CLAHE enhanced",          step3,  "#e9d8fd"),
            ("5. Unsharp masked",          step4,  "#fefcbf"),
            ("6. Bilateral + final mask",  step6,  "#fed7d7"),
        ]

        cols = st.columns(3)
        for i, (label, img, bg) in enumerate(steps):
            with cols[i % 3]:
                st.markdown(
                    f"<p style='font-size:0.82rem;font-weight:600;"
                    f"background:{bg};padding:4px 8px;border-radius:6px;"
                    f"text-align:center;'>{label}</p>",
                    unsafe_allow_html=True,
                )
                st.image(img, use_container_width=True)


        st.markdown("---")
        st.markdown("### Before vs. After Comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original image**")
            st.image(img_np, use_container_width=True)
        with col_b:
            st.markdown("**Fully preprocessed image (model-ready)**")
            st.image(step5, use_container_width=True)


        st.markdown("---")
        st.markdown("### Pipeline summary")
        steps_info = [
            ("Stain normalisation",     "Per-channel mean/std correction to dataset-level target statistics"),
            ("Resizing",               "Lanczos4 interpolation → 224×224 px (VGG16 / EffNetB0 / ResNet50)"),
            ("CLAHE",                  "Clip=3.0, 8×8 tile grid on L* channel; improves nuclear contrast"),
            ("Unsharp masking",        "Original ×1.4 minus Gaussian blur ×0.4; sharpens cell boundaries"),
            ("Bilateral denoising",    "d=5, σ_colour=30, σ_space=30; removes noise, preserves edges"),
            ("Pixel normalisation",    "Raw float32 [0–255] — EfficientNetB0 handles rescaling internally"),
        ]
        for name, desc in steps_info:
            st.markdown(
                f"<div style='display:flex;gap:12px;padding:6px 0;border-bottom:1px solid #e2e8f0;'>"
                f"<span style='min-width:180px;font-weight:600;color:#2b6cb0;font-size:0.85rem;'>{name}</span>"
                f"<span style='color:#4a5568;font-size:0.85rem;'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    else:
        st.info("⬆️ Upload a blood smear image above to start the visualisation.")



elif page == "🩺 Live Patient Diagnosis":

    st.markdown("## 🩺 Live Patient Diagnosis")


    st.markdown("""
    <div class="warning-banner">
        ⚠️ <strong>Clinical Calibration Notice:</strong>
        This system is calibrated exclusively for <strong>100× magnification
        peripheral blood smear slides</strong> stained with the Jenner–Giemsa technique.
        Images acquired at other magnifications, with different staining protocols,
        or from non-haematological sources will produce unreliable results.
        This tool is intended to <em>assist</em> qualified haematologists — it does
        <strong>not</strong> replace clinical judgement or laboratory confirmation.
    </div>
    """, unsafe_allow_html=True)


    st.markdown("### 🗂️ Demo: Sample Patient Files")
    samples = get_sample_files()
    if samples:
        demo_options = ["— Select a sample patient file —"] + list(samples.keys())
        demo_choice  = st.selectbox("Choose from pre-loaded samples:", demo_options)
        demo_image   = None
        if demo_choice != demo_options[0]:
            with open(samples[demo_choice], "rb") as f:
                demo_image = f.read()
            st.success(f"✅ Loaded: {demo_choice}")
    else:
        st.info(
            "No sample files found. Create a folder named **`samples/`** in the same "
            "directory as `app.py` and add JPG/PNG blood smear images to enable demo mode."
        )
        demo_image = None

    st.markdown("---")


    st.markdown("### 📁 Upload Patient Image")
    uploaded_diag = st.file_uploader(
        "Upload a JPG or PNG blood smear slide image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="diagnosis_uploader",
    )


    active_image_bytes = None
    if uploaded_diag:
        active_image_bytes = uploaded_diag.read()
    elif demo_image:
        active_image_bytes = demo_image

    if active_image_bytes:
        pil_img = Image.open(io.BytesIO(active_image_bytes)).convert("RGB")

        col_img, col_pre = st.columns(2)
        with col_img:
            st.markdown("**Uploaded slide**")
            st.image(pil_img, use_container_width=True)
        with col_pre:
            st.markdown("**After preprocessing**")
            with st.spinner("Preprocessing…"):
                original_np, processed_np, model_input_np = full_preprocess(pil_img)
            st.image(processed_np, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔬 Run Diagnosis")

        run_btn = st.button("▶️ Run EfficientNetB0 Classification", type="primary")

        if run_btn:
            model, err = load_efficientnet()


            if err:

                st.warning(
                    f"Model not loaded ({err}). Showing a **simulated demo result** "
                    f"for presentation purposes."
                )

                brightness = np.mean(processed_np)

                prob = float(np.clip(1.0 - brightness / 180.0, 0.05, 0.97))
                predicted_class = 1 if prob > 0.5 else 0
                confidence      = prob if predicted_class == 1 else 1 - prob

                st.markdown(
                    "<p style='font-size:0.78rem;color:#718096;text-align:center;'>"
                    "⚠️ DEMO MODE — Place <code>efficientnetb0_leukemia.h5</code> "
                    "in the project directory for real predictions.</p>",
                    unsafe_allow_html=True,
                )

            else:
                with st.spinner("Running inference…"):
                    batch = np.expand_dims(model_input_np, axis=0).astype(np.float32)
                    raw   = model.predict(batch, verbose=0)
                    prob  = float(raw[0][0]) if raw.shape[-1] == 1 else float(raw[0][1])
                    predicted_class = 1 if prob > 0.5 else 0
                    confidence      = prob if predicted_class == 1 else 1 - prob


            st.markdown("<br>", unsafe_allow_html=True)
            if predicted_class == 1:
                st.markdown(f"""
                <div class="result-leukemia">
                    <h2>⚠️ LEUKAEMIA DETECTED</h2>
                    <p>Confidence: <strong>{confidence * 100:.1f}%</strong></p>
                    <p style="margin-top:10px;font-size:0.83rem;">
                        The model has identified this cell as a malignant lymphoblast.
                        <strong>Please refer to a qualified haematologist for confirmation.</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-healthy">
                    <h2>✅ HEALTHY CELL</h2>
                    <p>Confidence: <strong>{confidence * 100:.1f}%</strong></p>
                    <p style="margin-top:10px;font-size:0.83rem;">
                        The model has classified this cell as a normal haematogone (HEM).
                        Routine follow-up is advised.
                    </p>
                </div>
                """, unsafe_allow_html=True)


            st.markdown("<br>", unsafe_allow_html=True)
            col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
            with col_g2:
                fig_g, ax_g = plt.subplots(figsize=(4, 0.4))
                fig_g.patch.set_facecolor("none")
                color = "#e53e3e" if predicted_class == 1 else "#38a169"
                ax_g.barh([0], [confidence], color=color, height=0.4)
                ax_g.barh([0], [1],          color="#e2e8f0", height=0.4, zorder=0)
                ax_g.set_xlim(0, 1)
                ax_g.axis("off")
                ax_g.set_title(
                    f"Confidence: {confidence * 100:.1f}%",
                    fontsize=10, color="#2d3748", fontweight="600"
                )
                st.pyplot(fig_g)
                plt.close(fig_g)


            with st.expander("🔍 Technical details"):
                st.markdown(f"""
                | Parameter | Value |
                |---|---|
                | **Model** | EfficientNetB0 (ImageNet → CNMC fine-tuned) |
                | **Input size** | 224 × 224 px |
                | **Raw sigmoid output** | {prob:.4f} |
                | **Decision threshold** | 0.50 |
                | **Predicted class** | {'ALL (Malignant)' if predicted_class == 1 else 'HEM (Healthy)'} |
                | **Preprocessing steps** | Stain norm -> Resize 224 X 224 → CLAHE → Unsharp mask → Bilateral filter → Raw float32 input |
                | **Validation method** | 3-Fold Cross-Validation on CNMC 2019 |
                | **Training dataset** | CNMC 2019
                """)

    else:
        st.info(
            "Select a sample patient file from the dropdown above, "
            "or upload your own blood smear image to begin diagnosis."
        )
