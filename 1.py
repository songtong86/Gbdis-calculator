# -*- coding: utf-8 -*-
"""
Streamlit app: GBDT risk prediction with TreeSHAP Force Plot.
Layout strictly following the minimal single-column style from top medical journals.
"""
import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Plot style (journal-compatible)
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 120,
})

# ============================================================
# Paths and constants
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "GradientBoost.pkl"

FEATURE_ORDER = ["Hsp90α", "MCV", "LMR", "ALB"]
FEATURE_LABELS = {
    "Hsp90α": "90α (HSP90A)",
    "MCV": "MCV (fL)",
    "LMR": "LMR",
    "ALB": "ALB (g/L)",
}
LOCKED_THRESHOLD = 0.50


# ============================================================
# Model loading
# ============================================================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def get_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return load_model(str(MODEL_PATH))


def get_explainer(model):
    return shap.TreeExplainer(model)


model = get_model()
explainer = get_explainer(model)


def build_input(hsp90a, mcv, lmr, alb):
    return pd.DataFrame([[hsp90a, mcv, lmr, alb]], columns=FEATURE_ORDER)


def compute_shap(x_df):
    sv = explainer.shap_values(x_df)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    shap_vals = sv.reshape(-1).astype(float)

    ev = np.asarray(explainer.expected_value).reshape(-1)
    base_val = float(ev[1]) if ev.size > 1 else float(ev[0])
    return shap_vals, base_val


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="GBDT Risk Prediction",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 自定义 CSS 让按钮看起来更像文献里的样式（白底红框）
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ffffff;
        color: #ff4b4b;
        border: 2px solid #ff4b4b;
        border-radius: 4px;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff4b4b;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("GBDT Risk Prediction Model")
st.caption(
    "Gradient Boosting classifier · TreeSHAP attribution in log-odds space · "
    f"locked threshold = {LOCKED_THRESHOLD:.2f}"
)

# ============================================================
# Input Form (Single Column)
# ============================================================
with st.form("predict_form"):
    hsp90a = st.number_input(
        "90α (HSP90A)",
        min_value=0.0, max_value=2000.0, value=50.0, step=0.1, format="%.4f",
    )
    mcv = st.number_input(
        "MCV (fL)",
        min_value=40.0, max_value=130.0, value=90.0, step=0.1,
    )
    lmr = st.number_input(
        "LMR",
        min_value=0.0, max_value=50.0, value=4.0, step=0.01, format="%.4f",
    )
    alb = st.number_input(
        "ALB (g/L)",
        min_value=10.0, max_value=70.0, value=42.0, step=0.1,
    )

    st.write("")  # 添加一点间距
    # 按钮左对齐
    submitted = st.form_submit_button("Predict")

# ============================================================
# Results (Single Column)
# ============================================================
if submitted:
    X_df = build_input(hsp90a, mcv, lmr, alb)
    prob = float(model.predict_proba(X_df)[0, 1])

    # 1. 大号斜体结论字 (对标文献)
    st.markdown(
        f"<h3 style='font-style: italic; font-weight: normal; margin-top: 30px; margin-bottom: 20px;'>"
        f"Based on feature values, predicted possibility of Positive Class is {prob * 100:.2f}%"
        f"</h3>",
        unsafe_allow_html=True
    )

    # 2. 横向全宽的 SHAP 力图 (Force Plot)
    shap_vals, base_val = compute_shap(X_df)

    # 为了让力图里的特征名显示得更美观，我们将特征名替换为全称
    display_X = X_df.copy()
    display_X.columns = [FEATURE_LABELS[f] for f in FEATURE_ORDER]

    # 绘制力图，让 shap 自己决定画布尺寸
    shap.force_plot(
        base_val,
        shap_vals,
        display_X.iloc[0],
        matplotlib=True,
        show=False,
        text_rotation=0,
        figsize=(14, 3.5)  # 给一个稍宽的比例
    )

    # 捕获画布
    fig = plt.gcf()

    # === 核心修复：通过保存为图片流，强制应用 bbox_inches='tight' ===
    import io
    from PIL import Image

    buf = io.BytesIO()
    # 这一步会自动把所有超出的文字完美包容进来
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)

    # 在网页上渲染图片
    st.image(img, use_container_width=True)

    # 清理画布
    plt.close(fig)

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "Research use and clinical decision support only. This tool does not "
    "constitute a medical diagnosis."
)
