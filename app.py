# app.py — UI ringkas: hanya fitur yang paling berpengaruh + opsi lanjutan
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn, imblearn

st.set_page_config(page_title="Heart Failure Prediction", page_icon="❤️", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1rem;}
.result-card {background:#f8fafc;border:1px solid #e6e9ef;border-radius:14px;padding:16px}
.kpi {border:1px dashed #e5e7eb;border-radius:12px;padding:10px 12px;background:#fff}
.header {font-weight:800; font-size:2.0rem; letter-spacing:.2px}
.subtle {color:#6b7280}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource(show_spinner=True)
def load_model():
    return joblib.load("model_heart.pkl")
model = load_model()

# ---------- SAMPLE DATA (opsional) ----------
@st.cache_data(show_spinner=False)
def load_sample():
    try:
        df = pd.read_csv("heart.csv")
        return df.copy()
    except Exception:
        return None
sample_df = load_sample()

# Ambil default statistik agar kolom tersembunyi punya nilai aman
def_stats = {}
if sample_df is not None:
    def_stats["RestingBP"] = int(sample_df["RestingBP"].median()) if "RestingBP" in sample_df else 120
    def_stats["Cholesterol"] = int(sample_df["Cholesterol"].median()) if "Cholesterol" in sample_df else 200
    def_stats["Age"] = int(sample_df["Age"].median()) if "Age" in sample_df else 54
    def_stats["MaxHR"] = int(sample_df["MaxHR"].median()) if "MaxHR" in sample_df else 150
    def_stats["Oldpeak"] = float(sample_df["Oldpeak"].median()) if "Oldpeak" in sample_df else 1.5
    # mode untuk kategorikal
    def_stats["Sex"] = sample_df["Sex"].mode()[0] if "Sex" in sample_df else "M"
    def_stats["ChestPainType"] = sample_df["ChestPainType"].mode()[0] if "ChestPainType" in sample_df else "ATA"
    def_stats["RestingECG"] = sample_df["RestingECG"].mode()[0] if "RestingECG" in sample_df else "Normal"
    def_stats["ExerciseAngina"] = sample_df["ExerciseAngina"].mode()[0] if "ExerciseAngina" in sample_df else "N"
    def_stats["ST_Slope"] = sample_df["ST_Slope"].mode()[0] if "ST_Slope" in sample_df else "Up"
else:
    def_stats = dict(RestingBP=120, Cholesterol=200, Age=54, MaxHR=150, Oldpeak=1.5,
                     Sex="M", ChestPainType="ATA", RestingECG="Normal", ExerciseAngina="N", ST_Slope="Up")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    threshold = st.slider("Threshold (positif)", 0.10, 0.90, 0.50, 0.01)
    st.markdown("---")
    st.markdown("### ℹ️ Info Model")
    st.caption(f"sklearn: **{sklearn.__version__}**, imblearn: **{imblearn.__version__}**")
    st.caption("Fitur utama: Oldpeak, MaxHR, Age, FastingBS, Cholesterol (hasil heatmap).")
    st.markdown("---")
    st.markdown("### 📦 File")
    st.caption("• `model_heart.pkl` (wajib)  •  `heart.csv` (opsional untuk sample/testing)")

st.markdown("<div class='header'>❤️ Heart Failure Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Pakai fitur inti (paling berpengaruh) + opsi lanjutan bila perlu</div>", unsafe_allow_html=True)

# ---------- Helper ----------
CORE_FEATURES = ["Oldpeak", "MaxHR", "Age", "FastingBS", "Cholesterol"]  # dari heatmap (|corr| terbesar)
ADV_FEATURES = ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope","RestingBP"]  # disembunyikan default

def build_full_row(core: dict, adv: dict):
    """Gabungkan input core + advanced menjadi 1 row lengkap sesuai kolom training."""
    row = {
        # inti
        "Oldpeak": float(core["Oldpeak"]),
        "MaxHR": int(core["MaxHR"]),
        "Age": int(core["Age"]),
        "FastingBS": int(core["FastingBS"]),
        "Cholesterol": int(core["Cholesterol"]),
        # lanjutan (kolom yang tidak tampil di heatmap tapi dibutuhkan model)
        "Sex": adv.get("Sex", def_stats["Sex"]),
        "ChestPainType": adv.get("ChestPainType", def_stats["ChestPainType"]),
        "RestingECG": adv.get("RestingECG", def_stats["RestingECG"]),
        "ExerciseAngina": adv.get("ExerciseAngina", def_stats["ExerciseAngina"]),
        "ST_Slope": adv.get("ST_Slope", def_stats["ST_Slope"]),
        "RestingBP": int(adv.get("RestingBP", def_stats["RestingBP"])),
    }
    # urutan kolom bebas karena pipeline ColumnTransformer akan mencocokkan by-name
    return pd.DataFrame([row])

def predict_one(df_row: pd.DataFrame, thr: float):
    proba = model.predict_proba(df_row)[0, 1]
    return int(proba >= thr), proba

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔮 Single Predict (Fitur Inti)", "🧪 Sample 5 Baris & Batch Test"])

with tab1:
    st.markdown("#### Form (Fitur Inti)")
    # Prefill dari sample (opsional)
    core_defaults = dict(Oldpeak=def_stats["Oldpeak"], MaxHR=def_stats["MaxHR"],
                         Age=def_stats["Age"], FastingBS=1, Cholesterol=def_stats["Cholesterol"])
    adv_defaults  = dict(Sex=def_stats["Sex"], ChestPainType=def_stats["ChestPainType"],
                         RestingECG=def_stats["RestingECG"], ExerciseAngina=def_stats["ExerciseAngina"],
                         ST_Slope=def_stats["ST_Slope"], RestingBP=def_stats["RestingBP"])

    if sample_df is not None:
        with st.expander("📋 Prefill dari Sample (pilih 1 dari 5 baris pertama)"):
            head5 = sample_df.head(5)
            st.dataframe(head5, use_container_width=True, height=220)
            idx = st.number_input("Baris ke-", 0, 4, 0)
            if st.button("Isi Form dari Baris Ini"):
                s = head5.iloc[int(idx)]
                # isi core kalau tersedia
                core_defaults["Oldpeak"] = float(s.get("Oldpeak", core_defaults["Oldpeak"]))
                core_defaults["MaxHR"] = int(s.get("MaxHR", core_defaults["MaxHR"]))
                core_defaults["Age"] = int(s.get("Age", core_defaults["Age"]))
                core_defaults["FastingBS"] = int(s.get("FastingBS", 1))
                core_defaults["Cholesterol"] = int(s.get("Cholesterol", core_defaults["Cholesterol"]))
                # isi advanced
                for k in ADV_FEATURES:
                    if k in s:
                        adv_defaults[k] = s[k]
                st.success("Form telah diisi ulang dari sample.")

    c1, c2, c3 = st.columns(3)
    with c1:
        Oldpeak = st.number_input("Oldpeak", 0.0, 10.0, float(core_defaults["Oldpeak"]), 0.1)
        MaxHR = st.number_input("MaxHR", 60, 220, int(core_defaults["MaxHR"]))
    with c2:
        Age = st.number_input("Age", 18, 100, int(core_defaults["Age"]))
        FastingBS = st.selectbox("FastingBS", [0,1], index=int(core_defaults["FastingBS"]))
    with c3:
        Cholesterol = st.number_input("Cholesterol", 0, 700, int(core_defaults["Cholesterol"]))

    # ---- Advanced (opsional) ----
    with st.expander("⚙️ Fitur Lanjutan (opsional)"):
        ca, cb, cc = st.columns(3)
        with ca:
            Sex = st.selectbox("Sex", ["M","F"], index=0 if adv_defaults["Sex"]=="M" else 1)
            RestingECG = st.selectbox("RestingECG", ["Normal","ST","LVH"],
                                      index=["Normal","ST","LVH"].index(adv_defaults["RestingECG"]))
        with cb:
            ChestPainType = st.selectbox("ChestPainType", ["TA","ATA","NAP","ASY"],
                                         index=["TA","ATA","NAP","ASY"].index(adv_defaults["ChestPainType"]))
            ExerciseAngina = st.selectbox("ExerciseAngina", ["Y","N"],
                                          index=0 if adv_defaults["ExerciseAngina"]=="Y" else 1)
        with cc:
            ST_Slope = st.selectbox("ST_Slope", ["Up","Flat","Down"],
                                    index=["Up","Flat","Down"].index(adv_defaults["ST_Slope"]))
            RestingBP = st.number_input("RestingBP", 0, 250, int(adv_defaults["RestingBP"]))

    # Build rows
    core_inputs = dict(Oldpeak=Oldpeak, MaxHR=MaxHR, Age=Age, FastingBS=FastingBS, Cholesterol=Cholesterol)
    adv_inputs  = dict(Sex=Sex, ChestPainType=ChestPainType, RestingECG=RestingECG,
                       ExerciseAngina=ExerciseAngina, ST_Slope=ST_Slope, RestingBP=RestingBP)
    row = build_full_row(core_inputs, adv_inputs)

    if st.button("Prediksi", type="primary"):
        pred, proba = predict_one(row, threshold)
        label = "🧨 Heart Disease (1)" if pred==1 else "✅ Tidak (0)"
        st.markdown("#### Hasil Prediksi")
        cA, cB = st.columns(2)
        with cA:
            st.markdown(f"<div class='result-card'><b>Prediksi</b><br/>{label}</div>", unsafe_allow_html=True)
        with cB:
            st.markdown(f"<div class='result-card'><b>Probabilitas (positif)</b><br/>{proba*100:.2f}%</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("#### 5 Data Awal (dari heart.csv)")
    if sample_df is None:
        st.info("File `heart.csv` tidak ditemukan.")
    else:
        head5 = sample_df.head(5).copy()
        st.dataframe(head5, use_container_width=True, height=260)

        st.markdown("##### Prediksi Batch (5 Baris Pertama) — memakai threshold sidebar")
        # Susun full row untuk 5 baris (isi kolom yang mungkin hilang)
        rows = []
        for _, s in head5.iterrows():
            core = dict(
                Oldpeak=float(s.get("Oldpeak", def_stats["Oldpeak"])),
                MaxHR=int(s.get("MaxHR", def_stats["MaxHR"])),
                Age=int(s.get("Age", def_stats["Age"])),
                FastingBS=int(s.get("FastingBS", 1)),
                Cholesterol=int(s.get("Cholesterol", def_stats["Cholesterol"]))
            )
            adv = dict(
                Sex=s.get("Sex", def_stats["Sex"]),
                ChestPainType=s.get("ChestPainType", def_stats["ChestPainType"]),
                RestingECG=s.get("RestingECG", def_stats["RestingECG"]),
                ExerciseAngina=s.get("ExerciseAngina", def_stats["ExerciseAngina"]),
                ST_Slope=s.get("ST_Slope", def_stats["ST_Slope"]),
                RestingBP=int(s.get("RestingBP", def_stats["RestingBP"]))
            )
            rows.append(build_full_row(core, adv))
        Xb = pd.concat(rows, ignore_index=True)

        if st.button("Prediksi Batch (5 Baris)"):
            probs = model.predict_proba(Xb)[:,1]
            preds = (probs >= threshold).astype(int)
            out = Xb[CORE_FEATURES].copy()
            out["proba_pos"] = np.round(probs, 4)
            out["pred"] = preds
            if "HeartDisease" in head5.columns:
                out["actual"] = head5["HeartDisease"].values
            st.dataframe(out, use_container_width=True)
            st.markdown(f"<div class='kpi'>Positif (pred=1): <b>{(preds==1).mean()*100:.1f}%</b></div>", unsafe_allow_html=True)

st.caption("Menampilkan hanya fitur inti dari heatmap (Oldpeak, MaxHR, Age, FastingBS, Cholesterol). Fitur lain tersedia di 'Lanjutan' bila ingin diubah.")