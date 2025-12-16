# app.py  (FINAL UI-refined version)

import os
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import requests

# =====================================================
# 0) Feature order (DO NOT CHANGE)
# =====================================================
FEATURE_ORDER = [
    'admission_age', 'genderscore', 'los_hospital', 'los_icu',
    'heart_rate_24hfinal', 'sbp_ni_24hfinal', 'dbp_ni_24hfinal',
    'mbp_ni_24hfinal', 'spo2_24hfinal', 'temperature_24hfinal',
    'urineoutput_24hr', 'charlson',
    'wbc', 'bicarbonate', 'calcium', 'chloride', 'creatinine',
    'alt', 'ast', 'bilirubin_total', 'glucose', 'sodium',
    'potassium', 'inr', 'pt', 'ptt', 'hematocrit',
    'hemoglobin', 'albumin', 'mch', 'platelet', 'rbc', 'rdw',
    'lactate', 'ph', 'be', 'pao2', 'paco2',
    'o2_flow', 'mechanical_ventilation_time', 'invasive_ventilation'
]

# =====================================================
# 1) Page setup (compact)
# =====================================================
st.set_page_config(page_title="再入ICU风险预测工具", layout="wide")
st.markdown("""
<style>
body, .stApp { font-size: 0.85rem; line-height: 1.15; }
h1 { font-size: 1.15rem; }
h2, h3 { font-size: 1.0rem; }
label { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.title("再入 ICU 风险预测工具（ReAdmit）")
st.caption("⚠️ 本工具仅用于科研与教学目的，不构成临床决策依据")

# =====================================================
# 2) OCR（仅文本展示，不进模型）
# =====================================================
@st.cache_resource
def get_ocr_engine():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='ch')
    except Exception:
        return None

def extract_text_from_image(image_file, ocr_engine):
    if ocr_engine is None:
        return "⚠️ 当前环境不支持 OCR"
    img_bytes = image_file.getvalue()
    with open("temp.png", "wb") as f:
        f.write(img_bytes)
    result = ocr_engine.ocr("temp.png", cls=True)
    if not result or not result[0]:
        return ""
    text = " ".join([line[1][0] for line in result[0]])
    text = re.sub(r"\s+", " ", text).strip()
    return text

ocr_engine = get_ocr_engine()

# =====================================================
# 3) Model
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("model_1212.pkl")
    with open("threshold_1212.txt") as f:
        threshold = float(f.read().strip())
    return model, threshold

model, threshold = load_model()

# =====================================================
# 4) Charlson
# =====================================================
def calculate_charlson(age, selections):
    weights = {"g1": 1, "g2": 2, "g3": 3, "g4": 6}
    score = sum(weights[k] * len(v) for k, v in selections.items())
    if age and age >= 40:
        score += ((age - 40) // 10) + 1
    return score

# =====================================================
# 5) UI
# =====================================================
with st.form("form"):

    c1, c2, c3, c4 = st.columns(4)

    # ---------- 基本信息 ----------
    with c1:
        st.subheader("基本信息")
        age = st.number_input("年龄（岁）", min_value=0, max_value=120, value=None)
        gender = st.radio("性别", ["男", "女"], horizontal=True)
        genderscore = 1 if gender == "男" else 0
        los_hospital = st.number_input("住院天数（天）", min_value=0.0, value=None)
        los_icu = st.number_input("ICU 住院天数（天）", min_value=0.0, value=None)

    # ---------- 生命体征 ----------
    with c2:
        st.subheader("生命体征")
        hr = st.number_input("心率（次/分）", value=None)
        sbp = st.number_input("收缩压（mmHg）", value=None)
        dbp = st.number_input("舒张压（mmHg）", value=None)
        mbp = (sbp + 2 * dbp) / 3 if sbp and dbp else 0
        st.text(f"平均动脉压：{mbp:.1f} mmHg")
        spo2 = st.number_input("血氧饱和度（%）", value=None)
        temp = st.number_input("体温（℃）", value=None)

    # ---------- 其他 ----------
    with c3:
        st.subheader("其他情况")
        urine = st.number_input("24h 尿量（mL）", value=None)
        o2flow = st.number_input("吸氧流量（L/min）", value=None)
        invasive = st.radio("气管插管/切开", ["有", "无"], horizontal=True)
        invasive_flag = 1 if invasive == "有" else 0
        mech_time = st.number_input("机械通气时长（小时）", value=None)

    # ---------- Charlson ----------
    with c4:
        st.subheader("Charlson 合并症")
        g1 = st.multiselect("1 分", ["心肌梗死","充血性心衰","慢性肺病","糖尿病"])
        g2 = st.multiselect("2 分", ["肾功能不全","肿瘤"])
        g3 = st.multiselect("3 分", ["中重度肝病"])
        g4 = st.multiselect("6 分", ["转移癌","AIDS"])
        charlson = calculate_charlson(age, {"g1": g1, "g2": g2, "g3": g3, "g4": g4})
        st.text(f"Charlson 指数：{charlson}")

    # =================================================
    # 实验室检查（中文分组）
    # =================================================
    st.subheader("实验室检查")

    lab_inputs = {}

    def lab_block(title, items):
        with st.expander(title, expanded=True):
            cols = st.columns(4)
            for i, (cn, en) in enumerate(items):
                lab_inputs[en] = cols[i % 4].number_input(cn, value=None)

    lab_block("血常规", [
        ("白细胞", "wbc"), ("红细胞", "rbc"), ("血红蛋白", "hemoglobin"),
        ("红细胞压积", "hematocrit"), ("平均血红蛋白含量", "mch"),
        ("血小板", "platelet"), ("红细胞分布宽度", "rdw")
    ])

    lab_block("凝血功能", [
        ("INR", "inr"), ("凝血酶原时间 PT", "pt"), ("APTT", "ptt")
    ])

    lab_block("肝肾功 / 生化", [
        ("肌酐", "creatinine"), ("ALT", "alt"), ("AST", "ast"),
        ("总胆红素", "bilirubin_total"), ("白蛋白", "albumin"),
        ("葡萄糖", "glucose"), ("钠", "sodium"),
        ("钾", "potassium"), ("氯", "chloride"), ("钙", "calcium"),
        ("碳酸氢根", "bicarbonate")
    ])

    lab_block("血气分析", [
        ("pH", "ph"), ("BE", "be"), ("乳酸", "lactate"),
        ("PaO₂", "pao2"), ("PaCO₂", "paco2")
    ])

    # =================================================
    # OCR only
    # =================================================
    st.subheader("放射学 / 临床文本（仅 OCR 展示，不参与预测）")
    img = st.file_uploader("上传报告截图", type=["png","jpg","jpeg"])
    ocr_text = extract_text_from_image(img, ocr_engine) if img else ""
    if ocr_text:
        st.text_area("OCR 识别结果", ocr_text, height=120)

    submitted = st.form_submit_button("开始预测")

# =====================================================
# 6) Prediction
# =====================================================
if submitted:
    input_dict = {
        "admission_age": age, "genderscore": genderscore,
        "los_hospital": los_hospital, "los_icu": los_icu,
        "heart_rate_24hfinal": hr, "sbp_ni_24hfinal": sbp,
        "dbp_ni_24hfinal": dbp, "mbp_ni_24hfinal": mbp,
        "spo2_24hfinal": spo2, "temperature_24hfinal": temp,
        "urineoutput_24hr": urine, "charlson": charlson,
        "o2_flow": o2flow, "mechanical_ventilation_time": mech_time,
        "invasive_ventilation": invasive_flag,
        **lab_inputs
    }

    for f in FEATURE_ORDER:
        input_dict.setdefault(f, 0.0)

    X = np.array([[float(input_dict[f]) if input_dict[f] is not None else 0.0 for f in FEATURE_ORDER]])
    prob = model.predict_proba(X)[0, 1]
    risk = "高风险" if prob >= threshold else "低风险"

    st.subheader("预测结果")
    st.metric("再入 ICU 风险概率", f"{prob:.2%}")
    st.success(f"风险分层：{risk}")
