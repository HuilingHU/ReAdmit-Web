# app.py  (UI refined – compact clinical version)

import os
import re
import numpy as np
import joblib
import streamlit as st

# =====================================================
# Feature order (DO NOT CHANGE)
# =====================================================
FEATURE_ORDER = [
    'admission_age','genderscore','los_hospital','los_icu',
    'heart_rate_24hfinal','sbp_ni_24hfinal','dbp_ni_24hfinal',
    'mbp_ni_24hfinal','spo2_24hfinal','temperature_24hfinal',
    'urineoutput_24hr','charlson',
    'wbc','bicarbonate','calcium','chloride','creatinine',
    'alt','ast','bilirubin_total','glucose','sodium','potassium',
    'inr','pt','ptt','hematocrit','hemoglobin','albumin','mch',
    'platelet','rbc','rdw','lactate','ph','be','pao2','paco2',
    'o2_flow','mechanical_ventilation_time','invasive_ventilation'
]

# =====================================================
# Page setup – compact
# =====================================================
st.set_page_config(page_title="ReAdmit-再入ICU风险预测", layout="wide")
st.markdown("""
<style>
body, .stApp { font-size: 0.80rem; line-height: 1.10; }
h1 { font-size: 1.05rem; }
h2 { font-size: 0.95rem; }
label { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

st.title("ReAdmit-再入ICU风险预测")

# =====================================================
# OCR
# =====================================================
@st.cache_resource
def get_ocr():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang="ch")
    except:
        return None

def ocr_image(img, engine):
    if engine is None:
        return ""
    with open("tmp.png","wb") as f:
        f.write(img.getvalue())
    res = engine.ocr("tmp.png", cls=True)
    return " ".join([x[1][0] for x in res[0]]) if res and res[0] else ""

ocr_engine = get_ocr()

# =====================================================
# Model
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("model_1212.pkl")
    with open("threshold_1212.txt") as f:
        thr = float(f.read().strip())
    return model, thr

model, threshold = load_model()

# =====================================================
# Charlson
# =====================================================
def charlson(age, g1, g2, g3, g4):
    score = len(g1)*1 + len(g2)*2 + len(g3)*3 + len(g4)*6
    if age and age >= 40:
        score += ((age-40)//10)+1
    return score

# =====================================================
# UI
# =====================================================
with st.form("form"):

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.subheader("基本信息")
        age = st.number_input("年龄（岁）", value=None)
        gender = st.radio("性别", ["男","女"], horizontal=True)
        genderscore = 1 if gender=="男" else 0
        los_hospital = st.number_input("住院天数（天）", value=None)
        los_icu = st.number_input("ICU住院天数（天）", value=None)

    with c2:
        st.subheader("生命体征")
        hr = st.number_input("心率（次/分）", value=None)
        sbp = st.number_input("收缩压（mmHg）", value=None)
        dbp = st.number_input("舒张压（mmHg）", value=None)
        mbp = (sbp+2*dbp)/3 if sbp and dbp else 0
        spo2 = st.number_input("血氧饱和度（%）", value=None)
        temp = st.number_input("体温（℃）", value=None)

    with c3:
        st.subheader("其他情况")
        urine = st.number_input("24h尿量（mL）", value=None)
        o2flow = st.number_input("吸氧流量（L/min）", value=None)
        invasive = st.radio("气管插管/切开", ["有","无"], horizontal=True)
        invasive_flag = 1 if invasive=="有" else 0
        mech_time = st.number_input("机械通气时长（小时）", value=None)

    with c4:
        st.subheader("Charlson 合并症")
        g1 = st.multiselect("1 分", ["心肌梗死","充血性心衰","慢性肺病","糖尿病"])
        g2 = st.multiselect("2 分", ["肾功能不全","肿瘤"])
        g3 = st.multiselect("3 分", ["中重度肝病"])
        g4 = st.multiselect("6 分", ["转移癌","AIDS"])
        charl = charlson(age,g1,g2,g3,g4)
        st.text(f"Charlson 指数：{charl}")

    st.subheader("实验室检查")

    lab = {}
    def lab_group(title, items):
        st.markdown(f"**{title}**")
        cols = st.columns(6)
        for i,(cn,en) in enumerate(items):
            lab[en] = cols[i%6].number_input(cn, value=None)

    lab_group("血常规", [
        ("白细胞（×10⁹/L）","wbc"),
        ("红细胞（×10¹²/L）","rbc"),
        ("血红蛋白（g/L）","hemoglobin"),
        ("红细胞压积（%）","hematocrit"),
        ("平均红细胞血红蛋白含量（pg）","mch"),
        ("血小板（×10⁹/L）","platelet"),
        ("红细胞分布宽度（%）","rdw"),
    ])

    lab_group("凝血功能", [
        ("国际标准化比值","inr"),
        ("凝血酶原时间（秒）","pt"),
        ("活化部分凝血活酶时间（秒）","ptt"),
    ])

    lab_group("肝肾功 / 生化", [
        ("肌酐（μmol/L）","creatinine"),
        ("丙氨酸氨基转移酶（IU/L）","alt"),
        ("天冬氨酸氨基转移酶（IU/L）","ast"),
        ("总胆红素（μmol/L）","bilirubin_total"),
        ("白蛋白（g/L）","albumin"),
    ])

    lab_group("血气分析", [
        ("HCO₃⁻（mmol/L）","bicarbonate"),
        ("Ca²⁺（mmol/L）","calcium"),
        ("Cl⁻（mmol/L）","chloride"),
        ("血糖（mmol/L）","glucose"),
        ("Na⁺（mmol/L）","sodium"),
        ("K⁺（mmol/L）","potassium"),
        ("乳酸（mmol/L）","lactate"),
        ("pH值","ph"),
        ("碱剩余（mmol/L）","be"),
        ("氧分压（mmHg）","pao2"),
        ("二氧化碳分压（mmHg）","paco2"),
    ])

    st.subheader("影像学检查文本")
    img = st.file_uploader("上传影像学报告截图", type=["png","jpg","jpeg"])
    if img:
        txt = ocr_image(img, ocr_engine)
        st.text_area("OCR 识别结果", txt, height=100)

    submit = st.form_submit_button("开始预测")

# =====================================================
# Prediction
# =====================================================
if submit:
    data = {
        "admission_age":age,"genderscore":genderscore,
        "los_hospital":los_hospital,"los_icu":los_icu,
        "heart_rate_24hfinal":hr,"sbp_ni_24hfinal":sbp,
        "dbp_ni_24hfinal":dbp,"mbp_ni_24hfinal":mbp,
        "spo2_24hfinal":spo2,"temperature_24hfinal":temp,
        "urineoutput_24hr":urine,"charlson":charl,
        "o2_flow":o2flow,"mechanical_ventilation_time":mech_time,
        "invasive_ventilation":invasive_flag,
        **lab
    }
    X = np.array([[float(data.get(f,0) or 0) for f in FEATURE_ORDER]])
    prob = model.predict_proba(X)[0,1]
    risk = "高风险" if prob>=threshold else "低风险"
    st.metric("再入 ICU 风险概率", f"{prob:.2%}")
    st.success(f"风险分层：{risk}")
