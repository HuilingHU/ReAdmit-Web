# app.py  (FINAL – larger main title, colored section titles, uniform lab inputs)

import os
import re
import numpy as np
import joblib
import streamlit as st
import shap 

# =====================================================
# Feature order (MUST match training)
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

FEATURE_NAME_MAP = {
    "admission_age": "年龄",
    "genderscore": "性别",
    "los_hospital": "住院时长",
    "los_icu": "ICU住院时长",
    "heart_rate_24hfinal": "心率",
    "sbp_ni_24hfinal": "收缩压",
    "dbp_ni_24hfinal": "舒张压",
    "mbp_ni_24hfinal": "平均动脉压",
    "spo2_24hfinal": "血氧饱和度",
    "temperature_24hfinal": "体温",
    "urineoutput_24hr": "24小时尿量",
    "charlson": "Charlson 合并症指数",

    "wbc": "白细胞",
    "rbc": "红细胞",
    "hemoglobin": "血红蛋白",
    "hematocrit": "红细胞压积",
    "mch": "平均红细胞血红蛋白含量",
    "platelet": "血小板",
    "rdw": "红细胞分布宽度",

    "inr": "INR",
    "pt": "凝血酶原时间",
    "ptt": "活化部分凝血活酶时间",

    "creatinine": "肌酐",
    "alt": "丙氨酸氨基转移酶",
    "ast": "天冬氨酸氨基转移酶",
    "bilirubin_total": "总胆红素",
    "albumin": "白蛋白",

    "bicarbonate": "碳酸氢根",
    "calcium": "钙",
    "chloride": "氯",
    "glucose": "血糖",
    "sodium": "钠",
    "potassium": "钾",
    "lactate": "乳酸",
    "ph": "pH",
    "be": "碱剩余",
    "pao2": "氧分压",
    "paco2": "二氧化碳分压",

    "o2_flow": "吸氧流量",
    "mechanical_ventilation_time": "机械通气时长",
    "invasive_ventilation": "有创通气"
}

# =====================================================
# Page & global style
# =====================================================
st.set_page_config(page_title="ReAdmit-再入ICU风险预测", layout="wide")

st.markdown(
    "<style>"
    "body, .stApp { font-size: 0.82rem; line-height: 1.25; }"
    ".main-title { font-size: 1.35rem; font-weight: 700; color: #0b3c5d; margin-bottom: 0.5rem; }"
    ".title-basic { font-size: 0.9rem; font-weight: 600; color: #1f7a8c; }"
    ".title-vital { font-size: 0.9rem; font-weight: 600; color: #b23a48; }"
    ".title-other { font-size: 0.9rem; font-weight: 600; color: #5f4b8b; }"
    ".title-charlson { font-size: 0.9rem; font-weight: 600; color: #2f855a; }"
    ".title-lab { font-size: 0.9rem; font-weight: 600; color: #3b5b92; }"
    ".group-title { font-size: 0.82rem; font-weight: 600; color: #334e68; margin-top: 0.35rem; }"
    "div[data-baseweb='input'] { width: 100% !important; }"
    "</style>",
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">ReAdmit-再入ICU风险预测</div>', unsafe_allow_html=True)

# =====================================================
# OCR (text only)
# =====================================================
@st.cache_resource
def load_ocr():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang="ch")
    except Exception:
        return None

def run_ocr(img, engine):
    if engine is None or img is None:
        return ""
    with open("tmp.png", "wb") as f:
        f.write(img.getvalue())
    res = engine.ocr("tmp.png", cls=True)
    if not res or not res[0]:
        return ""
    text = " ".join([x[1][0] for x in res[0]])
    return re.sub(r"\s+", " ", text)

ocr_engine = load_ocr()

# =====================================================
# Model
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("model_1212.pkl")
    with open("threshold_1212.txt") as f:
        threshold = float(f.read().strip())
    return model, threshold

model, threshold = load_model()

# =====================================================
# Charlson
# =====================================================
def calculate_charlson_score(age, selections):
    score = 0
    weights = {"group1": 1, "group2": 2, "group3": 3, "group4": 6}
    for group, items in selections.items():
        score += weights[group] * len(items)
    if age is not None and age >= 40:
        score += ((age - 40) // 10) + 1
    return score

# =====================================================
# UI
# =====================================================
with st.form("icu_form"):
    col1, col2, col3, col4, col5 = st.columns([0.9,0.9,0.9,0.9,1.1], gap="small")

    # -------- 基本信息 --------
    with col1:
        st.markdown('<div class="title-basic">📝 基本信息</div>', unsafe_allow_html=True)
        age = st.number_input("年龄（岁）", min_value=0, max_value=120, value=None)
        gender = st.radio("性别", ["男", "女"])
        genderscore = 1 if gender == "男" else 0
        los_hospital = st.number_input("住院时长（天）", value=None)
        los_icu = st.number_input("ICU住院时长（天）", value=None)

    # -------- 生命体征 --------
    with col2:
        st.markdown('<div class="title-vital">❤️ 生命体征</div>', unsafe_allow_html=True)
        hr = st.number_input("心率（次/分）", value=None)
        sbp = st.number_input("收缩压（mmHg）", value=None)
        dbp = st.number_input("舒张压（mmHg）", value=None)
        mbp = (sbp + 2 * dbp) / 3 if sbp and dbp else 0
        st.number_input("平均动脉压（mmHg）", value=mbp, disabled=True)
        spo2 = st.number_input("血氧饱和度（%）", value=None)
        temp = st.number_input("体温（℃）", value=None)

    # -------- 其他体征 --------
    with col3:
        st.markdown('<div class="title-other">🌡 其他体征</div>', unsafe_allow_html=True)
        urine = st.number_input("最后24h尿量（mL）", value=None)
        o2flow = st.number_input("吸氧流量（L/min）", value=None)
        intubated = st.radio("是否气管插管/切开", ["有", "无"])
        invasive_flag = 1 if intubated == "有" else 0
        mech_time = st.number_input("机械通气时长（小时）", value=None)

    # -------- Charlson --------
    with col4:
        st.markdown('<div class="title-charlson">🧾 Charlson 合并症</div>', unsafe_allow_html=True)
        group1 = st.multiselect(
            "1 分",
            ["心肌梗死","充血性心衰","慢性肺病","糖尿病",
             "结缔组织病","周围血管疾病","脑血管疾病","痴呆",
             "溃疡病","轻度肝脏疾病"]
        )
        group2 = st.multiselect(
            "2 分",
            ["中重度肾脏疾病","白血病","偏瘫",
             "糖尿病伴有器官损害","原发性肿瘤","淋巴瘤"]
        )
        group3 = st.multiselect("3 分", ["中重度肝病"])
        group4 = st.multiselect("6 分", ["转移癌","获得性免疫缺陷综合征"])
        selections = {
            "group1": group1,
            "group2": group2,
            "group3": group3,
            "group4": group4,
        }
        charlson = calculate_charlson_score(age, selections)
        st.success(f"Charlson 合并症指数（含年龄加权）：{charlson}")

    # -------- 影像文本 --------
    with col5:
        st.markdown('<div class="title-basic">📄 影像学检查文本</div>', unsafe_allow_html=True)
        img = st.file_uploader("上传影像学报告截图", type=["png","jpg","jpeg"])
        if img:
            ocr_text = run_ocr(img, ocr_engine)
      

    st.divider()

    # ================= 实验室检查 =================
    st.markdown('<div class="title-lab">🧪 实验室检查</div>', unsafe_allow_html=True)

    # ---- 血常规 ----
    st.markdown('<div class="group-title">血常规</div>', unsafe_allow_html=True)
    cbc = st.columns(7)
    wbc = cbc[0].number_input("白细胞 ×10⁹/L", value=None)
    rbc = cbc[1].number_input("红细胞 ×10¹²/L", value=None)
    hemoglobin = cbc[2].number_input("血红蛋白 g/L", value=None)
    hematocrit = cbc[3].number_input("红细胞压积 %", value=None)
    mch = cbc[4].number_input("平均红细胞血红蛋白 pg", value=None)
    platelet = cbc[5].number_input("血小板 ×10⁹/L", value=None)
    rdw = cbc[6].number_input("红细胞分布宽度 %", value=None)

    # ---- 凝血 ----
    st.markdown('<div class="group-title">凝血功能</div>', unsafe_allow_html=True)
    coag = st.columns(3)
    inr = coag[0].number_input("INR", value=None)
    pt = coag[1].number_input("凝血酶原时间 秒", value=None)
    ptt = coag[2].number_input("活化部分凝血活酶时间 秒", value=None)

    # ---- 肝肾功 ----
    st.markdown('<div class="group-title">肝肾功 / 生化</div>', unsafe_allow_html=True)
    liver = st.columns(5)
    creatinine = liver[0].number_input("肌酐 μmol/L", value=None)
    alt = liver[1].number_input("ALT IU/L", value=None)
    ast = liver[2].number_input("AST IU/L", value=None)
    bilirubin_total = liver[3].number_input("总胆红素 μmol/L", value=None)
    albumin = liver[4].number_input("白蛋白 g/L", value=None)

    # ---- 血气 ----
    st.markdown('<div class="group-title">血气分析</div>', unsafe_allow_html=True)
    abg = st.columns(11)
    bicarbonate = abg[0].number_input("HCO₃⁻ mmol/L", value=None)
    calcium = abg[1].number_input("Ca²⁺ mmol/L", value=None)
    chloride = abg[2].number_input("Cl⁻ mmol/L", value=None)
    glucose = abg[3].number_input("血糖 mmol/L", value=None)
    sodium = abg[4].number_input("Na⁺ mmol/L", value=None)
    potassium = abg[5].number_input("K⁺ mmol/L", value=None)
    lactate = abg[6].number_input("乳酸 mmol/L", value=None)
    ph = abg[7].number_input("pH", value=None)
    be = abg[8].number_input("碱剩余 mmol/L", value=None)
    pao2 = abg[9].number_input("氧分压 mmHg", value=None)
    paco2 = abg[10].number_input("二氧化碳分压 mmHg", value=None)

    submitted = st.form_submit_button("🔍 进行风险预测")

# =====================================================
# Prediction
# =====================================================
if submitted:
    data = {
        "admission_age": age,
        "genderscore": genderscore,
        "los_hospital": los_hospital,
        "los_icu": los_icu,
        "heart_rate_24hfinal": hr,
        "sbp_ni_24hfinal": sbp,
        "dbp_ni_24hfinal": dbp,
        "mbp_ni_24hfinal": mbp,
        "spo2_24hfinal": spo2,
        "temperature_24hfinal": temp,
        "urineoutput_24hr": urine,
        "charlson": charlson,
        "o2_flow": o2flow,
        "mechanical_ventilation_time": mech_time,
        "invasive_ventilation": invasive_flag,
        "wbc": wbc,
        "rbc": rbc,
        "hemoglobin": hemoglobin,
        "hematocrit": hematocrit,
        "mch": mch,
        "platelet": platelet,
        "rdw": rdw,
        "inr": inr,
        "pt": pt,
        "ptt": ptt,
        "creatinine": creatinine,
        "alt": alt,
        "ast": ast,
        "bilirubin_total": bilirubin_total,
        "albumin": albumin,
        "bicarbonate": bicarbonate,
        "calcium": calcium,
        "chloride": chloride,
        "glucose": glucose,
        "sodium": sodium,
        "potassium": potassium,
        "lactate": lactate,
        "ph": ph,
        "be": be,
        "pao2": pao2,
        "paco2": paco2,
    }

    X = np.array([[float(data.get(f, 0) or 0) for f in FEATURE_ORDER]])
    prob = model.predict_proba(X)[0, 1]
    risk = "高风险" if prob >= threshold else "低风险"

    st.subheader("📊 预测结果")
    st.metric("再入 ICU 风险概率", f"{prob:.2%}")
    st.success(f"风险分层：{risk}")

    if risk == "高风险":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 二分类，取正类
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        top_idx = np.argsort(np.abs(vals))[::-1][:5]

        st.subheader("⚠️ 主要风险贡献因素（模型解释）")

        for i in top_idx:
            fname = FEATURE_ORDER[i]
            cname = FEATURE_NAME_MAP.get(fname, fname)
            direction = "↑ 增加风险" if vals[i] > 0 else "↓ 降低风险"
            st.write(f"- **{cname}**：{direction}")
