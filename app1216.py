# app.py  (FINAL – compact titles, full features)

import os
import re
import numpy as np
import joblib
import streamlit as st

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

# =====================================================
# Page & global style (smaller titles)
# =====================================================
st.set_page_config(page_title="ReAdmit-再入ICU风险预测", layout="wide")

st.markdown("""
<style>
/* 全局字体 */
body, .stApp {
    font-size: 0.82rem;
    line-height: 1.25;
}

/* 页面主标题 */
.main-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

/* 各部分标题 */
.section-title {
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 0.2rem;
    margin-bottom: 0.3rem;
}

/* 小分组标题（血常规 / 凝血等） */
.group-title {
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 0.3rem;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

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
        st.markdown('<div class="section-title">📝 基本信息</div>', unsafe_allow_html=True)
        age = st.number_input("年龄（岁）", min_value=0, max_value=120, value=None)
        gender = st.radio("性别", ["男", "女"])
        genderscore = 1 if gender == "男" else 0
        los_hospital = st.number_input("住院时长（天）", value=None)
        los_icu = st.number_input("ICU住院时长（天）", value=None)

    # -------- 生命体征 --------
    with col2:
        st.markdown('<div class="section-title">❤️ 生命体征</div>', unsafe_allow_html=True)
        hr = st.number_input("心率（次/分）", value=None)
        sbp = st.number_input("收缩压（mmHg）", value=None)
        dbp = st.number_input("舒张压（mmHg）", value=None)
        mbp = (sbp + 2 * dbp) / 3 if sbp and dbp else 0
        st.number_input("平均动脉压（mmHg）", value=mbp, disabled=True)
        spo2 = st.number_input("血氧饱和度（%）", value=None)
        temp = st.number_input("体温（℃）", value=None)

    # -------- 其他体征 --------
    with col3:
        st.markdown('<div class="section-title">🌡 其他体征</div>', unsafe_allow_html=True)
        urine = st.number_input("最后24h尿量（mL）", value=None)
        o2flow = st.number_input("吸氧流量（L/min）", value=None)
        intubated = st.radio("是否气管插管/切开", ["有", "无"])
        invasive_flag = 1 if intubated == "有" else 0
        mech_time = st.number_input("机械通气时长（小时）", value=None)

    # -------- Charlson --------
    with col4:
        st.markdown('<div class="section-title">🧾 Charlson 合并症</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title">📄 影像学检查文本</div>', unsafe_allow_html=True)
        img = st.file_uploader("上传影像学报告截图", type=["png","jpg","jpeg"])
        if img:
            ocr_text = run_ocr(img, ocr_engine)
            st.text_area("OCR 识别结果", ocr_text, height=120)

    st.divider()

    # ================= 实验室检查 =================
    st.markdown('<div class="section-title">🧪 实验室检查</div>', unsafe_allow_html=True)

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
