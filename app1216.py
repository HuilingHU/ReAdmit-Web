# app.py  (Clinical-style UI, full features preserved)

import numpy as np
import joblib
import streamlit as st
import re

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
# Page style – clean clinical look
# =====================================================
st.set_page_config(page_title="ReAdmit-再入ICU风险预测", layout="wide")

st.markdown("""
<style>
body, .stApp {
    font-size: 0.82rem;
    line-height: 1.25;
    background-color: #f5f7fa;
    color: #1f2933;
}
h1 {
    font-size: 1.2rem;
    color: #0f4c5c;
}
.section {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
}
.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f4c5c;
    margin-bottom: 0.4rem;
}
label {
    font-size: 0.8rem;
    color: #374151;
}
</style>
""", unsafe_allow_html=True)

st.title("ReAdmit-再入ICU风险预测")

# =====================================================
# OCR
# =====================================================
@st.cache_resource
def load_ocr():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang="ch")
    except:
        return None

def run_ocr(img, engine):
    if engine is None or img is None:
        return ""
    with open("tmp.png","wb") as f:
        f.write(img.getvalue())
    res = engine.ocr("tmp.png", cls=True)
    if not res or not res[0]:
        return ""
    return re.sub(r"\s+", " ", " ".join([x[1][0] for x in res[0]]))

ocr_engine = load_ocr()

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
def calc_charlson(age, g1, g2, g3, g4):
    score = len(g1)*1 + len(g2)*2 + len(g3)*3 + len(g4)*6
    if age and age >= 40:
        score += ((age-40)//10)+1
    return score

# =====================================================
# UI
# =====================================================
with st.form("form"):

    # ---------- 上半部分 ----------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">基本信息</div>', unsafe_allow_html=True)
        age = st.number_input("年龄（岁）", value=None)
        gender = st.radio("性别", ["男","女"], horizontal=True)
        genderscore = 1 if gender=="男" else 0
        los_hospital = st.number_input("住院天数（天）", value=None)
        los_icu = st.number_input("ICU住院天数（天）", value=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">生命体征</div>', unsafe_allow_html=True)
        hr = st.number_input("心率（次/分）", value=None)
        sbp = st.number_input("收缩压（mmHg）", value=None)
        dbp = st.number_input("舒张压（mmHg）", value=None)
        mbp = (sbp+2*dbp)/3 if sbp and dbp else 0
        spo2 = st.number_input("血氧饱和度（%）", value=None)
        temp = st.number_input("体温（℃）", value=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">其他情况</div>', unsafe_allow_html=True)
        urine = st.number_input("24h尿量（mL）", value=None)
        o2flow = st.number_input("吸氧流量（L/min）", value=None)
        invasive = st.radio("气管插管/切开", ["有","无"], horizontal=True)
        invasive_flag = 1 if invasive=="有" else 0
        mech_time = st.number_input("机械通气时长（小时）", value=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Charlson 合并症</div>', unsafe_allow_html=True)

        g1 = st.multiselect("1分", [
            "心肌梗死","充血性心衰","慢性肺病","糖尿病",
            "结缔组织病","周围血管疾病","脑血管疾病","痴呆",
            "溃疡病","轻度肝脏疾病"
        ])

        g2 = st.multiselect("2分", [
            "中重度肾脏疾病","白血病","偏瘫",
            "糖尿病伴有器官损害","原发性肿瘤","淋巴瘤"
        ])

        g3 = st.multiselect("3分", ["中重度肝病"])

        g4 = st.multiselect("6分", ["转移癌","获得性免疫缺陷综合征"])

        charlson = calc_charlson(age,g1,g2,g3,g4)
        st.markdown(f"**Charlson 指数：{charlson}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- 实验室检查（完整，下方） ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">实验室检查</div>', unsafe_allow_html=True)

    l1, l2, l3, l4 = st.columns(4)

    with l1:
        st.markdown("**血常规**")
        wbc = st.number_input("白细胞 ×10⁹/L", value=None)
        rbc = st.number_input("红细胞 ×10¹²/L", value=None)
        hb = st.number_input("血红蛋白 g/L", value=None)
        hct = st.number_input("红细胞压积 %", value=None)
        mch = st.number_input("平均红细胞血红蛋白 pg", value=None)
        plt = st.number_input("血小板 ×10⁹/L", value=None)
        rdw = st.number_input("红细胞分布宽度 %", value=None)

    with l2:
        st.markdown("**凝血功能**")
        inr = st.number_input("INR", value=None)
        pt = st.number_input("凝血酶原时间 秒", value=None)
        ptt = st.number_input("活化部分凝血活酶时间 秒", value=None)

    with l3:
        st.markdown("**肝肾功 / 生化**")
        cr = st.number_input("肌酐 μmol/L", value=None)
        alt = st.number_input("ALT IU/L", value=None)
        ast = st.number_input("AST IU/L", value=None)
        tbil = st.number_input("总胆红素 μmol/L", value=None)
        alb = st.number_input("白蛋白 g/L", value=None)

    with l4:
        st.markdown("**血气分析**")
        hco3 = st.number_input("HCO₃⁻ mmol/L", value=None)
        ca = st.number_input("Ca²⁺ mmol/L", value=None)
        cl = st.number_input("Cl⁻ mmol/L", value=None)
        glu = st.number_input("血糖 mmol/L", value=None)
        na = st.number_input("Na⁺ mmol/L", value=None)
        k = st.number_input("K⁺ mmol/L", value=None)
        lac = st.number_input("乳酸 mmol/L", value=None)
        ph = st.number_input("pH", value=None)
        be = st.number_input("碱剩余 mmol/L", value=None)
        pao2 = st.number_input("PaO₂ mmHg", value=None)
        paco2 = st.number_input("PaCO₂ mmHg", value=None)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- OCR ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">影像学检查文本</div>', unsafe_allow_html=True)
    img = st.file_uploader("上传影像学报告截图", type=["png","jpg","jpeg"])
    if img:
        txt = run_ocr(img, ocr_engine)
        st.text_area("OCR 识别结果", txt, height=90)
    st.markdown('</div>', unsafe_allow_html=True)

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
        "urineoutput_24hr":urine,"charlson":charlson,
        "o2_flow":o2flow,"mechanical_ventilation_time":mech_time,
        "invasive_ventilation":invasive_flag,
        "wbc":wbc,"rbc":rbc,"hemoglobin":hb,"hematocrit":hct,
        "mch":mch,"platelet":plt,"rdw":rdw,
        "inr":inr,"pt":pt,"ptt":ptt,
        "creatinine":cr,"alt":alt,"ast":ast,"bilirubin_total":tbil,
        "albumin":alb,"bicarbonate":hco3,"calcium":ca,"chloride":cl,
        "glucose":glu,"sodium":na,"potassium":k,
        "lactate":lac,"ph":ph,"be":be,
        "pao2":pao2,"paco2":paco2
    }

    X = np.array([[float(data.get(f,0) or 0) for f in FEATURE_ORDER]])
    prob = model.predict_proba(X)[0,1]
    risk = "高风险" if prob>=threshold else "低风险"

    st.metric("再入 ICU 风险概率", f"{prob:.2%}")
    st.success(f"风险分层：{risk}")
