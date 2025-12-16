# app.py  (FINAL - deployable, structured-only model)
import os
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import requests

# ----------------------------
# 0) Feature order (MUST match model training)
# ----------------------------
FEATURE_ORDER = [
    'admission_age', 'genderscore', 'los_hospital', 'los_icu',
    'heart_rate_24hfinal', 'sbp_ni_24hfinal', 'dbp_ni_24hfinal',
    'mbp_ni_24hfinal', 'spo2_24hfinal', 'temperature_24hfinal',
    'urineoutput_24hr', 'charlson', 'wbc', 'bicarbonate', 'calcium',
    'chloride', 'creatinine', 'alt', 'ast', 'bilirubin_total',
    'glucose', 'sodium', 'potassium', 'inr', 'pt', 'ptt', 'hematocrit',
    'hemoglobin', 'albumin', 'mch', 'platelet', 'rbc', 'rdw',
    'lactate', 'ph', 'be', 'pao2', 'paco2', 'o2_flow',
    'mechanical_ventilation_time', 'invasive_ventilation'
]

# ----------------------------
# 1) Page setup
# ----------------------------
st.set_page_config(page_title="再入ICU风险预测工具 - ReAdmit (Online)", layout="wide")
st.markdown("""
<style>
body, .stApp { font-size: 0.9rem; line-height: 1.25; }
h1 { font-size: 1.25rem; }
h2, h3, h4 { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

st.title("再入ICU 风险预测工具 - ReAdmit（在线部署版）")
st.warning("⚠️ 上传任何截图/照片前请务必隐去姓名、住院号等敏感信息。")

# ----------------------------
# 2) Optional OCR: only for TEXT (not labs)
# ----------------------------
@st.cache_resource
def get_ocr_engine():
    """
    PaddleOCR is heavy; on some cloud environments it may fail due to missing deps.
    We keep it optional: if import fails, OCR will be disabled gracefully.
    """
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='ch')
    except Exception:
        return None

def extract_text_from_image(image_file, ocr_engine):
    """
    OCR only: return recognized text for user reference.
    """
    if ocr_engine is None:
        raise RuntimeError("OCR 引擎不可用（部署环境未安装 PaddleOCR 或依赖缺失）。")
    # Streamlit UploadedFile -> bytes
    img_bytes = image_file.getvalue()
    tmp_path = "temp_text.png"
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)

    result = ocr_engine.ocr(tmp_path, cls=True)
    if not result or not result[0]:
        return ""

    full_text = " ".join([line[1][0] for line in result[0]])

    # normalize text
    full_text = full_text.upper()
    full_text = re.sub(r"\s+", " ", full_text).strip()
    full_text = full_text.replace("⁺", "+").replace("－", "-").replace("–", "-")
    full_text = (full_text
                 .replace("０","0").replace("１","1").replace("２","2").replace("３","3")
                 .replace("４","4").replace("５","5").replace("６","6").replace("７","7")
                 .replace("８","8").replace("９","9"))
    full_text = full_text.replace("HC03", "HCO3")
    return full_text

# ----------------------------
# 3) Model loader
# ----------------------------
@st.cache_resource
def load_model_and_threshold():
    model_path = "model_1212.pkl"
    thr_path = "threshold_1212.txt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}（请把 model_1212.pkl 放在 app.py 同目录）")
    if not os.path.exists(thr_path):
        raise FileNotFoundError(f"未找到阈值文件：{thr_path}（请把 threshold_1212.txt 放在 app.py 同目录）")

    model = joblib.load(model_path)
    with open(thr_path, "r") as f:
        threshold = float(f.read().strip())
    return model, threshold

model, threshold = load_model_and_threshold()

# ----------------------------
# 4) Charlson calculator (same as your logic)
# ----------------------------
def calculate_charlson_score(age, selections):
    score = 0
    weights = {"group1": 1, "group2": 2, "group3": 3, "group4": 6}
    for group, items in selections.items():
        score += weights[group] * len(items)
    if age >= 40:
        score += ((age - 40) // 10) + 1
    return score

# ----------------------------
# 5) DeepSeek online call (optional)
# ----------------------------
def get_deepseek_key():
    # Streamlit Cloud recommended: st.secrets["DEEPSEEK_API_KEY"]
    # Local: export DEEPSEEK_API_KEY="..."
    if "DEEPSEEK_API_KEY" in st.secrets:
        return st.secrets["DEEPSEEK_API_KEY"]
    return os.getenv("DEEPSEEK_API_KEY", "")

def ask_deepseek_online(prompt: str) -> str:
    api_key = get_deepseek_key()
    if not api_key:
        return "未配置 DEEPSEEK_API_KEY：已跳过 LLM 解读。"

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个医学助手，提供风险解读和临床建议。请基于用户给出的信息，不要编造未提供的化验值或检查结果。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM 调用失败：{e}"

# ----------------------------
# 6) UI
# ----------------------------
ocr_engine = get_ocr_engine()

with st.form("icu_form"):
    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 1.2], gap="medium")

    # ---- Basic info
    with c1:
        st.subheader("📝 基本信息")
        admission_age = st.number_input("年龄 admission_age（岁）", min_value=0, max_value=120, value=60, step=1)
        gender = st.radio("性别", options=["男", "女"], horizontal=True)
        genderscore = 1 if gender == "男" else 0

        los_hospital = st.number_input("住院时长 los_hospital（天）", min_value=0.0, value=7.0, step=1.0)
        los_icu = st.number_input("ICU住院时长 los_icu（天）", min_value=0.0, value=3.0, step=1.0)

    # ---- Vitals
    with c2:
        st.subheader("❤️ 生命体征")
        heart_rate_24hfinal = st.number_input("心率 heart_rate_24hfinal（次/分）", min_value=0.0, value=90.0, step=1.0)
        sbp_ni_24hfinal = st.number_input("收缩压 sbp_ni_24hfinal（mmHg）", min_value=0.0, value=120.0, step=1.0)
        dbp_ni_24hfinal = st.number_input("舒张压 dbp_ni_24hfinal（mmHg）", min_value=0.0, value=70.0, step=1.0)
        mbp_ni_24hfinal = (sbp_ni_24hfinal + 2 * dbp_ni_24hfinal) / 3 if (sbp_ni_24hfinal > 0 and dbp_ni_24hfinal > 0) else 0.0
        st.number_input("平均动脉压 mbp_ni_24hfinal（自动计算）", value=float(mbp_ni_24hfinal), disabled=True)

        spo2_24hfinal = st.number_input("血氧饱和度 spo2_24hfinal（%）", min_value=0.0, max_value=100.0, value=96.0, step=1.0)
        temperature_24hfinal = st.number_input("体温 temperature_24hfinal（℃）", min_value=0.0, value=36.8, step=0.1)

    # ---- Other & ventilation
    with c3:
        st.subheader("🌡 其他体征/支持治疗")
        urineoutput_24hr = st.number_input("最后24h尿量 urineoutput_24hr（mL）", min_value=0.0, value=1500.0, step=50.0)
        o2_flow = st.number_input("吸氧流量 o2_flow（L/min）", min_value=0.0, value=2.0, step=0.5)
        invasive = st.radio("有无气管插管/切开（invasive_ventilation）", options=["有", "无"], horizontal=True)
        invasive_ventilation = 1 if invasive == "有" else 0
        mechanical_ventilation_time = st.number_input("机械通气时长 mechanical_ventilation_time（小时）", min_value=0.0, value=0.0, step=1.0)

    # ---- Charlson
    with c4:
        st.subheader("🧾 Charlson 合并症选择（用于计算 charlson）")
        group1 = st.multiselect("1 分（group1）", ["心肌梗死","充血性心力衰竭","周围血管疾病","脑血管疾病","痴呆","慢性肺部疾病","结缔组织病","溃疡病","轻度肝脏疾病","糖尿病"])
        group2 = st.multiselect("2 分（group2）", ["偏瘫","中度和重度肾脏疾病","糖尿病伴有器官损害","原发性肿瘤","白血病","淋巴瘤"])
        group3 = st.multiselect("3 分（group3）", ["中度和重度肝脏疾病"])
        group4 = st.multiselect("6 分（group4）", ["转移性肿瘤","获得性免疫缺陷综合征（艾滋病）"])
        selections = {"group1": group1, "group2": group2, "group3": group3, "group4": group4}
        charlson = calculate_charlson_score(admission_age, selections)
        st.success(f"Charlson（含年龄加权）= {charlson}")

    st.divider()

    # ---- Labs: manual inputs only
    st.subheader("🧪 实验室检查（全部手动输入；不再支持检验OCR提取）")
    labs_col1, labs_col2, labs_col3, labs_col4 = st.columns(4)

    # group for nicer UI (still uses FEATURE_ORDER for final ordering)
    lab_fields = [
        'wbc','bicarbonate','calcium','chloride','creatinine',
        'alt','ast','bilirubin_total','glucose','sodium','potassium',
        'inr','pt','ptt','hematocrit','hemoglobin','albumin','mch',
        'platelet','rbc','rdw','lactate','ph','be','pao2','paco2'
    ]

    # distribute into 4 columns
    lab_inputs = {}
    chunks = [lab_fields[i::4] for i in range(4)]
    for col, names in zip([labs_col1, labs_col2, labs_col3, labs_col4], chunks):
        with col:
            for name in names:
                lab_inputs[name] = st.number_input(f"{name}", value=0.0, step=0.1)

    st.divider()

    # ---- Text input (kept for user & LLM, NOT for model)
    st.subheader("📄 临床文本（保留输入，但不参与模型预测）")
    clinical_text = st.text_area(
        "可粘贴病程/影像描述/护理记录等（不会影响模型预测，仅用于展示/LLM解读）",
        height=120
    )

    st.subheader("📸 文本拍照识别（可选，仅用于文本）")
    text_image = st.file_uploader("上传文本截图（png/jpg/jpeg）", type=["png","jpg","jpeg"])
    ocr_text = ""
    if text_image is not None:
        if ocr_engine is None:
            st.info("当前部署环境 OCR 不可用（缺少 PaddleOCR 或依赖）。你仍可手动粘贴文本。")
        else:
            try:
                ocr_text = extract_text_from_image(text_image, ocr_engine)
                st.text_area("OCR 识别结果（可复制到上方文本框）", ocr_text, height=120)
            except Exception as e:
                st.error(f"OCR 失败：{e}")

    submitted = st.form_submit_button("🔍 进行风险预测")

# ----------------------------
# 7) Prediction
# ----------------------------
if submitted:
    try:
        # Build dict for all features
        input_dict = {
            'admission_age': float(admission_age),
            'genderscore': float(genderscore),
            'los_hospital': float(los_hospital),
            'los_icu': float(los_icu),
            'heart_rate_24hfinal': float(heart_rate_24hfinal),
            'sbp_ni_24hfinal': float(sbp_ni_24hfinal),
            'dbp_ni_24hfinal': float(dbp_ni_24hfinal),
            'mbp_ni_24hfinal': float(mbp_ni_24hfinal),
            'spo2_24hfinal': float(spo2_24hfinal),
            'temperature_24hfinal': float(temperature_24hfinal),
            'urineoutput_24hr': float(urineoutput_24hr),
            'charlson': float(charlson),
            'o2_flow': float(o2_flow),
            'mechanical_ventilation_time': float(mechanical_ventilation_time),
            'invasive_ventilation': float(invasive_ventilation),
            **{k: float(v) for k, v in lab_inputs.items()}
        }

        # Ensure all required features exist (fill missing with 0)
        for f in FEATURE_ORDER:
            input_dict.setdefault(f, 0.0)

        # Strict ordering
        X = np.array([[input_dict[f] for f in FEATURE_ORDER]], dtype=float)

        # Predict
        prob = float(model.predict_proba(X)[0][1])
        result = "高风险" if prob >= threshold else "低风险"

        st.subheader("📊 预测结果")
        st.metric("再入 ICU 风险概率", f"{prob:.2%}")
        if result == "高风险":
            st.error(f"风险分层：{result}（阈值 {threshold:.3f}）")
        else:
            st.success(f"风险分层：{result}（阈值 {threshold:.3f}）")

        # Show inputs (optional)
        with st.expander("查看本次用于模型预测的结构化输入（按特征顺序）"):
            df_show = pd.DataFrame({"feature": FEATURE_ORDER, "value": [input_dict[f] for f in FEATURE_ORDER]})
            st.dataframe(df_show, use_container_width=True)

        # LLM interpretation (optional)
        st.subheader("🤖 LLM 解读与建议（可选）")
        if clinical_text.strip() == "" and ocr_text.strip() != "":
            clinical_text_for_llm = ocr_text.strip()
        else:
            clinical_text_for_llm = clinical_text.strip()

        prompt = f"""
患者结构化信息已输入（模型仅使用结构化信息）。
模型预测：{result}（概率 {prob:.2%}，阈值 {threshold:.3f}）

补充临床文本（不参与模型预测，仅供参考）：
{clinical_text_for_llm if clinical_text_for_llm else "（未提供）"}

请输出：
1）对该风险结果的简要解释（明确哪些信息来自结构化输入，哪些来自文本）
2）3 条可执行的临床建议（每条一句理由）
3）如信息不足，请列出需要补充的 3 项关键数据（不要编造）
"""
        advice = ask_deepseek_online(prompt)
        st.markdown(advice)

    except Exception as e:
        st.error(f"预测出错：{e}")

# ----------------------------
# 8) Footer tips for deployment
# ----------------------------
with st.expander("✅ 部署提示（Streamlit Cloud）"):
    st.markdown("""
- 你需要把以下文件放在同一个 GitHub 仓库根目录：
  - `app.py`
  - `model_1212.pkl`
  - `threshold_1212.txt`
  - `requirements.txt`（建议加）
- 如果要启用 DeepSeek：
  - Streamlit Cloud → **Settings → Secrets** 添加：
    - `DEEPSEEK_API_KEY = "你的key"`
- 如果部署环境 OCR 不可用（PaddleOCR 依赖问题），程序会自动降级：仅手动粘贴文本也能正常预测。
""")