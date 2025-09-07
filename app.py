import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import time
# 加载模型
model = xgb.Booster()
model.load_model("xgboost_model.model")

# 设置页面宽度
st.set_page_config(layout="wide")

# 页面标题和简介
st.title("Simplified CPET Based Post-MI Heart Failure Probability Predictor")
st.markdown("""
This tool predicts the likelihood of heart failure (HF) after acute myocardial infarction (AMI) based on patient characteristics and CPET results.

**Instructions:**
- Fill in your details on the left.
- Click **Predict** to see your Post-MI HF probability and recommendations.
""")

# 创建两列布局，左侧输入，右侧显示预测结果
col1, col2 = st.columns([1, 1])  # 左侧 1/3, 右侧 2/3

# **左侧：输入参数**
with st.sidebar:
    st.header("Input Features")
    EF = st.sidebar.number_input("Ejection fraction (EF, %)", min_value=50.0, max_value=100.0, value=55.0, step=0.1)
    BNP = st.sidebar.number_input("NT-pro BNP (pg/mL)", min_value=0.0, max_value=100000.0, value=10.0, step=0.1)
    LDH = st.sidebar.number_input("Lactate dehydrogenase (LDH, U/L)", min_value=0.0, max_value=10000.0, value=100.0,step=0.1)
    LVEDD = st.sidebar.number_input("Left Ventricular End-Diastolic Diameter (LVEDD, mm)", min_value=10.0,max_value=100.0, value=45.0, step=0.1)
    age = st.sidebar.number_input("Age (years)", min_value=0, max_value=150, value=50, step=1)
    VO2KGPEAK = st.sidebar.number_input("Oxygen consumption peak (VO2 peak, ml/kg/min)", min_value=0.0, max_value=100.0, value=18.0, step=0.1)
    VEVCO2SLOPE = st.sidebar.number_input("Minute ventilation/carbon dioxide production slope (VE/VCO2 slope)",min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    PETCO2peak = st.sidebar.number_input("Peak prtial pressure of end tidal carbon dioxide (PETCO2peak, mmHg)",min_value=10.0, max_value=100.0, value=39.0, step=0.1)
    predict_button = st.sidebar.button("Predict")

# **右侧：显示预测结果**
if predict_button:
    with st.spinner("Calculating..."):
        time.sleep(1)  # 模拟计算时间
    st.success("Assessment complete!")

    # 特征编码
    feature_names = ["EF", "NT-pro BNP", "LDH", "LVEDD", "Age", "VO2 peak", "VE/VCO2 slope", "PETCO2 peak"]
    encoded_features = [EF, BNP, LDH, LVEDD, age, VO2KGPEAK, VEVCO2SLOPE, PETCO2peak]
    input_features = np.array(encoded_features).reshape(1, -1)
    dmatrix = xgb.DMatrix(input_features)

    # 预测概率
    probabilities = model.predict(dmatrix, iteration_range=(0, 57))
    predicted_probability = probabilities[0]

    # 风险分组逻辑
    if predicted_probability < 0.381445:
        risk_group = "Low Post-MI HF Probability"
        risk_color = "green"
        advice = "You have a low probability of HF."
    else:
        risk_group = "High Post-MI HF Probability"
        risk_color = "red"
        advice = (
            "You have a high probability of HF. Please consult a healthcare professional as soon as possible "
            "for detailed evaluation and treatment guidance."
        )

    # **显示结果在右侧**
    with col1:
        with st.container():
            st.header("Assessment Results")
            st.markdown(
                f"<h3 style='color:{risk_color};'>Risk Group: {risk_group}</h3>",
                unsafe_allow_html=True
            )
            st.write(advice)

            # **风险评分**
            risk_score = predicted_probability * 10
            st.markdown(f"**Your risk score is: {risk_score:.2f}**")

            # **SHAP 解释**
            st.markdown(
                f"<h3>Predicted probability of HF is {predicted_probability * 100:.2f}%</h3>",
                unsafe_allow_html=True
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame(input_features, columns=feature_names), tree_limit=57)

            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                feature_names=feature_names,
                data=input_features[0],  # 显示实际数值
                output_names=["HF Risk"]  # 定义输出维度
            )

            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(
                explanation,
                max_display=14,
                show=False
            )
            # 优化显示参数
            plt.title(f"Individualized HF Risk Explanation",
                      fontsize=14, pad=20)
            plt.xlabel("SHAP Value", fontsize=12)
            plt.xticks(fontsize=10, rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig("shap_waterfall.png", dpi=300, bbox_inches="tight")
            st.image("shap_waterfall.png")
