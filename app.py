import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. 页面基本配置
st.set_page_config(page_title="MUMPP进展为SMPP的风险预测计算器", layout="wide")
st.title("MUMPP进展为SMPP的风险预测计算器")
st.markdown("基于 GBM 算法与 SHAP 解释架构")

# 2. 加载训练好的模型
@st.cache_resource
def load_model():
    return joblib.load("gbm_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("无法加载模型，请确保您已经运行了 train_model.py 并在同级目录下生成了 gbm_model.pkl 文件。")
    st.stop()

# 3. 侧边栏：获取用户输入的特征数据 (改为直接数字输入)
st.sidebar.header("📝 输入患者指标")
st.sidebar.markdown("请在下方输入框中直接键入具体数值：")

dd_val = st.sidebar.number_input(
    "DD (D-二聚体，ng/mL)", 
    min_value=0.00, max_value=3000.00, value=518.00, step=0.01, format="%.2f"
)

crp_val = st.sidebar.number_input(
    "CRP (C反应蛋白，mg/L)", 
    min_value=0.00, max_value=200.00, value=21.16, step=0.01, format="%.2f"
)

ldh_val = st.sidebar.number_input(
    "LDH (乳酸脱氢酶，U/L)", 
    min_value=0.00, max_value=1000.00, value=251.00, step=0.01, format="%.2f"
)

nlr_val = st.sidebar.number_input(
    "NLR (中性粒细胞/淋巴细胞比)", 
    min_value=0.00, max_value=15.00, value=2.93, step=0.01, format="%.2f"
)

# 构建 DataFrame，注意列名必须和训练模型时的列名严格一致
input_data = pd.DataFrame({
    'DD': [dd_val],
    'CRP': [crp_val],
    'LDH': [ldh_val],
    'NLR': [nlr_val]
})

st.subheader("📊 当前输入的特征值")
st.dataframe(input_data.style.format("{:.2f}"), use_container_width=True)

# 4. 执行预测与 SHAP 可视化
if st.button("🚀 开始风险预测"):
    # 获取预测类别与概率
    pred_class = model.predict(input_data)[0]
    pred_prob = model.predict_proba(input_data)[0][1] # 取类别为 1 (SMPP阳性) 的概率
    
    # 结果展示
    st.markdown("---")
    st.subheader("💡 预测结果")
    col1, col2 = st.columns(2)
    with col1:
        if pred_class == 1:
            st.error(f"**高风险预警:** 模型预测患者有进展为 SMPP 的倾向 (分类: 1)")
        else:
            st.success(f"**低风险:** 模型预测患者进展为 SMPP 的风险较低 (分类: 0)")
    with col2:
        st.info(f"**进展为 SMPP 的概率 (Risk Score):** {pred_prob:.2%}")

    # SHAP 解释 (瀑布图)
    st.markdown("---")
    st.subheader("🔍 SHAP 个体化特征贡献分析")
    st.write("瀑布图展示了当前各项指标是如何将预测基线推向最终风险概率的：**红色箭头**代表该指标增加了进展风险，**蓝色箭头**代表降低了进展风险。")
    
    # 计算当前输入数据的 SHAP 值
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    
    # 渲染 matplotlib 图像到 streamlit
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    
    st.pyplot(fig)