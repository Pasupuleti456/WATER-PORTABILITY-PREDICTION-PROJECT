import streamlit as st
import numpy as np
import pickle
import random

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="🚰",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = pickle.load(open("water_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("""
<div style='text-align:center'>
    <h1>🚰 AquaGuard AI</h1>
    <h3>Smart Water Quality Assessment System</h3>
    <p>Analyze water quality and determine whether it is safe for drinking using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------
# SIDEBAR INPUTS
# -----------------------------------
st.sidebar.header("🧪 Enter Water Parameters")

ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)

hardness = st.sidebar.slider(
    "Hardness (mg/L)", 0.0, 500.0, 150.0
)

solids = st.sidebar.slider(
    "Solids (ppm)", 0.0, 50000.0, 20000.0
)

chloramines = st.sidebar.slider(
    "Chloramines (ppm)", 0.0, 15.0, 7.0
)

sulfate = st.sidebar.slider(
    "Sulfate (mg/L)", 0.0, 500.0, 330.0
)

conductivity = st.sidebar.slider(
    "Conductivity (μS/cm)", 0.0, 1000.0, 420.0
)

organic_carbon = st.sidebar.slider(
    "Organic Carbon (ppm)", 0.0, 30.0, 14.0
)

trihalomethanes = st.sidebar.slider(
    "Trihalomethanes (μg/L)", 0.0, 150.0, 66.0
)

turbidity = st.sidebar.slider(
    "Turbidity (NTU)", 0.0, 10.0, 4.0
)

# -----------------------------------
# DASHBOARD METRICS
# -----------------------------------
st.subheader("📊 Water Quality Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("pH", round(ph, 2))

with col2:
    st.metric("Hardness", round(hardness, 2))

with col3:
    st.metric("Turbidity", round(turbidity, 2))

# -----------------------------------
# HEALTH CHECK
# -----------------------------------
st.markdown("---")
st.subheader("🧪 Parameter Health Check")

c1, c2 = st.columns(2)

with c1:
    if 6.5 <= ph <= 8.5:
        st.success("✅ pH Level is within safe range")
    else:
        st.warning("⚠️ pH Level is outside safe range")

with c2:
    if turbidity < 5:
        st.success("✅ Turbidity Level is acceptable")
    else:
        st.warning("⚠️ Turbidity Level is high")

# -----------------------------------
# INPUT ARRAY
# -----------------------------------
input_data = np.array([[
    ph,
    hardness,
    solids,
    chloramines,
    sulfate,
    conductivity,
    organic_carbon,
    trihalomethanes,
    turbidity
]])

# -----------------------------------
# PREDICTION
# -----------------------------------
st.markdown("---")

if st.button("🚀 Analyze Water Quality", use_container_width=True):

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    try:
        probability = model.predict_proba(scaled_data)[0]
        confidence = max(probability) * 100
    except:
        confidence = 85

    st.subheader("🔍 Analysis Result")

    if prediction[0] == 1:

        st.balloons()

        st.markdown("""
        <div style='padding:20px;
                    border-radius:10px;
                    background-color:#d4edda;
                    color:black'>
        <h2>✅ SAFE TO DRINK</h2>
        <p>The water sample appears potable and suitable for drinking.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style='padding:20px;
                    border-radius:10px;
                    background-color:#f8d7da;
                    color:black'>
        <h2>❌ NOT SAFE TO DRINK</h2>
        <p>The water sample may contain unsafe characteristics.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Model Confidence")

    st.progress(int(confidence))

    st.write(f"**Confidence Score:** {confidence:.2f}%")

# -----------------------------------
# WATER FACTS
# -----------------------------------
st.markdown("---")

facts = [
    "💧 Only about 1% of Earth's water is suitable for drinking.",
    "🌍 Around 71% of Earth is covered with water.",
    "🚰 Safe drinking water helps prevent many diseases.",
    "🧪 Water quality depends on physical and chemical properties.",
    "💦 Turbidity measures the cloudiness of water.",
    "🌱 Clean water is essential for human health and agriculture."
]

st.subheader("🌍 Did You Know?")

st.info(random.choice(facts))

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption("🚰 AquaGuard AI | Machine Learning Based Water Potability Prediction")
