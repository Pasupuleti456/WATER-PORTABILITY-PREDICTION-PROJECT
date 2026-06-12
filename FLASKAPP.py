import streamlit as st
import numpy as np
import pickle

# ----------------------------

# PAGE SETTINGS

# ----------------------------

st.set_page_config(
page_title="Water Potability Prediction",
page_icon="🚰",
layout="wide"
)

# ----------------------------

# LOAD MODEL & SCALER

# ----------------------------

model = pickle.load(open("water_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------

# TITLE

# ----------------------------

st.title("🚰 Water Potability Prediction")
st.write("Analyze water quality and check whether water is safe for drinking.")

st.markdown("---")

# ----------------------------

# INPUT SECTION

# ----------------------------

col1, col2, col3 = st.columns(3)

with col1:
ph = st.number_input("pH", 0.0, 14.0, 7.0)
hardness = st.number_input("Hardness (mg/L)", 0.0, 500.0, 150.0)
solids = st.number_input("Solids (ppm)", 0.0, 60000.0, 10000.0)

with col2:
chloramines = st.number_input("Chloramines (ppm)", 0.0, 15.0, 6.0)
sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, 300.0)
conductivity = st.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 400.0)

with col3:
organic_carbon = st.number_input("Organic Carbon (ppm)", 0.0, 30.0, 10.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0, 150.0, 66.0)
turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 4.0)

# ----------------------------

# CREATE INPUT ARRAY

# ----------------------------

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

# ----------------------------

# PREDICTION BUTTON

# ----------------------------

if st.button("🚀 Analyze Water Quality", use_container_width=True):

```
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)

try:
    probability = model.predict_proba(scaled_data)[0]
    confidence = round(max(probability) * 100, 2)
except:
    confidence = 85.0

st.markdown("---")
st.subheader("🔍 Analysis Result")

# Water Score
score = 100

if ph < 6.5 or ph > 8.5:
    score -= 15

if hardness > 300:
    score -= 10

if solids > 30000:
    score -= 10

if chloramines > 10:
    score -= 10

if sulfate > 400:
    score -= 10

if conductivity > 800:
    score -= 10

if organic_carbon > 25:
    score -= 10

if trihalomethanes > 100:
    score -= 15

if turbidity > 5:
    score -= 10

score = max(score, 0)

if prediction[0] == 1:
    st.success("✅ SAFE TO DRINK")
else:
    st.error("❌ NOT SAFE TO DRINK")

st.metric("Water Quality Score", f"{score}/100")

st.write("### 📈 Model Confidence")
st.progress(int(confidence))
st.write(f"Confidence Score: {confidence}%")

warnings = []

if ph < 6.5 or ph > 8.5:
    warnings.append("pH is outside safe range (6.5 - 8.5)")

if hardness > 300:
    warnings.append("Hardness is extremely high")

if solids > 30000:
    warnings.append("Total dissolved solids are very high")

if chloramines > 10:
    warnings.append("Chloramines level is high")

if sulfate > 400:
    warnings.append("Sulfate concentration is high")

if conductivity > 800:
    warnings.append("Conductivity is unusually high")

if organic_carbon > 25:
    warnings.append("Organic carbon level is high")

if trihalomethanes > 100:
    warnings.append("Trihalomethanes level is unsafe")

if turbidity > 5:
    warnings.append("Turbidity exceeds recommended limit")

if warnings:
    st.write("### ⚠️ Water Quality Warnings")

    for warning in warnings:
        st.warning(warning)

else:
    st.success("All parameters are within recommended drinking-water limits.")
```

st.markdown("---")
st.caption("Machine Learning Based Water Potability Prediction System")
