import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="🚰",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("water_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------
# Header
# -------------------------------
st.markdown("""
# 🚰 Smart Water Potability Prediction System

This Machine Learning application predicts whether water is safe for drinking based on water quality parameters.

---
""")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🔬 Water Parameters")

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

# -------------------------------
# Create Input Array
# -------------------------------
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

# -------------------------------
# Layout Columns
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("📋 Input Summary")

    st.dataframe({
        "Parameter": [
            "pH",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic Carbon",
            "Trihalomethanes",
            "Turbidity"
        ],
        "Value": [
            ph,
            hardness,
            solids,
            chloramines,
            sulfate,
            conductivity,
            organic_carbon,
            trihalomethanes,
            turbidity
        ]
    })

with col2:

    st.subheader("ℹ️ About")

    st.info("""
    This model predicts water potability using:
    
    • Logistic Regression  
    • Decision Tree  
    • SVM  
    • KNN
    
    Best model selected after hyperparameter tuning.
    """)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict Water Quality"):

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)

    try:
        probability = model.predict_proba(scaled_input)[0]
        confidence = max(probability) * 100
    except:
        confidence = 0

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.success("✅ Water is SAFE for drinking")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

    else:

        st.error("❌ Water is NOT SAFE for drinking")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

# -------------------------------
# Water Quality Guide
# -------------------------------
st.markdown("---")

st.subheader("📖 Water Quality Guidelines")

guide1, guide2, guide3 = st.columns(3)

with guide1:
    st.success("""
    Recommended pH
    
    ✔ 6.5 - 8.5
    """)

with guide2:
    st.warning("""
    Turbidity
    
    ✔ Less than 5 NTU
    """)

with guide3:
    st.info("""
    Sulfate
    
    ✔ Less than 500 mg/L
    """)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")

st.markdown("""
### 👨‍💻 Project Details

**Dataset Size:** 3276 Samples

**Features Used:**
- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

**Target:** Potability (Safe / Unsafe)

Built using:
- Python
- Streamlit
- Scikit-Learn
- Machine Learning
""")
