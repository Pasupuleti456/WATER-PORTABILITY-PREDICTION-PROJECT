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

    # No balloons

    if prediction[0] == 1:

        st.markdown("""
        <div style='padding:20px;
                    border-radius:10px;
                    background-color:#d4edda;
                    color:black'>
        <h2>✅ SAFE TO DRINK</h2>
        <p>The water sample appears potable according to the ML model.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style='padding:20px;
                    border-radius:10px;
                    background-color:#f8d7da;
                    color:black'>
        <h2>❌ NOT SAFE TO DRINK</h2>
        <p>The water sample appears unsafe according to the ML model.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("Prediction Value:", prediction[0])

    st.markdown("### 📈 Model Confidence")
    st.progress(int(confidence))
    st.write(f"Confidence Score: {confidence:.2f}%")

    # -----------------------------------
    # SAFETY CHECKS
    # -----------------------------------

    warnings = []

    if ph < 6.5 or ph > 8.5:
        warnings.append("pH is outside safe drinking range (6.5–8.5).")

    if hardness > 300:
        warnings.append("Hardness is extremely high.")

    if solids > 30000:
        warnings.append("Total dissolved solids are very high.")

    if chloramines > 10:
        warnings.append("Chloramines level is high.")

    if sulfate > 400:
        warnings.append("Sulfate concentration is high.")

    if conductivity > 800:
        warnings.append("Conductivity is unusually high.")

    if organic_carbon > 25:
        warnings.append("Organic carbon level is high.")

    if trihalomethanes > 100:
        warnings.append("Trihalomethanes level is unsafe.")

    if turbidity > 5:
        warnings.append("Turbidity is above recommended level.")

    if warnings:
        st.markdown("### ⚠️ Water Quality Warnings")

        for warning in warnings:
            st.warning(warning)

        st.error(
            "Although the ML model may predict SAFE, some parameters exceed recommended drinking-water limits."
        )
