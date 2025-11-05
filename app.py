import streamlit as st
import pandas as pd
from joblib import load

st.title("🧠 Cyberbullying Detector")

model = load("/model_pipeline.joblib")

text = st.text_area("Enter a comment:")
if st.button("Analyze"):
    p = model.predict_proba([text])[0][1]
    result = "Toxic" if p >= 0.5 else "Safe"
    st.write("Prediction:", result)
    st.write("Confidence:", round(p, 2))
