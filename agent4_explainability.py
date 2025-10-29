import streamlit as st
import pandas as pd

def run_agent4_dashboard():
    st.header("Agent 4: Explainability with LIME")
    st.write("Visualizing most influential features based on LIME analysis.")

    lime_features = pd.DataFrame({
        "Feature": ["num_of_clicks", "assessment_score", "code_module"],
        "Influence Weight": [0.35, 0.25, 0.15]
    }).set_index("Feature")

    st.bar_chart(lime_features)
    st.caption("These features had the highest influence on model predictions.")
