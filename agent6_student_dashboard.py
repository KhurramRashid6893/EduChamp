import streamlit as st
import pandas as pd
import random

def run_agent6_dashboard():
    st.header("Agent 6: Student Summary Visualization Dashboard")

    st.write("""
    This section presents a unified visualization for individual student analytics — 
    combining predicted risk, explainability, and sustainability indicators.
    """)

    # Simulated student-level data (replace with your model output later)
    data = {
        "Student ID": [f"S{i:03d}" for i in range(1, 11)],
        "Predicted Risk Level": [random.choice(["High", "Medium", "Low"]) for _ in range(10)],
        "Key Factor": [random.choice(["Low VLE clicks", "Low assessment score", "Late submission"]) for _ in range(10)],
        "Sustainability Score": [round(random.uniform(85, 100), 2) for _ in range(10)]
    }
    df = pd.DataFrame(data)

    st.dataframe(df)

    st.write("### 🌿 Average Sustainability Score")
    avg_score = df["Sustainability Score"].mean()
    st.metric("Institutional Sustainability Score", f"{avg_score:.2f}")

    st.caption("This table integrates analytical insights with environmental metrics for each learner.")
