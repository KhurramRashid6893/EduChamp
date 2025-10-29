import streamlit as st
import pandas as pd
import os

def run_agent2_dashboard():
    base_path = os.getcwd()
    model_metrics = pd.read_csv(os.path.join(base_path, "model_performance_metrics.csv"))

    st.header("Agent 2: Model Performance Comparison")
    st.dataframe(model_metrics)

    st.bar_chart(model_metrics.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]])
    best_model = model_metrics.loc[model_metrics["Accuracy"].idxmax()]
    st.success(f"🏆 Best Performing Model: **{best_model['Model']}** (Accuracy: {best_model['Accuracy']:.2f})")
