import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_agent5_dashboard():
    base_path = os.getcwd()
    carbon_logs = pd.concat([
        pd.read_csv(os.path.join(base_path, "agent2_carbon_log.csv")),
        pd.read_csv(os.path.join(base_path, "agent3_carbon_log.csv"))
    ], ignore_index=True)

    model_metrics = pd.read_csv(os.path.join(base_path, "model_performance_metrics.csv"))

    st.header("Agent 5: Sustainability and Eco-Efficiency Metrics")

    st.dataframe(carbon_logs)

    st.write("### 📊 CO₂ Emissions Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(carbon_logs["Agent Task"], carbon_logs["CO₂ (g)"], color="seagreen")
    ax.set_xlabel("CO₂ (g)")
    st.pyplot(fig)

    st.write("### ⚡ Eco-Efficiency Analysis")
    merged = model_metrics.copy()
    merged["CO₂ (g)"] = carbon_logs["CO₂ (g)"][:len(model_metrics)]
    merged["Eco-Efficiency"] = merged["Accuracy"] / merged["CO₂ (g)"]

    st.dataframe(merged[["Model", "Accuracy", "CO₂ (g)", "Eco-Efficiency"]])

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(merged["CO₂ (g)"], merged["Accuracy"], s=100, color="royalblue")
    for i, row in merged.iterrows():
        ax2.text(row["CO₂ (g)"], row["Accuracy"], row["Model"], fontsize=9)
    ax2.set_xlabel("CO₂ (g)")
    ax2.set_ylabel("Accuracy")
    st.pyplot(fig2)
