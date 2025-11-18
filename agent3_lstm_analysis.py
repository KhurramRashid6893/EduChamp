import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os 

def run_agent3_dashboard():
    base_path = os.getcwd()
    lstm_acc = pd.read_csv(os.path.join(base_path, "lstm_accuracy_results.csv"))
    lstm_epoch = pd.read_csv(os.path.join(base_path, "lstm_best_epoch_summary.csv"))
    lstm_loss = pd.read_csv(os.path.join(base_path, "lstm_training_loss_values.csv"))

    st.header("Agent 3: LSTM Sequential Model Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Approx. Accuracy (%)", f"{lstm_acc.iloc[0, 3]:.2f}")
    col2.metric("Best Epoch", f"{lstm_epoch.iloc[0, 0]}")
    col3.metric("MSE", f"{lstm_acc.iloc[0, 1]:.5f}")

    st.write("### 📘 Best Epoch Summary")
    st.dataframe(lstm_epoch)

    st.write("### 📉 Training vs Validation Loss")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lstm_loss.index + 1, lstm_loss["loss"], label="Training Loss", marker='o')
    ax.plot(lstm_loss.index + 1, lstm_loss["val_loss"], label="Validation Loss", linestyle="--", marker='s')
    ax.legend(); ax.grid(True)
    st.pyplot(fig)
