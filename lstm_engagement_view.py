# lstm_engagement_view.py
# ===============================================
# 🤖 EduChampion - Module: LSTM Engagement Prediction
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---- Config: LSTM Engagement Prediction ----
SEQ_LEN = 10

def run_lstm_engagement_prediction(lstm_model):
    """
    Renders the UI for the LSTM Engagement Prediction feature in the MAIN PANEL.
    Takes the already-loaded LSTM model as input.
    """
    
    # --- All 'st.sidebar' calls are changed to 'st.' ---
    
    st.info(f"This tool uses the loaded LSTM to predict the 'next day clicks' based on a sequence of {SEQ_LEN} days synthesized from a single 'total_clicks' value.")
    
    uploaded_file = st.file_uploader(
        "Upload test data (must contain 'total_clicks' column)", 
        type=["csv"], 
        key="lstm_engagement_upload"
    )
    
    if uploaded_file:
        try:
            df_test = pd.read_csv(uploaded_file)
            
            if "total_clicks" not in df_test.columns:
                st.error("❌ 'total_clicks' column missing in uploaded file!")
                return
            
            # --- This 'with st.sidebar:' block is REMOVED ---
            
            # --- Spinner will now run in the main panel ---
            with st.spinner("Preparing sequences and running predictions..."):
                
                # --- Section 3: Prepare synthetic time-series data (from Colab) ---
                scaler = MinMaxScaler()
                # Fit scaler on the uploaded data to be consistent
                clicks_scaled = scaler.fit_transform(df_test[["total_clicks"]])

                X_test = []
                for c in clicks_scaled.flatten():
                    # Synthesize a sequence
                    noise = np.clip(np.random.normal(0, 0.05, SEQ_LEN), -0.1, 0.1)
                    seq = np.clip(c + noise, 0, 1)
                    X_test.append(seq)

                X_test = np.array(X_test).reshape(-1, SEQ_LEN, 1)
                st.caption(f"Synthesized {X_test.shape[0]} sequences of length {SEQ_LEN}.")

                # --- Section 4: Make predictions (from Colab) ---
                y_pred_scaled = lstm_model.predict(X_test, verbose=0)
                y_pred = scaler.inverse_transform(y_pred_scaled)

                df_test["predicted_next_day_clicks"] = y_pred.round(2)
                df_test["engagement_change_%"] = (
                    ((df_test["predicted_next_day_clicks"] - df_test["total_clicks"]) / df_test["total_clicks"]) * 100
                ).round(2).fillna(0) # Handle division by zero if total_clicks is 0

            # --- Show results *after* spinner, in main panel ---
            st.success("✅ Predictions generated.")
            st.dataframe(df_test.head())

            out_csv = df_test.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download engagement predictions", 
                data=out_csv, 
                file_name="lstm_engagement_predictions.csv", 
                mime="text/csv"
            )

            # --- Section 5: Visualization (from Colab) ---
            st.subheader("📊 Visualization (Top 10)")
            plt.figure(figsize=(10, 6))
            top10 = df_test.head(10)
            
            # Use student index for plotting
            student_indices = top10.index
            student_labels = [f"Student {i}" for i in student_indices]

            plt.plot(student_indices, top10["total_clicks"], marker='o', label="Actual Clicks (Basis)", linewidth=2)
            plt.plot(student_indices, top10["predicted_next_day_clicks"], marker='s', label="Predicted Next-Day Clicks", linewidth=2, linestyle="--")
            
            plt.xticks(student_indices, student_labels, rotation=45, ha="right")
            plt.title("Actual vs Predicted Engagement (Top 10 Students)")
            plt.xlabel("Students (from uploaded file)")
            plt.ylabel("Clicks Count")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Use st.pyplot() to display in Streamlit
            st.pyplot(plt.gcf())
            plt.clf() # Clear figure for next run

        except Exception as e:
            # Show error in main panel
            st.error(f"Failed during LSTM prediction: {e}")

