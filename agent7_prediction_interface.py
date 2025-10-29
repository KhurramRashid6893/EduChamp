# ===============================================
# 🤖 EduChampion - Agent 7: Prediction Interface (Safe, Lazy LSTM)
# ===============================================

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
# Removed matplotlib and MinMaxScaler, as they are now in the other file
from lstm_engagement_view import run_lstm_engagement_prediction

# ---- Config: raw feature names expected for manual / template CSV ----
RAW_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "total_clicks",
    "avg_assessment_score"
]

# ---- Config: LSTM Engagement Prediction ----
# SEQ_LEN has been moved to lstm_engagement_view.py

# ---- Helper: Result Mapping ----
# NOTE: This mapping must match how you encoded 'final_result' during training.
RESULT_MAPPING = {0: "Fail", 1: "Pass", 2: "Withdrawn", 3 : "Distinction"}


def load_models_and_preprocessor(base_path):
    """Load classical models and preprocessor. LSTM is loaded lazily if needed."""
    models = {}
    preprocessor = None

    # load preprocessor
    preproc_path = os.path.join(base_path, "preprocessor.pkl")
    if os.path.exists(preproc_path):
        preprocessor = joblib.load(preproc_path)
    else:
        raise FileNotFoundError("preprocessor.pkl not found in working directory. Please save it during Agent2 training.")

    # classical models
    for name, fname in [
        ("Decision Tree", "decision_tree_model.pkl"),
        ("Logistic Regression", "logistic_regression_model.pkl"),
        ("Random Forest", "random_forest_model.pkl")
    ]:
        p = os.path.join(base_path, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)

    return models, preprocessor

def try_load_lstm(base_path):
    """Try to load LSTM model — returns model or (None, error_message)."""
    lstm_path = os.path.join(base_path, "pattern_discovery_lstm.h5")
    if not os.path.exists(lstm_path):
        return None, f"LSTM file (pattern_discovery_lstm.h5) not found in working directory."

    try:
        # Lazy import of tensorflow/keras
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.losses import MeanSquaredError
        from tensorflow.keras.optimizers import Adam

        # Load the model
        m = load_model(lstm_path, compile=False)
        # Compile it as required
        m.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=["mae"])
        return m, None
    except ImportError:
        return None, "TensorFlow/Keras not installed. Please install it (e.g., pip install tensorflow)."
    except Exception as e:
        return None, f"Failed to import/load LSTM: {e}"

# --- This function has been moved to lstm_engagement_view.py ---
# def run_lstm_engagement_prediction(lstm_model):
#     ... (all logic moved)


def run_agent7_dashboard():
    st.header("🔮 Agent 7 — Prediction Interface (Interactive)")

    base_path = os.getcwd()

    # --- Load models and preprocessor (classical) ---
    try:
        models, preprocessor = load_models_and_preprocessor(base_path)
    except Exception as e:
        st.error(f"❌ Loading error: {e}")
        st.info("Make sure preprocessor.pkl and model .pkl files are in the app folder.")
        return

    st.success("✅ Preprocessor and classical ML models loaded.")

    # --- Sidebar for LSTM Loading and New Feature ---
    st.sidebar.title("🤖 EduChampion Menu")
    st.sidebar.divider()

    # --- Option to load LSTM lazily (Moved to Sidebar) ---
    load_lstm_checkbox = st.sidebar.checkbox("Load LSTM Model (Enables Engagement Prediction)", value=False)
    lstm_model = None  # Initialize
    
    if load_lstm_checkbox:
        # Use session state to cache the loaded model
        if 'lstm_model' not in st.session_state or st.session_state.lstm_model is None:
            
            # FIX: st.sidebar.spinner() does not exist.
            # We must wrap st.spinner() in a 'with st.sidebar:' block
            with st.sidebar:
                with st.spinner("Loading LSTM model..."):
                    model, err = try_load_lstm(base_path)
                
                # Check result and show message *after* spinner is done
                if model is None:
                    # No 'st.sidebar.' prefix needed here as we are in the 'with' block
                    st.warning(f"⚠ LSTM not available: {err}")
                    st.session_state.lstm_model = None
                else:
                    st.success("✅ LSTM loaded successfully.")
                    st.session_state.lstm_model = model
        
        lstm_model = st.session_state.lstm_model
    
    # --- Conditionally run LSTM Engagement UI ---
    if lstm_model:
        # Add the LSTM model to the main model dictionary for the *other* predictor
        models["LSTM (Final Result)"] = lstm_model
        
        st.sidebar.divider()
        # Pass the loaded model to the new function (which is now imported)
        # This function will also need to be fixed if it uses st.sidebar.spinner()
        run_lstm_engagement_prediction(lstm_model)
        st.sidebar.divider()

    # ----------------- Input Form (manual) -----------------
    st.subheader("📋 Manual Input (Final Result Prediction)")
    # Using a form to 'lock' the data for the prediction buttons
    with st.form("single_student_form"):
        col1, col2 = st.columns(2)
        code_module = col1.selectbox("Course Module", ["AAA","BBB","CCC","DDD"])
        code_presentation = col1.selectbox("Presentation", ["2013B","2014J"])
        gender = col2.selectbox("Gender", ["M","F"])
        region = col2.text_input("Region", "London")
        highest_education = col1.selectbox("Highest education", ["A Level or Equivalent","Lower Than A Level","HE Qualification","No Formal quals"])
        imd_band = col2.selectbox("IMD Band", ["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"])
        age_band = col1.selectbox("Age Band", ["0-35","35-55","55<="])
        disability = col2.selectbox("Disability", ["Y","N"])
        num_prev_attempts = col1.number_input("Previous attempts", min_value=0, value=0)
        studied_credits = col2.number_input("Studied credits", min_value=0, value=60)
        total_clicks = col1.number_input("Total VLE clicks", min_value=0, value=200)
        avg_score = col2.number_input("Average assessment score", min_value=0.0, max_value=100.0, value=60.0)
        
        # Form submission required to capture data
        form_submitted = st.form_submit_button("Lock Input Data")

    # Create dataframe only if form is submitted and store in session state
    if form_submitted:
        st.session_state.input_df = pd.DataFrame([{
            "code_module": code_module,
            "code_presentation": code_presentation,
            "gender": gender,
            "region": region,
            "highest_education": highest_education,
            "imd_band": imd_band,
            "age_band": age_band,
            "num_of_prev_attempts": num_of_prev_attempts,
            "studied_credits": studied_credits,
            "disability": disability,
            "total_clicks": total_clicks,
            "avg_assessment_score": avg_score
        }])
        st.info("Input data locked. You can now select a model and predict.")


    # --- Single Prediction Controls ---
    # Ensure input_df exists in session state before showing buttons
    if 'input_df' in st.session_state:
        
        # Model selector
        selected_model_name = st.selectbox("🔎 Choose model for 'Single (Selected)' prediction", list(models.keys()))

        # Prediction buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Predict Single (Selected Model)"):
                try:
                    input_df = st.session_state.input_df
                    # Preprocessing
                    Xp = preprocessor.transform(input_df)
                    if hasattr(Xp, "toarray"):
                        Xp = Xp.toarray()

                    # LSTM single-record handling
                    if "LSTM" in selected_model_name:
                        st.warning("⚠ LSTM (Final Result) prediction requires sequential time-series input. Cannot predict for single manual entry.")
                    else:
                        model = models[selected_model_name]
                        pred_code = model.predict(Xp)[0]
                        pred_label = RESULT_MAPPING.get(int(pred_code), str(pred_code))
                        
                        st.success(f"**{selected_model_name}** prediction: **{pred_label}**")
                        
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(Xp)[0]
                            st.info(f"Confidence: {max(proba)*100:.2f}%")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.info("Check that your preprocessor & model were trained on the same raw columns and order.")

        with col_btn2:
            if st.button("🔮 Predict with All Models (Single)"):
                st.subheader("📊 All Model Predictions (Single)")
                try:
                    input_df = st.session_state.input_df
                    # Preprocessing
                    Xp = preprocessor.transform(input_df)
                    if hasattr(Xp, "toarray"):
                        Xp = Xp.toarray()

                    results_data = []
                    for name, model in models.items():
                        if "LSTM" in name:
                            results_data.append({"Model": name, "Prediction": "Skipped (Requires Sequence)", "Confidence": "N/A"})
                            continue
                        
                        pred_code = model.predict(Xp)[0]
                        pred_label = RESULT_MAPPING.get(int(pred_code), str(pred_code))
                        
                        confidence = "N/A"
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(Xp)[0]
                            confidence = f"{max(proba)*100:.2f}%"
                        
                        results_data.append({"Model": name, "Prediction": pred_label, "Confidence": confidence})
                    
                    st.table(pd.DataFrame(results_data))

                except Exception as e:
                    st.error(f"Prediction failed for 'All Models': {e}")
    else:
        st.caption("Please fill the form and click 'Lock Input Data' to enable single student predictions.")


    # ----------------- Batch Prediction (CSV) -----------------
    st.subheader("📂 Batch Prediction (Final Result)")

    # Offer a template CSV for download
    st.markdown("If you do not have a CSV prepared, download this template, fill it, and re-upload.")
    template_df = pd.DataFrame(columns=RAW_FEATURES)
    # include one sample row
    template_df.loc[0] = ["AAA","2013B","M","London","A Level or Equivalent","0-10%","0-35",0,60,"N",200,60.0]

    csv_bytes = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV Template", data=csv_bytes, file_name="educhampion_prediction_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload CSV with raw features (same columns as template)", type=["csv"], key="batch_final_result_upload")
    
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())

            # Validate columns
            missing = [c for c in RAW_FEATURES if c not in batch_df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing}")
            else:
                # --- Batch Prediction Controls ---
                
                # Model selector for batch
                batch_model_name = st.selectbox("🔎 Choose model for 'Batch (Selected)' prediction", list(models.keys()), key="batch_model_select")
                
                b_col1, b_col2 = st.columns(2)
                
                with b_col1:
                    if st.button("🚀 Predict Batch (Selected Model)"):
                        X_batch = preprocessor.transform(batch_df)
                        if hasattr(X_batch, "toarray"):
                            X_batch = X_batch.toarray()

                        model = models[batch_model_name]
                        if "LSTM" in batch_model_name:
                            st.warning("⚠ LSTM batch prediction requires properly formatted sequences. Not supported in CSV template.")
                        else:
                            preds = model.predict(X_batch)
                            batch_df["Predicted_Result"] = [RESULT_MAPPING.get(int(p), str(p)) for p in preds]
                            
                            st.success("✅ Batch prediction completed.")
                            st.dataframe(batch_df.head(20))

                            out_csv = batch_df.to_csv(index=False).encode("utf-8")
                            st.download_button("⬇ Download predictions CSV", data=out_csv, file_name="educhampion_batch_predictions.csv", mime="text/csv", key="download_batch_single")

                with b_col2:
                    if st.button("🔮 Predict with All Models (Batch)"):
                        X_batch = preprocessor.transform(batch_df)
                        if hasattr(X_batch, "toarray"):
                            X_batch = X_batch.toarray()
                        
                        results_df = batch_df.copy()
                        st.subheader("📊 All Model Predictions (Batch)")

                        for name, model in models.items():
                            if "LSTM" in name:
                                results_df[f"Predicted_{name}"] = "Skipped (Requires Sequence)"
                                continue # Skip LSTM
                            
                            # --- FIX: Was X_row, changed to X_batch ---
                            preds = model.predict(X_batch) 
                            results_df[f"Predicted_{name}"] = [RESULT_MAPPING.get(int(p), str(p)) for p in preds]

                        st.success("✅ Batch 'All Model' prediction completed.")
                        st.dataframe(results_df.head(20))

                        out_csv_all = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇ Download all-model predictions CSV", data=out_csv_all, file_name="educhampion_batch_predictions_ALL.csv", mime="text/csv", key="download_batch_all")

        except Exception as e:
            st.error(f"Batch processing failed: {e}")

# Main entry point
if __name__ == "__main__":
    run_agent7_dashboard()

