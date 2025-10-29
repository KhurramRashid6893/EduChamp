import streamlit as st
import numpy as np
import os

# --- 1. Safe TensorFlow Import ---
# We try to import TensorFlow and the specific functions we need.
# If it fails, we set a flag and the rest of the app will know.

TENSORFLOW_AVAILABLE = False
load_model_func = None  # We'll store the load_model function here

try:
    # Try to import the necessary components
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    
    # If successful, assign the function and set the flag
    load_model_func = load_model
    TENSORFLOW_AVAILABLE = True

except ImportError:
    # If it fails, TENSORFLOW_AVAILABLE remains False, and
    # st.sidebar.warning() will be triggered later.
    pass
# --- End of Safe Import ---


# --- 2. Cached Model Loading Function ---
# This function will *only* be called if TENSORFLOW_AVAILABLE is True.
# We use @st.cache_resource to load the model once and keep it in memory.

@st.cache_resource
def load_safe_model(model_path):
    """
    Safely loads the Keras model using the imported load_model_func.
    Returns (model, error_message)
    """
    try:
        # We MUST use the 'load_model_func' variable we defined earlier
        model = load_model_func(model_path)
        return model, None  # Success!

    except (OSError, IOError):
        # This catches "file not found" errors
        return None, f"⚠ Model file not found. (Looking for: {model_path})"
    
    except Exception as e:
        # This catches other errors, like version mismatches or corrupt files
        # (e.g., the 'Unable to convert function' error you saw)
        return None, f"⚠ Model failed to load. Error: {e}"

# --- 3. Sidebar UI Function ---
# This function runs the actual ML part of the app in the sidebar.

def run_ml_sidebar():
    """
    Renders the ML prediction UI in the sidebar.
    This is only called if TensorFlow is available.
    """
    st.sidebar.success("✅ TensorFlow is available.")
    
    MODEL_PATH = "pattern_discovery_lstm.h5"  # Your model file
    
    # --- Safe Model Load Check ---
    # We call our cached function
    model, error = load_safe_model(MODEL_PATH)

    # If loading failed, show the error and stop the ML part.
    if error:
        st.sidebar.error(error)
        st.sidebar.info("The rest of the app will continue to work.")
        return
    
    # --- Model Loaded Successfully ---
    st.sidebar.success("✅ LSTM Model loaded successfully!")
    st.sidebar.header("Run Prediction")
    
    st.sidebar.write("Simulating LSTM input (10 timesteps):")
    # Example: A slider to get an input value
    data_input = st.sidebar.slider("Input Value", 0.0, 1.0, 0.5, 0.01)
    
    if st.sidebar.button("Predict"):
        try:
            # Format data for LSTM: (batch_size, timesteps, features)
            # This is just a dummy example; adjust to your model's needs.
            dummy_sequence = np.full((1, 10, 1), data_input)
            
            # Use the loaded model
            prediction = model.predict(dummy_sequence)
            
            # Display the result
            st.sidebar.metric("Model Prediction", f"{prediction[0][0]:.4f}")

        except Exception as e:
            st.sidebar.error(f"Prediction failed: {e}")

# --- 4. Main App ---
# This is the main entry point of your Streamlit app.

def main():
    st.title("Safe TensorFlow App Example")

    # --- A. Non-ML Part (Always Runs) ---
    st.header("This part always works!")
    st.write("""
    This main panel is not dependent on TensorFlow.
    You can use sliders, text inputs, and other widgets here,
    and they will work perfectly even if the ML model fails
    to load or if TensorFlow is not installed.
    """)
    
    st.slider("My Unrelated Slider", 0, 100, 50)
    st.text_input("My Unrelated Text Box", "Hello world!")

    # --- B. ML Part (Sidebar, Conditional) ---
    st.sidebar.title("ML Features")
    
    if TENSORFLOW_AVAILABLE:
        # If the import was successful, run the ML sidebar UI
        run_ml_sidebar()
    else:
        # If the import failed, show the fallback message
        st.sidebar.warning("⚠ TensorFlow not available. ML features are disabled.")
        st.sidebar.info("Please install TensorFlow to enable prediction.")

# --- Run the app ---
if __name__ == "__main__":
    main()
