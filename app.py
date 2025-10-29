# ===============================================
# 🎓 EduChampion Dashboard - Modular Integration
# Author: Khurram Rashid | Amity University Mumbai
# ===============================================

import streamlit as st
from agent2_model_performance import run_agent2_dashboard
from agent3_lstm_analysis import run_agent3_dashboard
from agent4_explainability import run_agent4_dashboard
from agent5_sustainability import run_agent5_dashboard
from agent6_student_dashboard import run_agent6_dashboard
from agent7_prediction_interface import run_agent7_dashboard

# Page setup
st.set_page_config(page_title="EduChampion Dashboard", layout="wide")
st.title("🎓 EduChampion: Sustainable & Explainable Learning Analytics Dashboard")
st.caption("Developed by Khurram Rashid | Amity University Mumbai")

# Tabs for each agent
tabs = st.tabs([
    "📊 Model Performance (Agent 2)",
    "🧠 LSTM Pattern Discovery (Agent 3)",
    "💡 Explainability (Agent 4)",
    "🌱 Sustainability (Agent 5)",
    "🎓 Student Visualization (Agent 6)",
    "🔮 Prediction Interface (Agent 7)"
])


with tabs[0]:
    run_agent2_dashboard()

with tabs[1]:
    run_agent3_dashboard()

with tabs[2]:
    run_agent4_dashboard()

with tabs[3]:
    run_agent5_dashboard()

with tabs[4]:
    run_agent6_dashboard()

with tabs[5]:
    run_agent7_dashboard()

