import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
# import openai
import google.generativeai as genai

# Load saved models and tools
rf_model = joblib.load("SavedModels/random_forest_model.pkl")
svm_model = joblib.load("SavedModels/svm_model.pkl")
lgb_model = joblib.load("SavedModels/lightgbm_model.pkl")
scaler = joblib.load("SavedModels/scaler.pkl")
label_encoders = joblib.load("SavedModels/label_encoders.pkl")

# Feature order 
feature_order = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

# Maternal-themed color palette
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }

    .stButton {
    display: flex;
    justify-content: center;
    }
    
    .stButton>button {
        background-color: #d63384 !important;
        color: white !important;
        font-weight: bold !important;
        border: none;
        border-radius: 5px;
        transition: none !important;
    }
    
    .stButton>button:hover {
        background-color: #c22572 !important;
    }

    .stButton>button:active {
        background-color: #a61e4d !important;
    }
     
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 5px;
        border: none !important;
        box-shadow: none !important;
    }

    .stMarkdown h1, .stMarkdown h3 {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""<h1 style='text-align: center; '>Maternal Health Risk Predictor</h1>""", unsafe_allow_html=True)
st.markdown("""<h4 style='text-align: center;'>Enter health parameters to predict the <b>maternal risk level</b> using multiple models.</h4>""", unsafe_allow_html=True)

# Input form
age = st.number_input("Age (in years)", min_value=15, max_value=55, value=30)
sbp = st.number_input("Systolic Blood Pressure", min_value=50, max_value=150, value=120)
dbp = st.number_input("Diastolic Blood Pressure", min_value=60, max_value=150, value=80)
bs = st.number_input("Blood Sugar Level", min_value=3.5, max_value=15.0, value=5.5, format="%.1f")
temp = st.number_input("Body Temperature (°F)", min_value=97.0, max_value=99.0, value=98.6, format="%.1f")
hr = st.number_input("Heart Rate (bpm)", min_value=60, max_value=100, value=75)

# Sidebar chatbot
st.sidebar.title("🤖 AI Assistant")

# Session state to store chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# HUGGINGFACE_API_TOKEN = st.secrets["huggingface"]["api_token"] 

def query_gemini(prompt):
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    
    model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-pro" if needed

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"


# Decode if necessary
def decode(pred):
    return (
        label_encoders['RiskLevel'].inverse_transform([pred])[0]
        if isinstance(pred, (np.integer, int)) else pred
    )

col_predict, col_analyze = st.columns(2)

with col_predict:
    predict_clicked = st.button("Predict Risk", key="predict_risk_button")

with col_analyze:
    analyze_clicked = st.button("Analyze with AI", key="analyze_risk_button")

if predict_clicked:
    
    errors = []

    if dbp > sbp:
        errors.append("Diastolic BP cannot be higher than Systolic BP.")

    if not (3.5 <= bs <= 15.0):
        errors.append("Blood sugar should be between 3.5 and 15.0 mmol/L.")

    if not (60 <= hr <= 200):
        errors.append("Heart rate seems out of expected range (60–200 bpm).")

    if not (97 <= temp <= 104):
        errors.append("Body temperature seems out of expected range (97–104°F).")

    if not (15 <= age <= 55):
        errors.append("Age should be between 15 and 55 years.")

    # Show error messages
    if errors:
        for error in errors:
            st.error(error)
    else:
        input_data = {
            "Age": age,
            "SystolicBP": sbp,
            "DiastolicBP": dbp,
            "BS": bs,
            "BodyTemp": temp,
            "HeartRate": hr,
        }

        user_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(user_df)

        # Predict with all models
        rf_pred = rf_model.predict(scaled_input)[0]
        svm_pred = svm_model.predict(scaled_input)[0]
        lgb_pred = lgb_model.predict(user_df)[0]  # unscaled for LGBM

        predictions = [decode(rf_pred), decode(svm_pred), decode(lgb_pred)]

        # Majority vote (or fallback to RF)
        prediction_label = max(set(predictions), key=predictions.count)

        # Color mapping
        risk_colors = {
            "Low Risk": "rgba(40, 167, 69, 0.5)",
            "Mid Risk": "rgba(255, 193, 7, 0.5)",
            "High Risk": "rgba(220, 53, 69, 0.5)",
        }
        risk_color = risk_colors.get(prediction_label, "#6c757d")

        # Show result
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                height: 60px;
                border-radius: 10px;
                background-color: {risk_color};
                color: white;
                text-align: center;.
                margin: 10px;
            ">
                <p style="margin: 0; font-size: 18px;">
                    Predicted Maternal Risk Level: <b>{prediction_label}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
elif analyze_clicked:
    user_inputs = (
        f"I have the following patient data:\n"
        f"- Age: {age} years\n"
        f"- Systolic BP: {sbp} mmHg\n"
        f"- Diastolic BP: {dbp} mmHg\n"
        f"- Blood Sugar: {bs} mmol/L\n"
        f"- Body Temperature: {temp} °F\n"
        f"- Heart Rate: {hr} bpm\n"
        f"Can you analyze these values of the pregnant person and explain what health risks might be present?"
    )

    st.session_state.chat_history.append(("You", user_inputs))

    with st.sidebar:
        with st.spinner("Analyzing..."):
            analysis_response = query_gemini(user_inputs)

    st.session_state.chat_history.append(("Assistant", analysis_response))




    
def handle_send():
    
    user_input = st.session_state.get("chat_input_form", "").strip()

    if user_input:
        st.session_state.chat_history.append(("You", user_input))
    
        with st.sidebar:
            with st.spinner("Thinking..."):
                if user_input:
                    bot_reply = query_gemini(user_input)
        st.session_state.chat_history.append(("Assistant", bot_reply))
        st.session_state["chat_input_form"] = ""
    else:
        st.warning("Please enter a message before sending.")
        
with st.sidebar:
    st.markdown("**Ask a question about maternal health:**")

    st.markdown("""
        <style>
        div[data-testid="stTextInput"] {
            margin-bottom: 0rem;
        }
        div[data-testid="column"] > div {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input(
            label="", 
            key="chat_input_form", 
            placeholder="Type your question...",
            label_visibility="collapsed"
        )

    with col2:
        submitted = st.button("Send", on_click=handle_send)

# Show chat history
for i in range(len(st.session_state.chat_history) - 1, -1, -2):
    if i >= 1:
        speaker1, message1 = st.session_state.chat_history[i - 1]
        speaker2, message2 = st.session_state.chat_history[i]
        st.sidebar.markdown(f"**{speaker1}:** {message1}")
        st.sidebar.markdown(f"**{speaker2}:** {message2}")
    else:
        speaker, message = st.session_state.chat_history[0]
        st.sidebar.markdown(f"**{speaker}:** {message}")
   