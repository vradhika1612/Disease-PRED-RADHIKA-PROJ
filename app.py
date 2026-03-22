import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import base64
import os
from genAI import get_health_advice

# ---------------- LOGIN SYSTEM ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email == "admin" and password == "123":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid email or password")
    st.stop()
# ----------------------------------------------

# Load Models
diabetes_model = pickle.load(open(r'Models/diabetes_pred.pkl', 'rb'))
heart_model = pickle.load(open(r'Models/heart_pred.pkl', 'rb'))
parkinson_model = pickle.load(open(r'Models/parkinson_pred.pkl', 'rb'))
stroke_model = joblib.load('Models/stroke_pred.joblib')

disease_pages = ["Home", "Diabetes", "Heart Disease", "Parkinson's", "Stroke"]

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# nav_select must be set BEFORE the selectbox widget is created.
# We use a pending value to queue changes, then apply before rendering.
if "nav_select" not in st.session_state:
    st.session_state["nav_select"] = "Home"

if "pending_nav" not in st.session_state:
    st.session_state["pending_nav"] = None

# Apply any pending nav change before the widget renders
if st.session_state["pending_nav"] is not None:
    st.session_state["nav_select"] = st.session_state["pending_nav"]
    st.session_state["pending_nav"] = None

def on_nav_change():
    st.session_state["page"] = st.session_state["nav_select"]

# ---- Sidebar ----
st.sidebar.title("Navigation")

st.sidebar.selectbox(
    "Choose Disease",
    disease_pages,
    key="nav_select",
    on_change=on_nav_change
)

st.sidebar.markdown("---")
st.sidebar.title("Documentation")
if st.sidebar.button("📖 User Manual", use_container_width=True):
    st.session_state["page"] = "User Manual"
    st.rerun()

st.sidebar.markdown("""<hr style="margin-top:200px; margin-bottom:10px;">""", unsafe_allow_html=True)
if st.sidebar.button("🚪 Logout", key="logout_btn"):
    st.session_state["authenticated"] = False
    st.rerun()


# ---- Helper: navigate to a main disease page ----
def nav_to(page):
    """Use pending_nav to queue selectbox update before next render."""
    st.session_state["page"] = page
    st.session_state["pending_nav"] = page
    st.rerun()


# ---- Page Routing ----

if st.session_state["page"] == "Home":
    st.title("Multi-Disease Prediction App")
    st.markdown("### Choose a disease to predict:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🩺 Diabetes Prediction", use_container_width=True):
            nav_to("Diabetes")
    with col2:
        if st.button("❤️ Heart Disease Prediction", use_container_width=True):
            nav_to("Heart Disease")

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🧠 Parkinson's Prediction", use_container_width=True):
            nav_to("Parkinson's")
    with col4:
        if st.button("🩸 Stroke Prediction", use_container_width=True):
            nav_to("Stroke")

elif st.session_state["page"] == "User Manual":
    if st.button("⬅️ Back to Home"):
        nav_to("Home")
    st.title("📘 User Manual")
    st.markdown("""
    Learn how to use each disease prediction tool with clear instructions and a sample report.
    Select a disease manual below:
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🩺 Diabetes Manual", use_container_width=True):
            st.session_state["page"] = "Diabetes Manual"
            st.rerun()
    with col2:
        if st.button("❤️ Heart Disease Manual", use_container_width=True):
            st.session_state["page"] = "Heart Disease Manual"
            st.rerun()
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🧠 Parkinson's Manual", use_container_width=True):
            st.session_state["page"] = "Parkinson's Manual"
            st.rerun()
    with col4:
        if st.button("🩸 Stroke Manual", use_container_width=True):
            st.session_state["page"] = "Stroke Manual"
            st.rerun()

elif st.session_state["page"] == "Diabetes Manual":
    if st.button("⬅️ Back to User Manual"):
        st.session_state["page"] = "User Manual"
        st.rerun()
    st.title("🩺 Diabetes Prediction Manual")
    st.markdown("""
    ### How to Use the Diabetes Prediction Tool
    1.  **Select your gender** – Choose either Male or Female.
    2.  **Enter your age** in years.
    3.  Indicate if you have **hypertension** and **heart disease** (Yes = 1, No = 0).
    4.  Select your **smoking history**: "never" or "other".
    5.  Enter your **BMI** (Body Mass Index).
    6.  Enter your **HbA1c level** (average blood sugar level over the past 3 months).
    7.  Enter your current **blood glucose level**.
    8.  Click **Predict Diabetes**.

    ### Result Interpretation
    - **0 = Not Diabetic** – your values don't indicate diabetes.
    - **1 = Diabetic** – you may have diabetes, seek medical confirmation.

    ### Note
    - You'll receive personalized advice and doctor recommendations.
    - The app only predicts diabetes risk based on your inputs.
    - It is always advised to consult a healthcare professional for a proper diagnosis.

    ### Sample Report:
    """)
    st.image("Sample_Parameters/diabetes_parameters.png", caption="Sample Diabetes Report")
    with open("Sample_Parameters/diabetes.pdf", "rb") as f:
        st.download_button("Download Sample Diabetes Report PDF", f.read(),
                           file_name="diabetes_sample_report.pdf", mime="application/pdf")

elif st.session_state["page"] == "Heart Disease Manual":
    if st.button("⬅️ Back to User Manual"):
        st.session_state["page"] = "User Manual"
        st.rerun()
    st.title("❤️ Heart Disease Prediction Manual")
    st.markdown("""
    ### How to Use the Heart Disease Prediction Tool
    1.  Enter your **age**.
    2.  Select your **sex**: 1 for Male, 0 for Female.
    3.  Choose your **chest pain type** (0–3).
    4.  Input your **resting blood pressure**.
    5.  Input your **cholesterol level**.
    6.  Indicate if your **fasting blood sugar > 120 mg/dl** (0 or 1).
    7.  Select your **ECG results** (0–2).
    8.  Enter your **maximum heart rate**.
    9.  Indicate if you have **exercise-induced angina** (0 or 1).
    10. Enter the **ST depression value** (oldpeak).
    11. Choose the **slope of ST segment** (0–2).
    12. Indicate number of **major vessels** (0–3).
    13. Select **thalassemia status** (1–3).

    ### Result Interpretation
    - **0 = No Heart Disease**
    - **1 = Heart Disease Present**

    ### Note
    - You'll get a medical explanation and when to consult a doctor.
    - The app predicts heart disease risk based on your inputs.
    - Always consult a healthcare professional for a proper diagnosis.

    ### Sample Report:
    """)
    st.image("Sample_Parameters/heart_parameters.png", caption="Sample Heart Disease Report")
    with open("Sample_Parameters/heart.pdf", "rb") as f:
        st.download_button("Download Sample Heart Disease Report PDF", f.read(),
                           file_name="heart_sample_report.pdf", mime="application/pdf")

elif st.session_state["page"] == "Parkinson's Manual":
    if st.button("⬅️ Back to User Manual"):
        st.session_state["page"] = "User Manual"
        st.rerun()
    st.title("🧠 Parkinson's Prediction Manual")
    st.markdown("""
    ### How to Use the Parkinson's Disease Prediction Tool
    1.  Input the following voice signal parameters:
    -   **MDVP:Shimmer**
    -   **MDVP:Shimmer(dB)**
    -   **Shimmer:DDA**
    -   **NHR (Noise-to-Harmonics Ratio)**
    -   **RPDE (Recurrence Period Density Entropy)**
    -   **DFA (Detrended Fluctuation Analysis)**
    -   **Spread1 and Spread2**
    -   **D2 and PPE**

    2.  Click **Predict Parkinson's Disease**.

    ### Result Interpretation
    - **0 = Unlikely to have Parkinson's**
    - **1 = Likely Parkinson's – consult a neurologist**

    ### Note
    - The app offers tailored lifestyle suggestions and care tips based on your voice indicators.
    - It predicts Parkinson's risk based on your inputs.
    - Always consult a healthcare professional for a proper diagnosis.

    ### Sample Report:
    """)
    st.image("Sample_Parameters/parkinson_parameters.png", caption="Sample Parkinson's Report")
    with open("Sample_Parameters/parkinsons_symptoms_diary.pdf", "rb") as f:
        st.download_button("Download Sample Parkinson's Report PDF", f.read(),
                           file_name="parkinsons_sample_report.pdf", mime="application/pdf")

elif st.session_state["page"] == "Stroke Manual":
    if st.button("⬅️ Back to User Manual"):
        st.session_state["page"] = "User Manual"
        st.rerun()
    st.title("🩸 Stroke Prediction Manual")
    st.markdown("""
    ### How to Use the Stroke Risk Prediction Tool
    1.  Select your **gender**.
    2.  Enter your **age**.
    3.  Indicate if you have **hypertension** and/or **heart disease** (0 or 1).
    4.  Select whether you have **ever been married**.
    5.  Choose your **work type**.
    6.  Choose your **residence type** (Urban/Rural).
    7.  Input your **average glucose level**.
    8.  Enter your **BMI** (Body Mass Index).
    9.  Select your **smoking status**.

    ### Result Interpretation
    - **0 = Low Stroke Risk**
    - **1 = High Stroke Risk – consult a doctor soon**

    ### Note
    - The model returns early lifestyle advice and flags when urgent care is needed.
    - It predicts stroke risk based on your inputs.
    - Always consult a healthcare professional for a proper diagnosis.

    ### Sample Report:
    """)
    st.image("Sample_Parameters/stroke_parameters.png", caption="Sample Stroke Report")
    with open("Sample_Parameters/Stroke-Risk-Assessment.pdf", "rb") as f:
        st.download_button("Download Sample Stroke Report PDF", f.read(),
                           file_name="stroke_sample_report.pdf", mime="application/pdf")

# ---- Diabetes Page ----
elif st.session_state["page"] == "Diabetes":
    if st.button("⬅️ Back to Home"):
        nav_to("Home")
    st.title("Diabetes Prediction")

    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking_history = st.selectbox("Smoking History", ["never", "other"])
    bmi = st.number_input("BMI", min_value=0.0, format="%.5f")
    hba1c = st.number_input("HbA1c Level", min_value=0.0, format="%.5f")
    glucose = st.number_input("Blood Glucose Level", min_value=0.0, format="%.5f")

    input_data = {
        "gender_Male": 1 if gender == "Male" else 0,
        "age": age, "hypertension": hypertension, "heart_disease": heart_disease,
        "smoking_history_never": 1 if smoking_history == "never" else 0,
        "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose
    }
    diabetes_features = ["gender_Male", "age", "hypertension", "heart_disease",
                         "smoking_history_never", "bmi", "HbA1c_level", "blood_glucose_level"]
    user_input_summary = (
        f"Gender: {gender}\nAge: {age}\nHypertension: {hypertension}\n"
        f"Heart Disease: {heart_disease}\nSmoking History: {smoking_history}\n"
        f"BMI: {bmi}\nHbA1c Level: {hba1c}\nBlood Glucose Level: {glucose}")
    input_df = pd.DataFrame([input_data])[diabetes_features]

    if st.button("Predict Diabetes"):
        try:
            prediction = diabetes_model.predict(input_df)[0]
            st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
            advice = get_health_advice(user_input_summary, prediction)
            st.markdown(f"### Prediction Result:\n**{prediction}**")
            st.markdown("### Health Advice & Recommendations:")
            st.write(advice)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ---- Heart Disease Page ----
elif st.session_state["page"] == "Heart Disease":
    if st.button("⬅️ Back to Home"):
        nav_to("Home")
    st.title("Heart Disease Prediction")

    age = st.number_input("Age", format="%.5f")
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", format="%.5f")
    chol = st.number_input("Cholesterol", format="%.5f")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", format="%.5f")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST depression induced by exercise", format="%.5f")
    slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0–3) colored by fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

    user_input_summary = (
        f"Age: {age}\nSex: {sex}\nChest Pain Type: {cp}\nResting Blood Pressure: {trestbps}\n"
        f"Cholesterol: {chol}\nFasting Blood Sugar: {fbs}\nResting ECG: {restecg}\n"
        f"Max Heart Rate: {thalach}\nExercise Induced Angina: {exang}\nOldpeak: {oldpeak}\n"
        f"Slope: {slope}\nCA: {ca}\nThalassemia: {thal}")
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    if st.button("Predict"):
        try:
            result = heart_model.predict(input_data)
            st.success(f"Prediction: {'Heart Disease' if result[0] == 1 else 'No Heart Disease'}")
            advice = get_health_advice(user_input_summary, result[0])
            st.markdown(f"### Prediction Result:\n**{result[0]}**")
            st.markdown("### Health Advice & Recommendations:")
            st.write(advice)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ---- Parkinson's Page ----
elif st.session_state["page"] == "Parkinson's":
    if st.button("⬅️ Back to Home"):
        nav_to("Home")
    st.title("Parkinson's Disease Prediction")

    shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, format="%.5f")
    shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, format="%.5f")
    shimmer_dda = st.number_input('Shimmer:DDA', min_value=0.0, format="%.5f")
    nhr = st.number_input('NHR (Noise-to-Harmonics Ratio)', min_value=0.0, format="%.5f")
    rpde = st.number_input('RPDE (Recurrence Period Density Entropy)', min_value=0.0, max_value=1.0, format="%.5f")
    dfa = st.number_input('DFA (Detrended Fluctuation Analysis)', min_value=-10.0, max_value=1.0, format="%.5f")
    spread1 = st.number_input('spread1', format="%.5f")
    spread2 = st.number_input('spread2', format="%.5f")
    d2 = st.number_input('D2', min_value=0.0, format="%.5f")
    ppe = st.number_input('PPE (Pitch Period Entropy)', min_value=0.0, format="%.5f")

    user_input_summary = (
        f"MDVP:Shimmer: {shimmer}\nMDVP:Shimmer(dB): {shimmer_db}\nShimmer:DDA: {shimmer_dda}\n"
        f"NHR: {nhr}\nRPDE: {rpde}\nDFA: {dfa}\nspread1: {spread1}\nspread2: {spread2}\n"
        f"D2: {d2}\nPPE: {ppe}")

    if st.button("Predict Parkinson's Disease"):
        try:
            input_data = np.array([[shimmer, shimmer_db, shimmer_dda, nhr, rpde, dfa,
                                    spread1, spread2, d2, ppe]])
            prediction = parkinson_model.predict(input_data)
            if prediction[0] == 1:
                st.error("The person is likely to have Parkinson's Disease.")
            else:
                st.success("The person is unlikely to have Parkinson's Disease.")
            advice = get_health_advice(user_input_summary, prediction[0])
            st.markdown(f"### Prediction Result:\n**{prediction[0]}**")
            st.markdown("### Health Advice & Recommendations:")
            st.write(advice)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ---- Stroke Page ----
elif st.session_state["page"] == "Stroke":
    if st.button("⬅️ Back to Home"):
        nav_to("Home")
    st.title("Stroke Prediction")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", format="%.5f")
    bmi = st.number_input("BMI", format="%.5f")
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    user_input_summary = (
        f"Gender: {gender}\nAge: {age}\nHypertension: {hypertension}\n"
        f"Heart Disease: {heart_disease}\nEver Married: {ever_married}\n"
        f"Work Type: {work_type}\nResidence Type: {residence_type}\n"
        f"Average Glucose Level: {avg_glucose_level}\nBMI: {bmi}\n"
        f"Smoking Status: {smoking_status}")

    if st.button("Predict Stroke Risk"):
        try:
            user_input = {
                'gender': gender, 'age': age, 'hypertension': hypertension,
                'heart_disease': heart_disease, 'ever_married': ever_married,
                'work_type': work_type, 'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level, 'bmi': bmi,
                'smoking_status': smoking_status
            }
            input_df = pd.DataFrame([user_input])
            prediction = stroke_model.predict(input_df)[0]
            st.success("🟥 High Stroke Risk" if prediction == 1 else "🟩 Low Stroke Risk")
            advice = get_health_advice(user_input_summary, prediction)
            st.markdown(f"### Prediction Result:\n**{prediction}**")
            st.markdown("### Health Advice & Recommendations:")
            st.write(advice)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
