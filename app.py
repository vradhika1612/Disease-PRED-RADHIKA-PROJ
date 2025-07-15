import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import base64
import os
# from pdf2image import convert_from_path # This is not needed if you're only using download buttons
from genAI import get_health_advice

# ---------------- LOGIN SYSTEM ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Replace this with your own credentials
    if st.button("Login"):
        if email == "admin" and password == "123":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid email or password")

    st.stop()
# ----------------------------------------------
# If authenticated, proceed to the main app

# Load Models
# Using forward slashes for cross-platform compatibility
diabetes_model = pickle.load(open(r'Models/diabetes_pred.pkl', 'rb'))
heart_model = pickle.load(open(r'Models/heart_pred.pkl', 'rb'))
parkinson_model = pickle.load(open(r'Models/parkinson_pred.pkl', 'rb'))
stroke_model = joblib.load('Models/stroke_pred.joblib')

# Define pages for dropdown (only main prediction pages)
disease_pages = ["Home", "Diabetes", "Heart Disease", "Parkinson's", "Stroke"]

# Sidebar Navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

st.sidebar.title("Navigation")

# Determine the initial index for the selectbox
current_selectbox_index = 0
if st.session_state["page"] in disease_pages:
    current_selectbox_index = disease_pages.index(st.session_state["page"])

sidebar_selection = st.sidebar.selectbox(
    "Choose Disease",
    disease_pages,
    index=current_selectbox_index,
    key="page_selector"
)

if sidebar_selection != st.session_state["page"] and st.session_state["page"] in disease_pages:
    st.session_state["page"] = sidebar_selection
    st.rerun()

# User Manual Button in sidebar
st.sidebar.markdown("---")
st.sidebar.title("Documentation")
if st.sidebar.button("📖 User Manual", use_container_width=True):
    st.session_state["page"] = "User Manual"
    st.rerun()

# Logout button
st.sidebar.markdown("""<hr style="margin-top:200px; margin-bottom:10px;">""", unsafe_allow_html=True)
logout_placeholder = st.sidebar.empty()
with logout_placeholder:
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.rerun()


# --- Consolidated Page Content Logic ---
if st.session_state["page"] == "Home":
    st.title("Multi-Disease Prediction App")
    st.markdown("### Choose a disease to predict:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🩺 Diabetes Prediction", use_container_width=True):
            st.session_state["page"] = "Diabetes"
            st.rerun()
    with col2:
        if st.button("❤️ Heart Disease Prediction", use_container_width=True):
            st.session_state["page"] = "Heart Disease"
            st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🧠 Parkinson's Prediction", use_container_width=True):
            st.session_state["page"] = "Parkinson's"
            st.rerun()
    with col4:
        if st.button("🩸 Stroke Prediction", use_container_width=True):
            st.session_state["page"] = "Stroke"
            st.rerun()

elif st.session_state["page"] == "User Manual":
    if st.button("⬅️ Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()
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
    - **0 = Not Diabetic** – your values don’t indicate diabetes.
    - **1 = Diabetic** – you may have diabetes, seek medical confirmation.

    ### Note
    - You'll receive personalized advice and doctor recommendations.
    - The app only predicts diabetes risk based on your inputs.
    - It is always advised to consult a healthcare professional for a proper diagnosis.

    ### Sample Report:
    """)
    st.image("Sample_Parameters/diabetes_parameters.png", caption="Sample Diabetes Report")
    pdf_path = "Sample_Parameters/diabetes.pdf"
    with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Sample Diabetes Report PDF",
                    data=pdf_file.read(),
                    file_name="diabetes_sample_report.pdf",
                    mime="application/pdf",
                    help="Click to download the sample diabetes report in PDF format.")
   
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
    - You’ll get a medical explanation and when to consult a doctor.
    - The app predicts heart disease risk based on your inputs.
    - Always consult a healthcare professional for a proper diagnosis.

    ### Sample Report:
    """)
    st.image("Sample_Parameters/heart_parameters.png", caption="Sample Diabetes Report")
    pdf_path = "Sample_Parameters/heart.pdf"
    with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Sample Diabetes Report PDF",
                    data=pdf_file.read(),
                    file_name="diabetes_sample_report.pdf",
                    mime="application/pdf",
                    help="Click to download the sample diabetes report in PDF format.")

elif st.session_state["page"] == "Parkinson's Manual":
    if st.button("⬅️ Back to User Manual"):
        st.session_state["page"] = "User Manual"
        st.rerun()
    st.title("🧠 Parkinson's Prediction Manual")
    st.markdown("""
    ### How to Use the Parkinson’s Disease Prediction Tool
    1.  Input the following voice signal parameters:
    -   **MDVP:Shimmer**
    -   **MDVP:Shimmer(dB)**
    -   **Shimmer:DDA**
    -   **NHR (Noise-to-Harmonics Ratio)**
    -   **RPDE (Recurrence Period Density Entropy)**
    -   **DFA (Detrended Fluctuation Analysis)**
    -   **Spread1 and Spread2**
    -   **D2 and PPE**

    2.  Click **Predict Parkinson’s Disease**.

    ### Result Interpretation
    - **0 = Unlikely to have Parkinson’s**
    - **1 = Likely Parkinson’s – consult a neurologist**

    ### Note
    - The app offers tailored lifestyle suggestions and care tips based on your voice indicators.
    - It predicts Parkinson’s risk based on your inputs.
    - Always consult a healthcare professional for a proper diagnosis.

    ### Sample Report:
    """)
    st.image("Sample_Parameters/parkinson_parameters.png", caption="Sample Diabetes Report")
    pdf_path = "Sample_Parameters/parkinsons_symptoms_diary.pdf"
    with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Sample Diabetes Report PDF",
                    data=pdf_file.read(),
                    file_name="diabetes_sample_report.pdf",
                    mime="application/pdf",
                    help="Click to download the sample diabetes report in PDF format.")
        
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
    st.image("Sample_Parameters/stroke_parameters.png", caption="Sample Diabetes Report")
    pdf_path = "Sample_Parameters/Stroke-Risk-Assessment.pdf"
    with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Sample Diabetes Report PDF",
                    data=pdf_file.read(),
                    file_name="diabetes_sample_report.pdf",
                    mime="application/pdf",
                    help="Click to download the sample diabetes report in PDF format.")
    

# Diabetes Page
elif st.session_state["page"] == "Diabetes":
    if st.button("⬅️ Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()
    st.title("Diabetes Prediction")

    # Collect inputs from user
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking_history = st.selectbox("Smoking History", ["never", "other"])
    bmi = st.number_input("BMI", min_value=0.0, format="%.5f")
    hba1c = st.number_input("HbA1c Level", min_value=0.0, format="%.5f")
    glucose = st.number_input("Blood Glucose Level", min_value=0.0, format="%.5f")

    # Build input dict matching selected features
    input_data = {
        "gender_Male": 1 if gender == "Male" else 0,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history_never": 1 if smoking_history == "never" else 0,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }

    diabetes_features = [
    "gender_Male",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history_never",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
    ]

    user_input_summary = (
        f"Gender: {gender}\n"
        f"Age: {age}\n"
        f"Hypertension: {hypertension}\n"
        f"Heart Disease: {heart_disease}\n"
        f"Smoking History: {smoking_history}\n"
        f"BMI: {bmi}\n"
        f"HbA1c Level: {hba1c}\n"
        f"Blood Glucose Level: {glucose}"
    )


    # Prepare input DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[diabetes_features]  # Ensure column order

    # Predict
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

# Heart Disease Page
elif st.session_state["page"] == "Heart Disease":
    if st.button("⬅️ Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()
    st.title("Heart Disease Prediction")

    age = st.number_input("Age", format="%.5f")
    sex = st.selectbox("Sex", [0, 1])  # 1=Male, 0=Female
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
        f"Age: {age}\n"
        f"Sex: {sex}\n"
        f"Chest Pain Type: {cp}\n"
        f"Resting Blood Pressure: {trestbps}\n"
        f"Cholesterol: {chol}\n"
        f"Fasting Blood Sugar: {fbs}\n"
        f"Resting ECG: {restecg}\n"
        f"Max Heart Rate: {thalach}\n"
        f"Exercise Induced Angina: {exang}\n"
        f"Oldpeak: {oldpeak}\n"
        f"Slope: {slope}\n"
        f"CA: {ca}\n"
        f"Thalassemia: {thal}"
    )

    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]]

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

# Parkinson's Page
elif st.session_state["page"] == "Parkinson's":
    if st.button("⬅️ Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()
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
        f"MDVP:Shimmer: {shimmer}\n"
        f"MDVP:Shimmer(dB): {shimmer_db}\n"
        f"Shimmer:DDA: {shimmer_dda}\n"
        f"NHR: {nhr}\n"
        f"RPDE: {rpde}\n"
        f"DFA: {dfa}\n"
        f"spread1: {spread1}\n"
        f"spread2: {spread2}\n"
        f"D2: {d2}\n"
        f"PPE: {ppe}"
    )

    # Prediction
    if st.button('Predict Parkinson\'s Disease'):
        try:
            input_data = np.array([[shimmer, shimmer_db, shimmer_dda, nhr, rpde, dfa, spread1, spread2, d2, ppe]])
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

# Stroke Page
elif st.session_state["page"] == "Stroke":
    if st.button("⬅️ Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()
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
        f"Gender: {gender}\n"
        f"Age: {age}\n"
        f"Hypertension: {hypertension}\n"
        f"Heart Disease: {heart_disease}\n"
        f"Ever Married: {ever_married}\n"
        f"Work Type: {work_type}\n"
        f"Residence Type: {residence_type}\n"
        f"Average Glucose Level: {avg_glucose_level}\n"
        f"BMI: {bmi}\n"
        f"Smoking Status: {smoking_status}"
    )

    # When predict button is clicked
    if st.button("Predict Stroke Risk"):
        try:
            # Create a dataframe in the same order as training data
            user_input = {
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }


            # Convert to DataFrame
            import pandas as pd
            input_df = pd.DataFrame([user_input])

            # Make prediction
            prediction = stroke_model.predict(input_df)[0]

            st.success("🟥 High Stroke Risk" if prediction == 1 else "🟩 Low Stroke Risk")
            advice = get_health_advice(user_input_summary, prediction)
            st.markdown(f"### Prediction Result:\n**{prediction}**")
            st.markdown("### Health Advice & Recommendations:")
            st.write(advice)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")