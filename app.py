import streamlit as st
import pandas as pd
import pickle

# Load your trained model
model = pickle.load(open('dropped.pkl', 'rb'))

st.set_page_config(page_title="Dropout Predictor", layout="centered")
st.title("🎓 Student Dropout Prediction")

st.markdown("Enter the student’s academic & financial details below:")

# Example inputs (you can adjust these to match your actual model features)
Age = st.slider("Age", 17, 50, 22)
avg_enrolled = st.number_input("Average Enrolled Units", min_value=0.0)
avg_approved = st.number_input("Average Approved Units", min_value=0.0)
avg_grade = st.number_input("Average Grade", min_value=0.0)
avg_without_evaluations = st.number_input("Average Without Evaluations", min_value=0.0)
Tuition_fees_up_to_date = st.selectbox("Tuition Fees Up-to-Date", [0, 1])
Debtor = st.selectbox("Is the Student a Debtor?", [0, 1])
Gender = st.selectbox("Gender", [0, 1])  # Adjust if needed

# Predict
if st.button("🔮 Predict Dropout"):
    input_df = pd.DataFrame([{
        'Age': Age,
        'avg_enrolled': avg_enrolled,
        'avg_approved': avg_approved,
        'avg_grade': avg_grade,
        'avg_without_evaluations': avg_without_evaluations,
        'Tuition_fees_up_to_date': Tuition_fees_up_to_date,
        'Debtor': Debtor,
        'Gender': Gender
    }])

    prediction = model.predict(input_df)[0]
    label = {0: "🎓 Will Graduate", 1: "⚠️ Likely to Dropout"}

    st.success(f"Prediction: {label[prediction]}")
