import streamlit as st
import pickle
import pandas as pd

st.title("EMPLOYEE SALARY PREDICTION")

model = pickle.load(open("salary_model.pkl", "rb"))
le_edu = pickle.load(open("le_edu.pkl", "rb"))
le_gen = pickle.load(open("le_gen.pkl", "rb"))
le_job = pickle.load(open("le_job.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.write("Enter employee details to predict salary")

age = st.number_input("Age", 18, 65)
experience = st.number_input("Experience (Years)", 0, 40)

education = st.selectbox("Education Level", le_edu.classes_)
gender = st.selectbox("Gender", le_gen.classes_)
job = st.selectbox("Job Title", le_job.classes_)

if st.button("Predict Salary"):
    input_data = {
        "Age": age,
        "Experience_Years": experience,
        "Education_Level": le_edu.transform([education])[0],
        "Gender": le_gen.transform([gender])[0],
        "Job_Title": le_job.transform([job])[0],
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]

    salary = model.predict(input_df)[0]

    st.success(f"Estimated Salary: ₹{int(salary)}")

st.caption("Salary is an estimate based on past employee data.")
