import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
# Ensure the repo_id and filename match your Hugging Face model repository
model_path = hf_hub_download(repo_id="Mahendra-ML/tourism_prediction_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the customer details below to get a prediction.
""")

# User input fields based on the tourism dataset features
# Numerical features
age = st.number_input("Age", min_value=18, max_value=90, value=30)
type_of_contact_map = {'Company Invited': 0, 'Self Inquiry': 1}
type_of_contact_input = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
type_of_contact = type_of_contact_map[type_of_contact_input]
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=60.0, value=10.0, step=0.5)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
preferred_property_star = st.number_input("Preferred Property Star (e.g., 3, 4, 5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
passport = st.selectbox("Passport Holder?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_children_visiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income (USD)", min_value=0.0, value=2500.0, step=100.0)

# Categorical features
occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business', 'Unemployed'])
gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male'])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Executive', 'VP', 'AVP', 'Director'])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])


if st.button("Predict Purchase"):
    # The model was trained with a classification threshold, so we apply it here.
    # Get probability of the positive class (1: ProdTaken)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Use the same classification threshold as during training
    classification_threshold = 0.45 # This was set in the train.py script
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]

    result = "Customer WILL purchase the Wellness Tourism Package!" if prediction == 1 else "Customer will NOT purchase the Wellness Tourism Package."
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
    st.info(f"Probability of Purchase: {prediction_proba[0]:.2f}")
