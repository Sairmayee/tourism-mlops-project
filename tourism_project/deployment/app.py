
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Model Hub
MODEL_REPO = "SriSair/tourism-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="best_model.pkl",
    repo_type="model"
)

model = joblib.load(model_path)

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    layout="centered"
)

st.title("Wellness Tourism Package Purchase Prediction")

st.write(
    "Enter customer details below to predict whether "
    "the customer is likely to purchase the Wellness Tourism Package."
)

Age = st.number_input("Age", min_value=18, max_value=100, value=30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    min_value=1.0,
    value=15.0
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    value=2
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    min_value=0.0,
    value=3.0
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3.0, 4.0, 5.0]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number Of Trips",
    min_value=0.0,
    value=2.0
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0.0,
    value=1.0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000.0,
    value=25000.0
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [Age],
        "TypeofContact": [TypeofContact],
        "CityTier": [CityTier],
        "DurationOfPitch": [DurationOfPitch],
        "Occupation": [Occupation],
        "Gender": [Gender],
        "NumberOfPersonVisiting": [NumberOfPersonVisiting],
        "NumberOfFollowups": [NumberOfFollowups],
        "ProductPitched": [ProductPitched],
        "PreferredPropertyStar": [PreferredPropertyStar],
        "MaritalStatus": [MaritalStatus],
        "NumberOfTrips": [NumberOfTrips],
        "Passport": [Passport],
        "PitchSatisfactionScore": [PitchSatisfactionScore],
        "OwnCar": [OwnCar],
        "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
        "Designation": [Designation],
        "MonthlyIncome": [MonthlyIncome]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(
            f"Customer is likely to purchase the package "
            f"(Probability: {probability:.2%})"
        )
    else:
        st.warning(
            f"Customer is unlikely to purchase the package "
            f"(Probability: {probability:.2%})"
        )

    st.subheader("Input Data")
    st.dataframe(input_data)
