import os
import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# --------------------------------------------------
st.set_page_config(
    page_title="Travel Package Purchase Prediction",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
# BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "tourism_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
FULL_PIPELINE_PATH = os.path.join(BASE_DIR, "full_pipeline.pkl")  # OPTIONAL

# --------------------------------------------------
# LOAD MODEL & PREPROCESSOR (CACHED)
# --------------------------------------------------
import pickle

@st.cache_resource
def load_artifacts():
    with open("tourism_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


try:
    model_or_pipeline, preprocessor, mode = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model files: {e}")
    st.stop()

# --------------------------------------------------
# PAGE HEADINGS
# --------------------------------------------------
st.title("🌍 Travel Package Purchase Prediction")
st.caption("Predict whether a customer will purchase a travel package")
st.divider()

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
user_input = {}

c1, c2, c3 = st.columns(3)
with c1:
    user_input["Age"] = st.number_input("Age", 18, 70, 36)
with c2:
    user_input["CityTier"] = st.number_input("City Tier (1–3)", 1, 3, 1)
with c3:
    user_input["DurationOfPitch"] = st.number_input("Duration Of Pitch (minutes)", 1, 60, 20)

c1, c2, c3 = st.columns(3)
with c1:
    user_input["NumberOfPersonVisiting"] = st.number_input("Number Of Persons Visiting", 1, 10, 3)
with c2:
    user_input["NumberOfFollowups"] = st.number_input("Number Of Followups", 0, 10, 4)
with c3:
    user_input["PreferredPropertyStar"] = st.number_input("Preferred Property Star", 1, 5, 3)

c1, c2, c3 = st.columns(3)
with c1:
    user_input["NumberOfTrips"] = st.number_input("Number Of Trips", 0, 20, 3)
with c2:
    user_input["PitchSatisfactionScore"] = st.number_input("Pitch Satisfaction Score", 1, 5, 3)
with c3:
    user_input["NumberOfChildrenVisiting"] = st.number_input("Number Of Children Visiting", 0, 10, 0)

c1, c2, c3 = st.columns(3)
with c1:
    user_input["MonthlyIncome"] = st.number_input("Monthly Income", 5000, 200000, 22418)
with c2:
    user_input["TypeofContact"] = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
with c3:
    user_input["Occupation"] = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

c1, c2, c3 = st.columns(3)
with c1:
    user_input["Gender"] = st.selectbox("Gender", ["Male", "Female"])
with c2:
    user_input["ProductPitched"] = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
with c3:
    user_input["MaritalStatus"] = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

c1, c2, c3 = st.columns(3)
with c1:
    user_input["Passport"] = st.selectbox("Passport", ["Yes", "No"])
with c2:
    user_input["OwnCar"] = st.selectbox("Own Car", ["Yes", "No"])
with c3:
    user_input["Designation"] = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

# --------------------------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------------------------
input_df = pd.DataFrame([user_input])

# Convert Yes/No → Binary (must match training)
input_df["Passport"] = input_df["Passport"].map({"Yes": 1, "No": 0})
input_df["OwnCar"] = input_df["OwnCar"].map({"Yes": 1, "No": 0})

st.subheader("📊 Customer Input Data")
st.dataframe(input_df, use_container_width=True)
st.divider()

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
l, c, r = st.columns([3, 2, 3])
with c:
    predict_btn = st.button("🚀 Predict", use_container_width=True)

# --------------------------------------------------
# PREDICTION (PRODUCTION SAFE)
# --------------------------------------------------
if predict_btn:
    try:
        # ---- CASE 1: FULL PIPELINE (BEST PRACTICE) ----
        if mode == "pipeline":
            prediction = model_or_pipeline.predict(input_df)[0]
            probability = model_or_pipeline.predict_proba(input_df)[0][1]

        # ---- CASE 2: SEPARATE PREPROCESSOR + MODEL ----
        else:
            # Enforce training schema
            if hasattr(preprocessor, "feature_names_in_"):
                expected_cols = preprocessor.feature_names_in_
                input_df = input_df.reindex(columns=expected_cols)
            else:
                st.error("❌ Preprocessor schema not found.")
                st.stop()

            X = preprocessor.transform(input_df)
            prediction = model_or_pipeline.predict(X)[0]
            probability = model_or_pipeline.predict_proba(X)[0][1]

        prob_percent = round(probability * 100, 2)

        if prediction == 1:
            st.success("✅ Customer is LIKELY to purchase the package")
            st.balloons()
        else:
            st.error("❌ Customer is UNLIKELY to purchase the package")

        st.metric("Purchase Probability", f"{prob_percent}%")
        st.progress(int(prob_percent))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
