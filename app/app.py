"""
Streamlit App for ML Model Deployment
=====================================

This is your Streamlit application that deploys both your regression and
classification models. Users can input feature values and get predictions.

HOW TO RUN LOCALLY:
    streamlit run app/app.py

HOW TO DEPLOY TO STREAMLIT CLOUD:
    1. Push your code to GitHub
    2. Go to share.streamlit.io
    3. Connect your GitHub repo
    4. Set the main file path to: app/app.py
    5. Deploy!

Author: Brodie Ellis
Dataset: NYC 2019 Airbnb prices
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_models():
    """Load all saved models and artifacts."""
    base_path = Path(__file__).parent.parent / "models"

    models = {}

    try:
        models['regression_model'] = joblib.load(base_path / "regression_model.pkl")
        models['regression_scaler'] = joblib.load(base_path / "regression_scaler.pkl")
        models['regression_features'] = joblib.load(base_path / "regression_features.pkl")

        models['classification_model'] = joblib.load(base_path / "classification_model.pkl")
        models['classification_scaler'] = joblib.load(base_path / "classification_scaler.pkl")
        models['label_encoder'] = joblib.load(base_path / "label_encoder.pkl")
        models['classification_features'] = joblib.load(base_path / "classification_features.pkl")

        try:
            models['binning_info'] = joblib.load(base_path / "binning_info.pkl")
        except Exception:
            models['binning_info'] = None

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Make sure you've trained and saved your models in the notebooks first!")
        return None

    return models


def get_user_inputs(key_prefix=""):
    """Create user-friendly input widgets and return the feature dict expected by the models."""
    col1, col2 = st.columns(2)

    with col1:
        room_type = st.selectbox(
            "Room Type",
            ["Entire home/apt", "Private room", "Shared room"],
            key=f"{key_prefix}room_type",
            help="The type of listing"
        )
        neighbourhood = st.selectbox(
            "Borough",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
            key=f"{key_prefix}neighbourhood",
            help="Which NYC borough is the listing in?"
        )
        reviews_per_month = st.number_input(
            "Reviews per Month",
            min_value=0.0,
            max_value=60.0,
            value=1.0,
            step=0.1,
            key=f"{key_prefix}reviews",
            help="Average number of reviews the listing receives per month"
        )

    with col2:
        availability_ratio = st.slider(
            "Availability Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key=f"{key_prefix}availability",
            help="Proportion of the year the listing is available (0 = never, 1 = always)"
        )
        host_listings = st.number_input(
            "Host's Total Listings",
            min_value=1,
            max_value=300,
            value=1,
            step=1,
            key=f"{key_prefix}host_listings",
            help="How many total listings the host has on Airbnb"
        )
        minimum_nights = st.number_input(
            "Minimum Nights",
            min_value=1,
            max_value=365,
            value=2,
            step=1,
            key=f"{key_prefix}min_nights",
            help="Minimum number of nights required for a booking"
        )

    # Convert user-friendly inputs into the one-hot encoded features the model expects
    input_values = {
        'room_type_Private room': room_type == "Private room",
        'neighbourhood_group_Manhattan': neighbourhood == "Manhattan",
        'reviews_per_month': reviews_per_month,
        'availability_ratio': availability_ratio,
        'room_type_Shared room': room_type == "Shared room",
        'calculated_host_listings_count': host_listings,
        'minimum_nights': minimum_nights,
    }

    return input_values


def make_regression_prediction(models, input_data):
    """Make a regression prediction."""
    input_scaled = models['regression_scaler'].transform(input_data)
    prediction = models['regression_model'].predict(input_scaled)
    return prediction[0]


def make_classification_prediction(models, input_data):
    """Make a classification prediction."""
    input_scaled = models['classification_scaler'].transform(input_data)
    prediction = models['classification_model'].predict(input_scaled)
    label = models['label_encoder'].inverse_transform(prediction)
    return label[0], prediction[0]


# =============================================================================
# SIDEBAR - Navigation
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📈 Price Prediction", "🏷️ Price Category"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This app uses machine learning models trained on the
    **NYC Airbnb 2019** dataset (46,000+ listings).

    - **Regression**: Predicts nightly price ($)
    - **Classification**: Predicts price category (Low / Medium / High)
    """
)
st.sidebar.markdown("**Built by:** Brodie Ellis")
st.sidebar.markdown("Full Stack Academy AI & ML Bootcamp")


# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.title("🏠 NYC Airbnb Price Predictor")
    st.markdown("### Welcome!")

    st.write(
        """
        This application predicts Airbnb listing prices in New York City
        using machine learning models trained on the 2019 NYC Airbnb dataset.

        **What you can do:**
        - 📈 **Price Prediction**: Enter listing details and get a predicted nightly price
        - 🏷️ **Price Category**: Find out if a listing would be Low, Medium, or High priced

        Use the sidebar to navigate between pages.
        """
    )

    st.markdown("---")
    st.markdown("### About This Project")
    st.write(
        """
        **Dataset:** NYC Airbnb Open Data (2019) — 46,000+ listings across all five boroughs

        **Problem Statement:** Can we predict the nightly price of an Airbnb listing
        based on its characteristics like room type, location, availability, and host activity?

        **Models Used:**
        - **Regression:** Gradient Boosting Regressor (R² = 0.45, RMSE = $66)
        - **Classification:** Gradient Boosting Classifier (Accuracy = 65%, F1 = 0.64)

        **Features Used:**
        Room type, borough, reviews per month, availability ratio,
        host listing count, and minimum nights.
        """
    )


# =============================================================================
# REGRESSION PAGE
# =============================================================================
elif page == "📈 Price Prediction":
    st.title("📈 Predict Nightly Price")
    st.write("Enter listing details below to get a predicted nightly price.")

    models = load_models()
    if models is None:
        st.stop()

    st.markdown("---")
    st.markdown("### Listing Details")

    input_values = get_user_inputs(key_prefix="reg_")

    st.markdown("---")

    if st.button("🔮 Predict Price", type="primary"):
        input_df = pd.DataFrame([input_values])
        prediction = make_regression_prediction(models, input_df)

        # Clamp to reasonable range
        prediction = max(prediction, 10)

        st.success(f"### Predicted Nightly Price: ${prediction:,.0f}")
        st.caption("Based on the NYC Airbnb 2019 dataset. Average listing price was $132/night.")

        with st.expander("View Input Summary"):
            display_df = pd.DataFrame([{
                'Room Type': [k for k, v in {
                    'Entire home/apt': not input_values['room_type_Private room'] and not input_values['room_type_Shared room'],
                    'Private room': input_values['room_type_Private room'],
                    'Shared room': input_values['room_type_Shared room']
                }.items() if v][0],
                'Borough': 'Manhattan' if input_values['neighbourhood_group_Manhattan'] else 'Other',
                'Reviews/Month': input_values['reviews_per_month'],
                'Availability': f"{input_values['availability_ratio']:.0%}",
                'Host Listings': input_values['calculated_host_listings_count'],
                'Min Nights': input_values['minimum_nights'],
            }])
            st.dataframe(display_df, hide_index=True)


# =============================================================================
# CLASSIFICATION PAGE
# =============================================================================
elif page == "🏷️ Price Category":
    st.title("🏷️ Predict Price Category")
    st.write("Enter listing details to find out the predicted price range.")

    models = load_models()
    if models is None:
        st.stop()

    class_labels = models['label_encoder'].classes_
    st.info(f"**Possible Categories:** {', '.join(class_labels)}")

    if models['binning_info']:
        with st.expander("How were categories created?"):
            binning = models['binning_info']
            st.write(f"The original **{binning['original_target']}** was split into categories:")
            for i, label in enumerate(binning['labels']):
                if i == 0:
                    st.write(f"- **{label}**: < ${binning['bins'][i+1]:,.0f}")
                elif i == len(binning['labels']) - 1:
                    st.write(f"- **{label}**: >= ${binning['bins'][i]:,.0f}")
                else:
                    st.write(f"- **{label}**: ${binning['bins'][i]:,.0f} to ${binning['bins'][i+1]:,.0f}")

    st.markdown("---")
    st.markdown("### Listing Details")

    input_values = get_user_inputs(key_prefix="class_")

    st.markdown("---")

    if st.button("🔮 Predict Category", type="primary"):
        input_df = pd.DataFrame([input_values])
        predicted_label, predicted_index = make_classification_prediction(models, input_df)

        color_map = {
            'Low': '🟢',
            'Medium': '🟡',
            'High': '🔴'
        }
        emoji = color_map.get(predicted_label, '🔵')

        st.success(f"### Predicted Category: {emoji} {predicted_label}")

        with st.expander("View Input Summary"):
            display_df = pd.DataFrame([{
                'Room Type': [k for k, v in {
                    'Entire home/apt': not input_values['room_type_Private room'] and not input_values['room_type_Shared room'],
                    'Private room': input_values['room_type_Private room'],
                    'Shared room': input_values['room_type_Shared room']
                }.items() if v][0],
                'Borough': 'Manhattan' if input_values['neighbourhood_group_Manhattan'] else 'Other',
                'Reviews/Month': input_values['reviews_per_month'],
                'Availability': f"{input_values['availability_ratio']:.0%}",
                'Host Listings': input_values['calculated_host_listings_count'],
                'Min Nights': input_values['minimum_nights'],
            }])
            st.dataframe(display_df, hide_index=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built by Brodie Ellis | Full Stack Academy AI & ML Bootcamp
    </div>
    """,
    unsafe_allow_html=True
)
