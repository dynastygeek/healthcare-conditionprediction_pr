#streamlit run app.py

# Import Libraries -
import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('best_drug_name_model.pkl')
vectorizer = joblib.load('best_drug_name_vectorizer.pkl')

# Function to clean a single review
def clean_review(review):
    # Convert HTML entities
    review = review.replace('&#039;', "'").replace('&amp;', '&')
    # Remove special characters using regex
    return re.sub(r'[^\w\s]', '', review)

# Function to load and predict using the best model
def predict_condition(new_review):
    # Clean the new review
    new_review_cleaned = clean_review(new_review)

    # Vectorize the cleaned review
    new_review_vectorized = vectorizer.transform([new_review_cleaned])

    # Predict the condition
    prediction = model.predict(new_review_vectorized)
    return prediction[0]

# Streamlit UI
st.title("Drug Name Predictor App")

# User input section
st.header("Single Input Prediction")
user_input = st.text_area("Enter your review here:")
if st.button("Submit"):
    if user_input:
        predicted_condition = predict_condition(user_input)
        st.success(f"Predicted Drug Name: {predicted_condition}")
    else:
        st.warning("Please enter a review to get a prediction.")

# Bulk upload section
st.header("Bulk Input Prediction")
st.subheader("Sample Input File Format")

# Display a sample table for reference
sample_data = pd.DataFrame({
    'review': [
        'Cancer',
        'Chronic respiratory diseases',
        'Hypertension',
        'Chronic kidney disease',
        'Reproductive health issues',
        'HIV/AIDS'
    ]
})
st.table(sample_data)  # Use st.dataframe() if you want an interactive table

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'review' in df.columns:
        # Clean and predict for all reviews in the DataFrame
        df['predicted_drug_name'] = df['review'].apply(predict_condition)
        
        # Display results
        st.write("Predictions for uploaded reviews:")
        st.dataframe(df)

        # Create a download link for the predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Predictions", data=csv, file_name='predicted_drug_names.csv', mime='text/csv')
    else:
        st.error("CSV file must contain a column named 'review'.")
