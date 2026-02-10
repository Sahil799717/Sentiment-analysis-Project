import streamlit as st
import requests

# Set page configuration for a better look and feel
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("📊 Sentiment Prediction App")
st.write("Enter a review and find out if it's **Positive** or **Negative**.")

# Text area for user input
review_text = st.text_area("Enter your review here:", height=150)

# Button to trigger the prediction
if st.button("Predict"):
    # Check if the text area is not empty
    if review_text.strip():
        try:
            # Send the user's review to the FastAPI backend for prediction
            response = requests.post(
                "https://sentiment-analysis-project-1-l46u.onrender.com/predict/",
                json={"review": review_text}
            )

            # Process the response from the backend
            if response.status_code == 200:
                sentiment = response.json().get("sentiment")
                if sentiment == "Positive":
                    st.success(f"✅ Sentiment: {sentiment}")
                elif sentiment == "Negative":
                    st.error(f"❌ Sentiment: {sentiment}")
                else:
                    st.warning("⚠️ Prediction was successful but sentiment is unknown.")
            else:
                # Display a user-friendly error message if the backend request fails
                st.error("Error: Could not get a prediction from the backend.")
                st.write(f"Status Code: {response.status_code}")
                st.write(f"Response: {response.text}")

        except requests.exceptions.ConnectionError:
            # Handle the case where the Streamlit app cannot connect to the FastAPI server
            st.error("Connection Error: Is the FastAPI backend server running?")
        except Exception as e:
            # Handle any other unexpected errors
            st.error(f"An unexpected error occurred: {e}")
    else:
        # Display a warning if the user clicks the button with no text
        st.warning("⚠ Please enter a review before predicting.")