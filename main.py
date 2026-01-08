from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and vectorizer
# Ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found. Please train a model and save it.")
    exit()

app = FastAPI()

# Pydantic model for request body validation
class ReviewRequest(BaseModel):
    review: str

@app.post("/predict/")
def predict_sentiment(request: ReviewRequest):
    """
    Predicts the sentiment of a given text review.
    """
    try:
        # Transform the input text using the loaded vectorizer
        # The vectorizer outputs a sparse matrix by default
        X_sparse = vectorizer.transform([request.review])

        # Convert the sparse matrix to a dense array for prediction
        # Some scikit-learn models (like SVC) require a dense array
        X_dense = X_sparse.toarray()

        # Make a prediction using the loaded model
        prediction = model.predict(X_dense)[0]

        # Determine sentiment based on the prediction (0 for Negative, 1 for Positive)
        sentiment = "Positive" if prediction == 1 else "Negative"

        return {"sentiment": sentiment}
    except Exception as e:
        # Return a server error if something goes wrong during prediction
        return {"error": str(e)}, 500