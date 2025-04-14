#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[1]:


from flask import Flask, request, jsonify  # for API
import joblib                              # to load model files
import re                                  # for text cleaning


# # Load the Trained Model and Vectorizer

# In[2]:


# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# # Initialize the Flask App

# In[3]:


# Initialize Flask app
app = Flask(__name__)


# # Define a Text Cleaning Function

# In[4]:


def clean_text(text):
    text = re.sub(r"http\S+", "", text)            # Remove URLs
    text = re.sub(r"@\w+", "", text)               # Remove mentions
    text = re.sub(r"#", "", text)                  # Remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)        # Remove special characters
    text = re.sub(r"\s+", " ", text)               # Replace multiple spaces
    return text.lower().strip()


# # Define a Route to Handle Predictions

# In[7]:


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        tweet = data.get("text", "")
        cleaned = clean_text(tweet)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        sentiment = "positive" if prediction == 1 else "negative"
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# # Run the Flask App

# In[8]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




