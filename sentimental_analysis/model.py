import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Load the intent data
with open('intents02.json') as file:
    intents_data = json.load(file)

# Prepare the training data
patterns = []
tags = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Create a TF-IDF vectorizer and logistic regression model pipeline
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
model.fit(patterns, tags)

# Save the trained model
joblib.dump(model, 'model.pkl')
