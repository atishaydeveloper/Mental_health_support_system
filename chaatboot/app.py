import os
import json
import pickle
import random
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request

# Disable certain TensorFlow optimizations (if needed)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load words and classes
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load intents from JSON
with open('intents1.json', 'r') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return bag

def predict_class(sentence, model):
    bow_vector = bow(sentence, words)
    cleaned_sentence = ' '.join(clean_up_sentence(sentence))  # Join list of words into a string
    prediction = model.predict_proba([cleaned_sentence])[0]  # Use predict_proba to get probabilities

    # Debugging prints
    print(f"Cleaned sentence: {cleaned_sentence}")
    print(f"Prediction: {prediction}")
    print(f"Prediction type: {type(prediction)}")
    print(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'N/A'}")

    try:
        predicted_class_index = prediction.argmax()
        predicted_class = classes[predicted_class_index]
        probability = prediction[predicted_class_index]
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Predicted class: {predicted_class}")
        print(f"Probability: {probability}")
    except (IndexError, AttributeError) as e:
        print(f"Error accessing prediction: {e}")
        return []

    return [{"intent": predicted_class, "probability": str(probability)}]

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']
                result = random.choice(responses)
                return result
    return "Sorry, I didn't understand that."

print('maitri is here for you!!')

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    print("Chatbot response: ", res)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("User text: " + userText)

    chatbot_response_text = chatbot_response(userText)
    return chatbot_response_text

if __name__ == "__main__":
    app.run()
