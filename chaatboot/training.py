# import json
# import pickle
# import random
# import nltk
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline

# # Initialize the lemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()

# # Load intents file
# with open('intents.json') as file:
#     intents = json.load(file)

# words = []
# classes = []
# documents = []
# ignore_words = ['?', '!']

# # Process each intent in the intents file
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         # Tokenize each word in the pattern
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         # Add documents in the corpus
#         documents.append((pattern, intent['tag']))
#         # Add to classes list if it's not already there
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize, lower each word, and remove duplicates
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))

# # Sort classes
# classes = sorted(list(set(classes)))

# # Print information about the dataset
# print(f"{len(documents)} documents")
# print(f"{len(classes)} classes: {classes}")
# print(f"{len(words)} unique lemmatized words")

# # Save words and classes for future use
# pickle.dump(words, open('texts.pkl', 'wb'))
# pickle.dump(classes, open('labels.pkl', 'wb'))

# # Prepare training data
# training_sentences = [doc[0] for doc in documents]
# training_labels = [doc[1] for doc in documents]

# # Convert class labels to numerical labels
# class_to_index = {cls: idx for idx, cls in enumerate(classes)}
# training_labels = [class_to_index[label] for label in training_labels]

# # Create a pipeline with CountVectorizer and MultinomialNB
# model = make_pipeline(CountVectorizer(), MultinomialNB())

# # Train the model
# model.fit(training_sentences, training_labels)

# # Save the trained model
# with open('model.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)

# print("Model created and saved.")

import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Initialize the lemmatizer and download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load intents file
with open('intents1.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process each intent in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((pattern, intent['tag']))
        # Add to classes list if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words and w not in stop_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print information about the dataset
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words")

# Save words and classes for future use
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Prepare training data
training_sentences = [doc[0] for doc in documents]
training_labels = [doc[1] for doc in documents]

# Convert class labels to numerical labels
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
training_labels = [class_to_index[label] for label in training_labels]

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Hyperparameter tuning
param_grid = {
    'tfidfvectorizer__max_df': [0.75, 1.0],
    'tfidfvectorizer__min_df': [1, 5],
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    'multinomialnb__alpha': [0.1, 1.0]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

# Train the model
grid_search.fit(training_sentences, training_labels)

# Save the best model
best_model = grid_search.best_estimator_
with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print("Best model created and saved.")
