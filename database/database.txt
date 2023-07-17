import sqlite3
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Load the labeled dataset from the database
def load_dataset_from_database():
    # Connect to the database
    conn = sqlite3.connect('dataset.db')
    cursor = conn.cursor()
    
    # Retrieve the labeled dataset from the database
    cursor.execute('SELECT text, label FROM dataset_table')
    dataset = cursor.fetchall()
    
    # Close the database connection
    cursor.close()
    conn.close()
    
    return dataset

# Convert text data into TF-IDF features
def extract_features(dataset):
    texts = [text for text, _ in dataset]
    labels = [label for _, label in dataset]
    
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    
    return features, labels

# Train a hate speech detection model
def train_model(features, labels):
    model = LogisticRegression()
    model.fit(features, labels)
    
    return model

# Evaluate the model
def evaluate_model(model, eval_features, eval_labels):
    eval_predictions = model.predict(eval_features)
    
    accuracy = accuracy_score(eval_labels, eval_predictions)
    precision = precision_score(eval_labels, eval_predictions)
    recall = recall_score(eval_labels, eval_predictions)
    f1 = f1_score(eval_labels, eval_predictions)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# Save the trained model
def save_model(model, filename):
    joblib.dump(model, filename)

# Load the trained model
def load_model(filename):
    return joblib.load(filename)

# Classify a new input using the loaded model
def classify_input(new_input, model, vectorizer):
    new_input_features = vectorizer.transform([preprocess_text(new_input)])
    prediction = model.predict(new_input_features)
    
    return prediction[0]

# Load the dataset from the database
dataset = load_dataset_from_database()

# Preprocess the dataset
preprocessed_dataset = [(preprocess_text(text), label) for text, label in dataset]

# Split the dataset into training and evaluation sets
train_data = preprocessed_dataset[:800]
eval_data = preprocessed_dataset[800:]

# Extract features from the training and evaluation sets
train_features, train_labels = extract_features(train_data)
eval_features, eval_labels = extract_features(eval_data)

# Train the hate speech detection model
model = train_model(train_features, train_labels)

# Evaluate the model
evaluate_model(model, eval_features, eval_labels)

# Save the trained model
save_model(model, 'hate_speech_model.pkl')

# Load the trained model
loaded_model = load_model('hate_speech_model.pkl')

# Example usage: classify a new input
new_input = "This is a hateful comment!"
new_input_preprocessed = preprocess_text(new_input)
prediction = classify_input(new_input_preprocessed, loaded_model, vectorizer)
print("Prediction:", prediction)
