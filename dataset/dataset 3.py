import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
import nltk
import re
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# Read the data into a DataFrame
data = pd.read_csv("labeled_data.csv")

print(data.head())

data["class"] = data["class"].map({0: "hate speech", 1: "offensive speech", 2: "neutral speech"})


#data["labels"]=data["class"].map({0:"Hate Speech", 1:"Offensive Speech", 2:"Neutral Speech"})

#data=data[["tweet", "labels"]]


# Preprocess the data and extract relevant columns
def clean(text):
    # Your data cleaning operations
    def clean(text):
        text = str(text).lower()
        text = re.sub('[.?]', '', text)
        text = re.sub('https?://\S+|www.\S+', '', text)
        text = re.sub('<.?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w\d\w', '', text)
        text = [word for word in text.split(' ') if word not in stopwords]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)

    return text

data["tweet"] = data["tweet"].apply(clean)
X = data["tweet"]
y = data["class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Vectorize the text data
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Initialize and train the SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Make predictions on the test data
svm_pred = svm.predict(X_test)

# Calculate evaluation metrics
svm_acc = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, average='macro')
svm_recall = recall_score(y_test, svm_pred, average='macro')
svm_f1_score = f1_score(y_test, svm_pred, average='macro')

# Print the evaluation metrics
print("Accuracy:", svm_acc)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1-score:", svm_f1_score)


i="I am happy!!"
i = cv.transform([i]).toarray()
print(svm.predict((i)))

"""
# Preprocess the input text
input_text = "Bill is a black ass nigga"
cleaned_text = clean(input_text)

# Transform the preprocessed text into a vector representation
input_vector = cv.transform([cleaned_text])

# Make the prediction
prediction = svm.predict(input_vector)

# Print the prediction result
print("Prediction:", prediction)
"""
