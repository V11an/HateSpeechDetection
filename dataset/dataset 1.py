import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


tweet_df = pd.read_csv('train.csv')

tweet_df["labels"]=tweet_df["label"].map({0:"Neutral Speech", 1:"Offensive Speech"})

print(tweet_df.head())

print(tweet_df.info())


#print(tweet_df['tweet'].iloc[0],"\n")
#print(tweet_df['tweet'].iloc[1],"\n")
#print(tweet_df['tweet'].iloc[2],"\n")
#print(tweet_df['tweet'].iloc[3],"\n")
#print(tweet_df['tweet'].iloc[4],"\n")


def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '',tweet)
    tweet = re.sub(r'ð', '', tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)

tweet_df.tweet = tweet_df['tweet'].apply(data_processing)

tweet_df = tweet_df.drop_duplicates('tweet')

# Lemmatization

lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    tweet = [lemmatizer.lemmatize(word) for word in data]
    return data

tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: lemmatizing(x))

#print(tweet_df['tweet'].iloc[0],"\n")
#print(tweet_df['tweet'].iloc[1],"\n")
#print(tweet_df['tweet'].iloc[2],"\n")
#print(tweet_df['tweet'].iloc[3],"\n")
#print(tweet_df['tweet'].iloc[4],"\n")

#Check size of each label(hatespeech and normal tweets)
print(tweet_df['label'].value_counts())

#fig = plt.figure(figsize=(5,5))
#sns.countplot(x='label', data = tweet_df)

#Model creation LOGISTIC REGRESSION USING Tfid vectorizer

vect = TfidfVectorizer(ngram_range=(1,2)).fit(tweet_df['tweet'])
feature_names = vect.get_feature_names()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features: \n{}".format(feature_names[:20]))

# Separating x and y
X = tweet_df['tweet']
Y = tweet_df['label']
X = vect.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_predict = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_predict, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

print(confusion_matrix(y_test, logreg_predict))
print("\n")
print(classification_report(y_test, logreg_predict))

# Preprocess the input text
input_text = "sheila is a bitch"
processed_text = data_processing(input_text)

# Vectorize the preprocessed text
input_vector = vect.transform([processed_text])

# Make predictions
prediction = logreg.predict(input_vector)

# Print the prediction result
print("Prediction:", prediction)
