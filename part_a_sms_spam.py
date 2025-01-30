import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Download stopwords if not already downloaded (uncomment if needed)
# import nltk
# nltk.download("stopwords")

# Define stopwords (English + Turkish)
stop_words = set(stopwords.words("english")) | {"ve", "bir", "bu", "da", "de", "için", "ile", "mı", "mu", "ne", "niye"}

# Initialize Stemmer
stemmer = PorterStemmer()

# Function for text preprocessing (cleaning + stemming)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenization
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Stemming + stopwords removal
    return " ".join(words)

# Load dataset
df = pd.read_csv("datasets/sms_spam_train.csv")

# Analyze message lengths
df["Message Length"] = df["Message"].apply(len)

# Plot histogram of message lengths (Spam vs. Ham)
plt.figure(figsize=(10, 5))
sns.histplot(df[df["Label"] == "ham"]["Message Length"], bins=50, color="blue", label="Ham", kde=True, alpha=0.6)
sns.histplot(df[df["Label"] == "spam"]["Message Length"], bins=50, color="red", label="Spam", kde=True, alpha=0.6)
plt.legend()
plt.title("Distribution of Message Lengths: Spam vs. Ham")
plt.xlabel("Message Length (characters)")
plt.ylabel("Number of Messages")
plt.show()

# Display statistics for message lengths per category
print(df.groupby("Label")["Message Length"].describe())

# Apply text preprocessing
df["Processed Message"] = df["Message"].apply(preprocess_text)

# Create WordClouds for Ham and Spam messages
plt.figure(figsize=(12, 6))

# WordCloud for Ham
plt.subplot(1, 2, 1)
plt.title("Most Common Words in Ham Messages")
wordcloud_ham = WordCloud(width=400, height=400, background_color="white").generate(" ".join(df[df["Label"] == "ham"]["Processed Message"]))
plt.imshow(wordcloud_ham, interpolation="bilinear")
plt.axis("off")

# WordCloud for Spam
plt.subplot(1, 2, 2)
plt.title("Most Common Words in Spam Messages")
wordcloud_spam = WordCloud(width=400, height=400, background_color="white", colormap="Reds").generate(" ".join(df[df["Label"] == "spam"]["Processed Message"]))
plt.imshow(wordcloud_spam, interpolation="bilinear")
plt.axis("off")

plt.show()

# Convert labels into numerical values (0 = Ham, 1 = Spam)
label_encoder = LabelEncoder()
df["Label Encoded"] = label_encoder.fit_transform(df["Label"])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["Processed Message"], df["Label Encoded"], test_size=0.2, random_state=42, stratify=df["Label Encoded"])

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Output of the dimensions after vectorization
print(f"Shape of X_train: {X_train_tfidf.shape}")
print(f"Shape of X_test: {X_test_tfidf.shape}")

# Train SVM model
svm_model = SVC(kernel="linear", C=1.0)  # Linear kernel, C is the regularization parameter
svm_model.fit(X_train_tfidf, y_train)

# Predictions on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model Performance:\n Accuracy: {accuracy:.4f}\n Precision: {precision:.4f}\n Recall: {recall:.4f}\n F1-Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# All Terminal Outputs for Part A:
#
#         count        mean        std   min    25%    50%    75%    max
# Label
# ham    6621.0   64.042441  53.765164   1.0   30.0   48.0   81.0  910.0
# spam   3379.0  164.527375  46.609803  13.0  143.0  157.0  162.0  453.0
# Shape of X_train: (8000, 5000)
# Shape of X_test: (2000, 5000)
# Model Performance:
#  Accuracy: 0.9790
#  Precision: 0.9847
#  Recall: 0.9527
#  F1-Score: 0.9684
#
# Classification Report:
#                precision    recall  f1-score   support
#
#            0       0.98      0.99      0.98      1324
#            1       0.98      0.95      0.97       676
#
#     accuracy                           0.98      2000
#    macro avg       0.98      0.97      0.98      2000
# weighted avg       0.98      0.98      0.98      2000
