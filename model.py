import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# =========================
# ENCODE LABEL
# =========================
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =========================
# SPLIT DATA
# =========================
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TEXT VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# TRAIN MODEL
# =========================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# SAVE MODEL & VECTORIZER
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
