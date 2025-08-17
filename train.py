
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return ' '.join(word for word in tokens if word not in STOPWORDS)

df = pd.read_csv("my-track-a.csv")
df['text'] = df['text'].apply(preprocess)

X_texts = df['text']
y_labels = df[['anger','fear', 'joy', 'sadness', 'surprise']]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_texts)

X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

base_model = LogisticRegression(max_iter=1000)
model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=y_labels.columns))

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

