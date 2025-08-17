import pandas as pd
import joblib
import string
import nltk
from nltk.corpus import stopwords
from pathlib import Path

# nltk.download('stopwords')
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))


def preprocess(text: str) -> str:
    """Clean and normalize the input text."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return ' '.join(word for word in tokens if word not in STOPWORDS)

def load_artifacts(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """Load the trained model and vectorizer from disk."""
    if not Path(model_path).exists() or not Path(vectorizer_path).exists():
        raise FileNotFoundError("Model or vectorizer file is missing.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def save_predictions(csv_file_path: str, output_file: str = "predictions.csv"):
    """Predict and save predictions alongside original text."""
    df = pd.read_csv(csv_file_path)

    if 'text' not in df.columns:
        raise ValueError("CSV must contain 'text' column.")

    df['text'] = df['text'].apply(preprocess)

    model, vectorizer = load_artifacts()
    features = vectorizer.transform(df['text'])
    predictions = model.predict(features)

    label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    pred_df = pd.DataFrame(predictions, columns=label_cols)

    result = pd.concat([df['text'], pred_df], axis=1)
    result.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to '{output_file}'")







