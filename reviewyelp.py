import re
import pickle
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).with_name("dataset_yelp.xlsx")
MODEL_PATH = Path(__file__).with_name("model.pkl")
VECTORIZER_PATH = Path(__file__).with_name("vectorizer.pkl")

_MODEL = None
_VECTORIZER = None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    df = pd.read_excel(DATA_PATH)
    df = df[["Review", "Spam(1) and Not Spam(0)"]].rename(
        columns={"Review": "text", "Spam(1) and Not Spam(0)": "label"}
    )
    df = df.dropna().drop_duplicates()
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["text_length"] = df["cleaned_text"].astype(str).apply(len)
    if len(df) > 30000:
        df = df.sample(25000, random_state=42).reset_index(drop=True)
    return df


def get_dataset_summary():
    df = load_dataset()
    label_counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    spam_count = int(label_counts.get(1, 0))
    non_spam_count = int(label_counts.get(0, 0))
    total_rows = int(len(df))

    label_distribution = [
        {
            "label": "Not Spam",
            "count": non_spam_count,
            "percentage": round((non_spam_count / total_rows) * 100, 1) if total_rows else 0,
        },
        {
            "label": "Spam",
            "count": spam_count,
            "percentage": round((spam_count / total_rows) * 100, 1) if total_rows else 0,
        },
    ]

    text_stats = df["text_length"].describe()

    return {
        "total_rows": total_rows,
        "label_distribution": label_distribution,
        "avg_text_length": round(float(text_stats["mean"]), 1),
        "max_text_length": int(text_stats["max"]),
        "min_text_length": int(text_stats["min"]),
        "preprocessing_steps": [
            "Lowercase normalization",
            "URL and number removal",
            "Punctuation stripping",
            "Whitespace cleanup",
        ],
    }


def train_model():
    global _MODEL, _VECTORIZER

    df = load_dataset()
    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_tfidf, y_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.5,
        solver="liblinear",
    )
    model.fit(X_train_sm, y_train_sm)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    with VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(tfidf, vectorizer_file)

    _MODEL = model
    _VECTORIZER = tfidf
    return model, tfidf


def load_model_artifacts():
    global _MODEL, _VECTORIZER

    if _MODEL is not None and _VECTORIZER is not None:
        return _MODEL, _VECTORIZER

    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        with MODEL_PATH.open("rb") as model_file:
            _MODEL = pickle.load(model_file)
        with VECTORIZER_PATH.open("rb") as vectorizer_file:
            _VECTORIZER = pickle.load(vectorizer_file)
        return _MODEL, _VECTORIZER

    return train_model()


def predict_review(text):
    model, vectorizer = load_model_artifacts()
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return "Not Spam"

    features = vectorizer.transform([cleaned_text])
    prediction = int(model.predict(features)[0])
    return "Spam" if prediction == 1 else "Not Spam"


if __name__ == "__main__":
    train_model()
    print("Model trained and saved.")
