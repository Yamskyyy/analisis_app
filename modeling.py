from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import streamlit as st

# Load tokenizer dan model BERT
model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Fungsi klasifikasi dengan BERT
def label_sentimen_bert(text):
    try:
        if pd.isna(text) or str(text).strip() == "":
            return "Netral"
        result = sentiment_classifier(text)[0]
        label = result['label']
        if label == 'positive':
            return 'Positif'
        elif label == 'negative':
            return 'Negatif'
        else:
            return 'Netral'
    except Exception as e:
        print(f"Error memproses: {text}, error: {e}")
        return 'Netral'

def run_models(df):
    df['text'] = df['cleaned']

    # Label otomatis
    df['klasifikasi'] = df['text'].apply(label_sentimen_bert)

    # CountVectorizer seperti TextBlob
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['klasifikasi']

    # Naive Bayes (alpha default 1.0 mirip TextBlob)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X, y)
    y_pred_nb = nb_model.predict(X)
    acc_nb = accuracy_score(y, y_pred_nb)

    # Random Forest
    rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
    rf_model.fit(X, y)
    y_pred_rf = rf_model.predict(X)
    acc_rf = accuracy_score(y, y_pred_rf)

    # Simpan prediksi
    df['nb_pred'] = y_pred_nb
    df['rf_pred'] = y_pred_rf

    final_table = df[['text', 'klasifikasi', 'nb_pred', 'rf_pred']]

    return {
        'akurasi': pd.DataFrame({
            'Model': ['Naive Bayes', 'Random Forest'],
            'Akurasi': [acc_nb, acc_rf]
        }),
        'final_table': final_table
    }
