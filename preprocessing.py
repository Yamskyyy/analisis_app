import html
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['full_text']].dropna().drop_duplicates()

    # Cleaning & normalisasi
    df['cleaned'] = df['full_text'].apply(clean_text)

    return df

def clean_text(text):
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) #menghapus mention
    text = re.sub(r'#\w+', '', text) #menghapus hastag
    text = re.sub(r'RT[\s]+', '', text) #menghapus retweet
    text = re.sub(r'https?://\S+', '', text) #menghapus URL
    text = re.sub(r'[^A-Za-z0-9 ]', '', text) #menghapus karakter
    text = re.sub(r'\s+', ' ', text).strip() #menghapus spasi yang lebih dari 1
    return stemming(remove_stopwords(text))

def remove_stopwords(text):
    stop_factory = StopWordRemoverFactory()
    stopwords = stop_factory.create_stop_word_remover()
    return stopwords.remove(text)

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)
