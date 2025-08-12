import streamlit as st
import pandas as pd
from crawl import crawl_twitter
from preprocessing import preprocess_data
from modeling import run_models
from visualisasi import show_visualizations
from utils import detect_overall_sentiment

st.set_page_config(page_title="Analisis Sentimen Twitter", layout="wide")

st.title("📊 Aplikasi Analisis Sentimen Twitter 'Kabinet Merah Putih'")
st.markdown("Analisis sentimen terhadap tweet menggunakan model Naive Bayes dan Random Forest.")

# --- Input user untuk crawling ---
with st.expander("1️⃣ Crawl Data Twitter"):
    token = st.text_input("Masukkan Auth Token Twitter", type="password")
    keyword = st.text_input("Kata Kunci Pencarian", "kabinet merah putih prabowo gibran")
    start_date = st.date_input("Tanggal Mulai")
    end_date = st.date_input("Tanggal Akhir")
    limit = st.number_input("Jumlah Tweet", min_value=100, max_value=5000, value=1000)

    if st.button("Mulai Crawl"):
        st.info("Sedang mengambil data...")
        crawl_twitter(token, keyword, start_date, end_date, limit)
        st.success("Selesai crawl dan simpan ke tweets-data/DataPenelitian.csv")
    try:
        df = pd.read_csv("tweets-data/DataPenelitian.csv")
        st.success(f"Jumlah data berhasil di-crawl: {len(df)} tweet")
    except Exception as e:
        st.error(f"Gagal membaca data hasil crawl: {e}")

# --- Preprocessing + Modeling ---
if st.button("🔍 Proses & Analisis"):
    df_cleaned = preprocess_data("tweets-data/DataPenelitian.csv")
    results = run_models(df_cleaned)

    st.subheader("📈 Akurasi Model")
    st.write(results['akurasi'])

    st.subheader("🧠 Hasil Klasifikasi")
    st.write(results['final_table'])

    st.subheader("📊 Visualisasi Hasil Klasifikasi")
    show_visualizations(results)

    overall = detect_overall_sentiment(results['final_table'])
    st.success(f"Hasil klasifikasi keseluruhan: **{overall.upper()}**")