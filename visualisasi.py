import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def show_visualizations(results):
    df = results['final_table']
    labels = ['Positif', 'Negatif', 'Netral']

    col1, col2 = st.columns(2)

    with col1:
        st.write("Distribusi Naive Bayes")
        plot_bar(df['nb_pred'])

    with col2:
        st.write("Distribusi Random Forest")
        plot_bar(df['rf_pred'])

    st.write("Perbandingan Akurasi")
    plot_line(results['akurasi'])

def plot_bar(series):
    counts = series.value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_line(akurasi_df):
    plt.plot(akurasi_df['Model'], akurasi_df['Akurasi'], marker='o')
    st.pyplot(plt.gcf())
    plt.clf()
