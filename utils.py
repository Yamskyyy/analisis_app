def detect_overall_sentiment(df):
    all_preds = df['nb_pred'].value_counts()
    return all_preds.idxmax()
