import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import metrics

df_news = pd.read_csv("news.csv")
df_news.set_index("Unnamed: 0", inplace=True)

y = df_news["label"]
X = df_news["text"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=53)

def add_text_features(text_series):
    df = pd.DataFrame()
    df['word_count'] = text_series.apply(lambda x: len(str(x).split()))
    df['char_count'] = text_series.apply(lambda x: len(str(x)))
    df['avg_word_len'] = df['char_count'] / (df['word_count']+1e-6)
    df['punct_count'] = text_series.apply(lambda x: sum([1 for c in str(x) if c in string.punctuation]))
    return df

text_feat_transformer = FunctionTransformer(add_text_features, validate=False)
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,2))

combined_features = ColumnTransformer([
    ("tfidf", tfidf_vectorizer, 0),
    ("text_features", Pipeline([
        ("extract", text_feat_transformer),
        ("scale", StandardScaler())
    ]), 0)
])

def train_eval(model, X_tr, X_te, y_tr, y_te, name="Model"):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = metrics.accuracy_score(y_te, preds)
    prec = metrics.precision_score(y_te, preds, pos_label="REAL")
    rec = metrics.recall_score(y_te, preds, pos_label="REAL")
    f1 = metrics.f1_score(y_te, preds, pos_label="REAL")
    
    print(f"{name} | Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")
    cm = metrics.confusion_matrix(y_te, preds, labels=["FAKE","REAL"])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["FAKE","REAL"], yticklabels=["FAKE","REAL"], cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"{name} Confusion Matrix")
    plt.show()
    return model

nb_pipeline = Pipeline([
    ("features", combined_features),
    ("nb", MultinomialNB(alpha=0.1))
])

pa_pipeline = Pipeline([
    ("features", combined_features),
    ("pa", PassiveAggressiveClassifier(max_iter=50))
])

train_eval(nb_pipeline, X_train, X_test, y_train, y_test, "TF-IDF + Features NB")
train_eval(pa_pipeline, X_train, X_test, y_train, y_test, "TF-IDF + Features PA")

def show_top_tfidf_features(vectorizer, classifier, n=20):
    if hasattr(classifier, "coef_"):
        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[0]
        top_fake = sorted(zip(coefs, feature_names))[:n]
        top_real = sorted(zip(coefs, feature_names))[-n:]
        print("\nTop FAKE words:")
        for coef, feat in top_fake:
            print(f"{feat}: {coef:.4f}")
        print("\nTop REAL words:")
        for coef, feat in reversed(top_real):
            print(f"{feat}: {coef:.4f}")

tfidf_part = pa_pipeline.named_steps["features"].named_transformers_["tfidf"]
pa_clf = pa_pipeline.named_steps["pa"]

show_top_tfidf_features(tfidf_part, pa_clf)
