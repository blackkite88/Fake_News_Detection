# ==========================================================
# Libraries
# ==========================================================
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics


# ==========================================================
# Load Dataset
# ==========================================================
news_data = pd.read_csv("news.csv")
news_data.set_index("Unnamed: 0", inplace=True)

labels = news_data["label"]
texts = news_data["text"]

# ==========================================================
# Train/Test Split
# ==========================================================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.33, random_state=53
)

# ==========================================================
# Vectorization: Count + TF-IDF
# ==========================================================
cv = CountVectorizer(stop_words="english")
X_train_cv = cv.fit_transform(train_texts)
X_test_cv = cv.transform(test_texts)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf = tfidf.transform(test_texts)

# Debug: Check feature differences
cv_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names_out())
tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf.get_feature_names_out())
print(set(cv_df.columns) - set(tfidf_df.columns))
print(cv_df.equals(tfidf_df))


# ==========================================================
# Confusion Matrix Plotter
# ==========================================================
def draw_conf_matrix(matrix, categories, normalize=False, title="Confusion Matrix"):
    plt.imshow(matrix, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_locations = np.arange(len(categories))
    plt.xticks(tick_locations, categories, rotation=45)
    plt.yticks(tick_locations, categories)

    if normalize:
        matrix = matrix.astype(float) / matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")

    threshold = matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ==========================================================
# Naive Bayes with TF-IDF
# ==========================================================
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, train_labels)
preds_tfidf = nb_tfidf.predict(X_test_tfidf)
acc_tfidf = metrics.accuracy_score(test_labels, preds_tfidf)
print(f"TF-IDF NB Accuracy: {acc_tfidf:.3f}")
cm_tfidf = metrics.confusion_matrix(test_labels, preds_tfidf, labels=["FAKE", "REAL"])
draw_conf_matrix(cm_tfidf, ["FAKE", "REAL"])
print(cm_tfidf)

# ==========================================================
# Naive Bayes with Count Vectors
# ==========================================================
nb_cv = MultinomialNB()
nb_cv.fit(X_train_cv, train_labels)
preds_cv = nb_cv.predict(X_test_cv)
acc_cv = metrics.accuracy_score(test_labels, preds_cv)
print(f"CountVec NB Accuracy: {acc_cv:.3f}")
cm_cv = metrics.confusion_matrix(test_labels, preds_cv, labels=["FAKE", "REAL"])
draw_conf_matrix(cm_cv, ["FAKE", "REAL"])
print(cm_cv)

# ==========================================================
# Passive Aggressive Classifier with TF-IDF
# ==========================================================
pa_tfidf = PassiveAggressiveClassifier(max_iter=50)
pa_tfidf.fit(X_train_tfidf, train_labels)
preds_pa = pa_tfidf.predict(X_test_tfidf)
acc_pa = metrics.accuracy_score(test_labels, preds_pa)
print(f"TF-IDF Passive Aggressive Accuracy: {acc_pa:.3f}")
cm_pa = metrics.confusion_matrix(test_labels, preds_pa, labels=["FAKE", "REAL"])
draw_conf_matrix(cm_pa, ["FAKE", "REAL"])
print(cm_pa)

# ==========================================================
# Hyperparameter Tuning for NB (TF-IDF)
# ==========================================================
best_nb = None
best_score = 0
for a in np.arange(0, 1, 0.1):
    model = MultinomialNB(alpha=a)
    model.fit(X_train_tfidf, train_labels)
    score = metrics.accuracy_score(test_labels, model.predict(X_test_tfidf))
    if score > best_score:
        best_score = score
        best_nb = model
    print(f"Alpha={a:.2f} | Accuracy={score:.5f}")

# ==========================================================
# Show Most Informative Features (Passive Aggressive)
# ==========================================================
def top_features(vectorizer, classifier, n=30):
    feat_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    top_fake = sorted(zip(coefs, feat_names))[:n]
    top_real = sorted(zip(coefs, feat_names))[-n:]
    
    print("\nTop FAKE Features:")
    for w, f in top_fake:
        print(f"{f}: {w}")
    print("\nTop REAL Features:")
    for w, f in reversed(top_real):
        print(f"{f}: {w}")

top_features(tfidf, pa_tfidf)

# ==========================================================
# Hashing Vectorizer Models
# ==========================================================
hv = HashingVectorizer(stop_words="english", alternate_sign=False)
X_train_hash = hv.fit_transform(train_texts)
X_test_hash = hv.transform(test_texts)

# Naive Bayes
nb_hash = MultinomialNB(alpha=0.01)
nb_hash.fit(X_train_hash, train_labels)
preds_hash = nb_hash.predict(X_test_hash)
print(f"Hash NB Accuracy: {metrics.accuracy_score(test_labels, preds_hash):.3f}")
draw_conf_matrix(metrics.confusion_matrix(test_labels, preds_hash, labels=["FAKE", "REAL"]), ["FAKE", "REAL"])

# Passive Aggressive
pa_hash = PassiveAggressiveClassifier(max_iter=50)
pa_hash.fit(X_train_hash, train_labels)
preds_hash_pa = pa_hash.predict(X_test_hash)
print(f"Hash PA Accuracy: {metrics.accuracy_score(test_labels, preds_hash_pa):.3f}")
draw_conf_matrix(metrics.confusion_matrix(test_labels, preds_hash_pa, labels=["FAKE", "REAL"]), ["FAKE", "REAL"])
