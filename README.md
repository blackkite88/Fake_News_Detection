# 📰 Fake News Detection

## 📌 Overview

This project implements a **Fake News Detection** pipeline using **Natural Language Processing (NLP)** and **Machine Learning** techniques.
It trains and evaluates multiple models to classify news articles as **FAKE** or **REAL** based on their textual content.

We use different text vectorization techniques—**Count Vectorization**, **TF-IDF**, and **Hashing Vectorization**—combined with classifiers like:

* **Naive Bayes (MultinomialNB)**
* **Passive Aggressive Classifier**

The project also includes **confusion matrix visualization**, **hyperparameter tuning**, and **feature importance extraction**.

---

## 📂 Dataset

* The dataset should be in a CSV format with at least the following columns:

  * `text` → News article text
  * `label` → Either `"FAKE"` or `"REAL"`
* Example: `news.csv` (from **Kaggle’s Fake News dataset** or similar)
* The script assumes the CSV contains an `Unnamed: 0` column (index), which is set as the DataFrame index.

---

## ⚙️ Features

* **Data Preprocessing**

  * Train/Test split (67/33 split)
  * Stopword removal using `stop_words="english"`
* **Vectorization Methods**

  * `CountVectorizer`
  * `TfidfVectorizer`
  * `HashingVectorizer`
* **Machine Learning Models**

  * Multinomial Naive Bayes (with hyperparameter tuning on `alpha`)
  * Passive Aggressive Classifier
* **Model Evaluation**

  * Accuracy score
  * Confusion matrix (with optional normalization)
* **Feature Analysis**

  * Top informative features for FAKE and REAL classification
* **Visualization**

  * Confusion matrix plotted using `matplotlib`

---

## 📊 Results Summary

| Vectorizer | Model                | Accuracy |
| ---------- | -------------------- | -------- |
| TF-IDF     | Naive Bayes          | 0.857    |
| CountVec   | Naive Bayes          | 0.893    |
| TF-IDF     | Passive Aggressive   | 0.937    |
| Hashing    | Naive Bayes (α=0.01) | 0.902    |
| Hashing    | Passive Aggressive   | 0.920    |

> **Best Performance:** Passive Aggressive with TF-IDF (93.7% accuracy)

---

## 📦 Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Create a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the script

```bash
python main.py
```

---

## 📜 Example Output

```
TF-IDF NB Accuracy: 0.857
CountVec NB Accuracy: 0.893
TF-IDF Passive Aggressive Accuracy: 0.937
Alpha=0.10 | Accuracy=0.89766
...
Top FAKE Features:
2016, october, hillary, share, article, ...
Top REAL Features:
said, tuesday, cruz, gop, marriage, ...
```

---

## 📚 Dependencies

* pandas
* numpy
* scikit-learn
* matplotlib

Install them via:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 📈 Future Improvements

* Add **deep learning models** (LSTMs, BERT)
* Perform **cross-validation**
* Incorporate **more preprocessing steps** (lemmatization, stemming)
* Deploy as a **web app** (Flask/Streamlit)

---

## 📜 License

This project is open-source and available under the **MIT License**.

---
