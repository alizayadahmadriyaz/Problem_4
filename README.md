
# 📰 Sports vs Politics Text Classification

## CSL7640 – Natural Language Understanding  
**Instructor:** Anand Mishra  
**Institute:** IIT Jodhpur  

---

## 📌 Project Overview

This project implements a binary text classification system that classifies news articles into two categories:

- 🏆 **Sport**
- 🏛 **Politics**

The goal is to design and compare multiple machine learning techniques using different feature representation methods such as:

- Bag of Words (BoW)
- TF-IDF
- n-grams

The system evaluates and compares at least three machine learning models quantitatively.

---

## 📂 Dataset

The dataset used is derived from the **BBC News Dataset**, obtained from Kaggle.

### Categories Used:
- `sport`
- `politics`

### Dataset Statistics

| Category  | Documents |
|------------|-----------|
| Sport      | ~500      |
| Politics   | ~400      |
| **Total**  | ~900      |

The dataset is relatively balanced, which reduces model bias.

---

## ⚙️ Project Pipeline

The system follows the standard NLP classification workflow:

Raw Text  
↓  
Preprocessing  
↓  
Feature Extraction (TF-IDF / BoW / n-grams)  
↓  
Model Training  
↓  
Evaluation  
↓  
Prediction  

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Lowercasing text
- Removal of punctuation
- Stopword removal
- Tokenization
- Optional bigram generation

---

## 🧠 Feature Engineering

### 1️⃣ Bag of Words (BoW)
- Represents documents as word frequency vectors.
- Ignores word order.
- Simple but effective for topic classification.

### 2️⃣ TF-IDF (Term Frequency–Inverse Document Frequency)
- Assigns weights to words based on importance.
- Reduces influence of common words.
- Used unigrams + bigrams for better contextual representation.

Formula:

TF-IDF = TF × IDF  
IDF = log(N / df)

---

### 3️⃣ n-grams
- Captures sequences of words.
- Helps identify meaningful phrases such as:
  - "prime minister"
  - "world cup"
- Improves contextual understanding.

---

## 🤖 Machine Learning Models Compared

Three supervised ML models were implemented:

### 1️⃣ Multinomial Naive Bayes
- Probabilistic classifier.
- Assumes conditional independence between features.
- Fast and memory efficient.

### 2️⃣ Logistic Regression
- Linear classifier.
- Works well with high-dimensional sparse data.
- Strong baseline for text classification.

### 3️⃣ Support Vector Machine (SVM)
- Maximizes margin between classes.
- Effective in high-dimensional feature spaces.
- Often achieves strong generalization performance.

---

## 📊 Experimental Setup

- Train/Test Split: 80% / 20%
- Feature Representation: TF-IDF (unigram + bigram)
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

## 📈 Results

### Model Accuracy Comparison

| Model               | Accuracy |
|--------------------|----------|
| Naive Bayes        | 1.00     |
| Logistic Regression| 1.00     |
| SVM                | 1.00     |

---

### Detailed Classification Report (SVM)

          precision    recall  f1-score   support

	politics       1.00      1.00      1.00        86
	   sport       1.00      1.00      1.00       100

	accuracy                           1.00       186


---

## 🔎 Discussion

All three models achieved 100% accuracy on the test dataset.

Possible reasons:

- Strong lexical separation between domains.
- Sports vocabulary (e.g., match, team, goal).
- Politics vocabulary (e.g., government, election, policy).
- Clean and well-labeled dataset.

Although impressive, such perfect accuracy may not generalize to noisier real-world data.

---

## ⚠️ Limitations

- Relies only on surface-level lexical patterns.
- Cannot detect sarcasm or deeper semantic meaning.
- Performance may degrade with overlapping topics.
- Dataset is relatively small and clean.

---


