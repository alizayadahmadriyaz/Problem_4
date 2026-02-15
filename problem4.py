import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------
# Load dataset
# -------------------------------

def load_data(folder_path):
    texts = []
    labels = []

    for label in ["sport", "politics"]:
        path = os.path.join(folder_path, label)

        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
            except UnicodeDecodeError:
                # Fallback to latin-1 encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    texts.append(f.read())
                    labels.append(label)

    return texts, labels



# -------------------------------
# Main
# -------------------------------

def main():

    print("Loading dataset...")
    texts, labels = load_data("dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("Feature Extraction using TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2)   # unigrams + bigrams
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -------------------------------
    # Model 1: Naive Bayes
    # -------------------------------
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_pred = nb.predict(X_test_vec)

    # -------------------------------
    # Model 2: Logistic Regression
    # -------------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    lr_pred = lr.predict(X_test_vec)

    # -------------------------------
    # Model 3: SVM
    # -------------------------------
    svm = LinearSVC()
    svm.fit(X_train_vec, y_train)
    svm_pred = svm.predict(X_test_vec)

    # -------------------------------
    # Evaluation
    # -------------------------------

    print("\n--- RESULTS ---")

    print("\nNaive Bayes Accuracy:",
          accuracy_score(y_test, nb_pred))

    print("\nLogistic Regression Accuracy:",
          accuracy_score(y_test, lr_pred))

    print("\nSVM Accuracy:",
          accuracy_score(y_test, svm_pred))

    print("\nClassification Report (SVM):")
    print(classification_report(y_test, svm_pred))


if __name__ == "__main__":
    main()