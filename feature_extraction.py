# TF-IDF ve N-gram gibi işlemleri yapan modül
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(texts, ngram_range=(1, 2)):
    """
    TF-IDF matrisini döndürür. ngram_range=(1,2) bigram dahil eder.
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
# The above code defines a function to extract TF-IDF features from a list of texts using the TfidfVectorizer from scikit-learn.
# It allows for specifying the n-gram range and limits the number of features to 5000.