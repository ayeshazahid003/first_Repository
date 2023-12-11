# 11-12-23
# CSC461- Assignment4 – NLP
# Ayesha Zahid
# Fa21-BSE-003

#Q1. Compute BoW, TF, IDF, and then TF.IDF values for each term in the following three sentences.
#S1: “data science is one of the most important courses in computer science”
#S2: “this is one of the best data science courses”
#S3: “the data scientists perform data analysis”

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import numpy as np

# Given sentences
sentences = [
    "data science is one of the most important courses in computer science",
    "this is one of the best data science courses",
    "the data scientists perform data analysis"
]

# Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
bow = X.toarray()

# Term Frequency (TF)
tf = np.divide(bow, np.sum(bow, axis=1, keepdims=True))

# Inverse Document Frequency (IDF)
idf = np.log(len(sentences) / np.sum(bow > 0, axis=0))

# TF.IDF
tfidf = tf * idf

# Print results
print("Bag of Words:")
print(vectorizer.get_feature_names_out())
print(bow)
print("\nTerm Frequency (TF):")
print(tf)
print("\nInverse Document Frequency (IDF):")
print(idf)
print("\nTF.IDF:")
print(tfidf)

# Q2: Compute similarity between S1, S2, and S3
cosine_sim = cosine_similarity(tfidf)
manhattan_dist = manhattan_distances(tfidf)
euclidean_dist = euclidean_distances(tfidf)

print("\nCosine Similarity:")
print(cosine_sim)

print("\nManhattan Distance:")
print(manhattan_dist)

print("\nEuclidean Distance:")
print(euclidean_dist)












